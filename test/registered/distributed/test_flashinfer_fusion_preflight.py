"""Real-distributed integration tests for FlashInfer fusion workspace preflight.

Spawns multiple processes on real CUDA devices (no mocks) and exercises
`_preflight_check_workspace_memory` from
`sglang.srt.layers.flashinfer_comm_fusion`:

- happy path: every rank's probe succeeds, vote returns PROCEED
- skewed path: rank 0 pre-pins enough memory that its probe must fail,
  every rank's vote returns SKIP (i.e. failure is *broadcast*)
"""

import multiprocessing as mp
import os
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-2-gpu-large")

WORLD_SIZE = 2


def _run_rank(rank, world_size, scenario, result_q):
    """Worker entrypoint. Each rank runs end-to-end in its own process."""
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        torch.cuda.set_device(rank)

        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
        )
        cpu_group = dist.group.WORLD

        from sglang.srt.layers.flashinfer_comm_fusion import (
            _preflight_check_workspace_memory,
        )

        # Mirror an 8-way TP lamport probe (~6 GiB after the MAX_COMM_SIZE cap
        # x3): big enough that the starvation scenario reliably exhausts the
        # rank-0 device pool, small enough not to OOM a healthy rank.
        probe_kwargs = dict(
            world_size=8,
            max_token_num=2048,
            hidden_dim=12288,
            dtype=torch.bfloat16,
            cpu_group=cpu_group,
        )

        held = None
        if scenario == "rank0_starved" and rank == 0:
            # Pin almost all of GPU 0's memory so the probe's cuMemCreate
            # has nowhere to land. Use cuMemCreate (matches the probe path)
            # rather than torch.empty so we starve the same allocator pool.
            from cuda import cuda as _cu

            free, _total = torch.cuda.mem_get_info(rank)
            # Leave only ~1 GiB so the ~6 GiB lamport probe must fail.
            target = max(free - (1 << 30), 0)

            prop = _cu.CUmemAllocationProp()
            prop.requestedHandleTypes = (
                _cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
            prop.type = _cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.location = _cu.CUmemLocation()
            prop.location.type = _cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            prop.location.id = rank
            prop.allocFlags.gpuDirectRDMACapable = 1

            err, gran = _cu.cuMemGetAllocationGranularity(
                prop,
                _cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
            assert err == _cu.CUresult.CUDA_SUCCESS, err
            aligned = (target // gran) * gran
            err, held = _cu.cuMemCreate(aligned, prop, 0)
            assert err == _cu.CUresult.CUDA_SUCCESS, (err, aligned)

        try:
            decision = _preflight_check_workspace_memory(**probe_kwargs)
        finally:
            if held is not None:
                from cuda import cuda as _cu

                _cu.cuMemRelease(held)

        # The vote must be unanimous. Make every rank report so the parent
        # can assert all of them, not just rank 0.
        result_q.put((rank, "ok", bool(decision)))
    except Exception as e:  # pragma: no cover - debug path
        result_q.put((rank, "err", repr(e)))
    finally:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def _spawn_and_collect(scenario, world_size=WORLD_SIZE, port=29512):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for r in range(world_size):
        p = ctx.Process(
            target=_run_rank,
            args=(r, world_size, scenario, q),
        )
        p.start()
        procs.append(p)

    results = {}
    for _ in range(world_size):
        rank, status, payload = q.get(timeout=300)
        results[rank] = (status, payload)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"rank exited with {p.exitcode}"
    return results


class TestFlashInferPreflightDistributed(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
            raise unittest.SkipTest(
                f"Need {WORLD_SIZE} CUDA devices, got {torch.cuda.device_count()}"
            )
        try:
            from cuda import cuda  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(f"cuda-python not importable: {e}")

    def test_happy_path_votes_proceed(self):
        """Normal probe: every rank's local probe succeeds -> PROCEED."""
        results = _spawn_and_collect("normal")
        for rank, (status, payload) in results.items():
            self.assertEqual(status, "ok", f"rank {rank}: {payload}")
            self.assertTrue(payload, f"rank {rank} voted SKIP unexpectedly")

    def test_starved_rank_broadcasts_skip(self):
        """One starved rank fails its probe; all ranks must agree to SKIP."""
        results = _spawn_and_collect("rank0_starved")
        for rank, (status, payload) in results.items():
            self.assertEqual(status, "ok", f"rank {rank}: {payload}")
            self.assertFalse(
                payload,
                f"rank {rank} voted PROCEED but rank 0 was starved -- "
                "vote did not propagate, this is the original hang bug",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
