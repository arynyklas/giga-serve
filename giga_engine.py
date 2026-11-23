"""Token-budget batcher + GPU loop for GigaEmbedModel.

One engine thread owns the GPU. Requests (`Job`, one per HTTP request, many sequences) queue
FIFO; each step takes sequences off the head until `max_tokens` / `max_seqs` would be exceeded
(a request may span steps), runs ONE forward, and hands the pooled vectors back. Steps up to
`graph_max_tokens` replay a CUDA graph captured for the (padded token count, max length) bucket;
bigger steps run eagerly, where the GPU time hides the kernel launches.

Normally one step is in flight: while it runs the queue fills and the next step is bigger. When
the queue already holds a full next step, that step is prepared and launched behind the running
one (two staging slots), so a saturated GPU does not idle between steps.

Any exception in the engine thread is fatal: every waiting request fails and `on_fatal` is
called (the server exits so a supervisor can restart it), as vLLM does when its EngineCore dies.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import TYPE_CHECKING, Callable, Literal, Protocol

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from giga_model import GigaEmbedModel

# Token buckets for graph capture; a step is padded up to the next one.
T_BUCKETS = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
# Max-sequence-length buckets: FA3's max_seqlen is baked into a graph and must be >= the real one.
L_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

TokenIds = npt.NDArray[np.int32]
Vectors = npt.NDArray[np.float32]
StepMode = Literal["graph", "eager"]


class EngineMetrics(Protocol):
    """What the engine reports; giga_serve.Metrics implements it."""

    def step(self, tokens: int, seconds: float, mode: StepMode) -> None: ...

    def aborted(self, n: int) -> None: ...


class Job:
    """One HTTP request: `seqs` token arrays in, `out` (normalised) and `raw` vectors back."""

    __slots__ = ("seqs", "out", "raw", "next", "pending", "cancelled", "done", "t_arrive", "t_start", "ntok")

    def __init__(self, seqs: list[TokenIds], dim: int, done: DoneCallback) -> None:
        self.seqs = seqs
        self.out: Vectors = np.empty((len(seqs), dim), dtype=np.float32)
        self.raw: Vectors = np.empty((len(seqs), dim), dtype=np.float32)
        self.next = 0
        self.pending = len(seqs)
        self.cancelled = False
        self.done = done  # called from the engine thread: done(job, error_or_None)
        self.t_arrive = time.monotonic()
        self.t_start: float | None = None
        self.ntok = int(sum(len(s) for s in seqs))


DoneCallback = Callable[[Job, BaseException | None], None]
StepItem = tuple[Job, int]  # (job, index of the sequence in job.seqs)


class EngineDead(RuntimeError):
    pass


class Slot:
    """Pinned host staging for one in-flight step."""

    def __init__(self, max_tokens: int, max_seqs: int, dim: int) -> None:
        self.h64 = torch.zeros(3 * max_tokens, dtype=torch.int64, pin_memory=True)  # ids | pos | seg
        self.h32 = torch.zeros(max_seqs + 1, dtype=torch.int32, pin_memory=True)  # cu_seqlens
        self.hout = torch.zeros(2, max_seqs, dim, dtype=torch.float32, pin_memory=True)  # raw, normalised
        self.done = torch.cuda.Event(blocking=True)  # type: ignore[no-untyped-call]
        self.items: list[StepItem] | None = None
        self.tot = 0
        self.mode: StepMode = "graph"
        self.t0 = 0.0


class Engine:
    def __init__(self, model: GigaEmbedModel, *, max_tokens: int, max_seqs: int, max_len: int,
                 graph_max_tokens: int, metrics: EngineMetrics,
                 on_fatal: Callable[[BaseException], None]) -> None:
        self.m = model
        self.dim = model.cfg.hidden
        self.max_tokens, self.max_seqs, self.max_len = max_tokens, max_seqs, max_len
        if max_len > max_tokens:
            raise ValueError("max_len must not exceed max_tokens: one sequence must fit one step")
        self.metrics = metrics
        self.on_fatal = on_fatal
        self.cv = threading.Condition()
        self.queue: collections.deque[Job] = collections.deque()
        self.queued_tokens = 0
        self.queued_seqs = 0
        self.running_seqs = 0
        self.stopping = False
        self.dead: BaseException | None = None
        self.dev = model.device
        self.thread = threading.Thread(target=self._run, name="giga-engine", daemon=True)
        self.slots = [Slot(max_tokens, max_seqs, self.dim) for _ in range(2)]
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.graph_max_tokens = graph_max_tokens
        self.t_buckets = [t for t in T_BUCKETS if t < graph_max_tokens] + [graph_max_tokens] if graph_max_tokens else []
        self.l_buckets = [l for l in L_BUCKETS if l < max_len] + [max_len]
        if self.t_buckets:
            Tg, Bg = graph_max_tokens, min(max_seqs, graph_max_tokens)
            self.g64 = torch.zeros(3 * Tg, dtype=torch.int64, device=self.dev)
            self.g32 = torch.zeros(Bg + 1, dtype=torch.int32, device=self.dev)
            self.g_attn = torch.zeros(Tg, model.cfg.heads, model.cfg.vdim, dtype=model.dtype, device=self.dev)
            self.g_out = torch.zeros(2, Bg + 1, self.dim, dtype=torch.float32, device=self.dev)
            self.pool = torch.cuda.graph_pool_handle()

    # ---- startup ----------------------------------------------------------------------------
    def _graph_body(self, Tp: int, Lp: int) -> None:
        Tg = self.graph_max_tokens
        Bp = min(self.max_seqs, Tp)
        ids, pos, seg = self.g64[:Tp], self.g64[Tg:Tg + Tp], self.g64[2 * Tg:2 * Tg + Tp]
        cu = self.g32[:Bp + 1]
        h = self.m.hidden(ids, pos, cu, Lp, attn_out=self.g_attn[:Tp])
        lens_f = (cu[1:] - cu[:-1]).float().clamp_(min=1.0)
        raw, out = self.m.pool(h, seg, lens_f, Bp + 1)
        self.g_out[0, :Bp + 1].copy_(raw)
        self.g_out[1, :Bp + 1].copy_(out)

    def capture(self) -> tuple[int, float]:
        """One graph per (token bucket, length bucket) a step can map to."""
        t0 = time.time()
        for Tp in self.t_buckets:
            for Lp in self.l_buckets:
                if Lp > Tp and Lp != self.l_buckets[0]:
                    continue
                n = min(Lp, Tp)
                lens = ([n] * (Tp // n) + ([Tp % n] if Tp % n else []))[: min(self.max_seqs, Tp)]
                self._stage_graph(self.slots[0], [np.full(k, 5, dtype=np.int32) for k in lens], Tp)
                for _ in range(2):  # pick the Triton/cuBLAS kernels for these shapes outside capture
                    self._graph_body(Tp, Lp)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self._graph_body(Tp, Lp)
                torch.cuda.synchronize()
                self.graphs[(Tp, Lp)] = g
        return len(self.graphs), time.time() - t0

    def warmup_eager(self) -> tuple[int, float]:
        """Compile/autotune the eager path for every step size above the graph range."""
        t0 = time.time()
        sizes = sorted({t for t in T_BUCKETS + (12288, 16384, 24576, 32768, self.max_tokens)
                        if self.graph_max_tokens < t <= self.max_tokens})
        slot = self.slots[0]
        for T in sizes:
            n = min(self.max_len, 64)
            lens = ([n] * (T // n) + ([T % n] if T % n else []))[: self.max_seqs]
            self._launch_eager(slot, [np.full(k, 5, dtype=np.int32) for k in lens])
        torch.cuda.synchronize()
        return len(sizes), time.time() - t0

    def start(self) -> None:
        self.thread.start()

    # ---- request side -----------------------------------------------------------------------
    def submit(self, job: Job) -> None:
        with self.cv:
            if self.dead is not None:
                raise EngineDead(str(self.dead))
            self.queue.append(job)
            self.queued_tokens += job.ntok
            self.queued_seqs += len(job.seqs)
            self.cv.notify()

    def cancel(self, job: Job) -> None:
        """Client went away: sequences not yet on the GPU are dropped."""
        with self.cv:
            if job.cancelled or job.pending == 0:
                return
            job.cancelled = True
            left = job.seqs[job.next:]
            self.queued_tokens -= int(sum(len(s) for s in left))
            self.queued_seqs -= len(left)
            job.pending -= len(left)
            job.next = len(job.seqs)
            self.metrics.aborted(len(left))

    def stop(self) -> None:
        with self.cv:
            self.stopping = True
            self.cv.notify()
        self.thread.join(timeout=30)

    # ---- engine thread ----------------------------------------------------------------------
    def _form(self) -> tuple[list[StepItem], int]:
        """Lock held. Pops (job, index) items off the queue head within the step budget."""
        items: list[StepItem] = []
        tot = 0
        while self.queue and len(items) < self.max_seqs:
            job = self.queue[0]
            while job.next < len(job.seqs) and len(items) < self.max_seqs:
                n = len(job.seqs[job.next])
                if items and tot + n > self.max_tokens:
                    break
                if job.t_start is None:
                    job.t_start = time.monotonic()
                items.append((job, job.next))
                tot += n
                job.next += 1
            if job.next >= len(job.seqs):
                self.queue.popleft()
            else:
                break
        self.queued_tokens -= tot
        self.queued_seqs -= len(items)
        return items, tot

    def _launch_eager(self, slot: Slot, seqs: list[TokenIds]) -> None:
        """Exact-size forward (no padding) into slot.hout; asynchronous."""
        B = len(seqs)
        lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=B)
        tot = int(lens.sum())
        h = slot.h64.numpy()
        h[:tot] = np.concatenate(seqs)
        seg = np.repeat(np.arange(B), lens)
        cu = slot.h32.numpy()
        cu[0] = 0
        np.cumsum(lens, out=cu[1:B + 1])
        h[tot:2 * tot] = np.arange(tot) - cu[:B][seg]
        h[2 * tot:3 * tot] = seg
        g64 = slot.h64[:3 * tot].to(self.dev, non_blocking=True)
        g32 = slot.h32[:B + 1].to(self.dev, non_blocking=True)
        lens_f = (g32[1:] - g32[:-1]).float()
        hid = self.m.hidden(g64[:tot], g64[tot:2 * tot], g32, int(lens.max()))
        raw, out = self.m.pool(hid, g64[2 * tot:3 * tot], lens_f, B)
        slot.hout[0, :B].copy_(raw, non_blocking=True)
        slot.hout[1, :B].copy_(out, non_blocking=True)

    def _stage_graph(self, slot: Slot, seqs: list[TokenIds], Tp: int) -> None:
        """Fills the static graph inputs for a step padded to Tp tokens (H2D on the stream)."""
        Tg = self.graph_max_tokens
        B, Bp = len(seqs), min(self.max_seqs, Tp)
        lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=B)
        tot = int(lens.sum())
        h = slot.h64.numpy()
        ids, pos, seg = h[:Tp], h[Tg:Tg + Tp], h[2 * Tg:2 * Tg + Tp]
        ids[:tot] = np.concatenate(seqs)
        ids[tot:] = 0
        seg[:tot] = np.repeat(np.arange(B), lens)
        seg[tot:] = Bp  # padding tokens pool into the scratch row
        cu = slot.h32.numpy()
        cu[0] = 0
        np.cumsum(lens, out=cu[1:B + 1])
        cu[B + 1:Bp + 1] = tot  # empty sequences: padding is in no attention segment
        pos[:tot] = np.arange(tot) - cu[:B][seg[:tot]]
        pos[tot:] = 0
        # ids | pos | seg sit at the same offsets in the pinned slot and in g64: one H2D copy for all three
        self.g64[:2 * Tg + Tp].copy_(slot.h64[:2 * Tg + Tp], non_blocking=True)
        self.g32[:Bp + 1].copy_(slot.h32[:Bp + 1], non_blocking=True)

    def _launch(self, slot: Slot, items: list[StepItem], tot: int) -> None:
        seqs = [job.seqs[i] for job, i in items]
        slot.items, slot.tot, slot.t0 = items, tot, time.perf_counter()
        if tot <= self.graph_max_tokens:
            maxlen = max(len(s) for s in seqs)
            Tp = next(t for t in self.t_buckets if t >= tot)
            Lp = next(l for l in self.l_buckets if l >= maxlen)
            if (Tp, Lp) not in self.graphs:  # only when Lp > Tp: take the smallest bucket that holds Lp
                Tp = next(t for t in self.t_buckets if t >= Lp)
            self._stage_graph(slot, seqs, Tp)
            self.graphs[(Tp, Lp)].replay()
            slot.hout[:, :len(seqs)].copy_(self.g_out[:, :len(seqs)], non_blocking=True)
            slot.mode = "graph"
        else:
            self._launch_eager(slot, seqs)
            slot.mode = "eager"
        slot.done.record()

    def _finish(self, slot: Slot) -> None:
        slot.done.synchronize()
        self.metrics.step(slot.tot, time.perf_counter() - slot.t0, slot.mode)
        items = slot.items
        assert items is not None
        B = len(items)
        raw, out = slot.hout[0, :B].numpy(), slot.hout[1, :B].numpy()
        finished: list[Job] = []
        with self.cv:
            for k, (job, i) in enumerate(items):
                if job.cancelled:
                    continue
                job.out[i] = out[k]
                job.raw[i] = raw[k]
                job.pending -= 1
                if job.pending == 0:
                    finished.append(job)
            self.running_seqs -= B
        slot.items = None
        for job in finished:
            job.done(job, None)

    def _run(self) -> None:
        torch.cuda.set_device(self.dev)
        inflight: collections.deque[Slot] = collections.deque()
        free: collections.deque[Slot] = collections.deque(self.slots)
        try:
            while True:
                items: list[StepItem] | None = None
                tot = 0
                with self.cv:
                    while not self.queue and not inflight and not self.stopping:
                        self.cv.wait()
                    if self.stopping and not self.queue and not inflight:
                        return
                    # a second step goes behind the running one only if it would be full anyway
                    full = self.queued_tokens >= self.max_tokens or self.queued_seqs >= self.max_seqs
                    if self.queue and free and (not inflight or full):
                        items, tot = self._form()
                        self.running_seqs += len(items)
                if items:
                    slot = free.popleft()
                    self._launch(slot, items, tot)
                    inflight.append(slot)
                    continue
                if inflight:
                    slot = inflight.popleft()
                    self._finish(slot)
                    free.append(slot)
        except BaseException as e:  # noqa: BLE001 -- any failure here leaves the GPU state unknown
            self._die(e, [it for s in self.slots if s.items for it in s.items])

    def _die(self, e: BaseException, inflight: list[StepItem]) -> None:
        import traceback

        traceback.print_exc()
        with self.cv:
            self.dead = e
            jobs: dict[int, Job] = {id(j): j for j, _ in inflight}
            jobs.update((id(j), j) for j in self.queue)
            self.queue.clear()
        for job in jobs.values():
            if not job.cancelled and job.pending > 0:
                job.pending = 0
                job.done(job, e)
        self.on_fatal(e)
