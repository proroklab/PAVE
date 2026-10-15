"""Generic parallel executors for eval scenarios.

Two sibling abstractions, both backed by ``multiprocessing.Pool`` with the
same per-worker thread-pinning initializer:

* :class:`EpisodeWorker` + :func:`run_episodes_parallel` — fans episode
  rollouts across processes; each worker loads models and builds the
  controller once, then runs many episodes.

* :class:`FigureWorker` + :func:`run_figures_parallel` — fans figure
  rendering (matplotlib + ffmpeg) across processes; each worker imports
  matplotlib once with the Agg backend, then renders many figures from
  pre-computed trajectory data.

Usage (rollouts)::

    specs = [EpisodeSpec(...), ...]
    results = run_episodes_parallel(
        specs,
        worker_cls=OAEpisodeWorker,
        worker_kwargs={"leader_path": ..., "follower_path": ...},
        max_workers=8,
    )

Usage (figures)::

    tasks = [FigureTask(output_path=..., payload={...}), ...]
    run_figures_parallel(
        tasks,
        worker_cls=OAEpisodeFigureWorker,
        worker_kwargs={"fps": 4, "leader_cfg_dict": ...},
        max_workers=8,
    )
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type


# ---------------------------------------------------------------------------
# Episode specification (serializable across processes)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeSpec:
    """Fully serializable specification for one episode-mode pair.

    All fields must be JSON-friendly (str, int, float, list, None) so that
    the spec can be passed through ``multiprocessing.Pool.map``.
    """

    scenario: str               # "obstacle_arena" | "warehouse" | "mobile_arm" | "household"
    mode: str                   # "unguided" | "value-only" | "full" | "sabotage"
    episode: int                # episode index (for ordering results)
    leader_seed: int
    follower_seeds: List[int] = field(default_factory=list)
    # Per-episode planner RNG seed. Sibling spawn of ``leader_seed`` /
    # ``follower_seeds[0]`` under the master ``SeedManager``'s
    # ``eval_episodes`` purpose root, so it's disjoint from both env-reset
    # streams. Consumed by ``_build_*_controller`` to seed the
    # per-episode ``CEMController._rng``. Paired across modes within one
    # episode — same triple is used for unguided/value-only/value-and-ood/
    # uniform so the only thing that differs across modes is the mode
    # itself (clean paired comparison).
    planner_seed: int = 0
    record: bool = False

    # Model paths (loaded inside worker)
    leader_model_path: str = ""
    follower_model_path: str = ""
    vae_path: Optional[str] = None
    alpha: Optional[float] = None

    # Scenario-specific initial conditions (obstacle_arena, household).
    # household repurposes ``obstacle_pos`` as the table centroid and adds
    # the ``*_theta`` orientations below.
    obstacle_pos: Optional[List[float]] = None
    leader_pos: Optional[List[float]] = None
    follower_pos: Optional[List[float]] = None
    obstacle_theta: Optional[float] = None
    leader_theta: Optional[float] = None
    follower_theta: Optional[float] = None


# ---------------------------------------------------------------------------
# Worker ABC
# ---------------------------------------------------------------------------

class EpisodeWorker(ABC):
    """Base class for scenario-specific episode workers.

    Subclasses implement :meth:`setup` (called once per worker process to
    load models and build the controller) and :meth:`run_episode` (called
    once per episode).

    The ``mode`` is set by the executor before :meth:`setup` is called.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._mode: str = kwargs.get("mode", "unguided")

    @property
    def mode(self) -> str:
        return self._mode

    @abstractmethod
    def setup(self) -> None:
        """Load models, build controller.  Called once per worker process."""

    @abstractmethod
    def run_episode(self, spec: EpisodeSpec) -> dict:
        """Execute one episode and return a metrics dict."""


# ---------------------------------------------------------------------------
# Process-pool machinery
# ---------------------------------------------------------------------------

# Global per-process worker reference, set by the pool initializer.
_worker: Optional[EpisodeWorker] = None


def _pool_initializer(worker_cls: Type[EpisodeWorker], worker_kwargs: dict) -> None:
    """Initializer for each pool worker process.

    Pins this worker to a single thread for PyTorch / numpy / BLAS to avoid
    oversubscription: with N workers × all-cores per worker the threads
    thrash. Per-step planner forward passes are tiny (single-obs networks),
    so single-thread per worker is much faster.

    Note: ``OMP_NUM_THREADS`` and friends are read at library load time, so
    setting them here only takes effect for libraries that haven't been
    imported yet in the spawned worker. ``torch.set_num_threads`` is the
    runtime knob that reliably caps torch's intra-op parallelism.
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch as _th
    except ImportError:
        # torch not installed — workers that don't use it can still proceed.
        pass
    else:
        try:
            _th.set_num_threads(1)
            _th.set_num_interop_threads(1)
        except RuntimeError as exc:
            # set_num_interop_threads must be called before any parallel work;
            # if torch already started its dispatcher, this raises. Log and
            # continue — set_num_threads alone still helps.
            print(
                "warning: could not pin torch threads ({exc}); worker will "
                "use default thread count.".format(exc=exc),
                file=sys.stderr,
            )

    global _worker
    _worker = worker_cls(**worker_kwargs)
    _worker.setup()


def _pool_run_episode(spec_dict: dict) -> dict:
    """Map function: deserialize spec, run episode, return result dict."""
    assert _worker is not None, "Worker not initialized (pool initializer not called)"
    spec = EpisodeSpec(**spec_dict)
    result = _worker.run_episode(spec)
    result["episode"] = spec.episode
    result["mode"] = spec.mode
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_episodes_parallel(
    specs: List[EpisodeSpec],
    worker_cls: Type[EpisodeWorker],
    worker_kwargs: Dict[str, Any],
    max_workers: Optional[int] = None,
) -> List[dict]:
    """Execute episode specs in parallel using a process pool.

    Groups specs by mode.  For each mode a fresh pool is created so that
    every worker in the pool builds the controller for that specific mode
    once, then processes its share of episodes sequentially.

    Args:
        specs: Episode specifications.
        worker_cls: ``EpisodeWorker`` subclass for the scenario.
        worker_kwargs: Keyword arguments forwarded to ``worker_cls.__init__``
            (should NOT include ``mode`` — it is set per-group).
        max_workers: Maximum parallel workers.  Defaults to ``os.cpu_count()``.

    Returns:
        List of result dicts (one per spec), in no guaranteed order.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    # Sequential fallback
    if max_workers <= 1:
        return _run_sequential(specs, worker_cls, worker_kwargs)

    # Group by mode so each pool builds a controller once per mode.
    by_mode: Dict[str, List[EpisodeSpec]] = defaultdict(list)
    for spec in specs:
        by_mode[spec.mode].append(spec)

    from tqdm import tqdm

    all_results: List[dict] = []
    pbar = tqdm(
        total=len(specs),
        desc="Episodes",
        file=sys.stderr,
        unit="ep",
    )
    try:
        for mode, group_specs in by_mode.items():
            n_workers = min(max_workers, len(group_specs))
            kwargs = {**worker_kwargs, "mode": mode}

            pbar.set_postfix_str(
                "{label} ({w}w)".format(label=mode, w=n_workers),
                refresh=True,
            )

            spec_dicts = [asdict(s) for s in group_specs]

            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                processes=n_workers,
                initializer=_pool_initializer,
                initargs=(worker_cls, kwargs),
            ) as pool:
                for result in pool.imap_unordered(_pool_run_episode, spec_dicts):
                    all_results.append(result)
                    pbar.update(1)
    finally:
        pbar.close()

    return all_results


def _run_sequential(
    specs: List[EpisodeSpec],
    worker_cls: Type[EpisodeWorker],
    worker_kwargs: Dict[str, Any],
) -> List[dict]:
    """Sequential fallback (``max_workers=1``).  Useful for debugging."""

    by_mode: Dict[str, List[EpisodeSpec]] = defaultdict(list)
    for spec in specs:
        by_mode[spec.mode].append(spec)

    from tqdm import tqdm

    all_results: List[dict] = []
    pbar = tqdm(
        total=len(specs),
        desc="Episodes",
        file=sys.stderr,
        unit="ep",
    )
    try:
        for mode, group_specs in by_mode.items():
            kwargs = {**worker_kwargs, "mode": mode}

            pbar.set_postfix_str(mode, refresh=True)

            worker = worker_cls(**kwargs)
            worker.setup()
            for spec in group_specs:
                result = worker.run_episode(spec)
                result["episode"] = spec.episode
                result["mode"] = spec.mode
                all_results.append(result)
                pbar.update(1)
    finally:
        pbar.close()

    return all_results


# ---------------------------------------------------------------------------
# Figure rendering parallelism
# ---------------------------------------------------------------------------
#
# Same machinery as the episode pool above, with a different worker shape:
#
# * :class:`FigureWorker` — :meth:`setup` loads scenario configs / sets the
#   matplotlib Agg backend once per worker; :meth:`render` consumes one
#   :class:`FigureTask` and writes a single output file (``.mp4`` / ``.png``).
# * :func:`run_figures_parallel` — best-effort: each task is wrapped in a
#   try/except inside the worker; failures are collected and reported at the
#   end. Raises :class:`FigureRenderError` if **any** task failed so the
#   caller still exits non-zero (CI signal preserved). Successful tasks are
#   not lost on a failure of one task.

class FigureRenderError(RuntimeError):
    """Raised by :func:`run_figures_parallel` when one or more figure tasks
    failed. Carries ``failures: list[(task, traceback_str)]`` so callers can
    introspect what went wrong without parsing log output."""

    def __init__(self, failures: List["_FigureFailure"]) -> None:
        self.failures = failures
        n = len(failures)
        first = failures[0].traceback_str if failures else ""
        super().__init__(
            "{n} figure render task(s) failed.\n\nFirst failure:\n{tb}".format(
                n=n, tb=first,
            )
        )


@dataclass
class FigureTask:
    """One figure-rendering job for :func:`run_figures_parallel`.

    ``output_path`` is shown in error messages and used to dedupe progress
    output. ``payload`` is whatever the worker's :meth:`FigureWorker.render`
    expects — typically a dict of per-episode trajectory arrays + scalars.
    All fields must be picklable (numpy arrays are; ``Path`` is too).
    """

    output_path: Path
    payload: Any = None


@dataclass
class _FigureFailure:
    """Internal: one failed task + its traceback string (str rather than
    raw exception so it survives pickling back from the worker)."""

    output_path: Path
    traceback_str: str


class FigureWorker(ABC):
    """Base class for scenario-specific figure renderers.

    Subclasses implement:

    * :meth:`setup` — called once per worker process. Set the matplotlib
      backend (``matplotlib.use("Agg")``), load any scenario configs, etc.
    * :meth:`render` — called once per task. Build the figure from the
      task's payload, save to ``task.output_path``, close the figure.

    Workers receive **only data** (the task payload). Anything that requires
    the live policy / env state must be precomputed in the main process and
    baked into the payload before the pool runs (see Q4 of the design
    grilling: workers don't import torch or load PPO models).
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    @abstractmethod
    def setup(self) -> None:
        """One-time per-process initialization (matplotlib backend, configs)."""

    @abstractmethod
    def render(self, task: FigureTask) -> None:
        """Render one figure to ``task.output_path``."""


# Per-process figure-worker reference, set by the figure pool initializer.
_figure_worker: Optional[FigureWorker] = None


def _figure_pool_initializer(
    worker_cls: Type[FigureWorker], worker_kwargs: dict,
) -> None:
    """Initializer for figure-pool workers.

    Pins thread libraries (mirroring :func:`_pool_initializer`) — even
    though the figure phase doesn't import torch, having a consistent
    pinning keeps NumPy / OpenBLAS used by matplotlib from oversubscribing
    when the rollout pool already pre-warmed those libraries.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    global _figure_worker
    _figure_worker = worker_cls(**worker_kwargs)
    _figure_worker.setup()


def _pool_render_figure(task: FigureTask) -> Optional[_FigureFailure]:
    """Map function: render one task; return ``None`` on success or a
    :class:`_FigureFailure` describing the error.

    Errors are caught here (best-effort policy) so a single bad task doesn't
    kill the pool. The traceback is stringified so it can be pickled back to
    the parent without dragging the original exception's traceback object
    (which doesn't pickle).
    """
    assert _figure_worker is not None, (
        "Figure worker not initialized (pool initializer not called)"
    )
    try:
        _figure_worker.render(task)
        return None
    except Exception:
        return _FigureFailure(
            output_path=task.output_path,
            traceback_str=traceback.format_exc(),
        )


def run_figures_parallel(
    tasks: List[FigureTask],
    worker_cls: Type[FigureWorker],
    worker_kwargs: Dict[str, Any],
    max_workers: Optional[int] = None,
    progress_desc: str = "Figures",
) -> List[Path]:
    """Render figure tasks in parallel using a process pool.

    Best-effort: on per-task failure the helper records the traceback and
    continues. After all tasks complete it raises :class:`FigureRenderError`
    if **any** failed (so the eval CLI still exits non-zero on bug regressions
    or environment issues like missing ffmpeg). Successful outputs are
    written to disk regardless of failures elsewhere in the batch.

    ``max_workers <= 1`` runs in-process (clean stack traces for debugging).

    Args:
        tasks: list of :class:`FigureTask` payloads, one per output file.
        worker_cls: :class:`FigureWorker` subclass.
        worker_kwargs: forwarded to ``worker_cls.__init__``.
        max_workers: pool size. Defaults to ``os.cpu_count()``.
        progress_desc: tqdm bar label.

    Returns:
        List of output paths that were successfully written.

    Raises:
        FigureRenderError: if one or more tasks failed (after rendering all
            successful ones).
    """
    if not tasks:
        return []

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    n_workers = min(max_workers, len(tasks))

    # Sequential fallback (debugging).
    if n_workers <= 1:
        return _run_figures_sequential(
            tasks, worker_cls, worker_kwargs, progress_desc=progress_desc,
        )

    from tqdm import tqdm

    successful: List[Path] = []
    failures: List[_FigureFailure] = []
    pbar = tqdm(
        total=len(tasks),
        desc=progress_desc,
        file=sys.stderr,
        unit="fig",
    )
    pbar.set_postfix_str("({w}w)".format(w=n_workers), refresh=True)
    try:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_figure_pool_initializer,
            initargs=(worker_cls, worker_kwargs),
        ) as pool:
            # imap_unordered returns results in completion order; each result
            # is None on success or a _FigureFailure carrying its own
            # output_path. We rebuild the set of failed paths after the loop.
            for failure in pool.imap_unordered(_pool_render_figure, tasks):
                if failure is not None:
                    failures.append(failure)
                pbar.update(1)
    finally:
        pbar.close()

    # Reconstruct successful list: every task minus those that failed.
    failed_paths = {f.output_path for f in failures}
    successful = [t.output_path for t in tasks if t.output_path not in failed_paths]

    if failures:
        # Print compact summary to stderr; full tracebacks live on the
        # exception so the caller can dump them if needed.
        print(
            "[run_figures_parallel] {ok}/{tot} figures saved; {bad} failed:".format(
                ok=len(successful), tot=len(tasks), bad=len(failures),
            ),
            file=sys.stderr,
        )
        for f in failures:
            # Just the first line of the traceback to keep stderr readable.
            first_line = (
                f.traceback_str.strip().splitlines()[-1]
                if f.traceback_str.strip()
                else "<no traceback>"
            )
            print(
                "  - {p}: {l}".format(p=f.output_path, l=first_line),
                file=sys.stderr,
            )
        raise FigureRenderError(failures)

    return successful


def _run_figures_sequential(
    tasks: List[FigureTask],
    worker_cls: Type[FigureWorker],
    worker_kwargs: Dict[str, Any],
    progress_desc: str = "Figures",
) -> List[Path]:
    """In-process fallback (``max_workers<=1``). Same best-effort contract."""
    from tqdm import tqdm

    worker = worker_cls(**worker_kwargs)
    worker.setup()

    successful: List[Path] = []
    failures: List[_FigureFailure] = []
    pbar = tqdm(
        total=len(tasks),
        desc=progress_desc,
        file=sys.stderr,
        unit="fig",
    )
    try:
        for task in tasks:
            try:
                worker.render(task)
                successful.append(task.output_path)
            except Exception:
                failures.append(
                    _FigureFailure(
                        output_path=task.output_path,
                        traceback_str=traceback.format_exc(),
                    )
                )
            pbar.update(1)
    finally:
        pbar.close()

    if failures:
        print(
            "[run_figures_parallel] {ok}/{tot} figures saved; {bad} failed:".format(
                ok=len(successful), tot=len(tasks), bad=len(failures),
            ),
            file=sys.stderr,
        )
        for f in failures:
            first_line = (
                f.traceback_str.strip().splitlines()[-1]
                if f.traceback_str.strip()
                else "<no traceback>"
            )
            print(
                "  - {p}: {l}".format(p=f.output_path, l=first_line),
                file=sys.stderr,
            )
        raise FigureRenderError(failures)

    return successful


__all__ = [
    "EpisodeSpec",
    "EpisodeWorker",
    "FigureRenderError",
    "FigureTask",
    "FigureWorker",
    "run_episodes_parallel",
    "run_figures_parallel",
]
