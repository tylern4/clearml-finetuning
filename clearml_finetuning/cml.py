"""ClearML integration helpers.

A small, wandb-free logging shim plus task utilities. The run logger reports
scalars per step, renders benchmark breakdown tables, uploads text generations
as debug samples, and emits a multi-panel summary figure on finish().
"""

import os
import time

_TERMINAL = {"completed", "failed", "stopped", "closed", "published"}


def prepare_vllm_nvrtc():
    """Expose the pip-provided CUDA NVRTC builtins to forked vLLM workers.

    vLLM's FP8 "humming" kernel JIT-compiles a weight-repack kernel through
    NVRTC at model load and needs libnvrtc-builtins.so, which the CUDA
    *runtime* docker image does not ship. The nvidia-cuda-nvrtc pip wheel
    provides it under site-packages/nvidia/cu13/lib (or nvidia/cuda_nvrtc/lib);
    prepend any such dir to LD_LIBRARY_PATH before vllm is imported (workers
    are forked, so they inherit this environment).
    """
    import glob
    import site

    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    for sp in roots:
        for libdir in glob.glob(os.path.join(sp, "nvidia", "*", "lib")):
            if not glob.glob(os.path.join(libdir, "libnvrtc-builtins*.so*")):
                continue
            if libdir not in cur.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = libdir + os.pathsep + cur
                cur = os.environ["LD_LIBRARY_PATH"]


class ClearMLRun:
    """Run logger with a minimal surface: log(), report_sample(), finish()."""

    def __init__(self, run_name=None, config=None, project=None,
                 task_type="training", output_uri=None):
        from clearml import Task

        self.task = Task.init(
            project_name=project or os.environ.get("CML_PROJECT")
            or os.environ.get("CLEARML_PROJECT", "clearml-finetuning"),
            task_name=run_name or "run",
            task_type=getattr(Task.TaskTypes, task_type),
            auto_connect_frameworks=False,
            reuse_last_task_id=False,
        )
        if output_uri or os.environ.get("WF_OUTPUT_URI"):
            uri = output_uri or os.environ.get("WF_OUTPUT_URI")
            self.task.output_uri = uri
            # debug samples default to the files server; keep them on the
            # task's output storage (e.g. S3) as well
            try:
                self.task.get_logger().set_default_upload_destination(uri)
            except Exception:
                pass
        if config is not None:
            cfg = config if isinstance(config, dict) else vars(config)
            self.task.connect(cfg, name="config")
        self._logger = self.task.get_logger()
        self._history = {}  # key -> [(step, value), ...]

    def log(self, payloads, step=None, **kwargs):
        """Report a dict of scalar metrics (and nested benchmark dicts)."""
        if step is None:
            step = (payloads or {}).get("step")
        iteration = int(step) if step is not None else 0

        flat = {}

        def flatten(prefix, obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    flatten(f"{prefix}/{key}" if prefix else str(key), value)
            elif isinstance(obj, (bool, int, float)):
                flat[prefix] = float(obj)

        for key, value in (payloads or {}).items():
            flatten(str(key), value)

        for key, value in flat.items():
            title, _, series = key.rpartition("/")
            self._history.setdefault(key, []).append((iteration, value))
            try:
                self._logger.report_scalar(
                    title=title or "metrics", series=series or key,
                    value=value, iteration=iteration,
                )
            except Exception as ex:
                print(f"[clearml] warning: failed reporting scalar {key}: {ex}")

        breakdown = (payloads or {}).get("benchmarks")
        if isinstance(breakdown, dict):
            try:
                self._logger.report_table(
                    title="benchmarks", series=f"step {iteration}",
                    iteration=iteration,
                    table=[["benchmark", "score"]]
                          + [[k, float(v)] for k, v in breakdown.items()],
                )
            except Exception as ex:
                print(f"[clearml] warning: failed reporting benchmark plot: {ex}")

    def report_sample(self, text, step=None, label="generation"):
        """Upload a text generation as a debug sample."""
        from io import BytesIO

        self._logger.report_media(
            title="samples", series=label,
            iteration=int(step) if step is not None else 0,
            stream=BytesIO(str(text).encode("utf-8")),
            file_extension="txt",
        )

    def _render_and_report(self, title, series_name, iteration):
        """Render the metric-panel figure and report it as a plot event."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as ex:
            print(f"[clearml] warning: matplotlib unavailable: {ex}")
            return
        if not self._history:
            return

        def series(key):
            pts = self._history.get(key) or []
            return [p[0] for p in pts], [p[1] for p in pts]

        panels = [
            [("train/loss", "train loss", None)],
            [("train/lr", "lr", None)],
            [("eval/bpb", "eval loss (bpb)", None), ("eval/pass1", "HumanEval pass@1", None)],
            [("train/tok_per_sec", "tok/sec", None), ("train/mfu", "MFU %", "tw")],
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle(title)
        used = set()
        for ax, panel in zip(axes.flat, panels):
            for key, label, mode in panel:
                xs, ys = series(key)
                if not xs:
                    continue
                used.add(id(ax))
                if mode == "tw":
                    ax2 = ax.twinx()
                    ax2.plot(xs, ys, color="tab:orange", alpha=0.85, label=label)
                    ax2.set_ylabel(label, color="tab:orange")
                else:
                    ax.plot(xs, ys, label=label)
            labeled = [l for l in ax.get_lines()
                       if l.get_label() and not l.get_label().startswith("_")]
            if labeled:
                ax.legend(loc="best", fontsize=8)
            ax.set_xlabel("step")
        for ax in axes.flat:  # drop empty panels instead of blank grids
            if id(ax) not in used:
                fig.delaxes(ax)
        if used:
            try:
                self._logger.report_matplotlib_figure(
                    title=title, series=series_name, figure=fig,
                    iteration=iteration,
                )
                # PNG conversion/upload trails the call by seconds; settle
                # so a soon-exiting process can't drop the event
                self.task.flush()
                time.sleep(1)
            except Exception as ex:
                print(f"[clearml] warning: failed reporting plot {title}: {ex}")
        plt.close(fig)

    def report_progress_plot(self, step=None):
        """Mid-run plot refresh; the UI Plots tab sliders across iterations."""
        self._render_and_report("progress", "curves",
                                int(step) if step is not None else 0)

    def report_figure(self, title, fig, step=None):
        """Report a caller-built matplotlib figure (latest-wins per title)."""
        try:
            self._logger.report_matplotlib_figure(
                title=title, series="plot", figure=fig,
                iteration=int(step) if step is not None else 0)
            # settle: conversion/upload trails the call by seconds
            self.task.flush()
            time.sleep(1)
        except Exception as ex:
            print(f"[clearml] warning: failed reporting figure {title}: {ex}")
        finally:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

    def _report_summary_plots(self):
        last_step = max((pts[-1][0] for pts in self._history.values() if pts),
                        default=0)
        self._render_and_report("training_summary", "curves", last_step)

    def finish(self):
        self._report_summary_plots()
        # do not close the task: a wrapping stage runner may still need to
        # upload artifacts after this process exits
        self.task.flush()
        # the summary figure's conversion/upload trails by seconds; give it
        # time so a promptly-exiting process doesn't drop it
        time.sleep(2)


def get_run_logger(run_name=None, config=None, project=None,
                   task_type="training", output_uri=None):
    """Return a ClearMLRun when CLEARML_RUN is set, else None."""
    if os.environ.get("CLEARML_RUN"):
        return ClearMLRun(run_name=run_name, config=config, project=project,
                          task_type=task_type, output_uri=output_uri)
    return None


def download_artifact(task_id, name):
    """Fetch a folder artifact from an upstream task; returns the local path."""
    from clearml import Task

    upstream = Task.get_task(task_id=task_id)
    art = upstream.artifacts.get(name)
    if art is None:
        raise RuntimeError(f"upstream task {task_id} has no artifact '{name}'")
    return art.get()


def wait_for_tasks(task_ids, timeout_s=86400, poll_s=30, on_status=None):
    """Poll tasks until terminal states; returns {task_id: status}."""
    import time

    status = {tid: None for tid in task_ids}
    deadline = time.time() + timeout_s
    pending = set(task_ids)
    while pending and time.time() < deadline:
        from clearml import Task

        for tid in list(pending):
            st = Task.get_task(task_id=tid).status
            status[tid] = st
            if on_status:
                on_status(tid, st)
            if st in _TERMINAL:
                pending.discard(tid)
        if pending:
            time.sleep(poll_s)
    for tid in pending:
        status[tid] = "timeout"
    return status
