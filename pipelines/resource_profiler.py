"""
A lightwight profiler for local computational resources.

Exported as `profiler`. Enabled with PROFILE_RESOURCES=1.
"""

import atexit
import logging
import os
import platform
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

MIB = 1024**2

ENABLED = os.environ.get("PROFILE_RESOURCES", "") not in (
    "",
    "0",
    "false",
    "False",
)
INTERVAL_S = float(os.environ.get("PROFILE_RESOURCES_INTERVAL", "0.05"))


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass

    return platform.processor() or platform.machine() or "unknown CPU"


class Sample(NamedTuple):
    """
    One sample of system state.
    """

    cpu: float  # Percentage, summed over cores, so 100% is one saturated core.
    rss: int  # Bytes. This process's resident set size.
    gpu_util: int  # Percentage.
    gpu_used: int  # Bytes. Includes memory used by other processes.
    torch_alloc: int  # Bytes.
    torch_reserved: int  # Bytes.


@dataclass
class Span:
    """
    A time-range and associated samples.
    """

    name: str
    on_gpu: bool = False
    t_start: float = 0.0
    t_end: float = 0.0
    samples: list[Sample] = field(default_factory=list)

    @property
    def wall_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def mean_cpu(self) -> float:
        if not self.samples:
            return float("nan")

        return sum(s.cpu for s in self.samples) / len(self.samples)

    @property
    def peak_rss(self) -> int:
        return max((s.rss for s in self.samples), default=0)

    @property
    def mean_gpu_util(self) -> float:
        if not self.samples:
            return float("nan")

        return sum(s.gpu_util for s in self.samples) / len(self.samples)

    @property
    def peak_torch_alloc(self) -> int:
        return max((s.torch_alloc for s in self.samples), default=0)

    def summary(self) -> str:
        if not self.samples:
            return f"{self.name}: {self.wall_s * 1e3:,.0f} ms (no GPU samples)"

        return (
            f"{self.name}: {self.wall_s * 1e3:,.0f} ms, "
            f"CPU {self.mean_cpu:.0f}%, "
            f"GPU {self.mean_gpu_util:.0f}% mean GPU util, "
            f"{self.peak_torch_alloc / MIB:,.0f} MiB peak torch alloc"
        )


class ResourceProfiler:
    """
    Samples system state on a background thread, per-user defined phase.
    """

    def __init__(self, device_index: int = 0, interval_s: float = INTERVAL_S):
        self.enabled = ENABLED
        self._interval_s = interval_s

        self._lock = threading.Lock()
        self._stack: list[Span] = []
        self._spans: list[Span] = []
        self._samples: list[Sample] = []

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start = self._t_end = 0.0
        self._baseline_gpu_util = float("nan")
        self.gpu = False

        if not self.enabled:
            return

        try:
            import psutil
        except ImportError as exc:
            self._disable(f"missing dependency ({exc})")
            return

        self._process = psutil.Process()
        self._process.cpu_percent()  # Prime; the first call always returns 0.0.
        self._cpu_count = psutil.cpu_count()
        self._physical_cpu_count = psutil.cpu_count(logical=False)
        self._total_ram = psutil.virtual_memory().total

        # GPU is optional.
        try:
            import pynvml
            import torch
        except ImportError as exc:
            logger.warning(
                f"GPU statistics disabled: missing dependency ({exc})"
            )
            return

        if not torch.cuda.is_available():
            logger.warning("GPU statistics disabled: CUDA is unavailable")
            return

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except pynvml.NVMLError as exc:
            logger.warning(f"GPU statistics disabled: NVML unavailable ({exc})")
            return

        self._pynvml, self._torch = pynvml, torch
        self.gpu = True

    def _measure_baseline_gpu_util(self, n: int = 8) -> float:
        readings = []
        for _ in range(n):
            readings.append(self._read().gpu_util)
            time.sleep(self._interval_s)

        return sum(readings) / len(readings)

    def _disable(self, reason: str) -> None:
        """
        Disable potential usage of the profiler.
        """

        logger.warning(f"Resource profiling disabled: {reason}")
        self.enabled = False

    def start(self) -> None:
        """
        Start the profiler.
        """

        if not self.enabled or self._thread is not None:
            return

        if self.gpu:
            self._torch.cuda.reset_peak_memory_stats()
            self._baseline_gpu_util = self._measure_baseline_gpu_util()

        self._t_start = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="gpu-sampler"
        )
        self._thread.start()

    @contextmanager
    def phase(self, name: str, on_gpu: bool = False):
        """
        Tag the enclosed work as `name` and attribute samples to it.
        """

        if not self.enabled:
            yield Span(name)
            return

        span = Span(name, on_gpu, t_start=time.perf_counter())
        with self._lock:
            self._stack.append(span)

        try:
            yield span
        finally:
            span.t_end = time.perf_counter()
            with self._lock:
                self._stack.pop()
                self._spans.append(span)

    def stop(self) -> None:
        """
        Stop the profiler.
        """

        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        self._t_end = time.perf_counter()

    def _read(self) -> Sample:
        if self.gpu:
            gpu_util = self._pynvml.nvmlDeviceGetUtilizationRates(
                self._handle
            ).gpu
            gpu_used = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
            torch_alloc = self._torch.cuda.memory_allocated()
            torch_reserved = self._torch.cuda.memory_reserved()
        else:
            gpu_util = gpu_used = torch_alloc = torch_reserved = 0

        return Sample(
            cpu=self._process.cpu_percent(),  # Usage since previous call.
            rss=self._process.memory_info().rss,
            gpu_util=gpu_util,
            gpu_used=gpu_used,
            torch_alloc=torch_alloc,
            torch_reserved=torch_reserved,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                sample = self._read()
            # Deliberately broad: stop sampling if something goes wrong.
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"GPU sampling stopped: {exc}")
                return

            with self._lock:
                self._samples.append(sample)
                if self._stack:
                    self._stack[-1].samples.append(sample)

    def report(self) -> None:
        """
        Log the collected profile as a single multi-line record.
        """

        if not self.enabled:
            return

        self.stop()
        wall_s = self._t_end - self._t_start

        if not self._samples:
            logger.info(f"GPU profile: no samples in {wall_s:,.1f} s.")
            return

        per_phase_header = (
            f"{'phase':<12}{'calls':>7}{'wall s':>10}{'% wall':>8}"
            f"{'cpu':>9}{'peak rss':>12}{'gpu util':>10}{'peak gpu':>13}"
        )
        rule, sub_rule = (
            "=" * len(per_phase_header),
            "-" * len(per_phase_header),
        )

        # Section 1: resources.
        if self.gpu:
            device_name = self._pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode()
            total_gpu_mem = self._pynvml.nvmlDeviceGetMemoryInfo(
                self._handle
            ).total
            gpu_description = f"{device_name} ({total_gpu_mem / MIB:,.0f} MiB)"
        else:
            total_gpu_mem = 0
            gpu_description = "none"

        # fmt: off
        lines = [
            "",
            rule,
            "Resources",
            sub_rule,
            f"CPU  {cpu_name()} ({self._physical_cpu_count} cores / {self._cpu_count} threads)",
            f"RAM  {self._total_ram / MIB:,.0f} MiB",
            f"GPU  {gpu_description}",
            "",
        ]
        # fmt: on

        # Section 2: top-line statistics.
        gpu_utils = sorted(s.gpu_util for s in self._samples)

        def gpu_util_pctl(q: float) -> int:
            return gpu_utils[min(int(q * len(gpu_utils)), len(gpu_utils) - 1)]

        peak_gpu_used = max(s.gpu_used for s in self._samples)
        peak_rss = max(s.rss for s in self._samples)
        mean_cpu = sum(s.cpu for s in self._samples) / len(self._samples)
        on_gpu_s = sum(s.wall_s for s in self._spans if s.on_gpu)

        # fmt: off
        lines += [
            rule,
            "Top-line statistics",
            sub_rule,
            f"Wall clock time         {wall_s:,.1f} s",
            f"Samples                 {len(self._samples):,} @ {self._interval_s * 1e3:.0f} ms",
            f"Mean CPU usage          {mean_cpu:,.0f}% of {self._cpu_count * 100:,}% ({mean_cpu / 100:.1f} of {self._cpu_count} cores busy)",
            f"Peak RSS usage          {peak_rss / MIB:,.0f} / {self._total_ram / MIB:,.0f} MiB ({peak_rss / self._total_ram:.0%})",
            "",
        ]
        # fmt: on

        if self.gpu:
            # fmt: off
            lines[-1:] = [
                f"Time in GPU phases      {on_gpu_s / wall_s:5.1%} ({on_gpu_s:,.1f} s)",
                f"GPU usage p50/p90/max   {gpu_util_pctl(0.5)}%/{gpu_util_pctl(0.9)}%/{gpu_utils[-1]}% (idle baseline: {self._baseline_gpu_util:.0f}%)",
                f"Peak GPU used           {peak_gpu_used / MIB:,.0f} / {total_gpu_mem / MIB:,.0f} MiB ({peak_gpu_used / total_gpu_mem:.0%}, incl. other processes)",
                f"Peak torch allocated    {self._torch.cuda.max_memory_allocated() / MIB:,.0f} MiB",
                f"Peak torch reserved     {max(s.torch_reserved for s in self._samples) / MIB:,.0f} MiB",
                "",
            ]
            # fmt: on

        # Section 3: per-phase statistics.
        lines += [
            rule,
            "Per-phase statistics",
            "",
            per_phase_header,
            sub_rule,
        ]

        by_phase: dict[str, list[Span]] = {}
        for span in self._spans:
            by_phase.setdefault(span.name, []).append(span)

        for name, spans in sorted(
            by_phase.items(), key=lambda kv: -sum(s.wall_s for s in kv[1])
        ):
            total_s = sum(s.wall_s for s in spans)
            samples = [s for span in spans for s in span.samples]
            gpu_util = (
                f"{sum(s.gpu_util for s in samples) / len(samples):.0f}%"
                if samples
                else "n/a"
            )
            cpu = (
                f"{sum(s.cpu for s in samples) / len(samples):,.0f}%"
                if samples
                else "n/a"
            )
            peak_gpu = max((s.torch_alloc for s in samples), default=0) / MIB
            rss = max((s.rss for s in samples), default=0) / MIB

            lines.append(
                f"{name:<12}{len(spans):>7,}{total_s:>10,.2f}{total_s / wall_s:>8.1%}"
                f"{cpu:>9}{rss:>8,.0f} MiB{gpu_util:>10}{peak_gpu:>9,.0f} MiB"
            )

        accounted = sum(s.wall_s for s in self._spans)
        lines += [
            sub_rule,
            (
                f"{'accounted':<12}{'':>7}{accounted:>10,.2f}{accounted / wall_s:>8.1%}"
            ),
            "",
        ]

        # Section 4: notes.
        lines += [
            rule,
            "Notes",
            sub_rule,
            '"cpu" column is for this process, summed over cores. 100% is one busy core.',
            '"gpu util" column is coarse (1s driver window) and over all processes.',
            '"accounted" row should be 100%. If more/less, there\'s over/under counting.',
            "",
        ]

        logger.info("\n".join(lines))


profiler = ResourceProfiler()

if profiler.enabled:
    profiler.start()
    atexit.register(profiler.report)
