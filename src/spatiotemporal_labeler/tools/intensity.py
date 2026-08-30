from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CardiacPhaseFrames:
    """Estimated phase markers for one periodic 4D image sequence.

    All indices are zero-based. ``diastole_start`` is the first frame after
    the high-signal systolic interval and may be before ``systole_start`` when
    the interval crosses the time-axis boundary.
    """

    systole_start: int
    peak: int
    diastole_start: int
    confidence: float


def _temporal_signal(data: NDArray[np.number]) -> np.ndarray:
    """Build the global finite-magnitude signal for each time frame.

    This intentionally matches :func:`strongest_signal_frame`: every finite
    voxel contributes its absolute magnitude, so phase detection and initial
    frame selection use the same whole-volume measurement.
    """
    sequence = np.asarray(data)
    if sequence.ndim != 4 or sequence.shape[3] < 1:
        raise ValueError("Temporal signal extraction requires nonempty 4D data")
    signal = np.zeros(sequence.shape[3], dtype=np.float64)
    for frame in range(sequence.shape[3]):
        values = np.asarray(sequence[..., frame]).ravel()
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        dtype = np.complex128 if np.iscomplexobj(values) else np.float64
        finite_values = np.asarray(values[finite], dtype=dtype)
        signal[frame] = np.abs(finite_values).sum(dtype=np.float64)
    return signal


def _circular_smooth(signal: np.ndarray) -> np.ndarray:
    if signal.size < 3:
        return signal.copy()
    window = min(7, signal.size if signal.size % 2 else signal.size - 1)
    if window < 3:
        return signal.copy()
    radius = window // 2
    padded = np.concatenate((signal[-radius:], signal, signal[:radius]))
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def detect_cardiac_phases(
    data: NDArray[np.number],
    *,
    systolic_fraction: float = 0.35,
) -> CardiacPhaseFrames:
    """Estimate systole start, signal peak, and diastole start.

    The estimate is circular in time: the systolic interval may cross the
    last/first frame boundary. ``systolic_fraction`` is a hysteresis-like
    fraction of the peak-above-baseline amplitude used to identify that
    interval. It intentionally returns a suggestion for UI review rather than
    replacing user-selected label keyframes.
    """
    fraction = float(systolic_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("Systolic fraction must be between 0 and 1")
    raw_signal = _temporal_signal(data)
    signal = _circular_smooth(raw_signal)
    if signal.size < 3:
        raise ValueError("At least three frames are required for phase detection")
    # The peak is the maximum of the global signal itself, exactly matching
    # the frame chosen by ``strongest_signal_frame``. Smoothing is retained
    # only for the hysteresis threshold used to find systolic boundaries.
    peak = int(np.argmax(raw_signal))
    baseline = float(np.percentile(signal, 25.0))
    peak_value = float(signal[peak])
    amplitude = peak_value - baseline
    scale = max(abs(peak_value), abs(baseline), 1.0)
    if not np.isfinite(amplitude) or amplitude <= np.finfo(float).eps * scale:
        raise ValueError("The temporal signal has no detectable cardiac peak")
    threshold = baseline + fraction * amplitude
    high = signal >= threshold
    if bool(np.all(high)):
        raise ValueError("The temporal signal has no detectable systolic interval")

    systole_start = peak
    for _ in range(signal.size):
        previous = (systole_start - 1) % signal.size
        if not high[previous]:
            break
        systole_start = previous

    last_systolic = peak
    for _ in range(signal.size):
        following = (last_systolic + 1) % signal.size
        if not high[following]:
            break
        last_systolic = following
    diastole_start = (last_systolic + 1) % signal.size
    confidence = float(np.clip(amplitude / scale, 0.0, 1.0))
    return CardiacPhaseFrames(
        systole_start=systole_start,
        peak=peak,
        diastole_start=diastole_start,
        confidence=confidence,
    )


def strongest_signal_frame(data: NDArray[np.number]) -> int:
    """Return the earliest frame with the largest finite absolute-signal sum."""
    return int(np.argmax(_temporal_signal(data)))
