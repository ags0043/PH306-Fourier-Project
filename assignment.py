"""Calculation of a Pulsar's frequency, specficially 
the Crab Pulsar, using fourier analysis
"""

# --- Imports --- #
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_pulsar_signal(file_path: str | Path) -> np.ndarray:
    """Load the intensity column from a tab-separated pulsar data file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to a tab-separated file whose third column contains intensity
        measurements.

    Returns
    -------
    numpy.ndarray
        Finite intensity measurements in their original sample order.

    Raises
    ------
    ValueError
        If the file does not contain at least two finite intensity values.
    """
    signal = np.genfromtxt(
        file_path, dtype=float, delimiter="\t", usecols=(2,)
    )
    return signal[np.isfinite(signal)]


def estimate_pulsar_period(
    samples: np.ndarray, sample_interval: float = 1.0
) -> float:
    """Estimate a pulsar's repeating period from evenly sampled data.

    Parameters
    ----------
    samples : numpy.ndarray
        One-dimensional measurements containing a repeating pulsar signal.
    sample_interval : float, optional
        Time between adjacent samples, in the desired time unit.

    Returns
    -------
    float
        Estimated period of the strongest nonzero Fourier component, in the
        same time unit as ``sample_interval``.

    Raises
    ------
    ValueError
        If the samples are not one-dimensional, contain fewer than two
        values, or ``sample_interval`` is not positive.
    """
    samples = np.asarray(samples, dtype=float)

    centered_samples = samples - np.mean(samples)
    windowed_samples = centered_samples * np.hanning(samples.size)
    amplitudes = np.abs(np.fft.rfft(windowed_samples))
    amplitudes[0] = 0.0

    peak_index = int(np.argmax(amplitudes))
   
    frequency_index = float(peak_index)
    if peak_index < amplitudes.size - 1:
        left = amplitudes[peak_index - 1]
        center = amplitudes[peak_index]
        right = amplitudes[peak_index + 1]
        curvature = left - 2.0 * center + right
        if curvature != 0.0:
            offset = 0.5 * (left - right) / curvature
            frequency_index += float(np.clip(offset, -0.5, 0.5))

    frequency = frequency_index / (samples.size * sample_interval)
    return float(1.0 / frequency)


def estimate_crab_period(
    file_path: str | Path = "Crab_isaac_12ms_03Feb12.txt",
    sample_interval: float = 0.012,
) -> float:
    """Estimate the Crab pulsar period from its uploaded sample file.

    Parameters
    ----------
    file_path : str or pathlib.Path, optional
        Path to the Crab data file. Its third column is treated as intensity.
    sample_interval : float, optional
        Time between samples, in seconds. The uploaded file uses 12 ms.

    Returns
    -------
    float
        Estimated Crab pulsar period in seconds.
    """
    signal = load_pulsar_signal(file_path)
    return estimate_pulsar_period(signal, sample_interval)


def plot_crab_analysis(
    file_path: str | Path = "Crab_isaac_12ms_03Feb12.txt",
    sample_interval: float = 0.012,
    output_path: str | Path = "crab_fourier_analysis.png",
) -> None:
    """Save a time-domain and Fourier-spectrum plot for the Crab data.

    Parameters
    ----------
    file_path : str or pathlib.Path, optional
        Path to the Crab data file.
    sample_interval : float, optional
        Time between samples, in seconds.
    output_path : str or pathlib.Path, optional
        Location where the PNG plot will be saved.
    """
    signal = load_pulsar_signal(file_path)
    period = estimate_pulsar_period(signal, sample_interval)
    centered_signal = signal - np.mean(signal)
    windowed_signal = centered_signal * np.hanning(signal.size)
    amplitudes = np.abs(np.fft.rfft(windowed_signal))
    frequencies = np.fft.rfftfreq(signal.size, d=sample_interval)

    time = np.arange(signal.size) * sample_interval
    frequency_limit = min(100.0, frequencies[-1])
    time_limit = min(2.0, time[-1])

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(time, signal, color="navy", linewidth=0.8)
    axes[0].set_xlim(0.0, time_limit)
    axes[0].set_title("Crab Pulsar Signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(alpha=0.3)

    axes[1].plot(frequencies, amplitudes, color="darkred", linewidth=0.8)
    axes[1].axvline(
        1.0 / period,
        color="black",
        linestyle="--",
        label=f"Detected: {1.0 / period:.3f} Hz",
    )
    axes[1].set_xlim(0.0, frequency_limit)
    axes[1].set_title("Fourier Amplitude Spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    period = estimate_crab_period()
    print(f"Estimated Crab pulsar period: {period:.8f} s")
    print(f"Estimated rotation frequency: {1.0 / period:.6f} Hz")
    plot_crab_analysis()
    print("Saved visualization to crab_fourier_analysis.png")
