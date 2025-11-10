import numpy as np
from abc import ABC, abstractmethod
from numpy.fft import ifft
from math import ceil, log2

def make_broadband_predictor(freqs, f_low, f_high, *, exponent=1.0, knee=0.0):
    freqs = np.asarray(freqs)
    mask  = (freqs >= f_low) & (freqs <= f_high) & (freqs > 0)
    shape = np.zeros_like(freqs, dtype=float)
    shape[mask] = 1.0 / (knee + np.abs(freqs[mask])**exponent)
    return shape

def make_gaussian_bump_predictor(freqs, f_low, f_high, *, center, sigma):
    freqs = np.asarray(freqs)
    mask  = (freqs >= f_low) & (freqs <= f_high)
    shape = np.zeros_like(freqs, dtype=float)
    shape[mask] = np.exp(-((freqs[mask] - center)**2) / (2.0 * sigma**2))
    return shape

def simulate_from_psd(PSD, fs, n_fft, n_time, random_seed=None, lambda_0=0.0):
    import numpy as np
    from numpy.fft import ifft

    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random

    if len(PSD) != n_fft:
        raise ValueError(f"PSD length ({len(PSD)}) must match n_fft={n_fft}.")

    halfM = n_fft // 2
    U = np.zeros(n_fft, dtype=np.complex128)

    # DC
    U[0] = np.sqrt(max(PSD[0], 0.0)) * rng.randn()

    if n_fft % 2 == 0:
        # EVEN n_fft: positives 1..halfM-1
        pos_psd = PSD[1:halfM] / 2.0
        amp = np.sqrt(np.maximum(pos_psd, 0.0))
        U[1:halfM] = amp * (rng.randn(len(amp)) + 1j * rng.randn(len(amp)))
        # Nyquist (pure real)
        U[halfM] = np.sqrt(max(PSD[halfM], 0.0)) * rng.randn()
        # negatives mirror of 1..halfM-1
        U[halfM+1:] = np.conj(U[1:halfM][::-1])
    else:
        # ODD n_fft: positives 1..halfM
        pos_psd = PSD[1:halfM+1] / 2.0
        amp = np.sqrt(np.maximum(pos_psd, 0.0))
        U[1:halfM+1] = amp * (rng.randn(len(amp)) + 1j * rng.randn(len(amp)))
        # negatives mirror of 1..halfM
        U[halfM+1:] = np.conj(U[1:halfM+1][::-1])

    # IFFT (unshifted)
    signal_freq_domain = np.sqrt(fs * n_fft) * ifft(U)
    time_signal = np.real(signal_freq_domain[:n_time])
    time_signal += lambda_0
    return time_signal


def simulate_from_psd_legacy(PSD, fs, n_fft, n_time, random_seed=None, lambda_0=0.0):
    import numpy as np
    from numpy.fft import ifft

    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random

    if len(PSD) != n_fft:
        raise ValueError(f"PSD length ({len(PSD)}) must match n_fft={n_fft}.")

    halfM = n_fft // 2
    U = np.zeros(n_fft, dtype=np.complex128)

    # DC
    U[0] = np.sqrt(max(PSD[0], 0.0)) * rng.randn()

    if n_fft % 2 == 0:
        # EVEN n_fft
        # positives 1..halfM-1
        pos_psd = PSD[1:halfM] / 2.0
        amp = np.sqrt(np.maximum(pos_psd, 0.0))
        U[1:halfM] = amp * (rng.randn(len(amp)) + 1j * rng.randn(len(amp)))
        # Nyquist at halfM (pure real)
        U[halfM] = np.sqrt(max(PSD[halfM], 0.0)) * rng.randn()
        # negatives mirror of 1..halfM-1
        U[halfM+1:] = np.conj(U[1:halfM][::-1])
    else:
        # ODD n_fft
        # positives 1..halfM
        pos_psd = PSD[1:halfM+1] / 2.0
        amp = np.sqrt(np.maximum(pos_psd, 0.0))
        U[1:halfM+1] = amp * (rng.randn(len(amp)) + 1j * rng.randn(len(amp)))
        # no Nyquist bin
        # negatives mirror of 1..halfM
        U[halfM+1:] = np.conj(U[1:halfM+1][::-1])

    # IFFT (unshifted)
    signal_freq_domain = np.sqrt(fs * n_fft) * ifft(U)
    time_signal = np.real(signal_freq_domain[:n_time])
    time_signal += lambda_0
    return time_signal


def simulate_from_psd_legacy(PSD, fs, n_fft, n_time, random_seed=None, lambda_0=0.0):
    """
    Draw a real time-domain signal whose (two-sided, unshifted) PSD is `PSD`.
    `PSD` must be in *unshifted* FFT order (DC at 0, +freqs, then -freqs).
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random

    if len(PSD) != n_fft:
        raise ValueError(f"PSD length ({len(PSD)}) must match n_fft={n_fft}.")

    halfM = n_fft // 2
    U = np.zeros(n_fft, dtype=np.complex128)

    # DC
    U[0] = np.sqrt(max(PSD[0], 0.0)) * rng.randn()

    # Positive frequencies (1..halfM-1 if even; 1..halfM if odd)
    pos_stop = halfM if (n_fft % 2 == 0) else (halfM + 1)
    pos_psd = PSD[1:pos_stop] / 2.0
    amp = np.sqrt(np.maximum(pos_psd, 0.0))
    U[1:pos_stop] = amp * (rng.randn(len(amp)) + 1j * rng.randn(len(amp)))

    # Nyquist (if even)
    if n_fft % 2 == 0:
        U[halfM] = np.sqrt(max(PSD[halfM], 0.0)) * rng.randn()

    # Negative freqs (Hermitian symmetry)
    U[pos_stop:] = np.conj(U[1:pos_stop][::-1])

    # IFFT in *unshifted* order — no extra shifts
    signal_freq_domain = np.sqrt(fs * n_fft) * ifft(U)
    time_signal = np.real(signal_freq_domain[:n_time])
    time_signal += lambda_0
    return time_signal

def _nextpow2(x):
    return 2**int(ceil(log2(x)))

class BaseSimulator(ABC):
    @abstractmethod
    def simulate(self):
        raise NotImplementedError

class CombinedSimulator(BaseSimulator):
    """
    Simulate broadband + rhythmic with either additive *or* multiplicative
    composition in *linear* PSD space.

      additive:        P(f) = P_bb(f) + P_rh(f)
      multiplicative:  P(f) = P_bb(f) * 10**G(f)     (G built from peaks in log10 units)

    Peaks are modeled as Gaussians at ±f0 on the two-sided frequency axis.
    """
    def __init__(
        self,
        sampling_rate,
        n_samples=None,
        duration=None,
        aperiodic_exponent=1.0,
        aperiodic_offset=1.0,
        knee=None,
        peaks=None,
        average_firing_rate=0.0,
        n_fft=None,
        target_df=0.01,
        random_state=None,
        mode: str = "additive",  # <— now supported, propagated from API
    ):
        self.sampling_rate = sampling_rate

        # time length
        if n_samples is None:
            if duration is None:
                raise ValueError("Must specify either n_samples or duration.")
            self.n_samples = int(duration * sampling_rate)
            if self.n_samples < 1:
                raise ValueError("Duration too short for the given sampling rate.")
        else:
            self.n_samples = int(n_samples)
            if duration is not None:
                expected_n = int(duration * sampling_rate)
                if expected_n != self.n_samples:
                    raise ValueError("n_samples and duration are inconsistent.")

        # big FFT grid
        if n_fft is None:
            required = int(ceil(self.sampling_rate / target_df))
            required = max(required, self.n_samples)
            self.n_fft = _nextpow2(required)
        else:
            self.n_fft = int(n_fft)

        self.aperiodic_exponent = float(aperiodic_exponent)
        self.aperiodic_offset = float(aperiodic_offset)
        self.knee = 0.0 if knee is None else float(knee)
        self.peaks = [] if peaks is None else list(peaks)
        self.average_firing_rate = float(average_firing_rate)
        self.random_state = random_state
        if mode not in ("additive", "multiplicative"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode

    def _build_two_sided_freqs(self):
        # Unshifted FFT freq order: [0, +f, ..., Nyq, -f,...]
        return np.fft.fftfreq(self.n_fft, d=1.0 / self.sampling_rate)

    def simulate(self):
        fs = self.sampling_rate
        n_time = self.n_samples
        n_fft = self.n_fft
        freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)

        # Aperiodic baseline (linear power)
        denom = (self.knee + np.abs(freqs) ** self.aperiodic_exponent)
        with np.errstate(divide="ignore", invalid="ignore"):
            P_bb = (10.0 ** self.aperiodic_offset) / denom
        P_bb[0] = 0.0

        # Helper: distinct seeds (safe if None)
        s0 = self.random_state
        seed_bb = s0
        seed_rh = None if s0 is None else s0 + 1
        seed_comb = None if s0 is None else s0 + 2

        if self.mode == "additive":
            # Rhythmic in *linear* power and two-sided (±f0)
            P_rh = np.zeros_like(freqs)
            for pk in self.peaks:
                f0 = float(pk["freq"]); amp = float(pk["amplitude"]); sigma = float(pk["sigma"])
                P_rh += amp * np.exp(-((freqs - f0) ** 2) / (2 * sigma ** 2))
                P_rh += amp * np.exp(-((freqs + f0) ** 2) / (2 * sigma ** 2))
            P_rh[0] = 0.0
            P_comb = P_bb + P_rh

            # Independent phase draws for each partial
            broadband_signal_big = simulate_from_psd(P_bb,   fs, n_fft, n_fft, random_seed=seed_bb, lambda_0=0.0)
            rhythmic_signal_big  = simulate_from_psd(P_rh,   fs, n_fft, n_fft, random_seed=seed_rh, lambda_0=0.0)
            combined_signal_big  = broadband_signal_big + rhythmic_signal_big

        else:  # multiplicative
            # Peaks as log10-power bumps: P = P_bb * 10**G
            G = np.zeros_like(freqs)
            for pk in self.peaks:
                f0 = float(pk["freq"]); a_log = float(pk["amplitude"]); sigma = float(pk["sigma"])
                G += a_log * np.exp(-((freqs - f0) ** 2) / (2 * sigma ** 2))
                G += a_log * np.exp(-((freqs + f0) ** 2) / (2 * sigma ** 2))
            P_comb = P_bb * np.power(10.0, G)
            P_comb[0] = 0.0

            # Combined signal draw (this is the *true* multiplicative model)
            combined_signal_big = simulate_from_psd(P_comb, fs, n_fft, n_fft, random_seed=seed_comb, lambda_0=0.0)

            # Optional: independent illustrative parts (won't sum to combined)
            broadband_signal_big = simulate_from_psd(P_bb, fs, n_fft, n_fft, random_seed=seed_bb, lambda_0=0.0)
            P_rh_lin = P_bb * (np.power(10.0, G) - 1.0)  # linear “excess” over baseline
            rhythmic_signal_big  = simulate_from_psd(P_rh_lin, fs, n_fft, n_fft, random_seed=seed_rh, lambda_0=0.0)

        # Slice to requested duration and add any DC offset in time
        combined_signal  = combined_signal_big[:n_time] + self.average_firing_rate
        broadband_signal = broadband_signal_big[:n_time]
        rhythmic_signal  = rhythmic_signal_big[:n_time]

        time = np.arange(n_time) / fs
        from spectral_decomposition.time_domain import TimeDomainData
        return TimeDomainData(
            time=time,
            combined_signal=combined_signal,
            broadband_signal=broadband_signal,
            rhythmic_signal=rhythmic_signal,
        )


    def simulate_legacy(self):
        fs = self.sampling_rate
        n_time = self.n_samples
        n_fft = self.n_fft
        freqs = self._build_two_sided_freqs()

        # Broadband in *linear* power: 10^offset / (k + |f|^chi)
        denom = (self.knee + np.abs(freqs) ** self.aperiodic_exponent)
        with np.errstate(divide="ignore", invalid="ignore"):
            P_bb = (10.0 ** self.aperiodic_offset) / denom
        # zero DC
        P_bb[0] = 0.0

        # Rhythmic field
        if self.mode == "additive":
            # linear-amplitude Gaussians added to linear PSD
            P_rh = np.zeros_like(freqs)
            for peak in self.peaks:
                f0 = float(peak["freq"])
                amp = float(peak["amplitude"])
                sigma = float(peak["sigma"])
                P_rh += amp * np.exp(-((freqs - f0) ** 2) / (2 * sigma ** 2))
                P_rh += amp * np.exp(-((freqs + f0) ** 2) / (2 * sigma ** 2))
            P_rh[0] = 0.0
            P_comb = P_bb + P_rh

        else:  # multiplicative
            # Peaks are *log10-power* bumps: P = P_bb * 10**G
            G = np.zeros_like(freqs)
            for peak in self.peaks:
                f0 = float(peak["freq"])
                a_log = float(peak["amplitude"])  # interpret as log10 height
                sigma = float(peak["sigma"])
                G += a_log * np.exp(-((freqs - f0) ** 2) / (2 * sigma ** 2))
                G += a_log * np.exp(-((freqs + f0) ** 2) / (2 * sigma ** 2))
            P_comb = P_bb * np.power(10.0, G)
            P_comb[0] = 0.0

        # Simulate from *unshifted* PSD
        combined_signal_big = simulate_from_psd(
            P_comb, fs, n_fft, n_fft, random_seed=self.random_state, lambda_0=0.0
        )

        # (optional) separate draws for components if you want to expose them
        # For now, keep backwards-compatible outputs by splitting via the same RNG seed
        broadband_signal_big = simulate_from_psd(
            P_bb, fs, n_fft, n_fft, random_seed=self.random_state, lambda_0=0.0
        )
        if self.mode == "additive":
            P_rh_lin = P_comb - P_bb
        else:
            P_rh_lin = P_bb * (np.power(10.0, G) - 1.0)
        rhythmic_signal_big = simulate_from_psd(
            P_rh_lin, fs, n_fft, n_fft, random_seed=self.random_state, lambda_0=0.0
        )

        combined_signal = combined_signal_big[:n_time] + self.average_firing_rate
        broadband_signal = broadband_signal_big[:n_time]
        rhythmic_signal = rhythmic_signal_big[:n_time]

        time = np.arange(n_time) / fs
        from spectral_decomposition.time_domain import TimeDomainData
        return TimeDomainData(
            time=time,
            combined_signal=combined_signal,
            broadband_signal=broadband_signal,
            rhythmic_signal=rhythmic_signal,
        )
