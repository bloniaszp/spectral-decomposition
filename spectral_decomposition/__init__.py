from spectral_decomposition.simulation import CombinedSimulator, BaseSimulator
from spectral_decomposition.decomposition import ParametricDecomposition, BaseDecomposition
from spectral_decomposition.time_domain import TimeDomainData
from spectral_decomposition.frequency_domain import FrequencyDomainData
from spectral_decomposition.plotting import PSDPlotter, BasePlotter
import numpy as np

def spectrum(
    sampling_rate: float = 1000.0,
    duration: float = 2.0,
    n_samples: int | None = None,
    aperiodic_exponent: float = 1.0,
    aperiodic_offset: float = 1.0,
    knee: float | None = None,
    peaks: list[dict] | None = None,
    direct_estimate: bool = False,
    plot: bool = False,
    average_firing_rate: float = 0.0,
    random_state: int | None = None,
    mode: str = "additive",
):
    """
    Simulate a time series and produce a theoretical (and optionally empirical) PSD.

    The simulator now respects `mode`:
      - "additive": time-domain signal generated from P = P_bb + P_rh
      - "multiplicative": time-domain signal generated from P = P_bb * 10**G
        (peaks interpreted as log10 heights)
    The theoretical PSD is constructed on the simulator's grid using the same mode.
    """
    # 1) Simulate time series using the chosen mode
    simulator = CombinedSimulator(
        sampling_rate=sampling_rate,
        n_samples=n_samples,
        duration=duration,
        aperiodic_exponent=aperiodic_exponent,
        aperiodic_offset=aperiodic_offset,
        knee=knee,
        peaks=peaks,
        average_firing_rate=average_firing_rate,
        random_state=random_state,
        mode=mode,  
    )
    time_data = simulator.simulate()

    # 2) Theoretical PSD on the same grid
    if mode == "additive":
        decomposer = ParametricDecomposition(
            sampling_rate=sampling_rate,
            n_fft=simulator.n_fft,
            aperiodic_exponent=aperiodic_exponent,
            aperiodic_offset=aperiodic_offset,
            knee=knee,
            peaks=peaks,
        )
        freq_data = decomposer.compute()

    elif mode == "multiplicative":
        # Build L and G on the same one-sided grid as the additive decomposer
        grid = ParametricDecomposition(
            sampling_rate=sampling_rate,
            n_fft=simulator.n_fft,
            aperiodic_exponent=aperiodic_exponent,
            aperiodic_offset=aperiodic_offset,
            knee=knee,
            peaks=[],
        ).compute()

        f = grid.frequencies
        kappa = 0.0 if knee is None else float(knee)

        with np.errstate(divide="ignore"):
            L = aperiodic_offset - np.log10(kappa + np.power(f, aperiodic_exponent))
        if f.size and f[0] == 0.0:
            L[0] = -np.inf

        G = np.zeros_like(f)
        for pk in (peaks or []):
            c = float(pk["freq"])
            a_log = float(pk["amplitude"])
            sigma = float(pk["sigma"])
            G += a_log * np.exp(-0.5 * ((f - c) / sigma) ** 2)

        P = np.power(10.0, L + G)
        if f.size and f[0] == 0.0:
            P[0] = 0.0

        freq_data = FrequencyDomainData(
            frequencies=f,
            broadband_spectrum=None,
            rhythmic_spectrum=None,
            combined_spectrum=P,
            empirical_spectrum=None,
        )

    else:
        raise ValueError(f"Unknown mode: {mode!r}; use 'additive' or 'multiplicative'.")

    # 3) Optional empirical PSD via spectral_connectivity
    if direct_estimate:
        try:
            from spectral_connectivity import Multitaper, Connectivity
        except ImportError:
            raise ImportError("Install 'spectral_connectivity' for direct_estimate=True.")

        signal = time_data.combined_signal
        m = Multitaper(
            time_series=signal,
            sampling_frequency=sampling_rate,
            time_halfbandwidth_product=2,
            n_tapers=3,
            n_fft_samples=len(time_data),
        )
        c = Connectivity.from_multitaper(m)
        power = c.power().squeeze()
        freqs_emp = c.frequencies

        pos = freq_data.frequencies >= 0
        freq_data.frequencies = freq_data.frequencies[pos]
        if getattr(freq_data, "broadband_spectrum", None) is not None:
            freq_data.broadband_spectrum = freq_data.broadband_spectrum[pos]
        if getattr(freq_data, "rhythmic_spectrum", None) is not None:
            freq_data.rhythmic_spectrum = freq_data.rhythmic_spectrum[pos]
        if getattr(freq_data, "combined_spectrum", None) is not None:
            freq_data.combined_spectrum = freq_data.combined_spectrum[pos]

        if freqs_emp[0] > 0 and freq_data.frequencies[0] == 0:
            freq_data.frequencies = freq_data.frequencies[1:]
            if getattr(freq_data, "broadband_spectrum", None) is not None:
                freq_data.broadband_spectrum = freq_data.broadband_spectrum[1:]
            if getattr(freq_data, "rhythmic_spectrum", None) is not None:
                freq_data.rhythmic_spectrum = freq_data.rhythmic_spectrum[1:]
            if getattr(freq_data, "combined_spectrum", None) is not None:
                freq_data.combined_spectrum = freq_data.combined_spectrum[1:]

        if len(freqs_emp) != len(freq_data.frequencies):
            of = freq_data.frequencies
            ob = getattr(freq_data, "broadband_spectrum", None)
            orh = getattr(freq_data, "rhythmic_spectrum", None)
            oc = getattr(freq_data, "combined_spectrum", None)

            freq_data.frequencies = freqs_emp
            if ob is not None:
                freq_data.broadband_spectrum = np.interp(freqs_emp, of, ob)
            if orh is not None:
                freq_data.rhythmic_spectrum = np.interp(freqs_emp, of, orh)
            if oc is not None:
                freq_data.combined_spectrum = np.interp(freqs_emp, of, oc)

        freq_data.empirical_spectrum = power.T

    # 4) Package
    params = dict(
        sampling_rate=sampling_rate,
        n_samples=len(time_data),
        duration=float(len(time_data)) / sampling_rate,
        aperiodic_exponent=aperiodic_exponent,
        aperiodic_offset=aperiodic_offset,
        knee=0.0 if knee is None else knee,
        peaks=peaks if peaks else [],
        direct_estimate=direct_estimate,
        average_firing_rate=average_firing_rate,
        random_state=random_state,
        mode=mode,
    )

    class SpectralDecompositionResult:
        def __init__(self, time_domain, frequency_domain, params):
            self.time_domain = time_domain
            self.frequency_domain = frequency_domain
            self.params = params

        def plot(self):
            fig = PSDPlotter().plot(self.frequency_domain)
            return fig

    result = SpectralDecompositionResult(time_data, freq_data, params)
    if plot:
        fig = result.plot()
        fig.show()
    return result
