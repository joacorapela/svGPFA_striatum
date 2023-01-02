import sys
import pickle
import argparse
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("error")

import gcnu_common.utils.neuralDataAnalysis
import gcnu_common.utils.spectral_analysis
import striatumUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_id", help="cluster_id to analyze", type=int,
                        default=0)
    parser.add_argument("--min_neuron_trials_avg_firing_rate",
                        help="minimum neuron average firing rate", type=float,
                        default=0.1)
    parser.add_argument("--max_trial_duration", help="maximum trial duration",
                        type=float, default=5.5)
    parser.add_argument("--bins_min_time", help="minimum time of all bins (ms)",
                        type=int, default=-200)
    parser.add_argument("--bins_max_time", help="maximum time of all bins (ms)",
                        type=int, default=5200)
    parser.add_argument("--bins_dt", help="bins dt", type=int, default=1)
    parser.add_argument("--window_length",
                        help="window length (sec) for spectogram",
                        type=float, default=.5)
    parser.add_argument("--step_size",
                        help="step size (sec) for spectogram",
                        type=float, default=.05)
    parser.add_argument("--min_freq_spectrogram",
                        help="minum frequency to plot in spectrograms (Hz)",
                        type=float, default=0.0)
    parser.add_argument("--max_freq_spectrogram",
                        help="maximum frequency to plot in spectrograms (Hz)",
                        type=float, default=50.0)
    parser.add_argument("--title_pattern", help="title pattern", type=str,
                        default="Cluster id: {:d}, Region: {:s}")
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spikes times filename pattern",
                        type=str,
                        default="../../results/spikes_times_epochedFirst2In_fixedDurationTrue.pickle")
    parser.add_argument("--ISIsHistFigFilenamePattern",
                        help="inter-spike intervals histogram filename pattern",
                        type=str,
                        default="../../figures/clusterID{:d}_ISIsHistogram_region{:s}.{:s}")
    parser.add_argument("--meanACFFigFilenamePattern",
                        help="mean autocorrelation functionfilename pattern",
                        type=str,
                        default="../../figures/clusterID{:d}_meanACF_region{:s}.{:s}")
    parser.add_argument("--meanSpectrumFigFilenamePattern",
                        help="mean spectrum filename pattern",
                        type=str,
                        default="../../figures/clusterID{:d}_meanSpectrum_region{:s}.{:s}")
    parser.add_argument("--spectrogramFigFilenamePattern",
                        help="spectrogramfilename pattern",
                        type=str,
                        default="../../figures/clusterID{:d}_spectrogram_region{:s}.{:s}")
    args = parser.parse_args()

    cluster_id = args.cluster_id
    min_neuron_trials_avg_firing_rate = args.min_neuron_trials_avg_firing_rate
    max_trial_duration = args.max_trial_duration
    bins_min_time = args.bins_min_time 
    bins_max_time = args.bins_max_time 
    bins_dt = args.bins_dt 
    window_length = args.window_length 
    step_size = args.step_size 
    min_freq_spectrogram = args.min_freq_spectrogram
    max_freq_spectrogram = args.max_freq_spectrogram
    title_pattern = args.title_pattern
    epoched_spikes_times_filename = args.epoched_spikes_times_filename
    ISIsHistFigFilenamePattern = args.ISIsHistFigFilenamePattern
    meanACFFigFilenamePattern = args.meanACFFigFilenamePattern
    meanSpectrumFigFilenamePattern = args.meanSpectrumFigFilenamePattern
    spectrogramFigFilenamePattern = args.spectrogramFigFilenamePattern

    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    clusters_ids = load_res["clusters_ids"]
    regions = load_res["regions"]
    trials_start_times = np.array(load_res["trials_start_times"])
    trials_end_times = np.array(load_res["trials_end_times"])

    n_neurons = len(spikes_times[0])
    n_trials = len(spikes_times)
    neurons_indices = np.arange(n_neurons)
    trials_indices = np.arange(n_trials)

    # remove units with low firing rate
    trials_durations = trials_end_times - trials_start_times
    spikes_times, neurons_indices = \
        gcnu_common.utils.neuralDataAnalysis.removeUnitsWithLessTrialAveragedFiringRateThanThr(
            spikes_times=spikes_times, neurons_indices=neurons_indices,
            trials_durations = trials_durations,
            min_neuron_trials_avg_firing_rate=min_neuron_trials_avg_firing_rate)
    clusters_ids = [clusters_ids[i] for i in neurons_indices]
    regions = [regions[i] for i in neurons_indices]

    # remove long trials
    spikes_times, trials_indices = \
        gcnu_common.utils.neuralDataAnalysis.removeTrialsLongerThanThr(
            spikes_times=spikes_times, trials_indices=trials_indices,
            trials_durations=trials_durations,
            max_trial_duration=max_trial_duration)

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    # convert spikes times from sec to msec
    spikes_times = [[spikes_times[r][n]*1000 for n in range(n_neurons)] for r in range(n_trials)]

    # get neuron spikes times
    neuron_spikes_times, region = striatumUtils.getNeuronSpikesTimesAndRegion(
        cluster_id=cluster_id, spikes_times=spikes_times,
        clusters_ids=clusters_ids, regions=regions)

    # remove trials with no spikes
    neuron_spikes_times = [neuron_spikes_times[r] for r in
                           range(len(neuron_spikes_times)) \
                     if np.sum(neuron_spikes_times[r])>0]

    n_trials = len(neuron_spikes_times)
    Fs = 1000

    # get neuron ISIs
    neuron_isis_by_trial = [np.diff(neuron_spikes_times[r])
                            for r in range(n_trials)]
    neuron_isis_concatenated = []
    for r in range(n_trials):
        neuron_isis_concatenated.extend(neuron_isis_by_trial[r])

    # bin neuron spikes times
    bins_edges = np.arange(bins_min_time, bins_max_time, bins_dt)
    n_bins = len(bins_edges) - 1
    neuron_binned_spikes = np.empty((n_trials, n_bins), dtype=np.double)
    for r in range(n_trials):
        neuron_binned_spikes[r,:] = np.histogram(neuron_spikes_times[r],
                                                 bins=bins_edges)[0]

    # compute acf
    acf, mean_acf, lags_ms = gcnu_common.utils.spectral_analysis.compute_acf(
        spikes_times=neuron_binned_spikes)

    # compute mean spectrum
    mean_spectrum, freqs = gcnu_common.utils.spectral_analysis.compute_mean_spectrum(
        x_trials=neuron_binned_spikes, Fs=Fs)

    # compute spectrogram
    bins_centers = bins_edges[:-1] + np.diff(bins_edges)
    spectogram, T, f = gcnu_common.utils.spectral_analysis.compute_spectogram(
        t=bins_centers, x_trials=neuron_binned_spikes,
        window_length=window_length, step_size=step_size, Fs=Fs,
        fpass=(min_freq_spectrogram, max_freq_spectrogram))

    # Let's plot now

    fig = go.Figure()
    trace = go.Histogram(
        x=neuron_isis_concatenated,
        xbins=dict(
            start=0.0,
            end=250.0,
            size=1.0,
        ),
    )
    fig.add_trace(trace)
    fig.update_xaxes(title_text="Time (msec)")
    fig.update_yaxes(title_text="Count")
    fig.update_layout(title=title_pattern.format(cluster_id, region))
    fig.write_image(ISIsHistFigFilenamePattern.format(cluster_id, region, "png"))
    fig.write_html(ISIsHistFigFilenamePattern.format(cluster_id, region, "html"))

    fig = go.Figure()
    trace = go.Scatter(x=lags_ms, y=mean_acf,
                       mode="markers",
                       error_y={'type': 'data',
                                 'symmetric': False,
                                 'arrayminus': mean_acf,
                                 'array': [0] * len(mean_acf),
                                 'width': 0
                               },
                      )
    fig.add_trace(trace)
    fig.add_hline(y=0.0, line_width=1, line_dash="solid", line_color="red")
    fig.update_xaxes(title_text="Lag (msec)")
    fig.update_yaxes(title_text="Autocorelation")
    fig.update_layout(title=title_pattern.format(cluster_id, region))
    fig.write_image(meanACFFigFilenamePattern.format(cluster_id, region, "png"))
    fig.write_html(meanACFFigFilenamePattern.format(cluster_id, region, "html"))

    fig = go.Figure()
    trace = go.Scatter(x=freqs, y=mean_spectrum)
    fig.add_trace(trace)
    fig.update_xaxes(title_text="Frequency (Hz)", range=(0, 100))
    fig.update_yaxes(title_text="Power")
    fig.update_layout(title=title_pattern.format(cluster_id, region))
    fig.write_image(meanSpectrumFigFilenamePattern.format(cluster_id, region, "png"))
    fig.write_html(meanSpectrumFigFilenamePattern.format(cluster_id, region, "html"))

    fig = go.Figure()
    trace = go.Contour(x=T, y=f, z=spectogram)
    fig.add_trace(trace)
    fig.update_yaxes(title_text="Frequency (Hz)")
    fig.update_xaxes(title_text="Time (msec)")
    fig.update_layout(title=f"Cluster id: {cluster_id}, Region: {region}")
    fig.write_image(spectrogramFigFilenamePattern.format(cluster_id, region, "png"))
    fig.write_html(spectrogramFigFilenamePattern.format(cluster_id, region, "html"))

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
