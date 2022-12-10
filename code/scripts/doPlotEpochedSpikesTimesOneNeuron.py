
import sys
import math
import argparse
import pickle
import numpy as np
import pandas as pd

import svGPFA.plot.plotUtilsPlotly
import striatumUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusterID_to_plot", help="clusterID to plot", type=int,
                        default=-1)
    parser.add_argument("--selected_regions",
                        help="only use neurons from these selected regions",
                        type=str, default="[striatum]")
    parser.add_argument("--max_trial_duration", help="maximum trial duration",
                        type=float, default=math.inf)
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epoched spikes times filename pattern",
                        type=str,
                        default="../../results/spikes_times_epochedFirst2In_regions{:s}.pickle")
    parser.add_argument("--vline_color", help="color of vertical line",
                        type=str, default="gray")
    parser.add_argument("--trials_timing_info_filename",
                        help="trials' timing info filename",
                        type=str,
                        default="../../results/trials_timing_info.csv")
    parser.add_argument("--fig_filename_pattern",
                        help="spikes times figure filename pattern",
                        type=str,
                        default="../../figures/epochedFirst2In_spikes_times_regions{:s}_neuron{:02d}_maxDuration{:f}.{:s}")
    args = parser.parse_args()

    clusterID_to_plot = args.clusterID_to_plot
    selected_regions = args.selected_regions[1:-1].split(",")
    max_trial_duration = args.max_trial_duration
    epoched_spikes_times_filename_pattern = args.epoched_spikes_times_filename_pattern
    vline_color = args.vline_color
    trials_timing_info_filename = args.trials_timing_info_filename
    fig_filename_pattern = args.fig_filename_pattern

    trials_timing_info = pd.read_csv(trials_timing_info_filename)

    selected_regions_str = "".join(selected_region + "_" for selected_region in selected_regions)
    epoched_spikes_times_filename = \
        epoched_spikes_times_filename_pattern.format(selected_regions_str)
    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    clusters_ids = load_res["clusters_ids"]
    trials_start_times = np.array(load_res["trials_start_times"])
    trials_end_times = np.array(load_res["trials_end_times"])


    trials_durations = trials_end_times - trials_start_times
    trials_to_keep = np.where(trials_durations<=max_trial_duration)[0]
    spikes_times = [spikes_times[trial_to_keep]
                    for trial_to_keep in trials_to_keep]
    n_trials = len(spikes_times)
    trials_durations = trials_durations[trials_to_keep]
    trial_spikes_rates = np.array([None for r in range(n_trials)],
                                  dtype=np.double)
    trials_indices = np.arange(n_trials)

    # neuron_indices = np.where(clusters_ids.to_numpy() == clusterID_to_plot)[0]
    try:
        neuron_index = clusters_ids.index(clusterID_to_plot)
    except ValueError:
        raise ValueError(f"{clusterID_to_plot} not found in cluster_id's " + \
                         ", ".join([str(cluster_id) for cluster_id in clusters_ids]))

    marked_events_times, marked_events_colors, marked_events_markers = \
        striatumUtils.buildMarkedEventsInfo(
            trials_timing_info=trials_timing_info,
            trials_indices=trials_indices,
        )

    align_event = np.array([marked_events_times[r][0] \
                            for r in range(n_trials)])

    for r in range(n_trials):
        spikes_times_rn = spikes_times[r][neuron_index]
        trial_spikes_rates[r] = len(spikes_times_rn)/trials_durations[r]
    trial_averaged_spike_rate = trial_spikes_rates.mean()
    title = "Neuron {:d}, Trial-Averaged Spike Rate {:f}".format(
        clusters_ids[neuron_index], trial_averaged_spike_rate)
    fig = svGPFA.plot.plotUtilsPlotly.getSpikesTimesPlotOneNeuron(
        spikes_times=spikes_times, neuron_index=neuron_index,
        trials_indices=trials_indices,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        align_event=align_event,
        title=title)
    fig.add_vline(x=0, line_color=vline_color)
    selected_regions_str = "".join(selected_region + "_" for selected_region in selected_regions)
    spikes_times_png_filename = fig_filename_pattern.format(
        selected_regions_str, clusters_ids[neuron_index], max_trial_duration,
        "png")
    spikes_times_html_filename = fig_filename_pattern.format(
        selected_regions_str, clusters_ids[neuron_index], max_trial_duration,
        "html")
    fig.write_image(spikes_times_png_filename)
    fig.write_html(spikes_times_html_filename)
    fig.show()

    breakpoint()

if __name__=="__main__":
    main(sys.argv)
