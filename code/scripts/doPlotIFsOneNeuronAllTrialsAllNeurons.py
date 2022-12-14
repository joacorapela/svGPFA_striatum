
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import argparse

import svGPFA.plot.plotUtilsPlotly
import striatumUtils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", help="estimation result number",
                        type=int)
    parser.add_argument("--n_neurons_to_plot",
                        help="number of neurons to plot",
                        type=int, default=18)
    parser.add_argument("--n_time_steps_CIF", help="number of stime steps in "
                        "CIF plots", type=int, default=100)
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename",
                        type=str, default="../../data/Transition_data_sync.csv")
    parser.add_argument("--CIFs_oneNeuron_all_trials_fig_filename_pattern",
                        help="figure filename for a CIF",
                        type=str,
                        default="../../figures/{:08d}_CIFsOneNeuronAllTrials_neuron{:03d}.{:s}")

    args = parser.parse_args()
    est_res_number = args.est_res_number
    n_neurons_to_plot = args.n_neurons_to_plot
    n_time_steps_CIF = args.n_time_steps_CIF
    transitions_data_filename = args.transitions_data_filename
    model_save_filename_pattern = args.model_save_filename_pattern
    CIFs_oneNeuron_all_trials_fig_filename_pattern = \
        args.CIFs_oneNeuron_all_trials_fig_filename_pattern

    model_save_filename = model_save_filename_pattern.format(est_res_number)
    with open(model_save_filename, "rb") as f:
        estResults = pickle.load(f)
    spikes_times = estResults["spikes_times"]
    model = estResults["model"]
    trials_ids = estResults["trials_ids"]
    clusters_ids = estResults["clusters_ids"]
    regions = estResults["regions"]
    n_trials = len(trials_ids)
    trials_start_times = estResults["trials_start_times"]
    trials_end_times = estResults["trials_end_times"]

    n_neurons = len(spikes_times[0])
    if n_neurons_to_plot > n_neurons:
        n_neurons_to_plot = n_neurons

    trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
        start_times=np.squeeze(trials_start_times),
        end_times=np.squeeze(trials_end_times),
        n_steps=n_time_steps_CIF)

    transitions_data = pd.read_csv(transitions_data_filename)
    marked_events_times, marked_events_colors, marked_events_markers = \
        striatumUtils.buildMarkedEventsInfoFromTransitions(
            transitions_data=transitions_data,
            trials_ids=trials_ids,
        )

    align_event = np.array([marked_events_times[r][0] \
                            for r in range(n_trials)])

    # calculate expected CIF values (for KS test and CIF plots)
    with torch.no_grad():
        cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)

    for neuron_to_plot in range(n_neurons_to_plot):
        print(f"Processing cluster_id {clusters_ids[neuron_to_plot]}")
        title = f"Cluster ID: {clusters_ids[neuron_to_plot]}, Region: {regions[neuron_to_plot]}"
        fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
            trials_times=trials_times,
            cif_values=cif_values,
            neuron_index=neuron_to_plot,
            spikes_times=spikes_times,
            trials_ids=trials_ids,
            align_event=align_event,
            marked_events_times=marked_events_times,
            marked_events_colors=marked_events_colors,
            marked_events_markers=marked_events_markers,
            title=title,
        )
        fig.write_image(
            CIFs_oneNeuron_all_trials_fig_filename_pattern.format(
                est_res_number, neuron_to_plot, "png"))
        fig.write_html(
            CIFs_oneNeuron_all_trials_fig_filename_pattern.format(
                est_res_number, neuron_to_plot, "html"))


if __name__ == "__main__":
    main(sys.argv)
