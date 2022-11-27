
import sys
import argparse
import configparser
import pickle
import numpy as np
import pandas as pd
import scipy.io

import svGPFA.utils.miscUtils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--before_trial_pad",
                        help="pad (in seconds) to be added at begining of "
                        "trial before the beginning of the sequence",
                        type=float, default=0.2)
    parser.add_argument("--after_trial_pad",
                        help="pad (in seconds) to be added at end of "
                        "trial after the end of the sequence",
                        type=float, default=0.2)
    args = parser.parse_args()

    before_trial_pad = args.before_trial_pad
    after_trial_pad = args.after_trial_pad

    correct_seqs_start_and_end_indices = 
    for r in range(n_trials):
        seq_start_index = correct_seqs_start_and_end_indices[r, 0]
        seq_end_index = correct_seqs_start_and_end_indices[r, 1]
        trial_start_time = transitions_data.loc[seq_start_index,
                                                port_in_ephys_ts_col_name]
        trial_start_time -= before_trial_pad
        trial_end_time = transitions_data.loc[seq_end_index,
                                              port_in_ephys_ts_col_name]
        trial_end_time += after_trial_pad
        for n in range(n_neurons):
            unit_spikes_times = units_df.loc[n, spikes_times_col_name]
            spikes_times[r][n] = unit_spikes_times[
                np.logical_and(trial_start_time <= unit_spikes_times,
                               unit_spikes_times < trial_end_time)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", help="electrode region", type=str,
                        default="GPe")
    parser.add_argument("--block_types_indices", help="block types indices",
                        default="[0]")
    parser.add_argument("--clusters_info_group", help="selected group in clusters_info",
                        type=str, default="good")
    parser.add_argument("--epoch_times_colname",
                        help="epoch times column name", type=str,
                        default="aligned__last_center_out")
    parser.add_argument("--centerIn_times_column_name", help="centerIn times column name",
                        type=str,
                        default="aligned__last_center_in")
    parser.add_argument("--sideIn_times_column_name", help="sideIn times column name",
                        type=str,
                        default="aligned__side_in_after_last_center_out")
    parser.add_argument("--centerIn_trialStart_pad",
                        help="trials start centerIn_trialStart_pad seconds "
                        "before the centerIn event",
                        type=float, default=0.2)
    parser.add_argument("--sideIn_trialEnd_pad",
                        help="trials end at the minimum between "
                        "sideIn_trialEnd_pad seconds after the sideIn event "
                        "and the start of the next trial",
                        type=float, default=3.0)
    parser.add_argument("--cluster_info_filename", help="clusters info filename",
                        type=str,
                        default="../../data/022822/s008_tab_m1113182_LR__20210516_173805__ephys_pxi.cluster_info.df.csv")
    parser.add_argument("--spike_clusters_filename", help="spike clusters filename",
                        type=str,
                        default="../../data/022822/s008_tab_m1113182_LR__20210516_173805__ephys_pxi.spike_clusters.npy")
    parser.add_argument("--spike_times_seconds_filename",
                        help="spikes times in seconds filename",
                        type=str,
                        default="../../data/022822/s008_tab_m1113182_LR__20210516_173805__ephys_pxi.spike_times_seconds.npy")
    parser.add_argument("--events_times_filename",
                        help="events times filename",
                        type=str,
                        default="../../data/022822/s008_tab_m1113182_LR__20210516_173815__probabilistic_switching.df.csv")
    parser.add_argument("--region_epoched_spike_times_metadata_filename_pattern",
                        help="region epched spikes times meta data filename pattern",
                        type=str,
                        default="../../results/00000000_region{:s}_spikes_times_epoched{:s}.ini")
    parser.add_argument("--region_epoched_spike_times_filename_pattern",
                        help="region epched spikes times filename pattern",
                        type=str,
                        default="../../results/00000000_region{:s}_blockTypeIndices{:s}_spikes_times_epoched{:s}.{:s}")
    args = parser.parse_args()

    region = args.region
    block_types_indices = \
        [int(str) for str in args.block_types_indices[1:-1].split(",")]
    clusters_info_group = args.clusters_info_group
    epoch_times_colname = args.epoch_times_colname
    centerIn_times_column_name = args.centerIn_times_column_name
    sideIn_times_column_name = args.sideIn_times_column_name
    centerIn_trialStart_pad = args.centerIn_trialStart_pad
    sideIn_trialEnd_pad = args.sideIn_trialEnd_pad
    cluster_info_filename = args.cluster_info_filename
    spike_clusters_filename = args.spike_clusters_filename
    spike_times_seconds_filename = args.spike_times_seconds_filename
    events_times_filename = args.events_times_filename
    region_epoched_spike_times_filename_pattern = args.region_epoched_spike_times_filename_pattern
    region_epoched_spike_times_metadata_filename_pattern = args.region_epoched_spike_times_metadata_filename_pattern

    cluster_info = pd.read_csv(cluster_info_filename) 
    neurons_in_region = cluster_info[(cluster_info["group"]==clusters_info_group) & (cluster_info["region_acronym"]==region)]["id"]
    print("Found {:d} neurons in region {:s}".format(len(neurons_in_region), region))

    events_times = pd.read_csv(events_times_filename)

    trials_indices = [r for r in range(len(events_times))
                      if events_times.iloc[r]["block_type_index"] in
                      block_types_indices]
    epochs_times = events_times[epoch_times_colname]
    epochs_times = epochs_times.iloc[trials_indices].to_numpy()
    n_trials = len(trials_indices)

    centerIn_times = events_times.iloc[trials_indices][centerIn_times_column_name].to_numpy()
    sideIn_times = events_times.iloc[trials_indices][sideIn_times_column_name].to_numpy()

#     trials_start_times_abs = centerIn_times - centerIn_trialStart_pad
    trials_start_times_abs = epochs_times - 0.3
#     trials_end_times_abs = np.array([min(sideIn_times[r] + sideIn_trialEnd_pad, centerIn_times[r+1] - centerIn_trialStart_pad)
#                                      if r < n_trials - 1
#                                      else sideIn_times[r] + sideIn_trialEnd_pad
#                                      for r in range(n_trials)])
    trials_end_times_abs = epochs_times + 1.0
    spike_clusters = np.load(spike_clusters_filename)
    spike_times_seconds = np.load(spike_times_seconds_filename)
    neurons_spikes_times = []
    for neuron in neurons_in_region:
        neuron_spikes_times = spike_times_seconds[spike_clusters==neuron]
        neurons_spikes_times.append(neuron_spikes_times)
    epoched_spikes_times = svGPFA.utils.miscUtils.separateNeuronsSpikeTimesByTrials(
        neurons_spike_times=neurons_spikes_times,
        epochs_times=epochs_times,
        trials_start_times=trials_start_times_abs,
        trials_end_times=trials_end_times_abs)

    metadata_config = configparser.ConfigParser()
    metadata_config["selection_info"] = \
        {"region": region,
         "clusters_info_group": clusters_info_group}
    metadata_config["epoch_info"] = \
        {"epoch_times_colname": epoch_times_colname}
#          "epoch_elapsed_time_before": epoch_elapsed_time_before,
#          "epoch_elapsed_time_after": epoch_elapsed_time_after}
    metadata_config["data_filenames"] = \
        {"cluster_info_filename": cluster_info_filename,
         "spike_clusters_filename": spike_clusters_filename,
         "spike_times_seconds_filename": spike_times_seconds_filename}
    region_epoched_spike_times_metadata_filename = \
        region_epoched_spike_times_metadata_filename_pattern.format(
            region, epoch_times_colname)
    with open(region_epoched_spike_times_metadata_filename, "w") as f:
        metadata_config.write(f)
    trials_start_times_rel = trials_start_times_abs - epochs_times
    trials_end_times_rel = trials_end_times_abs - epochs_times
    results_to_save = {"region": region, "neurons": neurons_in_region,
                       "trials_indices": trials_indices,
                       "epochs_times": epochs_times,
                       "spikes_times": epoched_spikes_times,
                       "trials_start_times": trials_start_times_rel,
                       "trials_end_times": trials_end_times_rel,
                      }
    breakpoint()
    block_types_indices_str = "_".join([str(index) for index in
                                        block_types_indices])
    region_epoched_spikes_times_pickle_filename = \
        region_epoched_spike_times_filename_pattern.format(
            region, block_types_indices_str, epoch_times_colname, "pickle")
    with open(region_epoched_spikes_times_pickle_filename, "wb") as f:
        pickle.dump(results_to_save, f)
    print("Saved results to {:s}".format(region_epoched_spikes_times_pickle_filename))
    region_epoched_spikes_times_mat_filename = \
    region_epoched_spike_times_filename_pattern.format(
        region, block_types_indices_str, epoch_times_colname, "mat")
    scipy.io.savemat(region_epoched_spikes_times_mat_filename, results_to_save)
    print("Saved results to {:s}".format(region_epoched_spikes_times_mat_filename))
    breakpoint()

if __name__=="__main__":
    main(sys.argv)
