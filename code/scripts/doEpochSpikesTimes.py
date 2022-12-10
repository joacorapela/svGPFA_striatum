
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

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
    parser.add_argument("--port_in_ephys_ts_col_name",
                        help="column name for port_in_ephys time stamp",
                        type=str, default="P1_IN_Ephys_TS")
    parser.add_argument("--port_out_ephys_ts_col_name",
                        help="column name for port_out_ephys time stamp",
                        type=str, default="P1_OUT_Ephys_TS")
    parser.add_argument("--spikes_times_col_name",
                        help="column name for spikes times",
                        type=str, default="Spike_times")
    parser.add_argument("--region_col_name",
                        help="region column name",
                        type=str, default="Region")
    parser.add_argument("--cluster_id_col_name",
                        help="cluster id column name",
                        type=str, default="cluster_id")
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename",
                        type=str, default="../../data/Transition_data_sync.csv")
    parser.add_argument("--units_info_filename",
                        help="filename with units information",
                        type=str, default="../../data/good_units_df.csv")
    parser.add_argument("--correct_sequences_start_and_end_indices_filename",
                        help="correct sequences filename",
                        type=str,
                        default="../../results/correctSequencesStartAndEndIndices.csv")
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spikes times filename",
                        type=str,
                        default="../../results/spikes_times_epochedFirst2In.pickle")
    args = parser.parse_args()

    before_trial_pad = args.before_trial_pad
    after_trial_pad = args.after_trial_pad
    port_in_ephys_ts_col_name = args.port_in_ephys_ts_col_name
    port_out_ephys_ts_col_name = args.port_out_ephys_ts_col_name
    spikes_times_col_name = args.spikes_times_col_name
    region_col_name = args.region_col_name
    cluster_id_col_name = args.cluster_id_col_name
    transitions_data_filename = args.transitions_data_filename
    units_info_filename = args.units_info_filename
    correct_sequences_start_and_end_indices_filename = \
        args.correct_sequences_start_and_end_indices_filename
    epoched_spikes_times_filename = args.epoched_spikes_times_filename

    transitions_data = pd.read_csv(transitions_data_filename)
    units_info = pd.read_csv(units_info_filename)

    correct_seqs_start_and_end_indices = \
        pd.read_csv(correct_sequences_start_and_end_indices_filename,
                    header=None)
    n_trials = correct_seqs_start_and_end_indices.shape[0]
    n_neurons = units_info.shape[0]

    epochs_times = [None for r in range(n_trials)]
    trials_start_times_rel = [None for r in range(n_trials)]
    trials_end_times_rel = [None for r in range(n_trials)]
    spikes_times_rel = [[None for n in range(n_neurons)] for r in range(n_trials)]
    for r in range(n_trials):
        print(f"Processing trial {r} ({n_trials-1})")
        seq_start_index = correct_seqs_start_and_end_indices.iloc[r, 0]
        seq_end_index = correct_seqs_start_and_end_indices.iloc[r, 1]
        first_2in_time = transitions_data.loc[seq_start_index,
                                              port_in_ephys_ts_col_name]
        trial_start_time = first_2in_time - before_trial_pad
        last_7out_time = transitions_data.loc[seq_end_index,
                                              port_out_ephys_ts_col_name]
        trial_end_time = last_7out_time + after_trial_pad
        epochs_times[r] = first_2in_time
        trials_start_times_rel[r] = -before_trial_pad
        trials_end_times_rel[r] = trial_end_time - first_2in_time
        for n in range(n_neurons):
            unit_spikes_times_str = units_info.iloc[n][spikes_times_col_name][1:-1].split(",")
            unit_spikes_times = np.array([float(unit_spike_times_str) for unit_spike_times_str in unit_spikes_times_str])
            spikes_times_rel[r][n] = unit_spikes_times[
                np.logical_and(trial_start_time <= unit_spikes_times,
                               unit_spikes_times < trial_end_time)] - first_2in_time
    clusters_ids = units_info[cluster_id_col_name].tolist()
    regions = units_info[region_col_name].tolist()
    results_to_save = {
        "epochs_times": epochs_times,
        "spikes_times": spikes_times_rel,
        "trials_start_times": trials_start_times_rel,
        "trials_end_times": trials_end_times_rel,
        "clusters_ids": clusters_ids,
        "regions": regions,
    }
    with open(epoched_spikes_times_filename, "wb") as f: pickle.dump(results_to_save, f)

    breakpoint()

if __name__=="__main__":
    main(sys.argv)
