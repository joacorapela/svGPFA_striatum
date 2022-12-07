
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_port_col_name",
                        help="column name for the start por",
                        type=str, default="Start_Port")
    parser.add_argument("--port_in_ephys_ts_col_name",
                        help="column name for port_in_ephys time stamp",
                        type=str, default="P1_IN_Ephys_TS")
    parser.add_argument("--port_out_ephys_ts_col_name",
                        help="column name for port_out_ephys time stamp",
                        type=str, default="P1_OUT_Ephys_TS")
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename",
                        type=str, default="../../data/Transition_data_sync.csv")
    parser.add_argument("--correct_sequences_start_and_end_indices_filename",
                        help="correct sequences filename",
                        type=str,
                        default="../../results/correctSequencesStartAndEndIndices.csv")
    parser.add_argument("--trials_timing_info_filename",
                        help="trials' timing info filename",
                        type=str,
                        default="../../results/trials_timing_info.csv")
    args = parser.parse_args()

    start_port_col_name = args.start_port_col_name
    port_in_ephys_ts_col_name = args.port_in_ephys_ts_col_name
    port_out_ephys_ts_col_name = args.port_out_ephys_ts_col_name
    transitions_data_filename = args.transitions_data_filename
    correct_sequences_start_and_end_indices_filename = \
        args.correct_sequences_start_and_end_indices_filename
    trials_timing_info_filename = args.trials_timing_info_filename

    transitions_data = pd.read_csv(transitions_data_filename)
    correct_seqs_start_and_end_indices = \
        pd.read_csv(correct_sequences_start_and_end_indices_filename,
                    header=None)
    n_trials = correct_seqs_start_and_end_indices.shape[0]

    trials = []
    ports = []
    stages = []
    timestamps = []
    for r in range(n_trials):
        seq_start_index = correct_seqs_start_and_end_indices.iloc[r, 0]
        seq_end_index = correct_seqs_start_and_end_indices.iloc[r, 1]
        for i in range(seq_start_index, seq_end_index+1):
            port = transitions_data.loc[i, start_port_col_name]
            in_time = transitions_data.loc[i, port_in_ephys_ts_col_name]
            out_time = transitions_data.loc[i, port_out_ephys_ts_col_name]
            trials.extend([r, r])
            ports.extend([port, port])
            stages.extend(["IN", "OUT"])
            timestamps.extend([in_time, out_time])
    df = pd.DataFrame(list(zip(trials, ports, stages, timestamps)),
                      columns=("trial, port, stage, timestamp"))
    df.to_csv(trials_timing_info_filename, index=False)

    breakpoint()

if __name__=="__main__":
    main(sys.argv)
