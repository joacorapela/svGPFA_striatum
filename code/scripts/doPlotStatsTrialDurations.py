
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objs as go

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile",
                        help="percentile to draw (in a 0-100 scale)",
                        type=float, default=90)
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
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename",
                        type=str, default="../../data/Transition_data_sync.csv")
    parser.add_argument("--correct_sequences_start_and_end_indices_filename",
                        help="correct sequences filename",
                        type=str,
                        default="../../results/correctSequencesStartAndEndIndices.csv")
    args = parser.parse_args()

    percentile = args.percentile
    before_trial_pad = args.before_trial_pad
    after_trial_pad = args.after_trial_pad
    port_in_ephys_ts_col_name = args.port_in_ephys_ts_col_name
    port_out_ephys_ts_col_name = args.port_out_ephys_ts_col_name
    transitions_data_filename = args.transitions_data_filename
    correct_sequences_start_and_end_indices_filename = \
        args.correct_sequences_start_and_end_indices_filename

    transitions_data = pd.read_csv(transitions_data_filename)
    correct_seqs_start_and_end_indices = \
        pd.read_csv(correct_sequences_start_and_end_indices_filename,
                    header=None)
    n_trials = correct_seqs_start_and_end_indices.shape[0]

    trials_durations = [None for r in range(n_trials)]
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
        trials_durations[r] = trial_end_time - trial_start_time

    percentile_90 = np.percentile(a=trials_durations, q=percentile)

    trace = go.Histogram(x=trials_durations)
    fig = go.Figure()
    fig.add_trace(trace)
    fig.add_vline(x=percentile_90)
    fig.show()

    breakpoint()

if __name__=="__main__":
    main(sys.argv)
