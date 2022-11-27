
import sys
import argparse
import pandas as pd

import striatumUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct_sequence", help="correct sequence",
                        type=str, default="21637")
    parser.add_argument("--start_port_colname", help="start_port_column nname",
                        type=str, default="Start_port")
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename",
                        type=str, default="../../data/Transition_data_sync.csv")
    parser.add_argument("--correct_sequences_start_and_end_indices_filename",
                        help="correct sequences filename",
                        type=str,
                        default="../../results/correctSequencesStartAndEndIndices.csv")
    args = parser.parse_args()
    correct_sequence = args.correct_sequence
    start_port_colname = args.start_port_colname
    transitions_data_filename = args.transitions_data_filename
    correct_sequences_start_and_end_indices_filename = \
        args.correct_sequences_start_and_end_indices_filename

    transitions_data = pd.read_csv(transitions_data_filename)
    correct_sequences_indices = \
        striatumUtils.findCorrectSequencesStartAndEndIndices(
            correct_sequence=correct_sequence,
            transitions_data=transitions_data)

    correct_sequences_indices_pd = pd.DataFrame(correct_sequences_indices)
    correct_sequences_indices_pd.to_csv(
        correct_sequences_start_and_end_indices_filename, header=False,
        index=False)

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
