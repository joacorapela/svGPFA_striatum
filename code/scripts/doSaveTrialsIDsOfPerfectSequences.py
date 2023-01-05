
import sys
import argparse
import pandas as pd

import striatumUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--perfect_sequence", help="correct sequence",
                        type=str, default="21637")
    parser.add_argument("--start_port_colname", help="start_port_column nname",
                        type=str, default="Start_port")
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename",
                        type=str, default="../../data/Transition_data_sync.csv")
    parser.add_argument("--save_filename",
                        help="filename where to save trials IDs",
                        type=str,
                        default="../../results/trialsIDsOfPerfectSecuences.csv")
    args = parser.parse_args()
    perfect_sequence = args.perfect_sequence
    start_port_colname = args.start_port_colname
    transitions_data_filename = args.transitions_data_filename
    save_filename = args.save_filename

    transitions_data = pd.read_csv(transitions_data_filename)
    trials_ids = striatumUtils.findTrialsIDsOfPerfectSequences(
        perfect_sequence=perfect_sequence, transitions_data=transitions_data)

    trials_ids_df = pd.DataFrame(trials_ids)
    trials_ids_df.to_csv(save_filename, header=False, index=False)

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
