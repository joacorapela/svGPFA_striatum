
import sys
import pandas as pd

sys.path.append("..")
import striatumUtils

def test_findCorrectSequencesStartAndEndIndices():
    correct_sequence = "21637"
    # data = [3,2,2,1,6,6,3,3,3,7,4]
    data = [3,2,2,1,6,6,3,3,3,7,4,2,1,6,3,3,2,1,6,3,7,2]

    transitions_data = pd.DataFrame(data, columns=["Start_Port"])
    correct_sequences_start_and_end_indices = \
       striatumUtils.findCorrectSequencesStartAndEndIndices(
           correct_sequence=correct_sequence,
           transitions_data=transitions_data)
    print(correct_sequences_start_and_end_indices)
    # assert(correct_sequences_start_and_end_indices[0] == (1,9))
    assert(correct_sequences_start_and_end_indices[0] == (1,9))
    assert(correct_sequences_start_and_end_indices[1] == (16,20))


def test_findTrialsIDsOfPerfectSequences():
    perfect_sequence = "21637"
    # data = [3,2,2,1,6,6,3,3,3,7,4]
    data_start_ports = [3,2,1,6,3,7,4,2,1,6,3,2,1,6,3,7,2]
    data_trials_ids  = [0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4]
    correct_trials_ids = [1, 3]
    transitions_data = pd.DataFrame({"Start_Port": data_start_ports,
                                     "Trial_id": data_trials_ids})
    trials_ids = striatumUtils.findTrialsIDsOfPerfectSequences(
        perfect_sequence=perfect_sequence, transitions_data=transitions_data)
    print(trials_ids)
    assert(len(correct_trials_ids) == len(trials_ids))
    for i in range(len(correct_trials_ids)):
        assert(correct_trials_ids[i] == trials_ids[i])


def main(argv):
    # test_findCorrectSequencesStartAndEndIndices()
    test_findTrialsIDsOfPerfectSequences()


if __name__ == "__main__":
    main(sys.argv)
