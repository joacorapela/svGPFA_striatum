
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


def main(argv):
    test_findCorrectSequencesStartAndEndIndices()


if __name__ == "__main__":
    main(sys.argv)
