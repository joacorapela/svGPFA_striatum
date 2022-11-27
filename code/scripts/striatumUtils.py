
def findCorrectSequencesStartAndEndIndices(correct_sequence,
                                           transitions_data,
                                           start_port_colname="Start_Port"):
    n_transitions = transitions_data.shape[0]
    correct_sequence0_int = int(correct_sequence[0])
    possible_start_indices = transitions_data.index[transitions_data[start_port_colname] == correct_sequence0_int].tolist()

    correct_sequences_start_and_end_indices = []
    tdi = possible_start_indices[0] # tdi == transition_data_index
    del possible_start_indices[0]
    while tdi < n_transitions - 1:
#         if tdi == 5767:
#             breakpoint()
        csi = 0 # csi == correct_sequence_index
        transition_sequence_start_index = tdi
        while csi < len(correct_sequence) and tdi < n_transitions:
            # at this point correct_sequence[csi] ==
            # transitions_data.loc[tdi, start_port_colname]
            tdi += 1
            # ignore repetitions of correct sequence elements
            while tdi < n_transitions and \
                  transitions_data.loc[tdi, start_port_colname] == \
                    int(correct_sequence[csi]):
                tdi += 1
            csi += 1
            # if all elements of correct sequence are matched
            # add a tuple to correct_sequences_start_and_end_indices
            if csi == len(correct_sequence):
                transition_sequence_end_index = tdi - 1
                correct_sequences_start_and_end_indices.append(
                    (transition_sequence_start_index,
                     transition_sequence_end_index)
                )
            # if the next transition_data does not match the next element of
            # the sequence, start searching again from the next possible_start
            #_indices
            if tdi < n_transitions and \
               (csi == len(correct_sequence) or \
                transitions_data.loc[tdi, start_port_colname] !=
                    int(correct_sequence[csi])):
                while len(possible_start_indices) > 0 and \
                        possible_start_indices[0] < tdi:
                    del possible_start_indices[0]
                if len(possible_start_indices) > 0:
                    tdi = possible_start_indices[0]
                else:
                    tdi = n_transitions
                break
    return correct_sequences_start_and_end_indices

