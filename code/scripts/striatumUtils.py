import pandas as pd
import numpy as np


def getNeuronSpikesTimesAndRegion(cluster_id, spikes_times, clusters_ids,
                                  regions):
    n_trials = len(spikes_times)
    cluster_index = clusters_ids.index(cluster_id)
    region = regions[cluster_index]
    neuron_spikes_times = [None for r in range(n_trials)]
    for r in range(n_trials):
        neuron_spikes_times[r] = spikes_times[r][cluster_index]
    return neuron_spikes_times, region

def findCorrectSequencesStartAndEndIndices(correct_sequence, transitions_data,
                                           start_port_colname="Start_Port"):
    n_transitions = transitions_data.shape[0]
    correct_sequence0_int = int(correct_sequence[0])
    possible_start_indices = transitions_data.index[transitions_data[start_port_colname] == correct_sequence0_int].tolist()

    correct_sequences_start_and_end_indices = []
    tdi = possible_start_indices[0] # tdi == transition_data_index
    del possible_start_indices[0]
    while tdi < n_transitions - 1:
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


def findTrialsIDsOfPerfectSequences(perfect_sequence, transitions_data,
                                    start_port_colname="Start_Port",
                                    trial_id_colname="Trial_id"):
    n_transitions = transitions_data.shape[0]
    trials_ids = transitions_data[trial_id_colname].unique()

    perfect_trials_ids = []
    for trial_id in trials_ids:
        # Let's look to the transition_data indicices where the trials starts
        tdi = np.where(transitions_data[trial_id_colname] == trial_id)[0][0]
        psi = 0 # psi == perfect_sequence_index
        # Let's advance on the transitions and perfect_sequence while they
        # match
        while psi < len(perfect_sequence) and \
              tdi < n_transitions and \
              transitions_data.loc[tdi, trial_id_colname] == trial_id and \
              transitions_data.loc[tdi, start_port_colname] == int(perfect_sequence[psi]):
            tdi += 1
            psi += 1
        # We have a correct trial if we advnaced until the end of the sequence
        # and (we reached the end of the transitions or the current trial_id of
        # the transtions is different from that in the previous sequence
        if psi == len(perfect_sequence) and \
           (tdi == n_transitions or \
            transitions_data.loc[tdi, trial_id_colname] != trial_id):
            perfect_trials_ids.append(transitions_data.loc[tdi-1, trial_id_colname])
    return perfect_trials_ids


def buildMarkedEventsInfo(trials_timing_info, trials_indices,
                           port_numbers=np.array((2, 1, 6, 3, 7)),
                           port_colors=np.array(("orange", "red", "green",
                                                 "blue", "black")),
                           stages=np.array(["IN", "OUT"]),
                           stage_markers=["cross", "circle"]):
    n_trials = len(trials_indices)
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    for trial_index, r in enumerate(trials_indices):
        trial_timing_info = \
            trials_timing_info[trials_timing_info.trial == r]
        trial_marked_events_times = []
        trial_marked_events_colors = []
        trial_marked_events_markers = []
        for i in range(trial_timing_info.shape[0]):
            trial_marked_events_times.append(
                trial_timing_info.iloc[i]["timestamp"],
            )

            port_number_index = np.where(
                port_numbers == trial_timing_info.iloc[i]["port"])[0].item()

            trial_marked_events_colors.append(port_colors[port_number_index])

            stage_index = np.where(
                stages == trial_timing_info.iloc[i]["stage"])[0].item()
            trial_marked_events_markers.append(stage_markers[stage_index])

        marked_events_times[trial_index] = trial_marked_events_times
        marked_events_colors[trial_index] = trial_marked_events_colors
        marked_events_markers[trial_index] = trial_marked_events_markers
    return marked_events_times, marked_events_colors, marked_events_markers


def buildMarkedEventsInfoFromTransitions(
        transitions_data, trials_ids, port_numbers=np.array((2, 1, 6, 3, 7)),
        port_colors=np.array(("orange", "red", "green", "blue", "black")),
        stage_markers=["cross", "circle"],
        trial_id_colname="Trial_id",
        port_in_time_colname="P1_IN_Ephys_TS",
        port_out_time_colname="P1_OUT_Ephys_TS",
        start_port_colname="Start_Port"):
    n_trials = len(trials_ids)
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    for r, trial_id in enumerate(trials_ids):
        trial_transition_data = \
            transitions_data[transitions_data[trial_id_colname]==trial_id]
        trial_marked_events_times = []
        trial_marked_events_colors = []
        trial_marked_events_markers = []
        for i in range(trial_transition_data.shape[0]):
            trial_marked_events_times.append(
                trial_transition_data.iloc[i][port_in_time_colname],
            )
            trial_marked_events_times.append(
                trial_transition_data.iloc[i][port_out_time_colname],
            )

            port_number_index = np.where(
                port_numbers == trial_transition_data.iloc[i][start_port_colname])[0].item()

            trial_marked_events_colors.append(port_colors[port_number_index])
            trial_marked_events_colors.append(port_colors[port_number_index])

            trial_marked_events_markers.append(stage_markers[0])
            trial_marked_events_markers.append(stage_markers[1])

        marked_events_times[r] = trial_marked_events_times
        marked_events_colors[r] = trial_marked_events_colors
        marked_events_markers[r] = trial_marked_events_markers
    return marked_events_times, marked_events_colors, marked_events_markers
