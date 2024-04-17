from fock_state_circuit.state import State
from fock_state_circuit.collection_of_states import CollectionOfStates
from fock_state_circuit.temporal_and_spectral_gate_functionality.column_of_states import ColumnOfStates
from fock_state_circuit.temporal_and_spectral_gate_functionality.interference_group import InterferenceGroup
from fock_state_circuit.temporal_and_spectral_gate_functionality.collection_of_state_columns import CollectionOfStateColumns
import numpy as np

"""
    add_time_delay_generic(collection_of_states, info, use_classical_control_channel):
            This gate will model the impact of a time delay for channels that have light (or photon) pulses with a finite 
            duration. The photon wave packets do not always overlap in this case (depending on the pulse width and time difference
            between the pulses.)

            This is a 'full' implementation suitable close to the mathematical formalism used to describe time
            dependent photon wave packets in quantum optics. In order to do this the optical states have to be decomposed in
            single photons states. This allows modelling successive time delay gates (i.e., if we first delay
            a photon for a time interval +t and then move it forward with same time interval -t we expect those 
            actions to cancel out.)

            The is a more 'quick' implementation for situations where time delays and pulse width are only set once
            in a circuit. See temporal_functions.add_quick_time_delay_generic for this.   

    add_time_delay(collection_of_states, parameters):
            This functions works as a shell to unwrap 'parameters' and call 
            temporal_functions.add_time_delay_generic 
            For documentation see temporal_functions.add_time_delay_generic.   
        
    add_time_delay_classical_control(collection_of_states, parameters):
            This functions works as a shell to unwrap 'parameters' and call 
            temporal_functions.add_time_delay_generic 
            For documentation see temporal_functions.add_time_delay_generic. 

    set_pulse_width(collection_of_states, parameters):
            Set pulse width for all channels, or for selected channels. 
            - All channels are set if parameters[0] is None or if parameters[0] is a list of all channels. In this csse the states
            are not changed. Pulse width is just added or updated.
            - If parameters[0] is the empty list [] no channels are set, the function returns the original collection
            - Otherwise just the channels in parameters[0] are set. In this case channels which are not set either keep
            the original pulse width, or if this is not present get the ault value for pulse width. The returned collection
            will consist of single photon states.
            
            Parameters[1] contains the pulse width to set

    photon_probability_function(photon_info_dict1, photon_info_dict2):
            Calculate the probability for detecting this photon pair depending on the overlap of the wave packets. If the photons have no overlap
            (for instance they have a narrow pulse and a large separation in time) the probability to detect them as a pair is zero. If they fully overlap
            (same pulse width and no time separation) the probability is 100%.

    add_quick_time_delay_generic(collection_of_states, info, use_classical_control_channel):
            This gate will model the impact of a time delay for channels that have light (or photon) pulses with a finite 
            duration. The photon wave packets do not always overlap in this case (depending on the pulse width and time difference
            between the pulses.)

            This is a 'quick' implementation suitable in situations where bandwidth and timing are only set
            once in a circuit. This will NOT work if we have successive time delay gates (i.e., if we first delay
            a photon for a time interval +t and then move it forward with same time interval -t we expect those 
            actions to cancel out. The implementation of time delay in this function will NOT model that correctly. 
            Use the full functionality in add_time_delay/add_time_delay_classical_control for that. 
            The penalty will be slower execution.)

            This gate splits every state in the collection in three parts:
            A part representing the temporal overlap and two states represent a 'ahead' and a 'behind' part. 
            The 'overlap' state and the 'ahead'/'behind' states will be in separate interference groups.
            The ratio (cumulative_probability) between the 'overlap' and 'ahead/behind' group is determined by the 
            overlap of the photon wave packets. If there is no 'overlap' we only have 'ahead/behind', if there is full 
            overlap (same pulse_width and no time delay) we only have the 'overlap' situation and probability for 
            'ahead/behind' will have probability zero.
    
    add_quick_time_delay(collection_of_states, parameters):
            This functions works as a shell to unwrap 'parameters' and call 
            temporal_functions.add_quick_time_delay_generic 
            For documentation see temporal_functions.add_quick_time_delay_generic. 

    add_quick_time_delay_classical_control(collection_of_states, parameters):
            This functions works as a shell to unwrap 'parameters' and call 
            temporal_functions.add_quick_time_delay_generic 
            For documentation see temporal_functions.add_quick_time_delay_generic. 

        Last modified: April 16th, 2024
"""
_VERSION = '1.0.0'

def add_time_delay_generic(collection_of_states: CollectionOfStates, info: dict, use_classical_control_channel: bool) -> CollectionOfStates:
    """ This gate will model the impact of a time delay for channels that have light (or photon) pulses with a finite 
        duration. The photon wave packets do not always overlap in this case (depending on the pulse width and time difference
        between the pulses.)

        This is a 'full' implementation suitable close to the mathematical formalism used to describe time
        dependent photon wave packets in quantum optics. In order to do this the optical states have to be decomposed in
        single photons states. This allows modelling successive time delay gates (i.e., if we first delay
        a photon for a time interval +t and then move it forward with same time interval -t we expect those 
        actions to cancel out.)

        The is a more 'quick' implementation for situations where time delays and pulse width are only set once
        in a circuit. See temporal_functions.add_quick_time_delay_generic for this.
    
        Args:
            collection_of_states: original collection of states
            info: dictionary in form {  'channels_to_be_delayed': list, 
                                        'time_delay': float, 
                                        'classical_channel_for_delay': integer,
                                        'pulse_width': float}
            use_classical_control_channel (bool): Indicates whether to take time_delay from the dict 'info', 
                                or from 'classical_channel_for_delay'

        Returns:
            CollectionOfStates: Updated collection with 'full' time delay implemented. 
    """
    # ensure channel_numbers is a list 
    channel_numbers = info['channels_to_be_delayed']
    if not isinstance(channel_numbers, list):
        channel_numbers = [channel_numbers]

    # create a collection for the return value of this function
    resulting_collection = collection_of_states.copy()
    resulting_collection.clear()
    
    collection_by_column = CollectionOfStateColumns(collection_of_states=collection_of_states)
    collection_by_column.split()
    collection_by_column.single_photon_states()
    for state in collection_by_column.by_state():
        if use_classical_control_channel:
            time_delay = state.classical_channel_values[info['classical_channel_for_delay']]
        else:
            time_delay = info['time_delay']
        # if there is not information in timing and pulse width first fill in the default values
        if 'photon_information' not in state.auxiliary_information['photon_resolution'].keys():
            state.auxiliary_information['photon_resolution']['photon_information'] = {'time_delay': 0, 'pulse_width':info['pulse_width']}

        # check if this state has a photon in one of the channels to be delayed
        optical_component = list(state.optical_components.keys())[0]
        values = state._dict_of_valid_component_names[optical_component]
        delay_this_state = False
        for channel in info['channels_to_be_delayed']:
            if (values[channel] != 0):
                delay_this_state = True
                break

        # apply the delay
        if delay_this_state:
            photon_info_dict = state.auxiliary_information['photon_resolution']['photon_information'] 
            photon_info_dict['time_delay'] += -1*time_delay
            state.auxiliary_information['photon_resolution']['photon_information'] = photon_info_dict 
        
        # set the pulse width
        if info['pulse_width'] is not None:
            state.auxiliary_information['photon_resolution']['photon_information']['pulse_width'] = info['pulse_width']

        # add state to resulting collection
        resulting_collection.add_state(state=state)
    
    return resulting_collection

def add_time_delay(collection_of_states: CollectionOfStates, parameters: list) -> CollectionOfStates:
    """ This functions works as a shell to unwrap 'parameters' and call 
        temporal_functions.add_time_delay_generic 
        For documentation see temporal_functions.add_time_delay_generic. 
    """
    info = {'channels_to_be_delayed': parameters[0], 'time_delay': parameters[1], 'pulse_width' : parameters[2]}
    return add_time_delay_generic(collection_of_states, info, use_classical_control_channel = False)

def add_time_delay_classical_control(collection_of_states: CollectionOfStates, parameters: list) -> CollectionOfStates:
    """ This functions works as a shell to unwrap 'parameters' and call 
        temporal_functions.add_time_delay_generic 
        For documentation see temporal_functions.add_time_delay_generic. 
    """
    info = {'channels_to_be_delayed': parameters[0], 'classical_channel_for_delay': parameters[1], 'pulse_width' : parameters[2]}
    return add_time_delay_generic(collection_of_states, info, use_classical_control_channel = True)


def set_pulse_width(collection_of_states: CollectionOfStates, parameters: list) -> CollectionOfStates:
    """ Set pulse width for all channels, or for selected channels. 
        - All channels are set if parameters[0] is None or if parameters[0] is a list of all channels. In this case the states
        are not changed. Pulse width is just added or updated.
        - If parameters[0] is the empty list [] no channels are set, the function returns the original collection
        - Otherwise just the channels in parameters[0] are set. In this case channels which are not set either keep
        the original pulse width, or if this is not present get the default value for pulse width. The returned collection
        will consist of single photon states. NOTE: Changing pulse width in some channels can lead to non-physical results. We do
        not take into account absorption in spectral filters or 'chirp' in stretched pulses. It is safer to keep pulse-width the same
        for all photons and all channels.
        
        Parameters[1] contains the pulse width to set

    Args:
        collection_of_states: original collection of states
        parameters (list): parameters[0] is list of channels, parameters[1] is the desired pulse width

    Returns:
        CollectionOfStates: collection of states with modified pulse width
    """
    info = {'channels_to_set_pulse_width': parameters[0], 'pulse_width': parameters[1]}


    # if no channels have to be set return the original collection
    if info['channels_to_set_pulse_width'] == []:
        return collection_of_states

    # check if all channels have to be set, in that case use fast option
    if set(info['channels_to_set_pulse_width']) == set(range(collection_of_states._no_of_optical_channels)):

        for state in collection_of_states:
            # if key 'photon_resolution' does not exist add this key
            if not 'photon_resolution' in state.auxiliary_information.keys():
                state.auxiliary_information.update({'photon_resolution': dict([])})

            # if there is not information in timing and pulse width first fill in the default values
            if 'photon_information' not in state.auxiliary_information['photon_resolution'].keys():
                state.auxiliary_information['photon_resolution']['photon_information'] = {'time_delay': 0, 'pulse_width':1}

            # update with desired pulse width
            state.auxiliary_information['photon_resolution']['photon_information']['pulse_width'] = info['pulse_width']

        return collection_of_states
    
    # otherwise (if we want to set specific channels) create a collection of single photon states and selectively set pulse width
    else:
        # create a collection for the return value of this function
        resulting_collection = collection_of_states.copy()
        resulting_collection.clear()
        
        collection_by_column = CollectionOfStateColumns(collection_of_states=collection_of_states)
        collection_by_column.split()
        collection_by_column.single_photon_states()
        for state_old in collection_by_column.by_state():
            state = state_old.copy()
            # if key 'photon_resolution' does not exist add this key
            if not 'photon_resolution' in state.auxiliary_information.keys():
                state.auxiliary_information.update({'photon_resolution': dict([])})

            # if there is not information in timing and pulse width first fill in the default values
            if 'photon_information' not in state.auxiliary_information['photon_resolution'].keys():
                state.auxiliary_information['photon_resolution']['photon_information'] = {'time_delay': 0, 'pulse_width':1}

            # check if this state has a photon in one of the channels where we want to set pulse_width
            optical_component = list(state.optical_components.keys())[0]
            values = state._dict_of_valid_component_names[optical_component]
            set_pulse_width_for_this_state = False
            for channel in info['channels_to_set_pulse_width']:
                if (values[channel] != 0):
                    set_pulse_width_for_this_state = True
                    break

            # apply the pulse width
            if set_pulse_width_for_this_state:
                photon_info_dict = state.auxiliary_information['photon_resolution']['photon_information'] 
                photon_info_dict['pulse_width'] = info['pulse_width']
                state.auxiliary_information['photon_resolution']['photon_information'] = photon_info_dict 
            
            # add state to resulting collection
            resulting_collection.add_state(state=state)
        return resulting_collection


def photon_probability_function(photon_info_dict1: dict, photon_info_dict2: dict) -> float:
    """ Calculate the probability for detecting this photon pair depending on the overlap of the wave packets. If the photons have no overlap
        (for instance they have a narrow pulse and a large separation in time) the probability to detect them as a pair is zero. If they fully overlap
        (same pulse width and no time separation) the probability is 100%.

    Args:
        photon_info_dict1 (dict): Dictionary with keys 'time_delay' and 'pulse_width'
        photon_info_dict2 (dict): Dictionary with keys 'time_delay' and 'pulse_width'

    Returns:
        float: Probability depending on overlap between wave packets
    """

    try:
        distance = photon_info_dict1['time_delay'] - photon_info_dict2['time_delay']
        width = np.sqrt(photon_info_dict1['pulse_width']**2 + photon_info_dict2['pulse_width']**2)
        probability = np.exp(-1*np.log(2)*(distance/width)**2) 
    except:
        probability = 1

    return probability

def add_quick_time_delay_generic(collection_of_states: CollectionOfStates, info: dict, use_classical_control_channel: bool) -> CollectionOfStates:
    """ This gate will model the impact of a time delay for channels that have light (or photon) pulses with a finite 
        duration. The photon wave packets do not always overlap in this case (depending on the pulse width and time difference
        between the pulses.)

        This is a 'quick' implementation suitable in situations where bandwidth and timing are only set
        once in a circuit. This will NOT work if we have successive time delay gates (i.e., if we first delay
        a photon for a time interval +t and then move it forward with same time interval -t we expect those 
        actions to cancel out. The implementation of time delay in this function will NOT model that correctly. 
        Use the full functionality in add_time_delay/add_time_delay_classical_control for that. 
        The penalty will be slower execution.)

        This gate splits every state in the collection in three parts:
        A part representing the temporal overlap and two states represent a 'ahead' and a 'behind' part. 
        The 'overlap' state and the 'ahead'/'behind' states will be in separate interference groups.
        The ratio (cumulative_probability) between the 'overlap' and 'ahead/behind' group is determined by the 
        overlap of the photon wave packets. If there is no 'overlap' we only have 'ahead/behind', if there is full 
        overlap (same pulse_width and no time delay) we only have the 'overlap' situation and probability for 
        'ahead/behind' will have probability zero.
    
        Args:
            collection_of_states: original collection of states
            info: dictionary in form {  'channels_to_be_delayed': list, 
                                        'time_delay': float, 
                                        'classical_channel_for_delay': integer,
                                        'pulse_width': float}
            use_classical_control_channel (bool): Indicates whether to take time_delay from the dict 'info', 
                                or from 'classical_channel_for_delay'

        Returns:
            CollectionOfStates: Updated collection with 'quick' time delay implemented.
    """
   
    channel_numbers = info['channels_to_be_delayed']
    if not isinstance(channel_numbers, list):
        channel_numbers = [channel_numbers]
    
    returned_collection = collection_of_states.copy()
    returned_collection.clear()

    collection_of_columns = CollectionOfStateColumns(collection_of_states=collection_of_states)
    collection_of_columns.split()

    interference_group_identifier = 0

    # loop through all  groups in the collection
    for group in collection_of_columns.by_group():

        # determine the time_delay for this state, all states with same initial_state (and all states in one
        # interference group) should have same classical values, so we can take a random state
        for input_state in group.by_state():
            if use_classical_control_channel:
                time_delay = input_state.classical_channel_values[info['classical_channel_for_delay']]
            else:
                time_delay = info['time_delay']
            break # stop loop after first state is used to get time_delay
        
        # calculate the overlap percentage
        overlap = np.exp(-1.0*np.log(2)*np.power(time_delay/info['pulse_width'], 2.0))
        separation = 1-overlap

        # Step 1: first create a new group for the overlap situation
        # update the numbering for interference group id to create a new group
        interference_group_identifier += 1
        list_of_state_columns = []
        for column in group.by_column():
            new_column = column.copy()
            new_column.group_cumulative_probability = column.group_cumulative_probability * overlap
            for state in new_column:
                state.cumulative_probability = state.cumulative_probability * overlap
            new_column.set_photon_information({'time_delay': 0, 'pulse_width':1})
            list_of_state_columns.append(new_column)
        group_overlap = InterferenceGroup(list_of_state_columns=list_of_state_columns,interference_group_identifier=interference_group_identifier)
        
        # Step 2: then add states for the non-overlap situation 'behind'
        # update the numbering for interference group id to create a new group
        interference_group_identifier += 1
        list_of_state_columns = []
        for column in group.by_column():
            list_of_states = []
            for original_state in column:
                # split the optical components for 'behind'
                behind_optical_components = []
                for name, amp_prob in original_state.optical_components.items():
                    values = original_state._dict_of_valid_component_names[name].copy()
                    amplitude = amp_prob['amplitude']
                    for channel_index, value in enumerate(values):
                        if channel_index in channel_numbers:
                            values[channel_index] = 0
                    behind_name = input_state._dict_of_optical_values[tuple(values)]
                    behind_optical_components.append((behind_name,amplitude)) 
                behind_state = original_state.copy()
                behind_state.optical_components = behind_optical_components
                behind_state.auxiliary_information['photon_resolution']['photon_information'] = {'time_delay': -100, 'pulse_width':1}
                behind_state.cumulative_probability = original_state.cumulative_probability * separation
                list_of_states.append(behind_state)

                # split the optical components for 'ahead'
                ahead_optical_components = []
                for name, amp_prob in original_state.optical_components.items():
                    values = original_state._dict_of_valid_component_names[name].copy()
                    amplitude = amp_prob['amplitude']
                    for channel_index, value in enumerate(values):
                        if channel_index not in channel_numbers:
                            values[channel_index] = 0
                    ahead_name = input_state._dict_of_optical_values[tuple(values)]
                    ahead_optical_components.append((ahead_name,amplitude)) 
                ahead_state = original_state.copy()
                ahead_state.optical_components = ahead_optical_components
                ahead_state.auxiliary_information['photon_resolution']['photon_information'] = {'time_delay': +100, 'pulse_width':1}
                ahead_state.cumulative_probability = original_state.cumulative_probability * separation
                list_of_states.append(ahead_state)

            # prepare the column for the split states
            new_column = ColumnOfStates(list_of_states=list_of_states)
            new_column.group_cumulative_probability =column.group_cumulative_probability * separation
            list_of_state_columns.append(new_column)
        group_separation = InterferenceGroup(list_of_state_columns=list_of_state_columns,interference_group_identifier=interference_group_identifier)
        
        # Step 4: add the states from the groups to the return collection
        for state in group_overlap.by_state():
            returned_collection.add_state(state)
        for state in group_separation.by_state():
            returned_collection.add_state(state)

    return returned_collection

def add_quick_time_delay(collection_of_states: CollectionOfStates, parameters: list) -> CollectionOfStates:
    """ This functions works as a shell to unwrap 'parameters' and call 
        temporal_functions.add_quick_time_delay_generic 
        For documentation see temporal_functions.add_quick_time_delay_generic. 
    """
    info = {'channels_to_be_delayed': parameters[0], 'time_delay': parameters[1], 'pulse_width': parameters[2]}
    return add_quick_time_delay_generic(collection_of_states, info, use_classical_control_channel = False)

def add_quick_time_delay_classical_control(collection_of_states: CollectionOfStates, parameters: list) -> CollectionOfStates:
    """ This functions works as a shell to unwrap 'parameters' and call 
        temporal_functions.add_quick_time_delay_generic 
        For documentation see temporal_functions.add_quick_time_delay_generic. 
    """
    info = {'channels_to_be_delayed': parameters[0], 'classical_channel_for_delay': parameters[1], 'pulse_width': parameters[2]}
    return add_quick_time_delay_generic(collection_of_states, info, use_classical_control_channel = True)

    