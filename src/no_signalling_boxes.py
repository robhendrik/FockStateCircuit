""" This module contains functions to explore 'no-signalling boxes' or 'Popescu-Rohrlich' boxes'.
    See https://robhendrik.github.io/superquantum 
"""

import sys  
sys.path.append("../src")
sys.path.append("../../../SW Projects/GitHubClones-NewPC/FockStateCircuit009/src")
import fock_state_circuit as fsc
import collection_of_states as cos
import importlib
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.colors
import random

def stokes_vector_from_amplitudes(amplitude_north_pole: complex, amplitude_south_pole: complex) -> tuple:
    """ Calculates the Stokes vector from amplitudes. The function returns a tuple containing
        a list with 3 floats representing x,y and z coordinates of the vector, and a float
        representing the normalization factor (the length of the vector before normalization). The vector
        represented by the list has length 1. The north-south pole is the z-axis (3rd coordinate). 
        If the phase difference between the amplitudes is zero the vector in the xy-plane is 
        pointing to the y-coordinate.

        As example: if northpole represents horizontal polarization and south pole vertical polarization 
        then the 'y-pole' represents diagonal polarization and the 'x-pole' circular polarization.

    Args:
        amplitude_north_pole (numpy complex): amplitude of a component of the wave functio
        amplitude_south_pole (numpy complex): amplitude of a component of the wave functio

    Returns:
        tuple: tuple (vector, normalization). vector is a list of 3 floats representing x,y and x coordinates 
        of the Stokes vector.

    Raises:
        Exception when both amplitudes have length zero
    """
    normalization_factor = np.abs(amplitude_north_pole)**2 + np.abs(amplitude_south_pole)**2
    if normalization_factor == 0:
        raise Exception("Stokes vector cannot have length zero")
    z_coordinate = (np.abs(amplitude_north_pole)**2 -np.abs(amplitude_south_pole)**2)/normalization_factor

    xy_length = np.sqrt(1-z_coordinate**2)
    if np.abs(amplitude_north_pole) == 0 or np.abs(amplitude_south_pole) == 0:
        phase_between_amplitudes = 0
    else:
        phase_between_amplitudes = np.angle(amplitude_north_pole/amplitude_south_pole)
    y_coordinate = np.cos(phase_between_amplitudes) * xy_length
    x_coordinate = np.sin(phase_between_amplitudes) * xy_length

    return ([x_coordinate,y_coordinate,z_coordinate] , np.sqrt(normalization_factor))

def repair_amplitudes(amplitudes):

    r = 8
    hh, hv, vh, vv = amplitudes
    
    non_zero_count = np.count_nonzero(np.round(np.abs(amplitudes),r) != 0)

    if non_zero_count == 0:
        # amplitudes are all zero
        return (0,0,0,0)
    if non_zero_count == 1:
        return amplitudes
    if non_zero_count == 2:
        valid_combinations = [(0,1),(2,3),(0,2),(1,3)]
        for combi in valid_combinations:
            two_elements = [amplitudes[combi[0]],amplitudes[combi[1]]]
            if np.count_nonzero(np.round(np.abs(two_elements),r) != 0) == 2:
                return amplitudes
        # if not returned from function we have an exception that needs to be corrected
        print('Warning: correction done on amplitudes. Values before: ', np.round(np.abs(amplitudes),r))
        for i in range(4):
            if np.round(np.abs(amplitudes[i]),r) == max(np.round(np.abs(amplitudes),r)):
                new_values = [0]*4
                new_values[i] = amplitudes[i]
                print('new_values: ', np.round(np.abs(new_values),r))
                return tuple(new_values)
    if non_zero_count == 3:
                mid_value = 1
                for i in range(4):
                    if np.round(np.abs(amplitudes[i]),r) == max(np.round(np.abs(amplitudes),r)):
                        max_value = amplitudes[i]
                    elif np.round(np.abs(amplitudes[i]),r) == 0:
                        zero_value = amplitudes[i]
                        zero_index = i
                    else:
                        mid_value = mid_value * amplitudes[i]
                smallest_value = mid_value/max_value
                if np.round(np.abs(smallest_value)) != 0:
                    print('Warning: correction done on amplitudes. Values before: ', np.round(np.abs(amplitudes),r))
                    new_values = amplitudes.copy()
                    new_values[zero_index] = smallest_value
                    print('new_values: ', np.round(np.abs(new_values),r))
                    return tuple(new_values)
                else:
                    return amplitudes
    else:
        return amplitudes
    
def stokes_vector_for_pair_from_amplitudes_hh_hv_vh_vv(amplitudes) -> tuple:
    """
        Args:
        amplitudes (_type_): _description_

    Returns:
        tuple: _description_

    """
    
    amplitudes = repair_amplitudes(amplitudes)
    
    #rounding paramater
    r = 8
    hh, hv, vh, vv = amplitudes
    
    h,v = [0,0,1],[0,0,-1]
    non_zero_count = np.count_nonzero(np.round(np.abs(amplitudes),r) != 0)

    if non_zero_count == 0:
        print("amplitudes given as input: ", amplitudes)
        print(np.round(np.abs(amplitudes),r))
        raise Exception('No-signalling input cannot be factored in two independent polarizations, all amps zero')
    elif non_zero_count == 1:
        if np.round(np.abs(hh),r) != 0:
            return (h,h,np.abs(hh))
        elif np.round(np.abs(hv),r) != 0:
            return (h,v,np.abs(hv))
        elif np.round(np.abs(vh),r) != 0:
            return (v,h,np.abs(vh))
        else: #np.round(np.abs(vv),r) != 0
            return (v,v,np.abs(vv))
    elif non_zero_count == 2:
        if np.round(np.abs(hh),r) != 0 and np.round(np.abs(vv),r) != 0:
            raise Exception('No-signalling input cannot be factored in two independent polarizations')
        elif np.round(np.abs(hv),r) != 0 and np.round(np.abs(vh),r) != 0:
            raise Exception('No-signalling input cannot be factored in two independent polarizations')
        elif np.round(np.abs(hh),r) != 0 and np.round(np.abs(vh),r) != 0:
            vector_1, length_of_amplitudes = stokes_vector_from_amplitudes(hh,vh)
            vector_2 = h
            return (vector_1, vector_2, length_of_amplitudes)
        elif np.round(np.abs(hv),r) != 0 and np.round(np.abs(vv),r) != 0:
            vector_1, length_of_amplitudes = stokes_vector_from_amplitudes(hv,vv)
            vector_2 = v
            return (vector_1, vector_2, length_of_amplitudes)
        elif np.round(np.abs(hh),r) != 0 and np.round(np.abs(hv),r) != 0:
            vector_1 = h
            vector_2, length_of_amplitudes = stokes_vector_from_amplitudes(hh,hv)
            return (vector_1, vector_2, length_of_amplitudes)
        elif np.round(np.abs(vh),r) != 0 and np.round(np.abs(vv),r) != 0:
            vector_1 = v
            vector_2, length_of_amplitudes = stokes_vector_from_amplitudes(vh,vv)
            return (vector_1, vector_2, length_of_amplitudes)
    else: # all 4 components are non zero
        length_of_amplitudes = np.sqrt(sum([np.abs(a)**2 for a in amplitudes]))
        if np.abs(hv)**2 + np.abs(vv)**2 > np.abs(hh)**2 + np.abs(vh)**2:
            vector_1, l1 = stokes_vector_from_amplitudes(hv,vv)
        else:
            vector_1, l1 = stokes_vector_from_amplitudes(hh,vh)

        if np.abs(hh)**2 + np.abs(hv)**2 > np.abs(vh)**2 + np.abs(vv)**2:
            vector_2, l2 = stokes_vector_from_amplitudes(hh,hv)
        else:
            vector_2, l2 = stokes_vector_from_amplitudes(vh,vv)

        return (vector_1, vector_2,length_of_amplitudes)


def probability_from_stokes_vectors(vector1, vector2,quantumness_value: int = 1) -> float:
    """ Calculates the correlation probability. This is the probability to get the same outcome behind 
        two detectors. Note that the probability to actually detect two photons is half this value, as 
        there is 50% to detect no photon behind either polarizer and 50% to actually detect two photons.

    Args:
        vector1 ([float]): Stokes vector indiction polarization orientation on Poincare sphere
        vector2 ([float]): Stokes vector indiction polarization orientation on Poincare sphere
        quantumness_value (int, optional): _description_. Defaults to 1.

    Returns:
        float: correlation probability
    """
    # the probability for coincidence detection (so probability to see a photon on both detectors)
    # is equal to the cosine of the angle difference on the Poincare sphere.
    # If this angle is zero the probability is 1, if this angle is 180 degree the probability is zero
    # 180 degree on Poincare sphere is 90 degree between polarization orientation, so orthogonal
    probability_amplitude = np.inner(vector1,vector2)/np.sqrt(np.inner(vector1,vector1)*np.inner(vector2,vector2))
    sign = np.sign(probability_amplitude)
    probability_amplitude = sign*np.power(sign*probability_amplitude,1/quantumness_value)

    return (1/2)*(probability_amplitude + 1) 

def create_collection_with_NS_boxes(state, ns_boxes) -> cos.CollectionOfStates:
    """_summary_

    Args:
        state (_type_): _description_
        ns_boxes (_type_): _description_

    Returns:
        cos.CollectionOfStates: _description_
    """
    number_of_states = 4**len(ns_boxes)
    identifier = "".join([str(random.randint(0,10)) for _ in range(50)])
    # configurations will be indicated by string of characters '0','1','2' and '3',
    # meaning hh, hv, vh and vv
    channel_values_for_config = [[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]]
    configurations = [np.base_repr(n, base=4, padding=len(ns_boxes)) for n in range(number_of_states)]
    length_of_string = len(ns_boxes)
    configurations = [ s[len(s) - length_of_string:len(s)] for s in configurations]
    quantumness_indicators = [box['quantumness_indicator'] for box in ns_boxes]
    new_collection = state._collection_of_states.copy()
    new_collection.clear()

    for configuration in configurations:       
        new_state = state.copy()
        new_state.cumulative_probability = 1
        new_state.initial_state = configuration
        new_optical_components = []
        for name, amp_prob in state.optical_components.items():
            values = new_collection._dict_of_valid_component_names[name].copy()
            for configuration_for_this_box,ns_box in zip(configuration,ns_boxes):
                channels_Ah_Av_Bh_Bv = ns_box['channels_Ah_Av_Bh_Bv']
                values_in_channels = channel_values_for_config[int(configuration_for_this_box)]
                for index,channel in enumerate(channels_Ah_Av_Bh_Bv):
                    values[channel] = values_in_channels[index]
            new_name = new_collection._get_state_name_from_list_of_photon_numbers(values)
            new_optical_components.append((new_name, amp_prob['amplitude']))
        new_state.optical_components = new_optical_components
        superentanglement_info = {  'superentanglement': identifier,
                                    'configuration' : configuration,
                                    'quantumness_indicator' : quantumness_indicators
                                }
        new_state.measurement_results.append(superentanglement_info)
        new_collection.add_state(new_state)
    return new_collection

def group_configurations_per_outcome(collection_of_states) -> dict:
    """ Determine the measurement outcomes from a COMPLETE measurement of the collection (i.e., measurement
        on all optical channels) with the related amplitude (so not probability which 
        would be np.abs(amplitude)**2). 
        
    Args:
        collection_of_states (cos.CollectionOfStates): collection of states for which to determine
            the possible measurement outcomes

    Returns:
        dict: Dictionary with outcomes as keys and the configurations with their amplitudes for this 
        outcome as values
    """
    # no_of_optical_channels = collection_of_states._no_of_optical_channels
    # no_of_classical_channels = no_of_optical_channels
    # length_of_fock_state = collection_of_states._length_of_fock_state
    # measurement_circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,
    #                                            no_of_optical_channels=no_of_optical_channels,
    #                                            no_of_classical_channels=no_of_classical_channels)
    # measurement_circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(no_of_optical_channels)],
    #                                                  classical_channels_to_be_written=[n for n in range(no_of_optical_channels)])
    # if collection_of_states._no_of_classical_channels < no_of_optical_channels:
    #     classical_channels_to_be_added = no_of_optical_channels - collection_of_states._no_of_classical_channels 
    #     collection_of_states.extend(extra_classical_channels = classical_channels_to_be_added)
    # result = measurement_circuit.evaluate_circuit(collection_of_states_input=collection_of_states)
    configurations_and_amplitudes_grouped_per_outcome = dict([])
    # for state in result:
    for state in collection_of_states:
        for name, amp_prob in state.optical_components.items():
            #name, amp_prob = list(state.optical_components.items())[0]
            outcome = name
            amplitude = amp_prob['amplitude'] * np.sqrt(state.cumulative_probability)
            list_of_ms_results = state.measurement_results
            for ms_res in list_of_ms_results[::-1]:
                if 'superentanglement' in ms_res.keys():
                    break
            configuration = ms_res['configuration']
            if outcome in configurations_and_amplitudes_grouped_per_outcome.keys():
                configurations_and_amplitudes_grouped_per_outcome[outcome].update({configuration:amplitude})
            else:
                configurations_and_amplitudes_grouped_per_outcome.update({outcome : {configuration:amplitude}})
    
    return configurations_and_amplitudes_grouped_per_outcome

def measure_collection_with_NS_boxes(collection) -> list:
    """_summary_

    Args:
        collection (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        dict: _description_
    """
    if len(collection) == 0:
        raise Exception('Empty collection passed as argument when measuring NS boxes')
    # take input on NS-boxes from first state in the circuit
    list_of_ms_results = list(collection._collection_of_states.values())[0].measurement_results
    for ms_res in list_of_ms_results[::-1]:
        if 'superentanglement' in ms_res.keys():
            break
    quantumness_indicators = ms_res['quantumness_indicator'].copy()
    #print('quantumness_indicators', quantumness_indicators)
    # configurations_and_amplitudes_grouped_per_outcome is a dictionary with as key the possible
    # measurement outcomes and as values the original configurations that contribute to that outcome 
    # and the amplitude of their contribution
    configurations_and_amplitudes_grouped_per_outcome = group_configurations_per_outcome(collection)
    # for key,val in configurations_and_amplitudes_grouped_per_outcome.items():
    #     print(key)
    #     for k,b in val.items():
    #         print(k,b)
    
    # histogram will be return in format {'output_state' : outcome, 'probability' : probability}
    histogram = []

    # per outcome adjust the probability based on configurations of ns-boxes
    # dict_of_configs_and_amps is a dictionary for a specific outcome, with as keys the c
    # configurations that contribute to that outcome as values the amplitude of that
    for outcome, dict_of_configs_and_amps in configurations_and_amplitudes_grouped_per_outcome.items():
        amplitudes_for_this_outcome = [amp for amp in dict_of_configs_and_amps.values()]

        # initialize the probability for this outcome from the total amplitudes
        probability = sum([np.abs(amp)**2 for amp in amplitudes_for_this_outcome])
        #print('for this outcome: ',outcome, 'these are the contributions: ', dict_of_configs_and_amps)
        #print('original probability: ', probability)
        # for every outcome iterate over all the ns boxes to come to factor in the 
        # probability caused by 'alignment' of that box, for that specific outcome
        for box_number, quantumness_indicator in enumerate(quantumness_indicators):
            #print('box_number', box_number)
            amplitudes_for_this_box = [0]*4
            # the configuration is a string of characters '0', '1', '2' and '3' indicating configs
            # hh, hv, vh and vv
            # the amplitudes have to be added to allow 'quantum interference' 
            for configuration, amplitude in dict_of_configs_and_amps.items():
                index_of_this_config = int(configuration[box_number])
                amplitudes_for_this_box[index_of_this_config] += amplitude
            #print('Amplitudes for this box: ', amplitudes_for_this_box)
            if np.round(sum([np.abs(amplitude)**2 for amplitude in amplitudes_for_this_box]),5) != 0:
                try:
                    vector1, vector2, length = stokes_vector_for_pair_from_amplitudes_hh_hv_vh_vv(tuple(amplitudes_for_this_box))
                except:
                    amplitudes = repair_amplitudes(amplitudes)
                    vector1, vector2, length = stokes_vector_for_pair_from_amplitudes_hh_hv_vh_vv(tuple(amplitudes_for_this_box))
                #print(stokes_vector_for_pair_from_amplitudes_hh_hv_vh_vv(tuple(amplitudes_for_this_box)))
                alignment_probability = probability_from_stokes_vectors(vector1, vector2,quantumness_indicator)
            else:
                alignment_probability = 0
            #print('Vectors for this box: ', vector1, vector2)
            #print('Probability correction from box no: ', box_number, 'prob: ', alignment_probability)
            probability = probability*alignment_probability
        #print("final result for this outcome ",{ 'output_state' : outcome, 'probability' : probability})
        histogram.append({ 'output_state' : outcome, 'probability' : probability})


    return histogram

def correlation_from_histogram(histogram, channel_combis_for_correlation, digits_per_channel: int = 1):
    """_summary_

    Args:
        histogram (_type_): _description_
        channel_combis_for_correlation (_type_): _description_
        digits_per_channel (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    correlations = []
    for combi in channel_combis_for_correlation:
        corr = 0
        count = 0
        for res_as_dict in histogram:
            this_outcome_correlates = True
            for index, channel in enumerate(combi):            
                start = digits_per_channel * channel
                end = digits_per_channel * (channel+1)
                value_new = int(res_as_dict['output_state'][start:end])
                if not index == 0:
                    if value_new != value_old:
                        this_outcome_correlates = False
                        break
                value_old = value_new 
            if this_outcome_correlates:
                corr += res_as_dict['probability']
            else:
                corr -= res_as_dict['probability']
            count += res_as_dict['probability']
        correlations.append(corr/count)
    return correlations

def plot_correlations(dict_for_plotting, info_for_bar_plot = dict([])):    

    # create a list of all possible outcomes across all initial states
    output_states = []
    for initial_state, list_of_outcomes in dict_for_plotting.items():
        for output_probability in list_of_outcomes:
            outcome = output_probability['output_state']
            if outcome not in output_states:
                output_states.append(outcome)

    # for states who do not lead to a certain outcome add the outcome with probability zero
    # this enables easier plotting of graphs. If the outcome is absent it will need to be corrected
    # later, so better to add the outcome with probability zero
    for initial_state, list_of_outcomes in dict_for_plotting.items():
        outcomes_for_this_initial_state = []
        for output_probability in list_of_outcomes:
            outcomes_for_this_initial_state.append(output_probability['output_state'])
        for outcome in list(set(output_states).difference(outcomes_for_this_initial_state)):
            dict_for_plotting[initial_state].append({'output_state': outcome, 'probability': 0})

    no_initial_states = len(dict_for_plotting)
    no_output_states = len(output_states)
    width = 0.8/no_output_states # spread the bars over 80% of the distance between ticks on x -axis
    mid = no_output_states//2
    # cycle through standard color list 
    cycle = info_for_bar_plot.get('colors', list(matplotlib.colors.TABLEAU_COLORS))
    greys = ['whitesmoke','whitesmoke']
    
    for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
        x = list(dict_for_plotting.keys()).index(initial_state)
        for i in range(no_output_states):
            plt.bar(x+(i-mid)*width, 
                    1.2,
                    color = greys[i%len(greys)],
                    width = width
                    )
            plt.bar(x+(i-mid)*width, 
                    -1.2,
                    color = greys[i%len(greys)],
                    width = width
                    )
    for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
        for outcome in outcomes_for_that_initial_state:
            x = list(dict_for_plotting.keys()).index(initial_state)
            i = output_states.index(outcome['output_state'])
            plt.bar(x+(i-mid)*width, 
                    outcome['probability'],
                    color = cycle[i%len(cycle)],
                    width = width
                    )
    custom_lines = [matplotlib.lines.Line2D([0], [0], color = cycle[i%len(cycle)], lw=4) for i in range(len(output_states))]
    plt.rcParams['figure.figsize'] = [15,6]
    plt.legend(custom_lines, [outcome for outcome in  output_states])
    plt.xticks(rotation=90)
    plt.xticks([x for x in range(no_initial_states)], list(dict_for_plotting.keys()))
    plt.ylabel(info_for_bar_plot.get('ylabel', 'Correlation'))
    text = info_for_bar_plot.get('title', 'correlations')
    plt.title(text) 
    plt.show()
    return

def inner_product(state1, state2):
    result = np.cdouble(0)
    for comp1, amp_prob1 in state1.optical_components.items():
        for comp2, amp_prob2 in state2.optical_components.items():
            if comp1 == comp2:
                result += np.conjugate(amp_prob1['amplitude']) * amp_prob2['amplitude']
    return result

def outer_product(state1, state2):
    vector1, basis1 = state1.translate_state_components_to_vector()
    vector2, basis2 = state2.translate_state_components_to_vector()
    return  np.outer(np.conjugate(vector1),vector2)

def tensor_product(state1, state2):
    if isinstance(state1,cos.State) and isinstance(state2,cos.State):
        new_optical_components = dict([])
        for comp1, amp_prob1 in state1.optical_components.items():
            for comp2, amp_prob2 in state2.optical_components.items():
                new_amp = amp_prob1['amplitude'] * amp_prob2['amplitude']
                new_optical_components.update({comp1 + comp2: {'amplitude': new_amp, 'probability': np.abs(new_amp)**2 }})
        return new_optical_components
    
    if isinstance(state1,list) and isinstance(state2,list):
        new_optical_components = dict([])
        for comp1, amp1 in state1:
            for comp2, amp2 in state2:
                new_amp = amp1 * amp2
                new_name = comp1 + comp2
                if new_name in new_optical_components.keys():
                    new_amp += new_optical_components[new_name]['amplitude']
                new_optical_components.update({comp1 + comp2: {'amplitude': new_amp, 'probability': np.abs(new_amp)**2 }})
        return new_optical_components
    
    if isinstance(state1,dict) and isinstance(state2,dict):
        new_optical_components = dict([])
        for comp1, amp_prob1 in state1.items():
            for comp2, amp_prob2 in state2.items():
                new_amp = amp_prob1['amplitude'] * amp_prob2['amplitude']
                new_optical_components.update({comp1 + comp2: {'amplitude': new_amp, 'probability': np.abs(new_amp)**2 }})
        return new_optical_components

