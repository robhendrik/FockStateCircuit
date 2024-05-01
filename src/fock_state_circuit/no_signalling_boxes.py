""" This module contains functions to explore 'no-signalling boxes' or 'Popescu-Rohrlich' boxes'.
    See https://robhendrik.github.io/superquantum 

            Example code:
            N = 20
            for angle_first in [0,np.pi/3]:
            dict_for_plotting = dict([])
            for angle in [(a/N)*np.pi/2 for a in range(N)]:
                # define the circuit on which the no-signalling boxes will be evaluated. We will have two no-signalling
                # boxes with each two photons. We need 8 optical channels.
                circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                # Rotate the first photon in first pair (channels 0 an 1)
                circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=angle_first)
                # Rotate the second photon in first pair (channels 2 an 3)
                circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                # Rotate the first photon in second pair (channels 4 an 5)
                circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=angle_first)
                # Rotate the second photon in first pair (channels 6 an 7)
                circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                
                # create a state as 'template' 
                collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                state = fsc.State(collection_of_states=collection)

                # define and create the no-signalling boxes. Each box requires 4 channels, since each box
                # consists of two polarized photons
                ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                            {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
                collection_with_boxes = fsc.create_collection_with_NS_boxes(state, ns_boxes)
            
                # run the collection with the no signalling boxes through the circuit
                output_collection = circuit.evaluate_circuit(collection_of_states_input=collection_with_boxes)

                # measure the collection after the circuit
                histogram = fsc.measure_collection_with_NS_boxes(output_collection)
        
                # determine correlation between the photons within the photon pair (i.e., between photons that together form 
                # a no-signalling box)
                channel_combis_for_correlation = [(0,2),(4,6)]
                correlations = fsc.correlations_from_measured_collection_with_NS_boxes(histogram, channel_combis_for_correlation, state)
                
                # prepare a dictionary for plotting
                lst = []
                lst.append({'output_state': 'first pair, K = 1', 'probability': correlations[0]})
                lst.append({'output_state': 'second pair, K = 5', 'probability': correlations[1]})
                dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})
            # plot the resulting correlations for the no-signalling boxes
            fsc.plot_correlations_for_NS_boxes(dict_for_plotting)

    Last modified: April 16th, 2024
"""

import sys  
sys.path.append("../src")
sys.path.append("../../../SW Projects/GitHubClones-NewPC/FockStateCircuit009/src")
import fock_state_circuit as fsc
import importlib
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.colors
import random



def _stokes_vector_from_amplitudes(amplitude_north_pole: complex, amplitude_south_pole: complex) -> tuple:
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

def _repair_amplitudes_for_photon_pair(amplitudes) -> tuple:
    """ This function is used to create a tuple for 4 amplitudes for the HH, HV, VH and VV polarization for a photon
        pair. Here HH is the product of amplitudes for the horizontal polarization components etc. 
        The resulting set of amplitudes should lead to a state where the polarizations for both photons are
        well defined, so the photons are not entangled. If the function makes a correction a warning will be printed.

        As examples: 
            - if all amplitudes are zero, or just one amplitude is non-zero the function will return original amplitudes
            - If two amplitudes are zero and the non-zero components are for instance (HH, HV) or (HH,VH) the function will
                return orginal
            - If two amplitudes are zero and the non-zero components are (HH, VV) or (HV,VH) we have an entangled state which
                cannot be factored in well-defined polarization angles for each photon in the pair. The function will try to get to 
                a close match but this might lead to erroneous results when using results.
    Args:
        amplitudes (tuple): Original amplitudes

    Returns:
        tuple: Amplitudes after correction
    """

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
    
def _stokes_vector_for_pair_from_amplitudes_hh_hv_vh_vv(amplitudes) -> tuple:
    """ Calculate the Stokes vector for each photon in a photon pair based on the amplitudes
        of the HH, HV, VH and VV polarization amplitudes (Here HH is the product of amplitudes 
        for the horizontal polarization components etc.).

        The function will return a tuple with the two vectors in format [x_coordinate,y_coordinate,z_coordinate],
        so return is (vector_1, vector_2,length_of_amplitudes). length_of_amplitudes is the length of the vector
        [HH, HV, VH, VV].

    Args:
        amplitudes (tuple): Amplitudes for photon pair in form (HH, HV, VH, VV)

    Raises:
        Exception: 'No-signalling input cannot be factored in two independent polarizations'

    Returns:
        tuple: (vector_1, vector_2,length_of_amplitudes)
    """
    
    amplitudes = _repair_amplitudes_for_photon_pair(amplitudes)
    
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
            vector_1, length_of_amplitudes = _stokes_vector_from_amplitudes(hh,vh)
            vector_2 = h
            return (vector_1, vector_2, length_of_amplitudes)
        elif np.round(np.abs(hv),r) != 0 and np.round(np.abs(vv),r) != 0:
            vector_1, length_of_amplitudes = _stokes_vector_from_amplitudes(hv,vv)
            vector_2 = v
            return (vector_1, vector_2, length_of_amplitudes)
        elif np.round(np.abs(hh),r) != 0 and np.round(np.abs(hv),r) != 0:
            vector_1 = h
            vector_2, length_of_amplitudes = _stokes_vector_from_amplitudes(hh,hv)
            return (vector_1, vector_2, length_of_amplitudes)
        elif np.round(np.abs(vh),r) != 0 and np.round(np.abs(vv),r) != 0:
            vector_1 = v
            vector_2, length_of_amplitudes = _stokes_vector_from_amplitudes(vh,vv)
            return (vector_1, vector_2, length_of_amplitudes)
    else: # all 4 components are non zero
        length_of_amplitudes = np.sqrt(sum([np.abs(a)**2 for a in amplitudes]))
        if np.abs(hv)**2 + np.abs(vv)**2 > np.abs(hh)**2 + np.abs(vh)**2:
            vector_1, l1 = _stokes_vector_from_amplitudes(hv,vv)
        else:
            vector_1, l1 = _stokes_vector_from_amplitudes(hh,vh)

        if np.abs(hh)**2 + np.abs(hv)**2 > np.abs(vh)**2 + np.abs(vv)**2:
            vector_2, l2 = _stokes_vector_from_amplitudes(hh,hv)
        else:
            vector_2, l2 = _stokes_vector_from_amplitudes(vh,vv)

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

def _group_configurations_per_outcome(collection_of_states) -> dict:
    """ Determine the measurement outcomes from a COMPLETE measurement of the collection (i.e., measurement
        on all optical channels) with the related amplitude (so not probability which 
        would be np.abs(amplitude)**2). 
        
    Args:
        collection_of_states (fsc.CollectionOfStates): collection of states for which to determine
            the possible measurement outcomes

    Returns:
        dict: Dictionary with outcomes as keys and the configurations with their amplitudes for this 
        outcome as values
    """
    configurations_and_amplitudes_grouped_per_outcome = dict([])
    for state in collection_of_states:
        for name, amp_prob in state.optical_components.items():
            outcome = name
            amplitude = amp_prob['amplitude'] * np.sqrt(state.cumulative_probability)
            
            try:
                superentanglement_info = state.auxiliary_information['no_signalling_box']
                configuration = superentanglement_info ['configuration']
            except:
                raise Exception("Could not find 'superentanglement_info' in state.auxiliary information. Has the collection been prepared with create_collection_with_NS_boxes") 


            if outcome in configurations_and_amplitudes_grouped_per_outcome.keys():
                configurations_and_amplitudes_grouped_per_outcome[outcome].update({configuration:amplitude})
            else:
                configurations_and_amplitudes_grouped_per_outcome.update({outcome : {configuration:amplitude}})
    
    return configurations_and_amplitudes_grouped_per_outcome

def create_collection_with_NS_boxes(state, ns_boxes) -> any:
    """ Create a CollectionOfStates which can be processed in a FockStateCircuit. This collection
        represents  set of 'No signalling boxes' which can have 'superquantum' correlations.

        The argument 'state' is a state matching the circuit on which the returned collection will
        be evaluated. The exact content of the state is not important, it is used as a 'template' 
        to create the collection from.

        The argument ns_boxes should be of form:
            [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
        Here the 'channels' are indices for the optical channels. The quantumness indicator determines
        how strong the 'superquantum' correlation will be. 

        Example code:
            N = 20
            for angle_first in [0,np.pi/3]:
            dict_for_plotting = dict([])
            for angle in [(a/N)*np.pi/2 for a in range(N)]:
                # define the circuit on which the no-signalling boxes will be evaluated. We will have two no-signalling
                # boxes with each two photons. We need 8 optical channels.
                circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                # Rotate the first photon in first pair (channels 0 an 1)
                circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=angle_first)
                # Rotate the second photon in first pair (channels 2 an 3)
                circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                # Rotate the first photon in second pair (channels 4 an 5)
                circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=angle_first)
                # Rotate the second photon in first pair (channels 6 an 7)
                circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                
                # create a state as 'template' 
                collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                state = fsc.State(collection_of_states=collection)

                # define and create the no-signalling boxes. Each box requires 4 channels, since each box
                # consists of two polarized photons
                ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                            {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
                collection_with_boxes = fsc.create_collection_with_NS_boxes(state, ns_boxes)
            
                # run the collection with the no signalling boxes through the circuit
                output_collection = circuit.evaluate_circuit(collection_of_states_input=collection_with_boxes)

                # measure the collection after the circuit
                histogram = fsc.measure_collection_with_NS_boxes(output_collection)
        
                # determine correlation between the photons within the photon pair (i.e., between photons that together form 
                # a no-signalling box)
                channel_combis_for_correlation = [(0,2),(4,6)]
                correlations = fsc.correlations_from_measured_collection_with_NS_boxes(histogram, channel_combis_for_correlation, state)
                
                # prepare a dictionary for plotting
                lst = []
                lst.append({'output_state': 'first pair, K = 1', 'probability': correlations[0]})
                lst.append({'output_state': 'second pair, K = 5', 'probability': correlations[1]})
                dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})
            # plot the resulting correlations for the no-signalling boxes
            fsc.plot_correlations_for_NS_boxes(dict_for_plotting)

    Args:
        state (State): State (of type FockStateCircuit State) used as basis for the returned collection
        ns_boxes (list): list like [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]

    Returns:
        fsc.CollectionOfStates: CollectionOfStates which can be executed on FockStateCircuit.
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
            new_name = new_collection._dict_of_optical_values[tuple(values)]
            new_optical_components.append((new_name, amp_prob['amplitude']))
        new_state.optical_components = new_optical_components
        superentanglement_info = {  'superentanglement': identifier,
                                    'configuration' : configuration,
                                    'quantumness_indicator' : quantumness_indicators
                                }
        new_state.auxiliary_information.update({'no_signalling_box': superentanglement_info})
        new_collection.add_state(new_state)
    return new_collection

def measure_collection_with_NS_boxes(collection) -> list:
    """ Measure a collection after evaluation in a FockStateCircuit. The collection has to be created with
        the function 'create_collection_with_NS_boxes'

        The returned value will be a list of the form
        [{ 'output_state' : outcome1, 'probability' : probability1}, 
        { 'output_state' : outcome2, 'probability' : probability2},
        { 'output_state' : outcome3, 'probability' : probability3},
        ...]

        Example code:
            N = 20
            for angle_first in [0,np.pi/3]:
            dict_for_plotting = dict([])
            for angle in [(a/N)*np.pi/2 for a in range(N)]:
                # define the circuit on which the no-signalling boxes will be evaluated. We will have two no-signalling
                # boxes with each two photons. We need 8 optical channels.
                circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                # Rotate the first photon in first pair (channels 0 an 1)
                circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=angle_first)
                # Rotate the second photon in first pair (channels 2 an 3)
                circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                # Rotate the first photon in second pair (channels 4 an 5)
                circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=angle_first)
                # Rotate the second photon in first pair (channels 6 an 7)
                circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                
                # create a state as 'template' 
                collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                state = fsc.State(collection_of_states=collection)

                # define and create the no-signalling boxes. Each box requires 4 channels, since each box
                # consists of two polarized photons
                ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                            {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
                collection_with_boxes = fsc.create_collection_with_NS_boxes(state, ns_boxes)
            
                # run the collection with the no signalling boxes through the circuit
                output_collection = circuit.evaluate_circuit(collection_of_states_input=collection_with_boxes)

                # measure the collection after the circuit
                histogram = fsc.measure_collection_with_NS_boxes(output_collection)
        
                # determine correlation between the photons within the photon pair (i.e., between photons that together form 
                # a no-signalling box)
                channel_combis_for_correlation = [(0,2),(4,6)]
                correlations = fsc.correlations_from_measured_collection_with_NS_boxes(histogram, channel_combis_for_correlation, state)
                
                # prepare a dictionary for plotting
                lst = []
                lst.append({'output_state': 'first pair, K = 1', 'probability': correlations[0]})
                lst.append({'output_state': 'second pair, K = 5', 'probability': correlations[1]})
                dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})
            # plot the resulting correlations for the no-signalling boxes
            fsc.plot_correlations_for_NS_boxes(dict_for_plotting)

    Args:
        collection (CollectionOfStates): CollectionOfStates after evaluating with a FockStateCircuit. 

    Returns:
        list: List of output states and their probabilities
    """
    if len(collection) == 0:
        raise Exception('Empty collection passed as argument when measuring NS boxes')

    # look at first state in the collection for superentanglement information
    reference_state = list(collection._collection_of_states.values())[0]
    try:
        superentanglement_info = reference_state.auxiliary_information['no_signalling_box']
        quantumness_indicators = superentanglement_info['quantumness_indicator'].copy()
    except:
        raise Exception("Could not find 'superentanglement_info' in state.auxiliary information. Has the collection been prepared with create_collection_with_NS_boxes") 

    # configurations_and_amplitudes_grouped_per_outcome is a dictionary with as key the possible
    # measurement outcomes and as values the original configurations that contribute to that outcome 
    # and the amplitude of their contribution
    configurations_and_amplitudes_grouped_per_outcome = _group_configurations_per_outcome(collection)
    
    # histogram will be return in format {'output_state' : outcome, 'probability' : probability}
    histogram = []

    # per outcome adjust the probability based on configurations of ns-boxes
    # dict_of_configs_and_amps is a dictionary for a specific outcome, with as keys the c
    # configurations that contribute to that outcome as values the amplitude of that
    for outcome, dict_of_configs_and_amps in configurations_and_amplitudes_grouped_per_outcome.items():
        amplitudes_for_this_outcome = [amp for amp in dict_of_configs_and_amps.values()]

        # initialize the probability for this outcome from the total amplitudes
        probability = sum([np.abs(amp)**2 for amp in amplitudes_for_this_outcome])

        # for every outcome iterate over all the ns boxes to come to factor in the 
        # probability caused by 'alignment' of that box, for that specific outcome
        for box_number, quantumness_indicator in enumerate(quantumness_indicators):
            amplitudes_for_this_box = [0]*4
            # the configuration is a string of characters '0', '1', '2' and '3' indicating configs
            # hh, hv, vh and vv
            # the amplitudes have to be added to allow 'quantum interference' 
            for configuration, amplitude in dict_of_configs_and_amps.items():
                index_of_this_config = int(configuration[box_number])
                amplitudes_for_this_box[index_of_this_config] += amplitude

            if np.round(sum([np.abs(amplitude)**2 for amplitude in amplitudes_for_this_box]),5) != 0:
                try:
                    vector1, vector2, length = _stokes_vector_for_pair_from_amplitudes_hh_hv_vh_vv(tuple(amplitudes_for_this_box))
                except:
                    amplitudes = _repair_amplitudes_for_photon_pair(amplitudes)
                    vector1, vector2, length = _stokes_vector_for_pair_from_amplitudes_hh_hv_vh_vv(tuple(amplitudes_for_this_box))

                alignment_probability = probability_from_stokes_vectors(vector1, vector2,quantumness_indicator)
            else:
                alignment_probability = 0

            probability = probability*alignment_probability

        histogram.append({ 'output_state' : outcome, 'probability' : probability})

    return histogram


def perform_measurement_no_signalling_boxes(collection_of_states, optical_channels_to_measure,classical_channels_to_write_to):
    """ Performs a FULL measurement (i.e., measures all optical channels) where the detection probability is determined
        from the 'no-signalling boxes'. The collection of states needs to be build by using the gate 'create_no_signalling_boxes'.

    Args:
        collection_of_states (fsc.CollectionOfStates): Collection of states to be measured
        optical_channels_to_be_measured (list[int]): list of of optical channel numbers to be measured
        classical_channels_to_be_written (list[int]): list of classical channel numbers to write the measurement result to

    Returns:
        fsc.CollectionOfStates: Collection with 'collapsed' states and classical channels written

    """
    output_collection = collection_of_states.copy()
    output_collection.clear()

    histo = measure_collection_with_NS_boxes(collection_of_states)
    for state in collection_of_states:
        reference_state = state
        break
    del reference_state.auxiliary_information['no_signalling_box']
    reference_state.initial_state = 'no_signalling_boxes'
    for outcome in histo:
        new_state = reference_state.copy()
        new_state.cumulative_probability = outcome['probability']
        new_state.optical_components = [(outcome['output_state'],1)]
        new_classical_channel_values = [val for val in new_state.classical_channel_values]
        optical_channel_values = new_state._dict_of_valid_component_names[outcome['output_state']]
        for classical_channel, optical_channel in zip(classical_channels_to_write_to, optical_channels_to_measure):
            new_classical_channel_values[classical_channel] = optical_channel_values[optical_channel]
        new_state.classical_channel_values = new_classical_channel_values
        # prepare the attribute 'measurement_results' for the collapsed state. If there are already
        # measurement results in the input state copy these to the new collapsed state
        if state.measurement_results and state.measurement_results is not None:
            measurement_results = [previous_values for previous_values in state.measurement_results]   
            measurement_results.append({'measurement_results':new_state.classical_channel_values.copy(), 'probability': outcome['probability']})   
        else:
            measurement_results = [{'measurement_results':new_state.classical_channel_values.copy(), 'probability': outcome['probability']}]
        new_state.measurement_results = measurement_results
        output_collection.add_state(state = new_state)
    return output_collection

def correlations_from_measured_collection_with_NS_boxes(list_of_outcomes: list, 
                                                        channel_combis_for_correlation: list,
                                                        state: any) -> list:
    """ This function will return a list of correlations for the provided 'list_of_outcomes'. 

        list_of_outcomes is of the form       
                [{ 'output_state' : outcome1, 'probability' : probability1}, 
                { 'output_state' : outcome2, 'probability' : probability2},
                { 'output_state' : outcome3, 'probability' : probability3},
                ...]

        channel_combis_for_correlation is of the form:
            [(0,2),(4,6)]
        Here the tuples indicate a pair of channels between which to calculate correlations. In the example two pairs are
        indicated. 

        The return value is a list of correlations. For every pair in 'channel_combis_for_correlation' a correlation 
        will be provided in the list. For the example the returned list could be [-1.0,-1.0] or [1.0, 0.0]

        state is a template state. This state is only used to translate the 'output states' in the list of outcomes
        from a string (i.e., '101010') to a tuple of values (i.e., (1,0,1,0,1,0)) 
    
        Example code:
            N = 20
            for angle_first in [0,np.pi/3]:
            dict_for_plotting = dict([])
            for angle in [(a/N)*np.pi/2 for a in range(N)]:
                # define the circuit on which the no-signalling boxes will be evaluated. We will have two no-signalling
                # boxes with each two photons. We need 8 optical channels.
                circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                # Rotate the first photon in first pair (channels 0 an 1)
                circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=angle_first)
                # Rotate the second photon in first pair (channels 2 an 3)
                circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                # Rotate the first photon in second pair (channels 4 an 5)
                circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=angle_first)
                # Rotate the second photon in first pair (channels 6 an 7)
                circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                
                # create a state as 'template' 
                collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                state = fsc.State(collection_of_states=collection)

                # define and create the no-signalling boxes. Each box requires 4 channels, since each box
                # consists of two polarized photons
                ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                            {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
                collection_with_boxes = fsc.create_collection_with_NS_boxes(state, ns_boxes)
            
                # run the collection with the no signalling boxes through the circuit
                output_collection = circuit.evaluate_circuit(collection_of_states_input=collection_with_boxes)

                # measure the collection after the circuit
                histogram = fsc.measure_collection_with_NS_boxes(output_collection)
        
                # determine correlation between the photons within the photon pair (i.e., between photons that together form 
                # a no-signalling box)
                channel_combis_for_correlation = [(0,2),(4,6)]
                correlations = fsc.correlations_from_measured_collection_with_NS_boxes(histogram, channel_combis_for_correlation, state)
                
                # prepare a dictionary for plotting
                lst = []
                lst.append({'output_state': 'first pair, K = 1', 'probability': correlations[0]})
                lst.append({'output_state': 'second pair, K = 5', 'probability': correlations[1]})
                dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})
            # plot the resulting correlations for the no-signalling boxes
            fsc.plot_correlations_for_NS_boxes(dict_for_plotting)

    Args:
        list_of_outcomes (list): list of outcomes as returned from 'measure_collection_with_NS_boxes'
        channel_combis_for_correlation (list): list of tuples where each tuple indicates a pair of channels.
        state (fsc.State): _description_

    Returns:
        list: list of correlations, for every channel combi a correlation will be returned
    """

    correlations = []
    for combi in channel_combis_for_correlation:
        corr = 0
        count = 0
        for res_as_dict in list_of_outcomes:
            values = state._dict_of_valid_component_names[res_as_dict['output_state']].copy()
            if (values[combi[0]] == values[combi[1]]):
                corr += res_as_dict['probability']
            else:
                corr -= res_as_dict['probability']
            count += res_as_dict['probability']
        correlations.append(corr/count)
    return correlations

def plot_correlations_for_NS_boxes(dict_for_plotting) -> None:
    """ Plot an histogram of correlations. 

        The input dictionary should be of form 
        {initial_state1 : [{'output_state':'1101', 'correlation': 0.5}, {'output_state':'3000', 'correlation': 0.5}] }
        So for every initial state there should be a list of possible outcomes. The elements in the list are of the form
        {'output_state':'1101', 'correlation': 0.5}. The value for 'correlation' should be between 1 and -1 

        NOTE: For backwards compatibility the function also accepts 'probability' instead of 'correlation' as key in the outcomes.
     
        Example code:
            N = 20
            for angle_first in [0,np.pi/3]:
            dict_for_plotting = dict([])
            for angle in [(a/N)*np.pi/2 for a in range(N)]:
                # define the circuit on which the no-signalling boxes will be evaluated. We will have two no-signalling
                # boxes with each two photons. We need 8 optical channels.
                circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                # Rotate the first photon in first pair (channels 0 an 1)
                circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=angle_first)
                # Rotate the second photon in first pair (channels 2 an 3)
                circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                # Rotate the first photon in second pair (channels 4 an 5)
                circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=angle_first)
                # Rotate the second photon in first pair (channels 6 an 7)
                circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                
                # create a state as 'template' 
                collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                state = fsc.State(collection_of_states=collection)

                # define and create the no-signalling boxes. Each box requires 4 channels, since each box
                # consists of two polarized photons
                ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                            {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
                collection_with_boxes = fsc.create_collection_with_NS_boxes(state, ns_boxes)
            
                # run the collection with the no signalling boxes through the circuit
                output_collection = circuit.evaluate_circuit(collection_of_states_input=collection_with_boxes)

                # measure the collection after the circuit
                histogram = fsc.measure_collection_with_NS_boxes(output_collection)
        
                # determine correlation between the photons within the photon pair (i.e., between photons that together form 
                # a no-signalling box)
                channel_combis_for_correlation = [(0,2),(4,6)]
                correlations = fsc.correlations_from_measured_collection_with_NS_boxes(histogram, channel_combis_for_correlation, state)
                
                # prepare a dictionary for plotting
                lst = []
                lst.append({'output_state': 'first pair, K = 1', 'probability': correlations[0]})
                lst.append({'output_state': 'second pair, K = 5', 'probability': correlations[1]})
                dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})
            # plot the resulting correlations for the no-signalling boxes
            fsc.plot_correlations_for_NS_boxes(dict_for_plotting)

    Args:
        dict_for_plotting (dict): {initial_state1 : [{'output_state':'1101', 'probability': 0.5}, {'output_state':'3000', 'probability': 0.5}] }
    """

    _DEFAULT_PLOT_SETTINGS = {
        'figsize' : (16,8)
    }
    plt.rcParams['figure.figsize'] = [15,6]
    # create a list of all possible outcomes across all initial states
    output_states = []
    for initial_state, list_of_outcomes in dict_for_plotting.items():
        for output_probability in list_of_outcomes:
            outcome = output_probability['output_state']
            if outcome not in output_states:
                output_states.append(outcome)
    info_for_bar_plot = dict([])
    no_initial_states = len(dict_for_plotting)
    no_output_states = len(output_states)
    width = 0.8/no_output_states # spread the bars over 80% of the distance between ticks on x -axis
    mid = no_output_states//2
    # cycle through standard color list 
    cycle = info_for_bar_plot.get('colors', list(matplotlib.colors.TABLEAU_COLORS))
    greys = ['whitesmoke','whitesmoke']


    fig, ax = plt.subplots(1, 1,figsize = _DEFAULT_PLOT_SETTINGS['figsize'])

    for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
        x = list(dict_for_plotting.keys()).index(initial_state)
        for i in range(no_output_states):
            ax.bar(x+(i-mid)*width, 
                    1.2,
                    color = greys[i%len(greys)],
                    width = width
                    )
            ax.bar(x+(i-mid)*width, 
                    -1.2,
                    color = greys[i%len(greys)],
                    width = width
                    )
    for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
        for outcome in outcomes_for_that_initial_state:
            if 'probability' in outcome.keys():
                value_to_plot = outcome['probability']
            else:
                value_to_plot = outcome['correlation']
            x = list(dict_for_plotting.keys()).index(initial_state)
            i = output_states.index(outcome['output_state'])
            ax.bar(x+(i-mid)*width, 
                    value_to_plot,
                    color = cycle[i%len(cycle)],
                    width = width
                    )
    custom_lines = [matplotlib.lines.Line2D([0], [0], color = cycle[i%len(cycle)], lw=4) for i in range(len(output_states))]
    ax.legend(custom_lines, [outcome for outcome in  output_states])
    plt.xticks(rotation=90)
    plt.xticks([x for x in range(no_initial_states)], list(dict_for_plotting.keys()))
    plt.ylabel(info_for_bar_plot.get('ylabel', 'Probability'))
    text = info_for_bar_plot.get('title', 'Probabilities for going from input state to measurement result')
    plt.title(text) 
    plt.show()
    return
