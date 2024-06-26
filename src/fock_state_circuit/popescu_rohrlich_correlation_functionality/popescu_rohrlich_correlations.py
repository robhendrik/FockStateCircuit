""" This module contains functions to explore super-quantum correlations between 'Popescu-Rohrlich' photon pairs.
    The entanglement between these photons is beyond the Tsirelson bound, so in a CHSH correlation experiment
    we can achieve correlations up to 4. Also we can use these photons for communication beyond what is allowed in 
    information theory, as proposed by Marcin Pawlosky and also van Dam.
    
    See 
    * https://robhendrik.github.io/superquantum 
    * S. Popescu and D. Rohrlich, “Quantum Nonlocality as an Axiom,” Foundations of Physics 24, 397 (1994).
    * M. Pawłowski, T. Paterek, D. Kaszlikowski, V. Scarani , A. Winter and M. Zukowski, 
    “Information causality as a physical principle,” Nature 461, 1101 (2009). https://doi.org/10.1038/nature08400
    * W. van Dam, “Implausible consequences of superstrong nonlocality,” 
    Nat Comput 12, 9–12 (2013). https://doi.org/10.1007/s11047-012-9353-6

    The 'entry' function is 'execute_popescu_rohrlich_gate' and the 'exit' function is 'perform_measurement_popescu_rohrlich_correlation'.
    The input is a collection of states. For every state in this collection a new state is created with the PR photons pairs as specifief in 
    'pr-correlation' (here the channels_Ah_Av_Bh_Bv determines which optical channels carry the h(orizontal) and v(ertical) polarization for 
    photon A or B in the pair. The 'quantumness_indicator' determines how 'superquantum' the correlation is. For 'quantumness_indicator = 1' 
    we have regular entanglement, in the limit of this parameter to infinity the entanglement becomes maximally non-local (so you an try to run
    with 'quantumness_indicator = 1000' to see what happens. In practice 'quantumness_indicator = 10' already gives strong non-locality beyond quantum 
    mechanics).

    The label 'initial_state' is maintained and after 'perform_measurement_popescu_rohrlich_correlation' we have a regular collection of states, where
    the effect of the PR correlation is that the probabilities to find specific outcomes deviates from what would normally expected.

    Example:
        If we have one pair of entangled photons with quantumness 1 we use 
        '''pr_correlation = [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1} ]'''
        If then apply the gate
        '''circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)'''
        we can use normal gates like beamsplitters and waveplates to run the collection of states through the circuit.

        Because we use quantumness_indicator = 1 we will find the same result as for regular entanglement. So we also could
        have started with a state with optical channels 0.71 x ('1010'+ '0101')

    Note:
        1. The measurement will always be a complete measurement where all optical channels are measured. So after the measurement
            the state will collapse to a statistical mixture of pure photon number states, without superposition
        2. There are fundamental limits to this pr-correlation (which is probably why we do not find it in nature).You cannot create entanglement
            between photons in different pairs. We need to be able to determin from the measurement result what the original polarization in 
            the photon pair was. If this is not possible the system will generate and error, or at least erroneous results.

    Methods:
        perform_measurement_popescu_rohrlich_correlation(collection_of_states: CollectionOfStates, # type: ignore
                                                    optical_channels_to_measure: list,
                                                    classical_channels_to_write_to: list) -> CollectionOfStates: # type: ignore

                Performs a FULL measurement (i.e., measures all optical channels) where the detection probability is determined
                from the 'popescu_rohrich_correlation'. The collection of states needs to be build by using the gate 'popescu_rohrlich_correlation_gate'.

                Args:
                    collection_of_states (fsc.CollectionOfStates): Collection of states to be measured
                    optical_channels_to_be_measured (list[int]): list of of optical channel numbers to be measured
                    classical_channels_to_be_written (list[int]): list of classical channel numbers to write the measurement result to

                Returns:
                    fsc.CollectionOfStates: Collection with 'collapsed' states and classical channels written

        execute_popescu_rohrlich_gate(collection: CollectionOfStates, 
                                        pr_correlation) -> CollectionOfStates: # type: ignore

                For each state in collection create a set of Popescu-Rohrlich photon pairs. The resulting collection will
                contain all states needed to process the Popescu_Rohrlich correlation

                The argument pr_correlation should be of form:
                    [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                        {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
                Here the 'channels' are indices for the optical channels. The quantumness indicator determines
                how strong the 'superquantum' correlation will be. 

                Args:
                    collection (CollectionOfStates): Collection used as basis for the returned collection
                    pr_correlation (list[dict]): list like [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                            {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]

                Returns:
                    fsc.CollectionOfStates: CollectionOfStates which can be executed on FockStateCircuit.
    

    Last modified: June 12th, 2024
"""
from __future__ import annotations
import fock_state_circuit as fsc
from fock_state_circuit.popescu_rohrlich_correlation_functionality.popescu_rohrlich_photon_pair_amplitude_functionality import PopescuRohrlichPhotonPairAmplitudes
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.colors
import random

_VERSION = '1.0.2'

def _probability_from_stokes_vectors(vector1, vector2,quantumness_value: int = 1) -> float:
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

def _group_configurations_per_outcome(collection_of_states: CollectionOfStates) -> dict: # type: ignore
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
                superentanglement_info = state.auxiliary_information['popescu_rohrlich_correlation']
                configuration = superentanglement_info ['configuration']
            except:
                raise Exception("Could not find 'superentanglement_info' in state.auxiliary information. Has the collection been prepared with create_collection_with_NS_boxes") 


            if outcome in configurations_and_amplitudes_grouped_per_outcome.keys():
                configurations_and_amplitudes_grouped_per_outcome[outcome].update({configuration:amplitude})
            else:
                configurations_and_amplitudes_grouped_per_outcome.update({outcome : {configuration:amplitude}})
    
    return configurations_and_amplitudes_grouped_per_outcome

def execute_popescu_rohrlich_gate(collection: CollectionOfStates, pr_correlation) -> CollectionOfStates: # type: ignore
    """ For each state in collection create a set of Popescu-Rohrlich photon pairs. The resulting collection will
        contain all states needed to process the Popescu_Rohrlich correlation

        The argument pr_correlation should be of form:
            [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
        Here the 'channels' are indices for the optical channels. The quantumness indicator determines
        how strong the 'superquantum' correlation will be. 

        The input is a collection of states. For every state in this collection a new state is created with the PR photons pairs as specifief in 
        'pr-correlation' (here the channels_Ah_Av_Bh_Bv determines which optical channels carry the h(orizontal) and v(ertical) polarization for 
        photon A or B in the pair. The 'quantumness_indicator' determines how 'superquantum' the correlation is. For 'quantumness_indicator = 1' 
        we have regular entanglement, in the limit of this parameter to infinity the entanglement becomes maximally non-local (so you an try to run
        with 'quantumness_indicator = 1000' to see what happens. In practice 'quantumness_indicator = 10' already gives strong non-locality beyond quantum 
        mechanics).

        The label 'initial_state' is maintained and after 'perform_measurement_popescu_rohrlich_correlation' we have a regular collection of states, where
        the effect of the PR correlation is that the probabilities to find specific outcomes deviates from what would normally expected.

        Example:
            If we have one pair of entangled photons with quantumness 1 we use 
            '''pr_correlation = [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1} ]'''
            If then apply the gate
            '''circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)'''
            we can use normal gates like beamsplitters and waveplates to run the collection of states through the circuit.

            Because we use quantumness_indicator = 1 we will find the same result as for regular entanglement. So we also could
            have started with a state with optical channels 0.71 x ('1010'+ '0101')

        Note:
            1. The measurement will always be a complete measurement where all optical channels are measured. So after the measurement
                the state will collapse to a statistical mixture of pure photon number states, without superposition
            2. There are fundamental limits to this pr-correlation (which is probably why we do not find it in nature).You cannot create entanglement
                between photons in different pairs. We need to be able to determin from the measurement result what the original polarization in 
                the photon pair was. If this is not possible the system will generate and error, or at least erroneous results.

        Args:
            collection (CollectionOfStates): Collection used as basis for the returned collection
            pr_correlation (list[dict]): list like [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                    {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]

        Returns:
            fsc.CollectionOfStates: CollectionOfStates which can be executed on FockStateCircuit.
    """
    return_collection = collection.copy()
    
    return_collection.clear()
    for input_state in collection:
        intermediate_collection = create_collection_with_NS_boxes(input_state, pr_correlation)
        for output_state in intermediate_collection:
                return_collection.add_state(output_state)

    return return_collection

def _create_collection_with_popescu_rohrlich_correlations(state: State, pr_correlation) -> CollectionOfStates: # type: ignore
    """ Create a CollectionOfStates which can be processed in a FockStateCircuit. This collection
        represents  set of photons pairs with 'Popescu-Rohrlich entanglement' which can have 
        'superquantum' correlations.

        The argument 'state' is a state matching the circuit on which the returned collection will
        be evaluated. The exact content of the state is not important, it is used as a 'template' 
        to create the collection from.

        The argument pr_correlation should be of form:
            [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
        Here the 'channels' are indices for the optical channels. The quantumness indicator determines
        how strong the 'superquantum' correlation will be. 

    Args:
        state (State): State (of type FockStateCircuit State) used as basis for the returned collection
        pr_correlation (list[dict]): list like [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]

    Returns:
        fsc.CollectionOfStates: CollectionOfStates which can be executed on FockStateCircuit.
    """
    
    number_of_states = 4**len(pr_correlation)
    identifier = "".join([str(random.randint(0,10)) for _ in range(50)])
    # configurations will be indicated by string of characters '0','1','2' and '3',
    # meaning hh, hv, vh and vv
    channel_values_for_config = [[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]]
    configurations = [np.base_repr(n, base=4, padding=len(pr_correlation)) for n in range(number_of_states)]
    length_of_string = len(pr_correlation)
    configurations = [ s[len(s) - length_of_string:len(s)] for s in configurations]
    quantumness_indicators = [box['quantumness_indicator'] for box in pr_correlation]
    new_collection = state._collection_of_states.copy()
    new_collection.clear()

    for configuration in configurations:       
        new_state = state.copy()
        new_state.cumulative_probability = (1/2)**len(pr_correlation)
        new_state.initial_state = state.initial_state
        new_optical_components = []
        for name, amp_prob in state.optical_components.items():
            values = new_collection._dict_of_valid_component_names[name].copy()
            for configuration_for_this_box,ns_box in zip(configuration,pr_correlation):
                channels_Ah_Av_Bh_Bv = ns_box['channels_Ah_Av_Bh_Bv']
                values_in_channels = channel_values_for_config[int(configuration_for_this_box)]
                for index,channel in enumerate(channels_Ah_Av_Bh_Bv):
                    values[channel] = values_in_channels[index]
            new_name = new_collection._dict_of_optical_values[tuple(values)]
            new_optical_components.append((new_name, amp_prob['amplitude']))
        new_state.optical_components = new_optical_components
        superentanglement_info = {  'superentanglement': identifier,
                                    'configuration' : configuration,
                                    'quantumness_indicator' : quantumness_indicators,
                                    'initial_state_name_for_output' : state.initial_state,
                                    'original_cumulative_probability': state.cumulative_probability
                                }
        new_state.auxiliary_information.update({'popescu_rohrlich_correlation': superentanglement_info})
        new_collection.add_state(new_state)
    return new_collection


def _measure_collection_with_pr_correlation(collection: CollectionOfStates) -> list: #type: ignore
    """ Measure a collection after evaluation in a FockStateCircuit. The collection has to be created with
        the function '_create_collection_with_popescu_rohrlich_correlations'

        The returned value will be a list of the form
        [{ 'output_state' : outcome1, 'probability' : probability1}, 
        { 'output_state' : outcome2, 'probability' : probability2},
        { 'output_state' : outcome3, 'probability' : probability3},
        ...]

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
        superentanglement_info = reference_state.auxiliary_information['popescu_rohrlich_correlation']
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
    # dict_of_configs_and_amps is a dictionary for a specific outcome, with as keys the 
    # configurations that contribute to that outcome as values the amplitude of that
    for outcome, dict_of_configs_and_amps in configurations_and_amplitudes_grouped_per_outcome.items():
        # the class PopescuRohrlichPhotonPairAmplitudes handles the 'unraveling' of the total state
        # into the states of the individual photons and their polarization and indicates
        # whether the state is 'entangled' of not. First creat the instance for this
        # measurement outcome.
        PR_amplitudes = PopescuRohrlichPhotonPairAmplitudes(input_dictionary=dict_of_configs_and_amps)

        # first check if we can determine the state of the individual photons. If the overall state
        # is entangled this is not possible and we have to stop
        if not PR_amplitudes.is_dict_factorable() and PR_amplitudes.are_all_photon_pairs_factorable():
            raise Exception("Cannot calculate Popescu_Rohrlich correlation since entanglement has been generated")

        # initialize the probability for this outcome from the total amplitudes. This is the highest
        # probability in case all photons are aligned in polarization to give correlation 1
        probability = PR_amplitudes.overall_probability

        # for every outcome iterate over all the photon pairs to come to factor in the 
        # probability caused by 'alignment' of that pair, for that specific outcome
        for box_number, quantumness_indicator in enumerate(quantumness_indicators):
            vector1 = PR_amplitudes.stokes_vectors[PR_amplitudes.photon_indices_per_pair[box_number][0]][0]
            vector2 = PR_amplitudes.stokes_vectors[PR_amplitudes.photon_indices_per_pair[box_number][1]][0]
            alignment_probability = _probability_from_stokes_vectors(vector1, vector2,quantumness_indicator)
            probability = probability*alignment_probability
      
        # add the outcome and the probability to the histogram of possible outcomes
        histogram.append({ 'output_state' : outcome, 'probability' : probability*superentanglement_info['original_cumulative_probability']})

    return histogram


def perform_measurement_popescu_rohrlich_correlation(collection_of_states: CollectionOfStates, # type: ignore
                                                    optical_channels_to_measure: list,
                                                    classical_channels_to_write_to: list) -> CollectionOfStates: # type: ignore
    """ Performs a FULL measurement (i.e., measures all optical channels) where the detection probability is determined
        from the 'popescu_rohrich_correlation'. The collection of states needs to be build by using the gate 'popescu_rohrlich_correlation_gate'.

        NOTE: This is the function called from the measurement gate in FockStateCircuit
    Args:
        collection_of_states (fsc.CollectionOfStates): Collection of states to be measured
        optical_channels_to_be_measured (list[int]): list of of optical channel numbers to be measured
        classical_channels_to_be_written (list[int]): list of classical channel numbers to write the measurement result to

    Returns:
        fsc.CollectionOfStates: Collection with 'collapsed' states and classical channels written

    """
    output_collection = collection_of_states.copy()
    output_collection.clear()

    dict_of_states_group_by_superentanglement_identifier = dict([])

    for state in collection_of_states:
        identifier = state.auxiliary_information['popescu_rohrlich_correlation']['superentanglement']
        dict_of_states_group_by_superentanglement_identifier.setdefault(identifier,list()).append(state)

    for identifier, list_of_states in dict_of_states_group_by_superentanglement_identifier.items():
        intermediate_collection = fsc.CollectionOfStates(fock_state_circuit= collection_of_states._fock_state_circuit,
                                                         input_collection_as_a_dict={str(n):state for n,state in enumerate(list_of_states)})
    
        histo = _measure_collection_with_pr_correlation(intermediate_collection)
        for state in intermediate_collection:
            reference_state = state
            break
        initial_state_name_for_output_collection = reference_state.auxiliary_information['popescu_rohrlich_correlation'].get('initial_state_name_for_output','popescu_rohrlich_correlation')
        del reference_state.auxiliary_information['popescu_rohrlich_correlation']
        reference_state.initial_state = initial_state_name_for_output_collection
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


# -----------------------------------------------------------------------------------------------------------
# Legacy functions for backward compatibility. These functions should not be used in any new code
# -------------------------------------------------------------------------------------------------
def perform_measurement_no_signalling_boxes(collection_of_states, optical_channels_to_measure,classical_channels_to_write_to):
    """ DEPRECATED: This function only exists for backwards compatibility. In any new code use a 
        FockStateCircuit to perform a regular measurement and then analyse and plot the resulting 
        collection as usual. No dedicated function to determine correlations from 'superquantum'
        entanglement is needed.
    """
    return perform_measurement_popescu_rohrlich_correlation(collection_of_states, optical_channels_to_measure,classical_channels_to_write_to)

def measure_collection_with_NS_boxes(collection) -> list:
    """ DEPRECATED: This function only exists for backwards compatibility. In any new code use a 
        FockStateCircuit to perform a regular measurement and then analyse and plot the resulting 
        collection as usual. No dedicated function to determine correlations from 'superquantum'
        entanglement is needed.
    """
    return _measure_collection_with_pr_correlation(collection)

def create_collection_with_NS_boxes(state, ns_boxes) -> CollectionOfStates: # type: ignore
    """ DEPRECATED: This function only exists for backwards compatibility. In any new code use a 
        FockStateCircuit to perform a regular measurement and then analyse and plot the resulting 
        collection as usual. No dedicated function to determine correlations from 'superquantum'
        entanglement is needed.
    """
    return _create_collection_with_popescu_rohrlich_correlations(state = state, pr_correlation = ns_boxes)


def correlations_from_measured_collection_with_NS_boxes(list_of_outcomes: list, 
                                                        channel_combis_for_correlation: list,
                                                        state: any) -> list:
    """ DEPRECATED: This function only exists for backwards compatibility. In any new code use a 
        FockStateCircuit to perform a regular measurement and then analyse and plot the resulting 
        collection as usual. No dedicated function to determine correlations from 'superquantum'
        entanglement is needed.


    
        This function will return a list of correlations for the provided 'list_of_outcomes'. 

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
    """ DEPRECATED: This function only exists for backwards compatibility. In any new code use a 
        FockStateCircuit to perform a regular measurement and then analyse and plot the resulting 
        collection as usual. No dedicated function to determine correlations from 'superquantum'
        entanglement is needed.

        Plot an histogram of correlations. 

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

# --------- BACKGROUND INFORMATION ----------------

# # %% [markdown]
# # # No_signalling_boxes.py
# # ## start with simple situation:
# # 
# # 2 boxes each with a pair of photons. One box channel 01 and 45, the other 23 and 67
# # ```
# # popescu_rohrlich_correlations = [
# #     {'channels_Ah_Av_Bh_Bv':[0,1,4,5],'quantumness_indicator':quantumness},
# #     {'channels_Ah_Av_Bh_Bv':[2,3,6,7],'quantumness_indicator':quantumness}
# #         ]
# # ```
# # Create the boxes: The resulting collection has 16 states in total
# # 
# # Look at an example state representing input hv for first box and vh for second box. So state '10010110' with name '12'
# # ```
# # Initial state: '12'
# # Cumulative probability: 1.00
# # Classical values: ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00']
# # Optical components: 
# # 	Component: '10010110' Amplitude: (1.00 - -0.00i), Probability: 1.00
# # Auxiliary information: 
# # 	no_signalling_box
# # ```
# # auxiliary info:
# # ```
# # {'no_signalling_box': {'superentanglement': '2322109048621975210537033154710105111938714641093732323', 
# # 'configuration': '12', 
# # 'quantumness_indicator': [1, 1],
# # 'initial_state_name_for_output': '00000000'}}
# # ```
# # 
# # we detect with polarizer 
# # * photon 1 (channel 0 and 1) at 0 degrees, 
# # * photon 2 (channel 2 and 3) at 45 degrees, 
# # * photon 3 (channel 4 and 5) at 22.5 degrees,
# # * photon 4 (channel 6 and 7) at 22.5 degrees,
# # 
# # The circuit is this:
# # ```
# # quantumness=1
# # # define the wave plate angles for the measurements
# # S_alice = np.pi/8 # polarization double wave plate angle, so at 45 degrees
# # T_alice = 0  # polarization double wave plate angle, so at 0 degrees
# # S_bob = np.pi/16  # polarization double wave plate angle, so at 22.5 degrees
# # T_bob = np.pi*(3/16) # polarization double wave plate angle, so at 67.5 degrees
# # 
# # # define the PR photon pairs
# # popescu_rohrlich_correlations = [
# #     {'channels_Ah_Av_Bh_Bv':[0,1,4,5],'quantumness_indicator':quantumness},
# #     {'channels_Ah_Av_Bh_Bv':[2,3,6,7],'quantumness_indicator':quantumness}
# #         ]
# # 
# # 
# # circuit0 = fsc.FockStateCircuit(length_of_fock_state=2,
# #                             no_of_optical_channels=8,
# #                             no_of_classical_channels=8,
# #                             use_full_fock_matrix=False)
# # 
# # # first create the photon pairs
# # circuit0.create_no_signalling_boxes(ns_boxes=popescu_rohrlich_correlations)
# # 
# # 
# # circuit = fsc.FockStateCircuit(length_of_fock_state=2,
# #                             no_of_optical_channels=8,
# #                             no_of_classical_channels=8,
# #                             use_full_fock_matrix=False)
# # 
# # # set the wave plates for Alice
# # 
# # circuit.half_wave_plate(channel_horizontal= 0,
# #                         channel_vertical= 1, 
# #                         angle = T_alice)
# # 
# # circuit.half_wave_plate(channel_horizontal= 2,
# #                         channel_vertical= 3, 
# #                         angle = S_alice)
# #     
# # 
# # # set the wave plates for Bob
# # 
# # circuit.half_wave_plate(    channel_horizontal= 4,
# #                             channel_vertical= 5, 
# #                             angle = S_bob)
# # 
# # 
# # circuit.half_wave_plate(    channel_horizontal= 6,
# #                             channel_vertical= 7, 
# #                             angle = S_bob)
# # ```
# # 
# # So for the 'boxes' box 1 is rotated over 22.5 and box 2 is rotated over -22.5
# # 
# # Now look at project on a certain outcome
# # 
# # evaluate outcome '10011010' equivalent to hvhh. This have contribution from 8 original states:
# # amplitdes are 1 for 0 degree, 0.71 for 45 degree, 0.92 for 22.5 degree and 0.38 for 67.5 degree
# # Outcome hvhh
# # * hhhh: 00: 1 * 0.71 * 0.92 * 0.92 = 0.60
# # * hhhv: 01: 1 * 0.71 * 0.92 * 0.38 = 0.25
# # * hvhh: 02: 1 * -0.71 * 0.92 * 0.92 = -0.60
# # * hvhv: 03: 1 * -0.71 * 0.92 * 0.38 = -0.25
# # * hhvh: 10: 1 * 0.71 * 0.38 * 0.92 = 0.25
# # * hhvv: 11: 1 * 0.71 * 0.38 * 0.38 = 0.10
# # * hvvh: 12: 1 * -0.71 * 0.38 * 0.92 = -0.25
# # * hvvv: 13: 1 * -0.71 * 0.38 * 0.38 = -0.10
# # 
# # 
# # We can reproduce this like
# # ```
# # photons = [[1,0],
# #            [np.cos(-np.pi/4), np.sin(-np.pi/4)],
# #            [np.cos(np.pi/8), np.sin(np.pi/8)],
# #            [np.cos(np.pi/8), np.sin(np.pi/8)]]
# # ```
# # ```
# # for number in range(2**4):
# #     amp, key = 1, ''
# #     for n in range(4):
# #         if (number >> n) & 1:
# #             key += 'h'
# #             amp *= photons[n][0]
# #         else:
# #             key += 'v'
# #             amp *= photons[n][1]
# #     print(key, ' : ', amp)
# # ```
# # resulting in :
# # ```
# # vvvv  :  -0.0
# # hvvv  :  -0.10355339059327377
# # vhvv  :  0.0
# # hhvv  :  0.10355339059327377
# # vvhv  :  -0.0
# # hvhv  :  -0.25
# # vhhv  :  0.0
# # hhhv  :  0.25
# # vvvh  :  -0.0
# # hvvh  :  -0.25
# # vhvh  :  0.0
# # hhvh  :  0.25
# # vvhh  :  -0.0
# # hvhh  :  -0.6035533905932737
# # vhhh  :  0.0
# # hhhh  :  0.6035533905932737
# # ```
# # Now we have to translate this in to the same dict we get from the code:
# # From the code we get:
# # 
# # ```
# # ('10011010', {
# # '00': (0.6035533432783801-3.086704949029365e-17j), 
# # '01': (0.24999998399588097-2.9354901997216757e-17j), 
# # '02': (-0.6035533432783801+8.313211995268274e-17j), 
# # '03': (-0.24999998399588097+5.1003803332378845e-17j), 
# # '10': (0.24999998399588097-2.9354901997216757e-17j), 
# # '11': (0.10355338545297983-1.9022448824460632e-17j), 
# # '12': (-0.24999998399588097+5.1003803332378845e-17j), 
# # '13': (-0.10355338545297983+2.7989717496887846e-17j)})
# # ```
# # lets first translate the box configuratiosn '00', '01' etc to photon orientations.
# # '00' is simply 'hhhh'. For '02' it is more complex. Box 0 is 'hh' and box 1 is 'vh'. With the ordering of
# # photons we get 'hvhh'. Indeed amplitude for '02' is -0.6 and amplitude for 'hvhh' is -0.6
# # 
# # ```
# # '00': 0.6035533905932737, 
# # '20': 0.0, 
# # '02': -0.6035533905932737, 
# # '22': -0.0, 
# # '10': 0.25, 
# # '30': 0.0, 
# # '12': -0.25, 
# # '32': -0.0, 
# # '01': 0.25, 
# # '21': 0.0, 
# # '03': -0.25, 
# # '23': -0.0, 
# # '11': 0.10355339059327377, 
# # '31': 0.0, 
# # '13': -0.10355339059327377, 
# # '33': -0.0
# # ```
# # So we recreate the dict
# # Now let's try to recreate the vectors from this dictionary
# # We want to get to:
# # * for box 0 [1,0] and [np.cos(np.pi/8), np.sin(np.pi/8)],
# # * for box 1 [np.cos(-np.pi/4), np.sin(-np.pi/4)] and [np.cos(np.pi/8), np.sin(np.pi/8)]
# # so,
# # Vector in 'hh','hv','vh','vv' should be [ 0.92387953  0.38268343  0.          0.        ]
# # Vector in 'hh','hv','vh','vv' should be [ 0.65328148  0.27059805 -0.65328148 -0.27059805]
# # 
# # We code:
# # ```
# # matrix = generate_matrix_from_dict(intermediate_dictionary)
# # swapped_matrix = np.swapaxes(matrix,0,0)
# # vector_box_0 = vector_from_matrix(swapped_matrix)
# # print(np.round(vector_box_0,2))
# # swapped_matrix = np.swapaxes(matrix,1,0)
# # vector_box_1 = vector_from_matrix(swapped_matrix)
# # print(np.round(vector_box_1,2))
# # ```
# # resulting in 
# # ```
# # [0.92+0.j 0.38-0.j 0.  +0.j 0.  +0.j]
# # [ 0.65+0.j  0.27-0.j -0.65-0.j -0.27-0.j]
# # ```
# # 
# # Now the stokes vectors:
# # * box 0
# # ```
# # amps for box [(0.9238795638084412+0j), (0.3826834261417389-2.5363271334106264e-17j), 0j, 0j]
# # vectors: [0, 0, 1] [4.6865213626847704e-17, 0.70710675611745, 0.7071068062556441]
# # alignment prob 0.85
# # ```
# # So we have first vector [0,0,1], point to pole S3. This is horizontal
# # The second stokes vector is [0,sqrt(1/2),sqrt(1,2)]. This vector is pointing 45 degree up on the Poincare sphere. So 'real' angle difference is 22.5 degree
# # Final probability should be 85% (we take quantumness 1, so regular quantum theory)
# # * box 1
# # ```
# # amps for box [(0.6532815098762512+0j), (0.27059805393218994-1.7934541700850227e-17j), (-0.6532815098762488-5.7111685718412646e-08j), (-0.27059805393218894-2.3656434138349884e-08j)]
# # vectors: [8.742277980456489e-08, -0.9999999999999962, 0.0] [7.162670304757166e-24, 0.7071067673170348, 0.7071067950560599]
# # alignment prob 0.14
# # ```
# # So we have first vector [0,-1,0], so in horizontal plane pointing to S2 pole
# # The second vector is [0,sqrt(1/2),sqrt(1,2)]. The angle on poincare sphere is 135 degrees, corresponding to real angle of 67,5 degrees
# # The alignment probability is 15%
# # 
# # The total probability for this outcome should be 0.15 * 0.85 = 0.1275
# # In the histogram we see
# # ```
# # {'output_state': '10011010', 'probability': 0.12499998920911817}
# # ```
# # 
# # If we change quantumness to 1000 we get
# # ```
# # box 0
# # amps for box [(0.9238795638084412+0j), (0.3826834261417389-2.5363271334106264e-17j), 0j, 0j]
# # vectors 10011010 [0, 0, 1] [4.6865213626847704e-17, 0.70710675611745, 0.7071068062556441]
# # alignment prob 10011010 0.999826743247425
# # final 10011010 0.9998265949074604
# # box 1
# # amps for box [(0.6532815098762512+0j), (0.27059805393218994-1.7934541700850227e-17j), (-0.6532815098762488-5.7111685718412646e-08j), (-0.27059805393218894-2.3656434138349884e-08j)]
# # vectors 10011010 [8.742277980456489e-08, -0.9999999999999962, 0.0] [7.162670304757166e-24, 0.7071067673170348, 0.7071067950560599]
# # alignment prob 10011010 0.00017325678009916246
# # final 10011010 0.00017322673649117624
# # ```
# #  
# #  
# # ### Now we add a third box where we simply detect both photons in h/v orientation. Quantumness is again 1000
# # 
# #  ```
# #  box 0
# # amps for box [(0.9238795638084412+0j), (0.3826834261417389-2.5363271334106264e-17j), 0j, 0j]
# # vectors 100110101010 [0, 0, 1] [4.6865213626847704e-17, 0.70710675611745, 0.7071068062556441]
# # alignment prob 100110101010 0.853553403127822
# # final 100110101010 0.8535532764897995
# # box 1
# # amps for box [(0.6532815098762512+0j), (0.27059805393218994-1.7934541700850227e-17j), (-0.6532815098762488-5.7111685718412646e-08j), (-0.27059805393218894-2.3656434138349884e-08j)]
# # vectors 100110101010 [8.742277980456489e-08, -0.9999999999999962, 0.0] [7.162670304757166e-24, 0.7071067673170348, 0.7071067950560599]
# # alignment prob 100110101010 0.14644661634148387
# # final 100110101010 0.12499998920911817
# # box 2
# # amps for box [(1+0j), 0j, 0j, 0j]
# # vectors 100110101010 [0, 0, 1] [0, 0, 1]
# # alignment prob 100110101010 1.0
# # final 100110101010 0.12499998920911817
# # ```
# # We see the unchanged results for first two boxes and also the expected result for third box. 
# # Now as final attempt add a fourth box and flip H/V polarization for both photons
# # 
# # ### Now change configuration
# # We mix the boxes a bit to see if it still works. For the third box we place half wave plates at 45 degree, so we swap H and V polarization
# # ```
# # popescu_rohrlich_correlations = [
# #     {'channels_Ah_Av_Bh_Bv':[0,1,4,5],'quantumness_indicator':quantumness},
# #     {'channels_Ah_Av_Bh_Bv':[2,3,6,7],'quantumness_indicator':quantumness},
# #     {'channels_Ah_Av_Bh_Bv':[8,9,10,11],'quantumness_indicator':quantumness}
# #         ]
# # ```
# # We look at the  dictionary for the original state '100110101010'
# # We detect third box horizontal, so only vertical states for this photon pair at teh origin can contribute. Hence we always see 3 (meaning 'vv' as last digit in the keys)
# # For the rest it is the same dictionary as expected
# # ```
# # '003': (0.6035533432783801-1.04781019711606e-16j), 
# # '013': (0.24999998399588097-5.997107135190655e-17j), 
# # '023': (-0.6035533432783801+1.5704609017399508e-16j), 
# # '033': (-0.24999998399588097+8.161997268706864e-17j), 
# # '103': (0.24999998399588097-5.997107135190655e-17j), 
# # '113': (0.10355338545297983-3.170408158141368e-17j), 
# # '123': (-0.24999998399588097+8.161997268706864e-17j), 
# # '133': (-0.10355338545297983+4.067135025384089e-17j)
# # ```
# # The if we calculate probabilities we see same results for box 0 and box 1. For box 2 the stokes vectors are [0,0,-1] for both photons. This is the S3 south pole, meaning
# # vertical polarization The probability is 1 since the two photons in the pair are perfectly aligned.
# # 
# # ```
# # box 0
# # amps for box [(0.9238795638084412+1.0129547320808785e-23j), (0.3826834261417389-2.536326626933329e-17j), 0j, 0j]
# # vectors 100110101010 [0, 0, 1] [4.6865212021187035e-17, 0.70710675611745, 0.7071068062556441]
# # alignment prob 100110101010 0.853553403127822
# # final 100110101010 0.8535532764897995
# # box 1
# # amps for box [(0.6532815098762512+7.16267165908756e-24j), (0.27059805393218994-1.7934538119514803e-17j), (-0.6532815098762488-5.7111685718412646e-08j), (-0.27059805393218894-2.3656434138349884e-08j)]
# # vectors 100110101010 [8.742277980456489e-08, -0.9999999999999962, 0.0] [7.162670304757166e-24, 0.7071067673170348, 0.7071067950560599]
# # alignment prob 100110101010 0.14644661634148387
# # final 100110101010 0.12499998920911817
# # box 2
# # amps for box [0j, 0j, 0j, (1+1.0964142641117087e-23j)]
# # vectors 100110101010 [0, 0, -1] [0, 0, -1]
# # alignment prob 100110101010 1.0
# # final 100110101010 0.12499998920911817
# # ```
# # 
# # For this outcome, can we determine the angles for the boxes
# # First box:
# # * First photon: this was horizontally at 0 degree polarized as hxxx/vxxx = infinity
# # * Second photon: this was polarized at 45 degree as xhxx/xvxx is always -1
# # So the amplitudes should be 
# # 
# # Second box:
# # * Third photon: this was polarized at 22,5  degree as  is always 0.92/0.38
# # * Fourth photon: this was polarized at 22,5  degree as  is always 0.92/0.38
# # 
# # First box consists of first and third photon. Their angle difference is 22.5 degree. So probability is 85%
# # Second box consists of second and fourth photon. Their angle difference is also 22.5 degree, So probability is also 85%

# # %%
# print(0.15 * 0.85)
# intermediate_dictionary = {
# '00': (0.6035533432783801-3.086704949029365e-17j), 
# '01': (0.24999998399588097-2.9354901997216757e-17j), 
# '02': (-0.6035533432783801+8.313211995268274e-17j), 
# '03': (-0.24999998399588097+5.1003803332378845e-17j), 
# '10': (0.24999998399588097-2.9354901997216757e-17j), 
# '11': (0.10355338545297983-1.9022448824460632e-17j), 
# '12': (-0.24999998399588097+5.1003803332378845e-17j), 
# '13': (-0.10355338545297983+2.7989717496887846e-17j)}

# photons = [[1,0],
#            [np.cos(-np.pi/4), np.sin(-np.pi/4)],
#            [np.cos(np.pi/8), np.sin(np.pi/8)],
#            [np.cos(np.pi/8), np.sin(np.pi/8)]]
# print(photons)
# for number in range(2**4):
#     amp, key = 1, ''
#     for n in range(4):
#         if (number >> n) & 1 == 0:
#             key += 'h'
#             amp *= photons[n][0]
#         else:
#             key += 'v'
#             amp *= photons[n][1]
#     print(key, ' : ', amp)

# print()

# # box 0 is photons 0 and 2, box 1 is photons 1 and 3
# photon_numbers = {0:(0,2),1:(1,3)}
# all_combis = []
# for number in range(2**4):
#     key, amp = '',1
#     for n in range(4):
#         if (number >> n) & 1 == 0:
#             key += 'h'
#             amp *= photons[n][0]
#         else:
#             key += 'v'
#             amp *= photons[n][1]
#     all_combis.append((key,amp))
# configuration_coding = ['hh','hv','vh','vv']
# created_dictionary = dict([])
# for combi in all_combis:
#     key,amplitude = combi[0],combi[1]
#     new_key = ''
#     for box in photon_numbers.values():
#         new_key += str(configuration_coding.index(key[box[0]]+key[box[1]]))
#     created_dictionary.update({new_key:amplitude})
# print(created_dictionary)


# vector1 = np.array([photons[0][n]*photons[2][m] for n,m in [(0,0),(0,1),(1,0),(1,1)]])
# vector2 = np.array([photons[1][n]*photons[3][m] for n,m in [(0,0),(0,1),(1,0),(1,1)]])
# vectors = np.array([vector1,vector2])
# print(vectors)
# print()
# matrix = generate_matrix_from_dict(intermediate_dictionary)
# swapped_matrix = np.swapaxes(matrix,0,0)
# vector_box_0 = vector_from_matrix(swapped_matrix)
# print(np.round(vector_box_0,2))
# swapped_matrix = np.swapaxes(matrix,1,0)
# vector_box_1 = vector_from_matrix(swapped_matrix)
# print(np.round(vector_box_1,2))

# # %%
# # Generic imports
# import sys  
# import importlib
# import math
# import numpy as np
# %matplotlib inline
# import matplotlib.pyplot as plt
# import matplotlib.lines
# import matplotlib.colors
# # Modules for optical simulation and no-signalling boxes
# # See https://github.com/robhendrik/FockStateCircuit
# sys.path.append("../src")
# import fock_state_circuit as fsc

# quantumness=1
# # define the wave plate angles for the measurements
# S_alice = np.pi/8 # polarization double wave plate angle, so at 45 degrees
# T_alice = 0  # polarization double wave plate angle, so at 0 degrees
# S_bob = np.pi/16  # polarization double wave plate angle, so at 22.5 degrees
# T_bob = np.pi*(3/16) # polarization double wave plate angle, so at 67.5 degrees

# # define the PR photon pairs
# popescu_rohrlich_correlations = [
#     {'channels_Ah_Av_Bh_Bv':[0,1,4,5],'quantumness_indicator':quantumness},
#     {'channels_Ah_Av_Bh_Bv':[2,3,6,7],'quantumness_indicator':quantumness},
#     {'channels_Ah_Av_Bh_Bv':[8,9,10,11],'quantumness_indicator':quantumness}
#         ]


# circuit0 = fsc.FockStateCircuit(length_of_fock_state=2,
#                             no_of_optical_channels=12,
#                             no_of_classical_channels=8,
#                             use_full_fock_matrix=False)

# # first create the photon pairs
# circuit0.create_no_signalling_boxes(ns_boxes=popescu_rohrlich_correlations)


# circuit = fsc.FockStateCircuit(length_of_fock_state=2,
#                             no_of_optical_channels=12,
#                             no_of_classical_channels=8,
#                             use_full_fock_matrix=False)

# # set the wave plates for Alice

# circuit.half_wave_plate(channel_horizontal= 0,
#                         channel_vertical= 1, 
#                         angle = T_alice)

# circuit.half_wave_plate(channel_horizontal= 2,
#                         channel_vertical= 3, 
#                         angle = S_alice)

# circuit.half_wave_plate(    channel_horizontal= 8,
#                             channel_vertical= 9, 
#                             angle = np.pi/4)
    

# # set the wave plates for Bob

# circuit.half_wave_plate(    channel_horizontal= 4,
#                             channel_vertical= 5, 
#                             angle = S_bob)


# circuit.half_wave_plate(    channel_horizontal= 6,
#                             channel_vertical= 7, 
#                             angle = S_bob)


# circuit.half_wave_plate(    channel_horizontal= 10,
#                             channel_vertical= 11, 
#                             angle = np.pi/4)

# # total measurement
# #circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3,4,5,6,7],
# #                                        classical_channels_to_be_written=[0,1,2,3,4,5,6,7])
    
# circuit0.draw()
# circuit.draw()
# initial_collection = fsc.CollectionOfStates(fock_state_circuit=circuit)
# initial_collection.clear()
# state = fsc.State(initial_collection)
# initial_collection.add_state(state)
# intermediate = circuit0.evaluate_circuit(initial_collection)
# #for state in intermediate:
#     #print(state)
#     #print(state.auxiliary_information)
#     #print('---')

# result = circuit.evaluate_circuit(intermediate)

# histo = fsc.measure_collection_with_NS_boxes(result)
# # for item in histo:
# #     print(item)