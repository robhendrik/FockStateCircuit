import sys  
sys.path.append("./src")
import fock_state_circuit as fsc

from fock_state_circuit.popescu_rohrlich_correlation_functionality.popescu_rohrlich_correlations import _group_configurations_per_outcome,_probability_from_stokes_vectors
import importlib
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.colors
import random

from unittest.mock import Mock
from unittest.mock import patch
import matplotlib.testing.compare as plt_test
import pytest

generate_reference_images = False

def test_probability_from_stokes_vectors():
    vectors = [
        # aligned vectors
        ((0,0,1),(0,0,1),1.0),
        ((0,1,0),(0,1,0),1.0),
        ((1,0,0),(1,0,0),1.0),
        ((1,1,1),(1,1,1),1.0),
        # opposing vectors
        ((0,0,1),(0,0,-1),0.0),
        ((0,-1,0),(0,1,0),0.0),
        ((-1,0,0),(1,0,0),0.0),
        ((1,1,1),(-1,-1,-1),0.0),
        # no correlation
        ((0,0,1),(0,1,0),0.5),
        ((0,-1,0),(1,0,0),0.5),
        ((-1,0,1),(1,0,1),0.5),
        ((1,0,1),(1,1,-1),0.5),
        # 22.5 degrees
        ((0,0,1),(0,1,1),0.85355),
        ((1,0,0),(1,1,0),0.85355),
        ((0,0,-1),(1,0,-1),0.85355),
        ((1,1,0),(0,1,0),0.85355),
    ]
    no_error_found = True
    for data in vectors:
        vector1, vector2, expectation = data
        res = _probability_from_stokes_vectors(vector1, vector2)
        if not np.round(res,4) == np.round(expectation,4):
            no_error_found = False
            print(vector1, vector2, expectation )
    assert no_error_found

def test_create_collection_with_NS_boxes():
    # this is testing a legacy function. We should use the gate circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
    # to create  collection with 'superquantum' correlation.  Measurement will automatically be directed to the right function
    # without calling a specific gate
    no_error_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=8,no_of_classical_channels=2)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    state.optical_components = [('22222222',1)]
    state.measurement_results = [{'results':[1,4,3,2]}]
    ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':2}]
    collection_with_boxes = fsc.create_collection_with_NS_boxes(state, ns_boxes)
    no_error_found = no_error_found and (len(collection_with_boxes) == 16)
    for test_state in collection_with_boxes:
        break
    super_info = test_state.auxiliary_information['popescu_rohrlich_correlation']
    ms = test_state.measurement_results
    no_error_found = no_error_found and super_info['quantumness_indicator'] == [1,2]
    no_error_found = no_error_found and (ms[0]['results'] == [1,4,3,2])
    for test_state in collection_with_boxes:
        init_state = test_state.initial_state
        configuration = test_state.auxiliary_information['popescu_rohrlich_correlation']['configuration']
        no_error_found = no_error_found

    assert no_error_found


def test_group_configurations_per_outcome():
    no_error_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=8,no_of_classical_channels=2)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    state.optical_components = [('22222222',1)]
    state.measurement_results = [{'results':[1,4,3,2]}]
    ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':2}]
    # this is testing a legacy function. We should use the gate circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
    # to create  collection with 'superquantum' correlation.  Measurement will automatically be directed to the right function
    # without calling a specific gate
    collection_with_boxes = fsc.create_collection_with_NS_boxes(state, ns_boxes)
    configurations_and_amplitudes_grouped_per_outcome = _group_configurations_per_outcome(collection_with_boxes)

    no_error_found = no_error_found and '00' in configurations_and_amplitudes_grouped_per_outcome['10101010'].keys()
    no_error_found = no_error_found and '33' in configurations_and_amplitudes_grouped_per_outcome['01010101'].keys()
    no_error_found = no_error_found and '03' in configurations_and_amplitudes_grouped_per_outcome['10100101'].keys()
    no_error_found = no_error_found and not '00' in configurations_and_amplitudes_grouped_per_outcome['01010101'].keys()
    no_error_found = no_error_found and not '33' in configurations_and_amplitudes_grouped_per_outcome['10101010'].keys()
    no_error_found = no_error_found and not '03' in configurations_and_amplitudes_grouped_per_outcome['01010101'].keys()

    assert no_error_found

def test_measure_collection_with_NS_boxes():
    # this is testing a legacy function. We should use the gate circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
    # to create  collection with 'superquantum' correlation. Measurement will automatically be directed to the right function
    # without calling a specific gate
    no_error_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=8,no_of_classical_channels=2)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    state.optical_components = [('22222222',1)]
    state.measurement_results = [{'results':[1,4,3,2]}]
    ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':2}]
    collection_with_boxes = fsc.create_collection_with_NS_boxes(state, ns_boxes)
    histogram = fsc.measure_collection_with_NS_boxes(collection_with_boxes)
    #print("printing histogram:" , histogram)
    list_of_outcomes = []

    for res_as_dict in histogram:
        list_of_outcomes+= [res_as_dict['output_state']]
    for outcome in ['10101010','10100101','10010110','01100110']:
        no_error_found = no_error_found and outcome in list_of_outcomes

    for res_as_dict in histogram:
        if '10101010' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1/4
        if '01011010' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1/4
        if '10100101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1/4
        if '01010101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1/4
        if '01101010' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '10010110' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '10010101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '10010110' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0

    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=2)
    circuit.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    state.optical_components = [('22222222',1)]
    state.measurement_results = [{'results':[1,4,3,2]}]
    ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':2}]
    collection_with_boxes = fsc.create_collection_with_NS_boxes(state, ns_boxes)
    output_collection = circuit.evaluate_circuit(collection_of_states_input=collection_with_boxes)
    histogram = fsc.measure_collection_with_NS_boxes(output_collection)
    #print("printing histogram:" , histogram)
    list_of_outcomes = []

    for res_as_dict in histogram:
        list_of_outcomes+= [res_as_dict['output_state']]
    for outcome in ['10101010','10100101','10010110','01100110']:
        no_error_found = no_error_found and outcome in list_of_outcomes

    for res_as_dict in histogram:
        if '10101010' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '01011010' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '10100101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '01010101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '01101010' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1/4
        if '10010110' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '10010101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1/4
        if '10010110' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '01100101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1/4
        if '10011010' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1/4
    assert no_error_found

def test_pr_gates():
    img1 = "./tests/test_drawings/testdrawing_pr_gate_histogram.png"
    img2 = img1.replace("testdrawing","reference")
    generate_reference_images = False
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:
        N = 20
        for angle_first in [0]:
            dict_for_plotting = dict([])
            for angle in [(a/N)*np.pi/2 for a in range(N)]:
                # define and create the popescu-rohrlich correlation. Each photon pair requires 4 channels, since each pair
                # consists of two polarized photons
                pr_correlation = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                            {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
                
                # define the circuit. We will have two photon pairs.
                # We need 8 optical channels.
                circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                # create teh correlation
                circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
                # Rotate the first photon in first pair (channels 0 an 1)
                circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=angle_first)
                # Rotate the second photon in first pair (channels 2 an 3)
                circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                # Rotate the first photon in second pair (channels 4 an 5)
                circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=angle_first)
                # Rotate the second photon in first pair (channels 6 an 7)
                circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                # Perform the measurement
                circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(8)], classical_channels_to_be_written=[n for n in range(8)])
                
                # create a state as 'template' 
                collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                state = fsc.State(collection_of_states=collection)
                collection.add_state(state = state)

                # run the collection with the no signalling boxes through the circuit
                output_collection = circuit.evaluate_circuit(collection_of_states_input=collection)

                channel_combis_for_correlation = [(0,2),(4,6)]
                correlations = output_collection.plot_correlations(channel_combis_for_correlation=channel_combis_for_correlation,correlation_output_instead_of_plot=True)

                # prepare a dictionary for plotting
                lst = []
                lst.append({'output_state': 'first pair, K = 1', 'probability': correlations['00000000'][0]})
                lst.append({'output_state': 'second pair, K = 5', 'probability': correlations['00000000'][1]})
                dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})
            # plot the resulting correlations for the no-signalling boxes
            fsc.plot_correlations_from_dict(dict_for_plotting)
            assert plt_test.compare_images(img1, img2, 0.001) is None

def test_no_signalling_box_application_with_legacy_functions():
    # this is testing a legacy function. We should use the gate circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
    # to create  collection with 'superquantum' correlation. Measurement will automatically be directed to the right function
    # without calling a specific gate
    img1 = "./tests/test_drawings/testdrawing_no_signalling_histogram.png"
    img2 = img1.replace("testdrawing","reference")
    generate_reference_images = False
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:
        N = 10
        angle_first = 0
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
            # function should accept both 'correlation' as well as 'probability'
            lst.append({'output_state': 'second pair, K = 5', 'correlation': correlations[1]}) 
            dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})
        # plot the resulting correlations for the no-signalling boxes
        fsc.plot_correlations_for_NS_boxes(dict_for_plotting)
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_no_signalling_box_application_with_gates_based_on_legacy_functions():
    # this is testing a legacy function. We should use the gate circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
    # to create  collection with 'superquantum' correlation. Measurement will automatically be directed to the right function
    # without calling a specific gate
    img1 = "./tests/test_drawings/testdrawing_no_signalling_histogram_from_gates.png"
    img2 = img1.replace("testdrawing","reference")
    generate_reference_images = False
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:
        N = 10
        angles = [0]
        circuit_is_already_drawn = False

        ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                    {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
        for angle_first in angles:
            dict_for_plotting = dict([])
            for angle in [(a/N)*np.pi/2 for a in range(N)]:
                circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                circuit.create_no_signalling_boxes(ns_boxes=ns_boxes)
                circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=angle_first)
                circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=angle_first)
                circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(8)],classical_channels_to_be_written=[n for n in range(8)])

                if not circuit_is_already_drawn:
                    circuit.draw()
                    circuit_is_already_drawn = True

                collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                state = fsc.State(collection_of_states=collection)
                state.initial_state = 'pr_correlations'
                collection.add_state(state)
                output_collection = circuit.evaluate_circuit(collection_of_states_input=collection)

                channel_combis_for_correlation = [(0,2),(4,6)]
                label_for_channel_combinations = ['first pair, K = 1','second pair, K = 5']
                
                correlations = output_collection.plot_correlations(channel_combis_for_correlation=channel_combis_for_correlation,
                                                                correlation_output_instead_of_plot=True)['pr_correlations']
                lst = []
                lst.append({'output_state': 'first pair, K = 1', 'probability': correlations[0]})
                lst.append({'output_state': 'second pair, K = 5', 'probability': correlations[1]})
                dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})
            # plot the resulting correlations for the no-signalling boxes
            fsc.plot_correlations_from_dict(dict_for_plotting)
            assert plt_test.compare_images(img1, img2, 0.001) is None

def test_guessing_game():
    def number_to_bit_list(number, bit_string_length):
        ''' Generate list of bits (as [1,0,0] with least significant bit first. So the reverse of a string notation)'''
        return [((number >> i) & 1) for i in range(bit_string_length)]

    def bit_list_to_number(bit_list):
        ''' Generate an integer from a list of bits. Least significant bit in list at index 0'''
        bit_string = "".join([str(n) for n in bit_list])[::-1]
        return int(bit_string,2)

    def merge_two_lists_on_even_indices(list_1, list_2):
        ''' Create a new list from two lists. New list has half size. If two lists have equal value at even index new list is 0, else 1.
            Example:
                list_1 = [1,0,1,0,1,0,1,0]
                list_2 = [1,0,0,0,0,0,1,0]
                outcome = [0,1,1,0]
        '''
        return [int(value_1 != value_2) for index, (value_1, value_2) in enumerate(zip(list_1,list_2)) if index%2 ==0]
        
    def length_of_bitstring(number):
        ''' Function to calculate the bit string length for guessing a specific number'''
        length = int(np.ceil(np.log2(number+1)))
        n = int(2**int(np.ceil(np.log2(length))))
        return n
    
    def determine_wave_plate_settings_for_alice(current_values, new_input_values, affected_channels):
        ''' Current values will be all classical channels, new_input_values is not used. Affected_channels will be 
            a list with Alice's channels and Charlie's channel
            Example: If bit string length is 8, charlie's number for Alice is 11 and charlie's bit index for bob is 2 then
            for this function input is 
                current_values = [b,b,b,b,b,b,b,b,0,0,0,0,0,0,0,0,2,11,0] (with b bob's channels)
                affected_channels = [0,1,2,3,4,5,6,7,16,17,18]
                return values = [pi,0,pi,pi/16,pi,0,pi,0,b,b,b,b,b,b,b,b,2,11,0]
        '''
        # define the wave plate angles for the measurements
        S_alice = np.pi/8 # polarization double wave plate angle, so at 45 degrees
        T_alice = 0  # polarization double wave plate angle, so at 0 degrees
        S_bob = np.pi/16  # polarization double wave plate angle, so at 22.5 degrees
        T_bob = np.pi*(3/16) # polarization double wave plate angle, so at 67.5 degrees
        # for all combinations relative angle is 22.5 degrees, except TT which has relative angle of 67.5 degrees

        # affected channels will be the list of all Alice's classical channels and Charlie's classical channels
        # the second channel of Charlie contains the number representing Alice's bit string
        charlies_number = current_values[affected_channels[-2]]
        # generate the bit string with least significant bit left
        bit_string_length = len(affected_channels) - 3
        alices_bit_list = number_to_bit_list(charlies_number, bit_string_length)
        
        # we now determine the settings for the wave plates for Alice
        # example: Charlie number is 11 and bit string length is 8. The bit string is 11010000
        # (remember LSB left). The first photon should be measured "S" since first
        # two bits are equal, the second photon should be measured "T"
        new_values = []
        previous_bit_value = None
        for index, this_bit_value in enumerate(alices_bit_list):
            if index%2 == 1:
                if previous_bit_value is not None and this_bit_value == previous_bit_value:
                    measurement = S_alice
                else:
                    measurement = T_alice
                new_values.append(measurement)

            else:
                new_values.append(np.pi) # it will be a half wave plate so phase shift is 180 degrees, or pi radians
                    
            previous_bit_value = this_bit_value
        
        for index, value in enumerate(new_values):
            current_values[affected_channels[index]] = value
        #print('alice out', current_values)
        return current_values

    def determine_wave_plate_settings_for_bob(current_values, new_input_values, affected_channels):
            ''' current values will be all classical channels, new_input_values is not used. Affected_channels will be 
                a list with Bob's channels and Charlie's channel
                Example: If bit string length is 8, charlie's number for Alice is 11 and charlie's bit index for bob is 2 then
                for this function input is 
                    current_values = [0,0,0,0,0,0,0,0,a,a,a,a,a,a,a,a,2,11,0] (with a alice's channels)
                    affected_channels = [8,9,10,11,12,13,14,15,16,17,18]
                    return values = [a,a,a,a,a,a,a,a,pi,0,pi,0,pi,0,pi,0,2,11,0]
            '''
            # define the wave plate angles for the measurements
            S_alice = np.pi/8 # polarization double wave plate angle, so at 45 degrees
            T_alice = 0  # polarization double wave plate angle, so at 0 degrees
            S_bob = np.pi/16  # polarization double wave plate angle, so at 22.5 degrees
            T_bob = np.pi*(3/16) # polarization double wave plate angle, so at 67.5 degrees
            # for all combinations relative angle is 22.5 degrees, except TT which has relative angle of 67.5 degrees

            # affected channels will be the list of all Alice's classical channels and Charlie's classical channels
            # the first channel of Charlie contains the number representing the index for which Bob has to guess the value
            charlies_bit_index = current_values[affected_channels[-3]]
            pair, position = divmod(charlies_bit_index,2)
            if position == 0:
                measurement = S_bob
            else:
                measurement = T_bob

            # we now determine the settings for the wave plates for Bob
            # if the index is even, measurement should be "S" and if the index is odd measurement shoudl be "T"
            new_values = []
            for index in range(len(affected_channels[:-3])):
                if index%2 == 0:
                    new_values.append(np.pi) # it will be a half wave plate so phase shift is 180 degrees, or pi radians
                else:
                    new_values.append(measurement)
            
            for index, value in enumerate(new_values):
                current_values[affected_channels[index]] = value
            #print('bob out', current_values)
            return current_values

    def prepare_for_next_stage_to_generate_answer(current_values, new_input_values, affected_channels):
        ''' Function as final step of a 'stage' in the guessing game to re-order the classical channels.
            Function has to do three things:
                1) Take Alice's result and turn that in a new number for the next stage of the game 
                2) take Bob's bit index and determine the new index for the next stage.
                3) take Bob's guess so far and store the value

        '''
        
        # Split the classical channel values so the processing is a bit more clear
        length_of_bit_string = int((len(current_values)-3)/2)
        channels_for_alice = current_values[:length_of_bit_string]
        channels_for_bob = current_values[length_of_bit_string:-3]
        channels_for_charlie = current_values[-3:]

        # the bit index for Bob has to be adjusted for bit_string that is half the size
        charlies_bit_index = channels_for_charlie[0]
        pair, position = divmod(charlies_bit_index,2)

        # the bit index for Bob has to be adjusted for bit_string that is half the size
        channels_for_charlie[0] = pair

        # Alice's results are stored in the first even classical channels
        # the algorith is that if for each measurement the outcome is the same as the first bit in the pair we store 0, else we store 1
        # first regenerate the bit string from the number as input to this stage
        charlies_input_number = channels_for_charlie[1]
        input_bit_list = number_to_bit_list(charlies_input_number, length_of_bit_string)
        channels_for_charlie[1] = bit_list_to_number(merge_two_lists_on_even_indices(channels_for_alice,input_bit_list))

        # read the measurement from Bob and add that to the number in measurement result
        bobs_measurement_result = channels_for_bob[pair*2]
        channels_for_charlie[2] += bobs_measurement_result

        if length_of_bit_string == 2: #this was the last stage, we have to generate Bob's guess
            bit_value_that_alice_will_communicate_to_bob = channels_for_charlie[1]
            bobs_results_so_far = channels_for_charlie[2]
            bobs_guess = (bit_value_that_alice_will_communicate_to_bob + bobs_results_so_far)%2
            channels_for_charlie[2] = bobs_guess

        return channels_for_alice + channels_for_bob + channels_for_charlie


    def shift_charlies_channels_to_place_for_next_stage(current_values, new_input_values, affected_channels):
        # if this was last stage do nothing
        if len(current_values) == 2 + 2 + 3:
            return current_values
        else:
            old_length = len(current_values) - 3
            new_length = old_length // 2
            for index, value in enumerate(current_values[-3:]):
                current_values[new_length+index] = value

        return current_values
    
    def generate_circuit_for_stage_in_guessing_game(length_of_bitstring, quantumness, run_with_operator_nodes: bool = False):

        # the actual length of the bitstring has to be a power of 2. Round up to the nearest power of 2
        actual_length = 2**int(np.ceil(np.log2(length_of_bitstring)))

        # we need one photon pair per 2 bits (so Alice has one photon per 2 bits at her side. Same for Bob)
        number_of_photon_pairs = length_of_bitstring//2

        # define the PR photon pairs
        popescu_rohrlich_correlations = [
            {'channels_Ah_Av_Bh_Bv':[n*2,n*2+1,number_of_photon_pairs*2 + n*2,number_of_photon_pairs*2 + n*2 + 1],
            'quantumness_indicator':quantumness} for n in range(number_of_photon_pairs)
            ]

        # make lists to easily identify the channel numbers Bob, Alice and Charlie
        optical_channels_for_alice = []
        for box in popescu_rohrlich_correlations:
            optical_channels_for_alice.append(box['channels_Ah_Av_Bh_Bv'][0])
            optical_channels_for_alice.append(box['channels_Ah_Av_Bh_Bv'][1])

        optical_channels_for_bob = []
        for box in popescu_rohrlich_correlations:
            optical_channels_for_bob.append(box['channels_Ah_Av_Bh_Bv'][2])
            optical_channels_for_bob.append(box['channels_Ah_Av_Bh_Bv'][3])

        # two channels per photon pair are needed for both Alice and Bob (they need one wave plate for each photon at their side, and need
        # two classical channels to control that waveplate)
        classical_channels_for_alice = [index for index in range(2*len(popescu_rohrlich_correlations))] # one channel to measure each box
        classical_channels_for_bob = [(1+ max(classical_channels_for_alice)+ index) for index in range(2*len(popescu_rohrlich_correlations))]

        # we then add two classical channels for 'Charlie'. One channel where the number represented by the original bit-string can be stored, and one channel
        # where the bit index is stored. So Charlie's channels contain the 'assignment' in: "Guess the n-bit in the binary representation of number m"
        classical_channels_for_charlie = [  1+max(classical_channels_for_bob),
                                            2+max(classical_channels_for_bob),
                                            3+max(classical_channels_for_bob)] # first channel to store result, second channel to give bit index as input

        # determine number of channels in this stage of the circuit by adding up the channels needed by each player
        number_of_optical_channels = len(optical_channels_for_alice) + len(optical_channels_for_bob)
        number_of_classical_channels = len(classical_channels_for_alice) + len(classical_channels_for_bob) + len(classical_channels_for_charlie)

        circuit = fsc.FockStateCircuit(length_of_fock_state=2,
                                    no_of_optical_channels=number_of_optical_channels,
                                    no_of_classical_channels=number_of_classical_channels,
                                    circuit_name ="Stage for \n length " + str(actual_length))

        # first create the photon pairs
        circuit.popescu_rohrlich_correlation_gate(pr_correlation=popescu_rohrlich_correlations)

        # execute the classical operation at Alice's side to set wave plates based bit string
        node_info = {'label' : "Alice"}
        circuit.classical_channel_function( function = determine_wave_plate_settings_for_alice,
                                            affected_channels=classical_channels_for_alice + classical_channels_for_charlie,
                                            node_info=node_info)
        
        # set the wave plates for Alice
        for index, box in enumerate(popescu_rohrlich_correlations):
            circuit.wave_plate_classical_control(  optical_channel_horizontal= box['channels_Ah_Av_Bh_Bv'][0],
                                                    optical_channel_vertical= box['channels_Ah_Av_Bh_Bv'][1], 
                                                    classical_channel_for_orientation= classical_channels_for_alice[index*2 + 1],
                                                    classical_channel_for_phase_shift= classical_channels_for_alice[index*2])
        
        # execute the classical operation at Bob's side to set wave plates based bit index
        node_info = {'label' : "Bob"}
        circuit.classical_channel_function(function = determine_wave_plate_settings_for_bob, 
                                        affected_channels=classical_channels_for_bob + classical_channels_for_charlie,
                                        node_info=node_info)

        # set the wave plates for Bob
        for index, box in enumerate(popescu_rohrlich_correlations):
            circuit.wave_plate_classical_control(  optical_channel_horizontal= box['channels_Ah_Av_Bh_Bv'][2],
                                                    optical_channel_vertical= box['channels_Ah_Av_Bh_Bv'][3], 
                                                    classical_channel_for_orientation= classical_channels_for_bob[index*2 + 1],
                                                    classical_channel_for_phase_shift= classical_channels_for_bob[index*2])

        # total measurement
        circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(circuit._no_of_optical_channels)],
                                            classical_channels_to_be_written=[n for n in range(circuit._no_of_optical_channels)])
        
        # prepare everything for a next iteration (i.e., the next stage) or generate the final answer if this was the last stage
        node_info = {'label' : "Prep next"}
        circuit.classical_channel_function(function = prepare_for_next_stage_to_generate_answer,
                                        affected_channels=classical_channels_for_alice +classical_channels_for_bob + classical_channels_for_charlie,
                                        node_info=node_info)

        # shift charlies channels to right place of next stage
        node_info = {'label' : "Shift Charlie"}
        circuit.classical_channel_function(function = shift_charlies_channels_to_place_for_next_stage,
                                        affected_channels=classical_channels_for_alice +classical_channels_for_bob + classical_channels_for_charlie,
                                        node_info=node_info)
        

        def group_states_by_marker(collection,parameters):
            output_collection = collection.copy(empty_template = True)
            dict_with_state_lists = dict([])
            dict_with_probabilities = dict([])
            for state in collection:
                marker = tuple(state.classical_channel_values[3:] + [state.initial_state])
                dict_with_state_lists.setdefault(marker,list()).append(state)
                dict_with_probabilities.setdefault(marker,list()).append(state.cumulative_probability)
            for marker in dict_with_state_lists.keys():
                new_state = dict_with_state_lists[marker][0].copy()
                new_state.cumulative_probability = sum(dict_with_probabilities[marker])
                output_collection.add_state(state=new_state)
            return output_collection
        
        circuit.generic_function_on_collection(group_states_by_marker, affected_optical_channels=[], affected_classical_channels=[])

        return circuit
    
    # test last stage for all possible combinations
    bit_string_length = 2
    circuit_draw_settings_dict = {'channel_labels_optical' : ['Alice ' + str(n) for n in range(bit_string_length)] +  ['Bob ' + str(n) for n in range(bit_string_length)],
                                'channel_labels_classical' :  ['Alice ' + str(n) for n in range(bit_string_length)] +  ['Bob ' + str(n) for n in range(bit_string_length)] + ['Charlie ' + str(n) for n in range(3)],
                                'number_of_nodes_on_a_line': 14, 
                                'spacing_between_lines_in_relation_to_spacing_between_nodes' : 1.5
                            }


    circuit =   generate_circuit_for_stage_in_guessing_game(length_of_bitstring=bit_string_length,quantumness=1000)

    resulting_collection = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    for bobs_index in [0,1]:
        for charlies_number in [0,1,2,3]:
            for stored_so_far in [0,1]:
                alices_bit_list = number_to_bit_list(charlies_number,bit_string_length)
                initial_collection = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
                state = fsc.State(initial_collection)
                
                state.classical_channel_values[-3] = bobs_index
                state.classical_channel_values[-2] = charlies_number
                state.classical_channel_values[-1] = stored_so_far
                expected = (alices_bit_list[bobs_index]+stored_so_far)%2
                state.initial_state = "Expected: " + str(expected) 
                # for each initial state we want the cumulative probability to add up to 1. We have 16 states over two initial state,
                # so every state has 12.5% probability
                state.cumulative_probability = 0.125
                initial_collection.add_state(state)

                result = circuit.evaluate_circuit(initial_collection)
                for state_out in result:
                        resulting_collection.add_state(state_out)

    histo = resulting_collection.plot(classical_channels=[-1], histo_output_instead_of_plot=True)
    for outcome in histo['Expected: 0']:
        if outcome['output_state'] == '0':
            if not outcome['probability'] > 0.99:
                assert False
    for outcome in histo['Expected: 1']:
        if outcome['output_state'] == '1':
            if not outcome['probability'] > 0.99:
                assert False

    # test two stages
    N = 10
    bit_string_length = 4
    circuit_draw_settings_dict = {'channel_labels_optical' : ['Alice ' + str(n) for n in range(bit_string_length)] +  ['Bob ' + str(n) for n in range(bit_string_length)],
                                'channel_labels_classical' :  ['Alice ' + str(n) for n in range(bit_string_length)] +  ['Bob ' + str(n) for n in range(bit_string_length)] + ['Charlie ' + str(n) for n in range(3)],
                                'number_of_nodes_on_a_line': 18, 
                                'spacing_between_lines_in_relation_to_spacing_between_nodes' : 1.5
                            }

    circuit_list = []
    bit_string_length_counter = bit_string_length
    while bit_string_length_counter > 1:
        circuit =   generate_circuit_for_stage_in_guessing_game(length_of_bitstring=bit_string_length_counter ,quantumness=100000)
        circuit_list.append(circuit)
        bit_string_length_counter = bit_string_length_counter //2

    compound_circuit = fsc.CompoundFockStateCircuit(list_of_circuits=circuit_list)
    #compound_circuit.draw()
    initial_collection = fsc.CollectionOfStates(fock_state_circuit=circuit_list[0], input_collection_as_a_dict=dict([]))

    for _ in range(N):
        charlies_number = np.random.randint(0,2**bit_string_length)
        bobs_index = np.random.randint(bit_string_length)     
        alices_bit_list = number_to_bit_list(charlies_number,bit_string_length)
        state = fsc.State(initial_collection)
        state.initial_state = "Expected: " + str(alices_bit_list[bobs_index])
        state.classical_channel_values[-3] = bobs_index
        state.classical_channel_values[-2] = charlies_number
        state.cumulative_probability = 1/N
        initial_collection.add_state(state)

    
    resulting_collection = compound_circuit.evaluate_circuit(initial_collection)


    histo = resulting_collection.plot(classical_channels=[-1], histo_output_instead_of_plot=True)

    total = 0
    for outcome in histo['Expected: 0']:
        if outcome['output_state'] == '1':
            if not outcome['probability'] < 0.01:
                assert False
        else:
            total += outcome['probability']
    for outcome in histo['Expected: 1']:
        if outcome['output_state'] == '0':
            if not outcome['probability'] < 0.01:
                assert False
        else:
            total += outcome['probability']
    assert total > 0.99

    assert True

def test_partial_measurement():
    # even if we perform a measurement on a few optical channels the system measures all channels if the collection of states
    # has PR correlations.
    pr_correlation = [  {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1}]

    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=6,no_of_classical_channels=4)
    circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
    circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=0)
    circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=0)
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(4)],classical_channels_to_be_written=[n for n in range(4)])

    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    state.optical_components = [('000010', np.sqrt(1/2)),('000001', np.sqrt(1/2))]
    state.initial_state = 'popescu_rohrlich_correlation'
    collection.add_state(state)
    output_collection = circuit.evaluate_circuit(collection_of_states_input=collection)
    output_collection.clean_up()
    assert len(output_collection) == 4

def test_double_digit_length_of_fock_state():
    pr_correlation = [  {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1}]

    circuit = fsc.FockStateCircuit(length_of_fock_state=11,no_of_optical_channels=4,no_of_classical_channels=4)
    circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
    circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=np.pi/16)
    circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=0)
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(4)],classical_channels_to_be_written=[n for n in range(4)])

    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    state.optical_components = [('00000000', 1)]
    state.initial_state = 'popescu_rohrlich_correlation'
    collection.add_state(state)
    output_collection = circuit.evaluate_circuit(collection_of_states_input=collection)
    output_collection.clean_up()
    no_error_found = []

    for state in output_collection:
        if list(state.optical_components.keys())[0] == '01000100':
            no_error_found.append(np.round(state.cumulative_probability,2) == np.round(0.43,2))

        if list(state.optical_components.keys())[0] == '00010001':
            no_error_found.append( np.round(state.cumulative_probability,2) == np.round(0.43,2))

        if list(state.optical_components.keys())[0] == '01000001':
            no_error_found.append( np.round(state.cumulative_probability,3) == np.round(0.073,3))

        if list(state.optical_components.keys())[0] == '00010100':
            no_error_found.append(np.round(state.cumulative_probability,3) == np.round(0.073,3))

    no_error_found.append(len(no_error_found)==4)
    assert  all(no_error_found)

def test_reverse_state_notation():
    pr_correlation = [  {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1}]

    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=5,no_of_classical_channels=4, channel_0_left_in_state_name=False)
    circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
    circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=np.pi/16)
    circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=0)
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(4)],classical_channels_to_be_written=[n for n in range(4)])

    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    state.optical_components = [('30000', 1)]
    state.initial_state = 'popescu_rohrlich_correlation'
    collection.add_state(state)
    output_collection = circuit.evaluate_circuit(collection_of_states_input=collection)
    output_collection.clean_up()
    no_error_found = []
  
    for state in output_collection:
        if list(state.optical_components.keys())[0] == '31010':
            no_error_found.append(np.round(state.cumulative_probability,2) == np.round(0.43,2))

        if list(state.optical_components.keys())[0] == '30101':
            no_error_found.append( np.round(state.cumulative_probability,2) == np.round(0.43,2))

        if list(state.optical_components.keys())[0] == '31001':
            no_error_found.append(np.round(state.cumulative_probability,3) == np.round(0.073,3))
      
        if list(state.optical_components.keys())[0] == '30110':
            no_error_found.append(np.round(state.cumulative_probability,3) == np.round(0.073,3))
      
    no_error_found.append(len(no_error_found)==4)
    assert  all(no_error_found)

def test_entangled_boxes():
    # check that the correct error is raised when we create entanglement between photons.
    with pytest.raises(Exception, match='Cannot calculate Popescu_Rohrlich correlation since entanglement has been generated'):
        def quantum_teleportation_with_two_pr_boxes(quantumness):
            # initialize the circuit 
            teleportation_circuit = fsc.FockStateCircuit(   length_of_fock_state = 3, 
                                                            no_of_optical_channels = 8,
                                                            no_of_classical_channels=14
                                                            )

            # prepare the source photon pair as Popescu-Rohrlich photon pair 
            pr_correlation=  [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':quantumness},
                            {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':quantumness}]
            teleportation_circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
            teleportation_circuit.swap(4,5)

            # ========= Starting the teleportation =================

            # Bell state measurement by sender on source photon and first photon of shared pair
            # write the result to classical channels 2-5
            teleportation_circuit.non_polarizing_50_50_beamsplitter(input_channels_a = (2,3), 
                                                                    input_channels_b = (4,5)
                                                                    )


            # ========= Teleportation complete ================= 
            # perform a measurement on the target photon in optical channel 6 and 7 
            teleportation_circuit.wave_plate_classical_control( optical_channel_horizontal = 6,
                                                                optical_channel_vertical = 7,
                                                                classical_channel_for_orientation = 12,
                                                                classical_channel_for_phase_shift = 13)



            # finally the sender can measure the photon that was originally entangled with the source, 
            # after optional manipulation in a phase plate
            teleportation_circuit.wave_plate_classical_control( optical_channel_horizontal = 0,
                                                                optical_channel_vertical = 1,
                                                                classical_channel_for_orientation = 0,
                                                                classical_channel_for_phase_shift = 1)

            # perform complete measurement
            teleportation_circuit.measure_optical_to_classical( optical_channels_to_be_measured=[0,1,2,3,4,5,6,7],
                                                                classical_channels_to_be_written=[0,1,2,3,4,5,10,11])
            
            # define communication in channels 6 and 7 to receiver based on outcome of Bell State measurement
            def define_communication_bits(input_list, new_values = [], affected_channels = []):
                # if a Bell state is detected the two bits are 0-0 or 0-1
                lookup_table = {(0,1,1,0) : (0,1),
                                (1,0,0,1) : (0,1)
                                }
                
                # default value for the two bits is 1-1
                communication = lookup_table.get(tuple(input_list[2:6]),(1,1))
                
                input_list[6], input_list[7] = communication[0], communication[1]
                return input_list

            teleportation_circuit.classical_channel_function(define_communication_bits, affected_channels=[2,3,4,5,6,7])
            
            return teleportation_circuit
        
        teleportation_circuit = quantum_teleportation_with_two_pr_boxes(1)
        dictionary_of_measurement_settings = {'SS': (2*np.pi/16,np.pi/16),'ST':(2*np.pi/16,3*np.pi/16), 'TS':(0,np.pi/16), 'TT':(0,3*np.pi/16)}
        initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=teleportation_circuit, input_collection_as_a_dict=dict([]))
        for name, setting in dictionary_of_measurement_settings.items():      
            state = fsc.State(collection_of_states=initial_collection_of_states)
            state.initial_state = name

            amp = 1
            state.optical_components = {'01010000' :  {'amplitude': amp, 'probability': amp**2}}
            state.classical_channel_values = [0]*14
            state.classical_channel_values[0] = setting[0]
            state.classical_channel_values[1] = np.pi
            state.classical_channel_values[12] = setting[1]
            state.classical_channel_values[13] = np.pi
            initial_collection_of_states.add_state(state)
        for quantumness in [1,5,15]:
            teleportation_circuit = quantum_teleportation_with_two_pr_boxes(quantumness)
            result = teleportation_circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
            result.filter_on_classical_channel(classical_channel_numbers=[6], values_to_filter=[0])
            channel_combinations_for_correlations = [(0,10)]
            correlation = result.plot_correlations(channel_combis_for_correlation=channel_combinations_for_correlations,correlation_output_instead_of_plot=True)

