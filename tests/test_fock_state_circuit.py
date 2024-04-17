import sys  
sys.path.append("./src")
import fock_state_circuit as fsc

import numpy as np
import math
import pytest
from unittest.mock import Mock
import matplotlib.pyplot as plt
from unittest.mock import patch
import matplotlib.testing.compare as plt_test

def test_circuit_initialization():

    circuit = fsc.FockStateCircuit(length_of_fock_state = 11, 
                                   no_of_optical_channels = 2, 
                                   no_of_classical_channels=3, 
                                   channel_0_left_in_state_name=False)

    pass_1 = True if circuit._length_of_fock_state == 11 else False
    pass_2 = True if circuit._no_of_optical_channels == 2 else False
    pass_3 = True if circuit._no_of_classical_channels == 3 else False
    pass_4 = True if circuit._channel_0_left_in_state_name == False else False
    pass_5 = True if circuit._use_full_fock_matrix == False else False
    assert all([pass_1, pass_2, pass_3, pass_4, pass_5])

def test_generate_state_list_as_words_for_double_digit_fock_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 11, no_of_optical_channels = 2)

    collection = fsc.CollectionOfStates(circuit)
    table = collection.generate_allowed_components_names_as_list_of_strings()
    assert len(table[0]) == 4 and table[1] == '0100' and table.count('0101') == 1 and circuit._channel_0_left_in_state_name == True

def test_generate_state_list_as_wordsfor_double_digit_fock_states_with_reversed_notation():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 11, no_of_optical_channels = 2, channel_0_left_in_state_name=False)

    collection = fsc.CollectionOfStates(circuit)
    table = collection.generate_allowed_components_names_as_list_of_strings()
    assert len(table[0]) == 4 and table[1] == '0001' and table.count('0101') == 1 and circuit._channel_0_left_in_state_name == False

def test_trivial_circuit_with_half_wave_plates():
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                   no_of_optical_channels = 4, 
                                   no_of_classical_channels=0, 
                                   )
    circuit2.half_wave_plate(channel_vertical=3,channel_horizontal=2)
    circuit2.half_wave_plate(channel_vertical=3,channel_horizontal=2)

    result = circuit2.evaluate_circuit()
    
    desired_result = dict([])
    list_of_states = ['0000', '1000', '0111', '1110']
    for state_identifier, state in result.items():
        if state['initial_state'] in list_of_states:
            desired_result.update({state_identifier: {state['initial_state']:{'amplitude': 1.0, 'probability': 1.0} }})
    
    bool_list = []
    threshold = circuit2._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_trivial_circuit_with_quarter_wave_plates():
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                   no_of_optical_channels = 2, 
                                   no_of_classical_channels=0, 
                                   )
    circuit2.quarter_wave_plate(channel_vertical=0,channel_horizontal=1)
    circuit2.quarter_wave_plate(channel_vertical=0,channel_horizontal=1)
    circuit2.quarter_wave_plate(channel_vertical=0,channel_horizontal=1)
    circuit2.quarter_wave_plate(channel_vertical=0,channel_horizontal=1)
    # net effect should be zero
    result = circuit2.evaluate_circuit()
    
    desired_result = dict([])
    list_of_states = ['00', '10', '01', '11']
    for state_identifier, state in result.items():
        if state['initial_state'] in list_of_states:
            desired_result.update({state_identifier: {state['initial_state']:{'amplitude': 1.0, 'probability': 1.0} }})
    
    bool_list = []
    threshold = circuit2._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_with_quarter_wave_plates_to_deliver_correct_amplitudes():
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                   no_of_optical_channels = 2, 
                                   no_of_classical_channels=0, 
                                   )
    circuit2.quarter_wave_plate_225(channel_vertical=1,channel_horizontal=0)
    circuit2.quarter_wave_plate_225(channel_vertical=1,channel_horizontal=0)
    
    result = circuit2.evaluate_circuit()
    
    x = math.sqrt(1/2)
    desired_result = dict([])
    list_of_states_unchanged = ['00', '11']
    for state_identifier, state in result.items():
        if state['initial_state'] in list_of_states_unchanged:
            desired_result.update({state_identifier: {state['initial_state']:{'amplitude': 1.0, 'probability': 1.0} }})
        if state['initial_state'] == '01':
            desired_result.update({state_identifier: {'01':{'amplitude': -x, 'probability': 0.5}, '10':{'amplitude': x, 'probability': 0.5} }})
        if state['initial_state'] == '10':
            desired_result.update({state_identifier: {'10':{'amplitude': x, 'probability': 0.5}, '01':{'amplitude': x, 'probability': 0.5} }})

    bool_list = []
    threshold = circuit2._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_with_mixed_wave_plates_to_swap_abc_into_bca():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                   no_of_optical_channels = 3, 
                                   no_of_classical_channels=0, 
                                   )
    circuit.half_wave_plate(channel_horizontal = 0, channel_vertical = 1, angle = math.pi/4, node_info = None)
    circuit.quarter_wave_plate(channel_horizontal = 1, channel_vertical = 2, angle = math.pi/4, node_info = None)
    circuit.quarter_wave_plate(channel_horizontal = 1, channel_vertical = 2, angle = math.pi/4, node_info = None)
    # circuit should first swap channel 0 and 1 and then channel 1 and 2. So abc should become bca

    result = circuit.evaluate_circuit()

    desired_result = dict([])
    list_of_states = ['000', '100', '010', '110', '001', '011', '101', '111']
    for state_identifier, state in result.items():
        if state['initial_state'] in list_of_states:
            old_state = state['initial_state']
            new_state = old_state[1]+old_state[2]+old_state[0]
            desired_result.update({state_identifier: {new_state:{'amplitude': 1.0, 'probability': 1.0} }})
    
    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_with_wave_plates_for_double_digit_fock_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 11, 
                                   no_of_optical_channels = 3, 
                                   no_of_classical_channels=0, 
                                   )
                                   
    circuit.half_wave_plate(channel_horizontal = 0, channel_vertical = 1, angle = math.pi/4, node_info = None)
    circuit.quarter_wave_plate(channel_horizontal = 1, channel_vertical = 2, angle = math.pi/4, node_info = None)
    circuit.quarter_wave_plate(channel_horizontal = 1, channel_vertical = 2, angle = math.pi/4, node_info = None)
    # circuit should first swap channel 0 and 1 and then channel 1 and 2. So abc should become bca
    circuit._use_full_fock_matrix = True
    result = circuit.evaluate_circuit()
    desired_result = {'000000': {'000000': {'amplitude': 1.0, 'probability': 1.0}},
                    '100000': {'000010': {'amplitude': 1.0, 'probability': 1.0}},
                    '000100': {'010000': {'amplitude': 1.0, 'probability': 1.0}},
                    '000009': {'000900': {'amplitude': 1.0, 'probability': 1.0}},
                    '001000': {'100000': {'amplitude': 1.0, 'probability': 1.0}},
                    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })
    
    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)    
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_with_wave_plates_for_double_digit_fock_states_for_reversed_state_notation():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 11, 
                                   no_of_optical_channels = 3, 
                                   no_of_classical_channels=0, 
                                   channel_0_left_in_state_name= False
                                   )
    circuit.half_wave_plate(channel_horizontal = 0, channel_vertical = 1, angle = math.pi/4, node_info = None)
    circuit.quarter_wave_plate(channel_horizontal = 1, channel_vertical = 2, angle = math.pi/4, node_info = None)
    circuit.quarter_wave_plate(channel_horizontal = 1, channel_vertical = 2, angle = math.pi/4, node_info = None)
    # circuit should first swap channel 0 and 1 and then channel 1 and 2. we have reversed notation,
    # So cba should become acb

    result = circuit.evaluate_circuit()
    desired_result = {'000000': {'000000': {'amplitude': 1.0, 'probability': 1.0}},
                    '100000': {'001000': {'amplitude': 1.0, 'probability': 1.0}},
                    '000100': {'000001': {'amplitude': 1.0, 'probability': 1.0}},
                    '000009': {'090000': {'amplitude': 1.0, 'probability': 1.0}},
                    '001000': {'000010': {'amplitude': 1.0, 'probability': 1.0}},
                    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })
    
    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_with_wave_plates_as_50_50_beamsplitter():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                   no_of_optical_channels = 3, 
                                   no_of_classical_channels=0, 
                                   )
    circuit.half_wave_plate(channel_horizontal = 0, channel_vertical = 1, angle = math.pi/4, node_info = None)
    circuit.quarter_wave_plate(channel_horizontal = 1, channel_vertical = 2, angle = math.pi/4, node_info = None)
    circuit.quarter_wave_plate(channel_horizontal = 1, channel_vertical = 2, angle = math.pi/4, node_info = None)
    # circuit should first swap channel 0 and 1 and then channel 1 and 2. So abc should become bca

    result = circuit.evaluate_circuit()

    desired_result = dict([])
    list_of_states = ['000', '100', '010', '110', '001', '011', '101', '111']
    for state_identifier, state in result.items():
        if state['initial_state'] in list_of_states:
            old_state = state['initial_state']
            new_state = old_state[1]+old_state[2]+old_state[0]
            desired_result.update({state_identifier: {new_state:{'amplitude': 1.0, 'probability': 1.0} }})
    
    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_swap_normal_state_notation():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=2, 
                                    )
    circuit.swap(first_channel = 0, second_channel = 1)
    
    result = circuit.evaluate_circuit()
    desired_result = {
        '010': {'100': {'amplitude': 1.0, 'probability': 1.0}},
        '001': {'001': {'amplitude': 1.0, 'probability': 1.0}},
        '011': {'101': {'amplitude': 1.0, 'probability': 1.0}},
        '221': {'221': {'amplitude': 1.0, 'probability': 1.0}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)    
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_swap_with_polarizing_beamsplitter_reversed_state_notation():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                   no_of_optical_channels = 6, 
                                   no_of_classical_channels=2, 
                                   channel_0_left_in_state_name= False
                                   )
    
    circuit.polarizing_beamsplitter(input_channels_a = (1,2), input_channels_b = (4,5))
    
    result = circuit.evaluate_circuit()
    desired_result = {
        '100000': {'000100':{'amplitude': 1.0, 'probability': 1.0}},
        '010000': {'010000':{'amplitude': 1.0, 'probability': 1.0}},
        '001000': {'001000':{'amplitude': 1.0, 'probability': 1.0}},
        '000100': {'100000':{'amplitude': 1.0, 'probability': 1.0}},
        '000010': {'000010':{'amplitude': 1.0, 'probability': 1.0}},
        '000001': {'000001':{'amplitude': 1.0, 'probability': 1.0}},
        '110000': {'010100':{'amplitude': 1.0, 'probability': 1.0}},
        '111000': {'011100':{'amplitude': 1.0, 'probability': 1.0}},
        '000002': {'000002':{'amplitude': 1.0, 'probability': 1.0}},
        '110000': {'010100':{'amplitude': 1.0, 'probability': 1.0}},
        '111000': {'011100':{'amplitude': 1.0, 'probability': 1.0}},
        '000002': {'000002':{'amplitude': 1.0, 'probability': 1.0}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)     
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_non_polarizing_beamsplitters():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                no_of_optical_channels = 4, 
                                no_of_classical_channels=0, 
                                )
    circuit.non_polarizing_beamsplitter(input_channels_a = (0,1), input_channels_b = (2,3), reflection = 0.25, transmission = 0.75)
    
    result = circuit.evaluate_circuit()
    desired_result = {
        '1000': {'1000': {'amplitude': 0.5, 'probability': 0.25}, '0010': {'amplitude': math.sqrt(0.75), 'probability': 0.75}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_mix_50_50_with_too_high_photon_number():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 5, 
                                   no_of_optical_channels = 2, 
                                   no_of_classical_channels=2, 
                                   )
    circuit.mix_50_50(first_channel = 0, second_channel = 1)
   
    result = circuit.evaluate_circuit()
    desired_result = {
        '41': {'41':{'amplitude': 1.0, 'probability': 1.0}},
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)     
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_mix_50_50_with_correct_photon_numbers():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 5, 
                                   no_of_optical_channels = 2, 
                                   no_of_classical_channels=2, 
                                   )
    circuit.mix_50_50(first_channel = 0, second_channel = 1)
   
    result = circuit.evaluate_circuit()
    desired_result = {
        '30': {
            '30': {'amplitude': 0.3535533845424652, 'probability': 0.12499999572143228},
            '21': {'amplitude': 0.6123724579811096, 'probability': 0.37500002729382587},
            '12': {'amplitude': 0.6123724579811096, 'probability': 0.37500002729382587},
            '03': {'amplitude': 0.3535533845424652, 'probability': 0.12499999572143228}
            }
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)     
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_phase_shift_single_channel():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=2, 
                                    )
    circuit.phase_shift_single_channel(channel_for_shift = 3, phase_shift = math.pi/4)
    
    result = circuit.evaluate_circuit()
    desired_result = {
        '1100': {'1100':{'amplitude': 1+0j, 'probability': 1.0}},
        '1111': {'1111':{'amplitude': math.sqrt(1/2)+ math.sqrt(1/2)*1j, 'probability': 1.0}},
        '0002': {'0002':{'amplitude': 1j, 'probability': 1.0}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_HOM_effect_on_50_50_beamsplitter():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                   no_of_optical_channels = 4, 
                                   no_of_classical_channels=0, 
                                   )
    circuit.non_polarizing_50_50_beamsplitter(input_channels_a = (0,1), input_channels_b = (2,3))
    
    result = circuit.evaluate_circuit()
    x = math.sqrt(1/2)
    desired_result = {
        '0000': {'0000': {'amplitude': 1.0, 'probability': 1.0}},
        '1000': {'1000': {'amplitude': x, 'probability': 0.5}, '0010': {'amplitude': x, 'probability': 0.5}}, 
        '0100': {'0100': {'amplitude': x, 'probability': 0.5}, '0001': {'amplitude': x, 'probability': 0.5}}, 
        '0010': {'0010': {'amplitude': -x, 'probability': 0.5}, '1000': {'amplitude': x, 'probability': 0.5}},
        '0001': {'0001': {'amplitude': -x, 'probability': 0.5}, '0100': {'amplitude': x, 'probability': 0.5}},
        '1010': {'2000': {'amplitude': x, 'probability': 0.5}, '0020': {'amplitude': -x, 'probability': 0.5}},
        '0101': {'0200': {'amplitude': x, 'probability': 0.5}, '0002': {'amplitude': -x, 'probability': 0.5}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = 0.001 #circuit.threshold_amplitude_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)     
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_read_write_classical_channels():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 2, 
                                    no_of_classical_channels = 4, 
                                    )
    circuit.set_classical_channels(list_of_values_for_classical_channels=[-9, 1+1j], list_of_classical_channel_numbers=[1,2])
    answer = circuit.evaluate_circuit()
    state_identifier = list(answer.keys())[0]
        
    assert answer[state_identifier]['classical_channel_values'] == [0, -9, 1+1j, 0]

def test_classical_channel_function():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                   no_of_optical_channels = 2, 
                                   no_of_classical_channels=3, 
                                   )
    def test_functie(input_list, new_values = [], affected_channels = []):
        return input_list[::-1]
    circuit._use_full_fock_matrix = True
    circuit.set_classical_channels(list_of_values_for_classical_channels=[4,2,1], list_of_classical_channel_numbers=[0,1,2])
    circuit.classical_channel_function(test_functie)
    answer = circuit.evaluate_circuit()
    state_identifier = list(answer.keys())[0]
    assert answer[state_identifier]['classical_channel_values'] == [1,2,4]

def test_classical_and_optical_mixed():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                   no_of_optical_channels = 2, 
                                   no_of_classical_channels=3, 
                                   )
    def test_functie(input_list, new_values = [], affected_channels = []):
        return input_list[::-1]
    circuit._use_full_fock_matrix = True
    circuit.phase_shift_single_channel(channel_for_shift = 1, phase_shift = 2*math.pi)
    circuit.set_classical_channels(list_of_values_for_classical_channels=[4,2,1], list_of_classical_channel_numbers=[0,1,2])
    circuit.half_wave_plate_45(channel_horizontal = 0, channel_vertical = 1)
    circuit.phase_shift_single_channel(channel_for_shift = 0, phase_shift = math.pi)
    circuit.classical_channel_function(test_functie)
    circuit.phase_shift_single_channel(channel_for_shift = 0, phase_shift = math.pi)

    answer = circuit.evaluate_circuit()
    for state_identifier, state in answer.items():
        if state['initial_state'] == '01':
            reference_identifier = state_identifier

    assert answer[reference_identifier]['classical_channel_values'] == [1,2,4] and answer[reference_identifier]['optical_components']['10']['probability'] == 1.0

def test_phase_shift_single_channel_classical_control():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=3, 
                                    )
    circuit.set_classical_channels(list_of_values_for_classical_channels=[math.pi/2], list_of_classical_channel_numbers=[2])
    circuit.quarter_wave_plate(channel_vertical=0,channel_horizontal=1)
    circuit.quarter_wave_plate(channel_vertical=0,channel_horizontal=1)
    circuit.quarter_wave_plate(channel_vertical=0,channel_horizontal=1)
    circuit.quarter_wave_plate(channel_vertical=0,channel_horizontal=1)
    circuit.phase_shift_single_channel_classical_control(optical_channel_to_shift = 0, classical_channel_for_phase_shift = 2)
    circuit.half_wave_plate(channel_vertical=2,channel_horizontal=3)
    circuit.half_wave_plate(channel_vertical=2,channel_horizontal=3)

    circuit._use_full_fock_matrix = True
    result = circuit.evaluate_circuit()
    desired_result = {
        '1100': {'1100':{'amplitude': 1j, 'probability': 1.0}},
        '1111': {'1111':{'amplitude': 1j, 'probability': 1.0}},
        '0002': {'0002':{'amplitude': 1.0, 'probability': 1.0}},
        '2000': {'2000':{'amplitude': -1.0, 'probability': 1.0}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })


    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_wave_plate_classical_control():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=3 
                                    )
    circuit.phase_shift_single_channel(channel_for_shift = 1, phase_shift = 0)
    circuit.set_classical_channels(list_of_values_for_classical_channels=[math.pi/4,math.pi], list_of_classical_channel_numbers=[0,2])
    circuit.phase_shift_single_channel(channel_for_shift = 2, phase_shift = 0)
    circuit.wave_plate_classical_control(optical_channel_horizontal = 0,
                                        optical_channel_vertical = 1,
                                        classical_channel_for_orientation = 0,
                                            classical_channel_for_phase_shift = 2)
    circuit.phase_shift_single_channel(channel_for_shift = 3, phase_shift = 0)
    
    result = circuit.evaluate_circuit()
    desired_result = {
        '1100': {'1100':{'amplitude': 1.0, 'probability': 1.0}},
        '0020': {'0020':{'amplitude': 1.0, 'probability': 1.0}},
        '1000': {'0100':{'amplitude': 1.0, 'probability': 1.0}},
        '0200': {'2000':{'amplitude': 1.0, 'probability': 1.0}},
        '0011': {'0011':{'amplitude': 1.0, 'probability': 1.0}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)    
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_wave_plate_classical_control_reversed_state_notation():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=3, 
                                    channel_0_left_in_state_name=False
                                    )
    circuit.phase_shift_single_channel(channel_for_shift = 1, phase_shift = 0)
    circuit.set_classical_channels(list_of_values_for_classical_channels=[math.pi/4,math.pi], list_of_classical_channel_numbers=[0,2])
    circuit.phase_shift_single_channel(channel_for_shift = 2, phase_shift = 0)
    circuit.wave_plate_classical_control(optical_channel_horizontal = 0,
                                        optical_channel_vertical = 1,
                                        classical_channel_for_orientation = 0,
                                            classical_channel_for_phase_shift = 2)
    circuit.phase_shift_single_channel(channel_for_shift = 3, phase_shift = 0)
    
    result = circuit.evaluate_circuit()
    desired_result = {
        '1100': {'1100':{'amplitude': 1.0, 'probability': 1.0}},
        '0020': {'0002':{'amplitude': 1.0, 'probability': 1.0}},
        '0001': {'0010':{'amplitude': 1.0, 'probability': 1.0}},
        '1000': {'1000':{'amplitude': 1.0, 'probability': 1.0}},
        '0200': {'0200':{'amplitude': 1.0, 'probability': 1.0}},
        '0011': {'0011':{'amplitude': 1.0, 'probability': 1.0}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_reversing_optical_channels():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=3, 
                                    channel_0_left_in_state_name=True
                                    )
    circuit._use_full_fock_matrix = False
    circuit.set_classical_channels(list_of_values_for_classical_channels=[math.pi/4,math.pi], list_of_classical_channel_numbers=[0,2])
    circuit.swap(first_channel=2,second_channel=3)
    circuit.polarizing_beamsplitter(input_channels_a=(0,1),input_channels_b=(4,5))
    circuit.polarizing_beamsplitter(input_channels_a=(1,0),input_channels_b=(5,4))
    circuit.wave_plate_classical_control(optical_channel_horizontal = 0,
                                        optical_channel_vertical = 1,
                                        classical_channel_for_orientation = 0,
                                        classical_channel_for_phase_shift = 2)
    circuit.half_wave_plate_45(channel_horizontal=4,channel_vertical=5)
    circuit.swap(first_channel=0,second_channel=1)

    name_list = ['000100','010000','000001','020001','100020','000020','002000','100000','110100','100101']
    input_collection = dict([])
    for name in name_list:
        input_collection.update({name: 
                                    {'optical_components': 
                                        {name: {'amplitude': (1+0j), 'probability': 1.0}}, 
                                    'classical_channel_values': [0,0,0], 
                                    'measurement_results': {}}
                                    })
        
    result = circuit.evaluate_circuit(collection_of_states_input=input_collection)
        
    desired_result = dict([])
    for state_name in name_list:
        desired_result.update({state_name:{state_name[::-1]:{'amplitude': 1.0, 'probability': 1.0}}})


    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_reversing_optical_channels_alternate_order_and_use_full_fock_matrix():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=3,
                                    channel_0_left_in_state_name=False
                                    )
    circuit._use_full_fock_matrix = True
    circuit.set_classical_channels(list_of_values_for_classical_channels=[math.pi/4,math.pi], list_of_classical_channel_numbers=[0,2])
    circuit.swap(first_channel=2,second_channel=3)
    circuit.wave_plate_classical_control(optical_channel_horizontal = 0,
                                        optical_channel_vertical = 1,
                                        classical_channel_for_orientation = 0,
                                            classical_channel_for_phase_shift = 2)
    circuit.polarizing_beamsplitter(input_channels_a=(0,1),input_channels_b=(2,3))
    circuit.polarizing_beamsplitter(input_channels_a=(1,0),input_channels_b=(3,2))   

    name_list = ['0020','0110','1020','0101','1020','2000','0020','1010','2001','3000']
    input_collection = dict([])
    for name in name_list:
        input_collection.update({name: 
                                    {'optical_components': 
                                        {name: {'amplitude': (1+0j), 'probability': 1.0}}, 
                                    'classical_channel_values': [0,0,0], 
                                    'measurement_results': {}}
                                    })
        

    selection_of_state_class = fsc.CollectionOfStates(circuit,input_collection_as_a_dict=input_collection)
    result1 = circuit.evaluate_circuit(collection_of_states_input=selection_of_state_class) 
    result = result1.collection_as_dict()
    
    desired_result = dict([])
    for state_name in name_list:
        desired_result.update({state_name:{state_name[::-1]:{'amplitude': 1.0, 'probability': 1.0}}})


    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_reversing_optical_channels():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=3, 
                                    channel_0_left_in_state_name=False
                                    )
    circuit._use_full_fock_matrix = False
    circuit.set_classical_channels(list_of_values_for_classical_channels=[math.pi/4,math.pi], list_of_classical_channel_numbers=[0,2])
    circuit.swap(first_channel=2,second_channel=3)
    circuit.polarizing_beamsplitter(input_channels_a=(0,1),input_channels_b=(4,5))
    circuit.polarizing_beamsplitter(input_channels_a=(1,0),input_channels_b=(5,4))
    circuit.wave_plate_classical_control(optical_channel_horizontal = 0,
                                        optical_channel_vertical = 1,
                                        classical_channel_for_orientation = 0,
                                        classical_channel_for_phase_shift = 2)
    circuit.half_wave_plate_45(channel_horizontal=4,channel_vertical=5)

    name_list = ['000100','010000','000001','020001','100020','000020','002000','100000','110100','100101']
    input_collection = dict([])
    for name in name_list:
        input_collection.update({name: 
                                    {'optical_components': 
                                        {name: {'amplitude': (1+0j), 'probability': 1.0}}, 
                                    'classical_channel_values': [0,0,0], 
                                    'measurement_results': {}}
                                    })
        
    
    selection_of_state_class = fsc.CollectionOfStates(circuit,input_collection_as_a_dict=input_collection)
    result1 = circuit.evaluate_circuit(collection_of_states_input=selection_of_state_class)
    result = result1.collection_as_dict()

    desired_result = dict([])
    for state_name in name_list:
        desired_result.update({state_name:{state_name[::-1]:{'amplitude': 1.0, 'probability': 1.0}}})


    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_conventions_sign_after_beamsplitter():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=0
                                    )
    circuit.non_polarizing_50_50_beamsplitter(input_channels_a=(0,1),input_channels_b=(2,3))
    result = circuit.evaluate_circuit()   
    x = math.sqrt(1/2)
    desired_result = {
    '1000': {'1000':{'amplitude': x, 'probability': 0.5}, '0010':{'amplitude': x, 'probability': 0.5}},
    '0100': {'0100':{'amplitude': x, 'probability': 0.5}, '0001':{'amplitude': x, 'probability': 0.5}},
    '0010': {'0010':{'amplitude': -x, 'probability': 0.5}, '1000':{'amplitude': x, 'probability': 0.5}},
    '0001': {'0001':{'amplitude': -x, 'probability': 0.5}, '0100':{'amplitude': x, 'probability': 0.5}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            print(res)
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_conventions_sign_and_orientation_angle_waveplates():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=0
                                    )
    circuit.half_wave_plate(channel_horizontal=0,channel_vertical=2,angle = math.pi/8)
    circuit.half_wave_plate(channel_horizontal=1,channel_vertical=3,angle = -math.pi/8)
    
    result = circuit.evaluate_circuit()   
    x = math.sqrt(1/2)
    desired_result = {
    '1000': {'1000':{'amplitude': x, 'probability': 0.5}, '0010':{'amplitude': x, 'probability': 0.5}},
    '0100': {'0100':{'amplitude': x, 'probability': 0.5}, '0001':{'amplitude': -x, 'probability': 0.5}},
    '0010': {'0010':{'amplitude': -x, 'probability': 0.5}, '1000':{'amplitude': x, 'probability': 0.5}},
    '0001': {'0001':{'amplitude': -x, 'probability': 0.5}, '0100':{'amplitude': -x, 'probability': 0.5}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            print(res)
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_conventions_phase_of_waveplate():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 5, 
                                    no_of_classical_channels=0
                                    )
    circuit.quarter_wave_plate(channel_horizontal=0,channel_vertical=1,angle = 0)
    circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle = 0)
    circuit.phase_shift_single_channel(channel_for_shift = 4, phase_shift = math.pi/2)
    
    result = circuit.evaluate_circuit()   
    x = math.sqrt(1/2)
    desired_result = {
    '10000': {'10000':{'amplitude': 1, 'probability': 1}},
    '01000': {'01000':{'amplitude': 1j, 'probability': 1}},
    '00100': {'00100':{'amplitude': 1, 'probability': 1}},
    '00010': {'00010':{'amplitude': -1, 'probability': 1}},
    '00001': {'00001':{'amplitude': 1j, 'probability': 1}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            print(res)
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_conventions_sign_after_swap():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=0
                                    )
    circuit.swap(first_channel=0,second_channel=1)
    circuit.half_wave_plate_45(channel_horizontal=2,channel_vertical=3)
    
    result = circuit.evaluate_circuit()   
    x = math.sqrt(1/2)
    desired_result = {
    '1000': {'0100':{'amplitude': 1, 'probability': 1}},
    '0100': {'1000':{'amplitude': 1, 'probability': 1}},
    '0010': {'0001':{'amplitude': 1, 'probability': 1}},
    '0001': {'0010':{'amplitude': 1, 'probability': 1}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            print(res)
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)

def test_circuit_conventions_sign_multi_photon_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 5, 
                                    no_of_classical_channels=0
                                    )
    circuit.quarter_wave_plate(channel_horizontal=0,channel_vertical=1,angle = 0)
    circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle = 0)
    circuit.phase_shift_single_channel(channel_for_shift = 4, phase_shift = math.pi/2)
    
    result = circuit.evaluate_circuit()   
    x = math.sqrt(1/2)
    desired_result = {
    '20000': {'20000':{'amplitude': 1, 'probability': 1}},
    '02000': {'02000':{'amplitude': 1j*1j, 'probability': 1}},
    '00200': {'00200':{'amplitude': 1, 'probability': 1}},
    '00020': {'00020':{'amplitude': -1*-1, 'probability': 1}},
    '00003': {'00003':{'amplitude': 1j*1j*1j, 'probability': 1}}
    }
    desired_result_old = desired_result
    desired_result = dict([])
    for state_identifier in result.keys():
        initial_state = result[state_identifier]['initial_state']
        if initial_state in desired_result_old.keys():
            desired_result.update({state_identifier : desired_result_old[initial_state] })

    bool_list = []
    threshold = circuit._threshold_probability_for_setting_to_zero
    for input in desired_result.keys():
        for output in desired_result[input].keys():
            res = np.cdouble(result[input]['optical_components'][output]['amplitude'])
            deviation = np.abs(np.cdouble(desired_result[input][output]['amplitude']) - res)   
            print(res)
            bool_list.append(deviation < threshold)
    assert len(bool_list) > 0 and all(bool_list)


def test_circuit_conventions_sign_multi_photon_states2():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 5, 
                                    no_of_optical_channels = 2, 
                                    no_of_classical_channels=0
                                    )
    circuit.quarter_wave_plate(channel_horizontal=0,channel_vertical=1, angle = 0)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit)
    collection.filter_on_initial_state(initial_state_to_filter=['01','02', '03','04'])
    result = circuit.evaluate_circuit(collection_of_states_input=collection)   
    state = result.get_state(initial_state='01')
    error1 = state.optical_components['01']['amplitude'] != 1j
    state = result.get_state(initial_state='02')
    error2 = state.optical_components['02']['amplitude'] != 1j
    state = result.get_state(initial_state='03')
    error3 = state.optical_components['03']['amplitude'] != 1j
    state = result.get_state(initial_state='04')
    error4 = state.optical_components['04']['amplitude'] != 1j
    assert all([error1,error2,error3,error4])

    
def test_different_classical_state_per_channel():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                no_of_optical_channels = 4, 
                                no_of_classical_channels=3
                                )
    circuit.wave_plate_classical_control(optical_channel_horizontal = 0,
                                        optical_channel_vertical = 1,
                                        classical_channel_for_orientation = 0,
                                            classical_channel_for_phase_shift = 2) 

    name_list = ['0200','0110','2010','0101','1000','2000','0020','1110','0001','1100']
    input_collection = dict([])
    for name in name_list[:4]:
        input_collection.update({name: 
                                    {'optical_components': 
                                        {name: {'amplitude': (1+0j), 'probability': 1.0}}, 
                                    'classical_channel_values': [0,0,0], 
                                    'measurement_results': {}}
                                    })
    for name in name_list[4:]:
        input_collection.update({name: 
                                    {'optical_components': 
                                        {name: {'amplitude': (1+0j), 'probability': 1.0}}, 
                                    'classical_channel_values': [math.pi/4,0,math.pi], 
                                    'measurement_results': {}}
                                    })
        
    selection_of_state_class = fsc.CollectionOfStates(circuit,input_collection_as_a_dict=input_collection)
    result1 = circuit.evaluate_circuit(collection_of_states_input=selection_of_state_class)
    result = result1.collection_as_dict()


    for index,name in enumerate(name_list):
        answer = list(result[name]['optical_components'].keys())[0]
        if index < 4:
            if answer != name:
                assert False
        if index > 4:
            if answer != name[1] + name[0] + name[2:]:
                assert False
    assert True
    
def test_measurement_combined_with_classical_control():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 4,
                                    no_of_classical_channels=6,
                                    )

    circuit.wave_plate_classical_control(optical_channel_horizontal=0,
                                        optical_channel_vertical=1,
                                        classical_channel_for_orientation=4,
                                        classical_channel_for_phase_shift=5)
    circuit.non_polarizing_50_50_beamsplitter(input_channels_a=(0,1), 
                                            input_channels_b=(2,3))
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3])

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
    reference_state = initial_collection_of_states.get_state(initial_state='1010')
    
    selection_of_states = dict([])
    angles = [i/12 * math.pi/4 for i in range(25)]

    for state_identifier,state in initial_collection_of_states.items():
        if state['initial_state'] == '1010':
            reference_identifier = state_identifier

    for state_number in range(25):
        state_name = 'state_' + str(state_number)
        angle = angles[state_number]
        selection_of_states[state_name] = dict([])
        selection_of_states[state_name]['initial_state'] = state_name
        selection_of_states[state_name]['classical_channel_values'] = [0,0,0,0,angle,math.pi]
        selection_of_states[state_name]['optical_components'] = reference_state.optical_components
        selection_of_states[state_name]['measurement_results'] = []

    selection_of_state_class = fsc.CollectionOfStates(circuit,input_collection_as_a_dict=selection_of_states)
    result1 = circuit.evaluate_circuit(collection_of_states_input=selection_of_state_class)
    result = result1.collection_as_dict()
   
    errors = []
    if not 'state_0-M01a-M02a-M03a-M04a-M05a-M06a-M07a-M08a-M09a-M10a-M11a-M12a' in result.keys():
        errors.append('issue with creating the right state name')
    first_state = result.get('state_0-M01a-M02a-M03a-M04a-M05a-M06a-M07a-M08a-M09a-M10a-M11a-M12a')
    if not (first_state.initial_state ==  'state_0' and first_state.cumulative_probability - 0.5 <= 0.0001):
        errors.append('issue with state definition')       
    if len(first_state.measurement_results) !=  12:
        errors.append('issue with measurement results')   
    other_state = result.get('state_2-M01c-M02a-M03a-M04a-M05a-M06a-M07a-M08a-M09a-M10a-M11a-M12a')

    if ('0110' not in other_state.optical_components.keys()) or (other_state.optical_components['0110']['amplitude'] + 1 > 0.0001):
        errors.append('issue with optical components')
    if (other_state.classical_channel_values[:4] != [0,1,1,0]):
        errors.append('issue with classical_channel_values')
    if other_state.cumulative_probability - 0.016746823190766324 > 0.0001:
        errors.append('issue with classical_channel_values')

    assert len(errors) == 0









def test_decoherence():
    no_failure_found = True
    circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                no_of_optical_channels = 3, 
                                no_of_classical_channels=2
                                )
    circuit1.half_wave_plate_225(channel_horizontal=0,channel_vertical=1)
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=3
                                    )
    circuit1.bridge(circuit2)
    circuit2.channel_coupling(control_channels=[0,1,2], target_channels=[3,4,5], coupling_strength=0)
    circuit3 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=2
                                    )
    circuit2.bridge(circuit3)
    circuit3.half_wave_plate_225(channel_horizontal=1,channel_vertical=0)
    circuit3.measure_optical_to_classical(optical_channels_to_be_measured=[0,1], classical_channels_to_be_written=[0,1])

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('100',1)]
    initial_collection_of_states.add_state(state=state)

    result = circuit1.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    no_failure_found = no_failure_found and (len(result)==1)
    result.clean_up()
    no_failure_found = no_failure_found and (len(result)==1)

    circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                no_of_optical_channels = 3, 
                                no_of_classical_channels=2
                                )
    circuit1.half_wave_plate_225(channel_horizontal=0,channel_vertical=1)
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=3
                                    )
    circuit1.bridge(circuit2)
    circuit2.channel_coupling(control_channels=[0,1,2], target_channels=[3,4,5], coupling_strength=1)
    circuit3 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=2
                                    )
    circuit2.bridge(circuit3)
    circuit3.half_wave_plate_225(channel_horizontal=1,channel_vertical=0)
    circuit3.measure_optical_to_classical(optical_channels_to_be_measured=[0,1], classical_channels_to_be_written=[0,1])

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('100',1)]
    initial_collection_of_states.add_state(state=state)

    result = circuit1.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    no_failure_found = no_failure_found and (len(result)==4)
    result.clean_up()
    no_failure_found = no_failure_found and (len(result)==3)


    circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                no_of_optical_channels = 3, 
                                no_of_classical_channels=2
                                )
    circuit1.half_wave_plate_225(channel_horizontal=0,channel_vertical=1)
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=3
                                    )
    circuit1.bridge(circuit2)
    circuit2.channel_coupling(control_channels=[0,1,2], target_channels=[3,4,5], coupling_strength=1)
    circuit3 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=2
                                    )
    circuit2.bridge(circuit3)
    circuit3.half_wave_plate_225(channel_horizontal=1,channel_vertical=0)
    circuit3.measure_optical_to_classical(optical_channels_to_be_measured=[0,1], classical_channels_to_be_written=[0,1])

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('100',1)]
    initial_collection_of_states.add_state(state=state)

    result = circuit1.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    no_failure_found = no_failure_found and (len(result)==4)

    result.clean_up()
    no_failure_found = no_failure_found and (len(result)==3)
    assert no_failure_found

def test_decoherence_larger_photon_numbers():
    no_failure_found = True
    circuit3 = fsc.FockStateCircuit(length_of_fock_state = 5, 
                            no_of_optical_channels = 4, 
                            no_of_classical_channels=2
                            )
    circuit3.channel_coupling(control_channels=[0,1], target_channels=[2,3], coupling_strength=1)

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit3, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('0001',1)]
    state.initial_state = '0001'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('1332',1)]
    state.initial_state = '1332'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('2203',1)]
    state.initial_state = '2203'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('4014',1)]
    state.initial_state = '4014'
    initial_collection_of_states.add_state(state=state)
    result = circuit3.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    state_out = result.get_state(initial_state='0001')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('0001' in oc_out.keys() )

    state_out = result.get_state(initial_state='1332')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('1340' in oc_out.keys() )

    state_out = result.get_state(initial_state='2203')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('2220' in oc_out.keys() )

    state_out = result.get_state(initial_state='4014')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('4004' in oc_out.keys() )
    #----------------------------
    circuit4 = fsc.FockStateCircuit(length_of_fock_state = 5, 
                            no_of_optical_channels = 4, 
                            no_of_classical_channels=2
                            )
    circuit4.channel_coupling(control_channels=[0,1], target_channels=[2,3], coupling_strength=0.5)

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit4, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('0001',1)]
    state.initial_state = '0001'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('1332',1)]
    state.initial_state = '1332'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('2203',1)]
    state.initial_state = '2203'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('4014',1)]
    state.initial_state = '4014'
    initial_collection_of_states.add_state(state=state)
    result = circuit4.evaluate_circuit(collection_of_states_input=initial_collection_of_states)

    state_out = result.get_state(initial_state='0001')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('0001' in oc_out.keys() )

    state_out = result.get_state(initial_state='1332')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 4 and ('1340' in oc_out.keys() )
    no_failure_found = no_failure_found and oc_out['1340']['probability'] < 0.25001 and oc_out['1340']['probability'] > 0.24999

    state_out = result.get_state(initial_state='2203')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 4 and ('2220' in oc_out.keys() )

    state_out = result.get_state(initial_state='4014')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 2 and ('4004' in oc_out.keys() )
    no_failure_found = no_failure_found and oc_out['4004']['probability'] < 0.5001 and oc_out['4004']['probability'] > 0.4999
        #----------------------------
    circuit5 = fsc.FockStateCircuit(length_of_fock_state = 5, 
                            no_of_optical_channels = 4, 
                            no_of_classical_channels=2
                            )
    circuit5.channel_coupling(control_channels=[0,1], target_channels=[2,3], coupling_strength=0.1)

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit5, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('0101',1)]
    state.initial_state = '0101'
    initial_collection_of_states.add_state(state=state)
    result = circuit5.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    state_out = result.get_state(initial_state='0101')

    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 2
    no_failure_found = no_failure_found and oc_out['0101']['probability'] < 0.90001 and oc_out['0101']['probability'] > 0.89999
    no_failure_found = no_failure_found and oc_out['0102']['probability'] < 0.10001 and oc_out['0102']['probability'] > 0.09999
    
    assert no_failure_found

def test_c_shift():
    no_failure_found = True
    circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                no_of_optical_channels = 3, 
                                no_of_classical_channels=2
                                )
    circuit1.c_shift(control_channels=[0], target_channels=[1])

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('100',1)]
    state.initial_state = '100'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('110',1)]
    state.initial_state = '110'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('000',1)]
    state.initial_state = '000'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('010',1)]
    state.initial_state = '010'
    initial_collection_of_states.add_state(state=state)

    result = circuit1.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    state_out = result.get_state(initial_state='100')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('110' in oc_out.keys() )

    state_out = result.get_state(initial_state='110')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('100' in oc_out.keys() )

    state_out = result.get_state(initial_state='000')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('000' in oc_out.keys() )

    state_out = result.get_state(initial_state='010')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('010' in oc_out.keys() )
    #--------------------------------------
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 5, 
                                no_of_optical_channels = 4, 
                                no_of_classical_channels=2
                                )
    circuit2.c_shift(control_channels=[3], target_channels=[0])

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit2, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('0001',1)]
    state.initial_state = '0001'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('1332',1)]
    state.initial_state = '1332'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('2203',1)]
    state.initial_state = '2203'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('4014',1)]
    state.initial_state = '4014'
    initial_collection_of_states.add_state(state=state)

    result = circuit2.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    state_out = result.get_state(initial_state='0001')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('1001' in oc_out.keys() )

    state_out = result.get_state(initial_state='1332')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('3332' in oc_out.keys() )

    state_out = result.get_state(initial_state='2203')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('0203' in oc_out.keys() )

    state_out = result.get_state(initial_state='4014')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('3014' in oc_out.keys() )
    assert no_failure_found
       #--------------------------------------
    circuit3 = fsc.FockStateCircuit(length_of_fock_state = 5, 
                                no_of_optical_channels = 4, 
                                no_of_classical_channels=2
                                )
    circuit3.c_shift(control_channels=[0,1], target_channels=[2,3])

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit3, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('0001',1)]
    state.initial_state = '0001'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('1332',1)]
    state.initial_state = '1332'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('2203',1)]
    state.initial_state = '2203'
    initial_collection_of_states.add_state(state=state)
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('4014',1)]
    state.initial_state = '4014'
    initial_collection_of_states.add_state(state=state)

    result = circuit3.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    state_out = result.get_state(initial_state='0001')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('0001' in oc_out.keys() )

    state_out = result.get_state(initial_state='1332')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('1340' in oc_out.keys() )

    state_out = result.get_state(initial_state='2203')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('2220' in oc_out.keys() )

    state_out = result.get_state(initial_state='4014')
    oc_out = state_out.optical_components
    no_failure_found = no_failure_found and (len(oc_out)) == 1 and ('4004' in oc_out.keys() )
    assert no_failure_found

def test_basis():
        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                no_of_optical_channels = 3, 
                                no_of_classical_channels=3
                                )
        bss = circuit1.basis()
        circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                        no_of_optical_channels = 3, 
                        no_of_classical_channels=3,
                        channel_0_left_in_state_name=False
                        )
        bss_rev = circuit2.basis()
        assert all([bss['100'] == [1,0,0],bss['000'] == [0,0,0],bss_rev['100'] == [0,0,1]])

def test_basis():
        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                no_of_optical_channels = 3, 
                                no_of_classical_channels=3
                                )
        bss = circuit1.basis()
        circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                        no_of_optical_channels = 3, 
                        no_of_classical_channels=3,
                        channel_0_left_in_state_name=False
                        )
        bss_rev = circuit2.basis()
        assert all([bss['100'] == [1,0,0],bss['000'] == [0,0,0],bss_rev['100'] == [0,0,1]])

def test_custom_fock_state_node_identity():
        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                no_of_optical_channels = 2, 
                                no_of_classical_channels=3
                                )
        bss = circuit1.basis()
        custom_matrix = np.identity(len(bss), dtype = np.cdouble)
        circuit1.custom_fock_state_node(custom_fock_matrix=custom_matrix)
        initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1,input_collection_as_a_dict=dict([]))
        state = fsc.State(collection_of_states=initial_collection_of_states)
        state.optical_components = [('20', 0.5),('11',0.5)]
        state.initial_state = 'initial_state'
        initial_collection_of_states.add_state(state=state)
        result = circuit1.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
        res_state = result.get_state(initial_state='initial_state')
        res_oc = res_state.optical_components
        assert len(result) == 1 and '20' in res_oc.keys() and '11' in res_oc.keys()

def test_custom_fock_state_node_swap():
        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                no_of_optical_channels = 2, 
                                no_of_classical_channels=3
                                )
        bss = circuit1.basis()
        custom_matrix = np.zeros((len(bss),len(bss)), dtype = np.cdouble)
        for input_index, input_values in enumerate(bss.values()):
            for output_index, output_values in enumerate(bss.values()):
                if input_values[0] == output_values[0] and input_values[0] == output_values[1]:
                    custom_matrix[output_index][input_index] = np.cdouble(1)
                else:
                    custom_matrix[output_index][input_index] = np.cdouble(0)
        circuit1.custom_fock_state_node(custom_fock_matrix=custom_matrix)
        initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1,input_collection_as_a_dict=dict([]))
        state = fsc.State(collection_of_states=initial_collection_of_states)
        state.optical_components = [('20', 0.5),('12',0.5)]
        state.initial_state = 'initial_state'
        initial_collection_of_states.add_state(state=state)
        result = circuit1.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
        res_state = result.get_state(initial_state='initial_state')
        res_oc = res_state.optical_components
        assert len(result) == 1 and '22' in res_oc.keys() and '11' in res_oc.keys() and '20' not in res_oc.keys() and '12' not in res_oc.keys()

def test_get_fock_state_matrix():

    circuit1 = fsc.FockStateCircuit(length_of_fock_state = 3, 
                no_of_optical_channels = 2, 
                no_of_classical_channels=3
                )

    bss = circuit1.basis()
    custom_matrix = np.zeros((len(bss),len(bss)), dtype = np.cdouble)
    for input_index, input_values in enumerate(bss.values()):
        for output_index, output_values in enumerate(bss.values()):
            if sum(input_values) > 2 or sum(output_values) > 2:
                if input_values[0] == output_values[0] and input_values[1] == output_values[1]:
                    custom_matrix[output_index][input_index] = np.cdouble(1)
                else:
                    custom_matrix[output_index][input_index] = np.cdouble(0)
            else:
                if input_values[0] == output_values[1] and input_values[1] == output_values[0]:
                    custom_matrix[output_index][input_index] = np.cdouble(1)
                else:
                    custom_matrix[output_index][input_index] = np.cdouble(0)



    circuit1.quarter_wave_plate_225(channel_horizontal=0,channel_vertical=1)
    circuit1.swap(first_channel=0,second_channel=1)
    circuit1.quarter_wave_plate_225(channel_horizontal=0,channel_vertical=1)

    retrieved_matrix = circuit1.get_fock_state_matrix(nodes_to_be_evaluated=[1])
    result = np.matmul(custom_matrix,retrieved_matrix) 
    no_failure_found = True
    for input_index, input_values in enumerate(bss.values()):
        for output_index, output_values in enumerate(bss.values()):
            if input_index == output_index:
                no_failure_found = no_failure_found and 0.9999 < result[input_index][output_index] < 1.0001
            else:
                no_failure_found = no_failure_found and result[input_index][output_index]**2 < 0.000000001
    assert no_failure_found

def test_compound_circuit():

    circuit1 = fsc.FockStateCircuit(length_of_fock_state = 3, 
                no_of_optical_channels = 2, 
                no_of_classical_channels=3
                )
    
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 5, 
            no_of_optical_channels = 3, 
            no_of_classical_channels=0
            )
    
    circuit3 = fsc.FockStateCircuit(length_of_fock_state = 2, 
            no_of_optical_channels = 5, 
            no_of_classical_channels=5
            )
    
    circuit4 = fsc.FockStateCircuit(length_of_fock_state = 3, 
            no_of_optical_channels = 2, 
            no_of_classical_channels=3
            )

    circuit_list = [circuit1, circuit2, circuit3, circuit4]
    compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)

    error = False

    error = error or len(compound_circuit.list_of_circuits) != 4
    error = error or len(compound_circuit._list_of_circuits_with_bridges) != 4

    compound_circuit.list_of_circuits.pop(2)

    error = error or len(compound_circuit.list_of_circuits) != 3
    error = error or len(compound_circuit._list_of_circuits_with_bridges) != 4

    compound_circuit.refresh()

    error = error or len(compound_circuit.list_of_circuits) != 3
    error = error or len(compound_circuit._list_of_circuits_with_bridges) != 3

    assert error == False

def test_shift():
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                no_of_optical_channels = 2,
                                no_of_classical_channels=4,
                                circuit_name = 'GHZ creation'
                                )
    circuit.shift(target_channels=[0,1], shift_per_channel=[1,2])
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['00', '22', '11'])
    result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    no_error = True
    state = result.get_state(initial_state='00')
    no_error = no_error and '12' in state.optical_components.keys()
    state = result.get_state(initial_state='22')
    no_error = no_error and '01' in state.optical_components.keys()
    state = result.get_state(initial_state='11')
    no_error = no_error and '20' in state.optical_components.keys()

    assert no_error

def test_GHZ_with_compound_circuit_direct_gates():
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'GHZ creation'
                                    )
    preparation = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 4,
                                    no_of_classical_channels=1,
                                    circuit_name = 'Preparation'
                                    )

    detection_1 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'D1@+,D2@-,D3@-'
                                    )

    detection_2 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'D1@+,D2@-,D3@+'
                                    )

    detection_3 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'D1@0,D2@-,D3@+'
                                    )
    detection_4 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'D1@0,D2@-,D3@+'
                                    )
    # Start in basis $|ah>|av>|bh>|bv>|BSh>|BSv>|PBSh>|PBSv>$
    preparation.shift_direct(target_channels=[0,1], shift_per_channel=[1,1])
    preparation.c_shift_direct(control_channels=[0,1],target_channels=[2,3])
    preparation.time_delay_classical_control(affected_channels=[0,1], classical_channel_for_delay=0, bandwidth=1)

    # The polarizing beamsplitter works between channels 0,1 and 6,7 
    # (so between channel 'a' and the PBS vacuum input)
    circuit.polarizing_beamsplitter(input_channels_a=(0,1),input_channels_b=(6,7))

    # We add the half wave plate behind the polarizing beamsplitter
    # This is for channels 6 and 7 representing horizontal and vertical polarization for this output of the PBS
    circuit.half_wave_plate_225(channel_horizontal=7,channel_vertical=6)

    # The non-polarizing beamsplitter in front of detector 3 works between channels 2,3 and 4,5 
    # (so between channel 'b' and the BS vacuum input)

    circuit.non_polarizing_50_50_beamsplitter(input_channels_a=(2,3),input_channels_b=(4,5))
    # We add the second non-polarizing beamsplitter mixing the output of the half wave plate and the first non-polarizing beamsplitter
    circuit.polarizing_beamsplitter(input_channels_a=(6,7),input_channels_b=(4,5))

    # bring the output to basis ThTbD1hD1vD2hD2vD3hD3v
    circuit.swap(4,2) 
    circuit.swap(5,3)
    circuit.swap(4,6)
    circuit.swap(5,7)

    # We add a half wave plate in front of detector D1 at +22.5 degree 
    detection_1.half_wave_plate(channel_horizontal=2,channel_vertical=3, angle = math.pi/8)
    detection_2.half_wave_plate(channel_horizontal=2,channel_vertical=3, angle = math.pi/8)
    detection_3.half_wave_plate(channel_horizontal=2,channel_vertical=3, angle = 0)
    detection_4.half_wave_plate(channel_horizontal=2,channel_vertical=3, angle = 0)

    # We add a half wave plate in front of detector D2 at -22.5 degree 
    detection_1.half_wave_plate(channel_horizontal=4, channel_vertical=5, angle = -1*math.pi/8)
    detection_2.half_wave_plate(channel_horizontal=4, channel_vertical=5, angle = -1*math.pi/8)
    detection_3.half_wave_plate(channel_horizontal=4, channel_vertical=5, angle = -1*math.pi/8)
    detection_4.half_wave_plate(channel_horizontal=4, channel_vertical=5, angle = -1*math.pi/8)

    # We add a half wave plate in front of detector D3 at +22.5 degree or -22.5 degree
    # for detection_1 with D3 at +45 degree we see 6% occurence of 4 fold correlation
    # for two photons channel a this should lead to a photon on Th and D2h, for two photons in b this should lead to a photon on D1h and D3h
    # for detection_2 with D3 at -45 degree we see no occurence of 4 fold correlation
    detection_1.half_wave_plate(channel_horizontal=6, channel_vertical=7, angle = +1*math.pi/8)
    detection_2.half_wave_plate(channel_horizontal=6, channel_vertical=7, angle = -1*math.pi/8)
    detection_3.half_wave_plate(channel_horizontal=6, channel_vertical=7, angle = +1*math.pi/8)
    detection_4.half_wave_plate(channel_horizontal=6, channel_vertical=7, angle = -1*math.pi/8)

    # We map the optical channels on the classical channels such that the result in the classical channels in the order Th,Tv,D1h,D1,D2,D2,D3h,D3v
    detection_1.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],classical_channels_to_be_written=[0,1,2,3])
    detection_2.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],classical_channels_to_be_written=[0,1,2,3])           
    detection_3.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],classical_channels_to_be_written=[0,1,2,3])
    detection_4.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],classical_channels_to_be_written=[0,1,2,3])
                      
    initial_collection_of_states_curve = fsc.CollectionOfStates(fock_state_circuit=preparation, input_collection_as_a_dict=dict([]))
    delays = [n/4.0 for n in range(-20,21)]
    for n in delays:
        state1 = fsc.State(collection_of_states=initial_collection_of_states_curve)
        state1.initial_state = 'delay' + str(n)
        state1.optical_components = [('0000', 1)]
        state1.classical_channel_values = [n]
        initial_collection_of_states_curve.add_state(state1)

    circuit_list = [preparation, circuit, detection_1]
    compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)
    result = compound_circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states_curve)

    histo = result.plot(histo_output_instead_of_plot=True)

    probs = []
    for initial_state in initial_collection_of_states_curve.initial_states_as_list():
        outcomes = histo.get(initial_state, None)
        if outcomes is not None:
            for xx in outcomes:
                if xx['output_state'] == '1111':
                    probs.append(xx['probability'])
                    break
        else:
            probs.append(0)

    assert 0.031 < probs[0] < 0.0315 and probs[19] < 0.0015 and 0.020 < probs[25] < 0.021

def test_GHZ_with_compound_circuit():
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'GHZ creation'
                                    )
    preparation = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 4,
                                    no_of_classical_channels=1,
                                    circuit_name = 'Preparation'
                                    )

    detection_1 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'D1@+,D2@-,D3@-'
                                    )

    detection_2 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'D1@+,D2@-,D3@+'
                                    )

    detection_3 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'D1@0,D2@-,D3@+'
                                    )
    detection_4 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'D1@0,D2@-,D3@+'
                                    )
    # Start in basis $|ah>|av>|bh>|bv>|BSh>|BSv>|PBSh>|PBSv>$
    preparation.shift(target_channels=[0,1], shift_per_channel=[1,1])
    preparation.c_shift(control_channels=[0,1],target_channels=[2,3])
    preparation.time_delay_classical_control(affected_channels=[0,1], classical_channel_for_delay=0, bandwidth=1)

    # The polarizing beamsplitter works between channels 0,1 and 6,7 
    # (so between channel 'a' and the PBS vacuum input)
    circuit.polarizing_beamsplitter(input_channels_a=(0,1),input_channels_b=(6,7))

    # We add the half wave plate behind the polarizing beamsplitter
    # This is for channels 6 and 7 representing horizontal and vertical polarization for this output of the PBS
    circuit.half_wave_plate_225(channel_horizontal=7,channel_vertical=6)

    # The non-polarizing beamsplitter in front of detector 3 works between channels 2,3 and 4,5 
    # (so between channel 'b' and the BS vacuum input)

    circuit.non_polarizing_50_50_beamsplitter(input_channels_a=(2,3),input_channels_b=(4,5))
    # We add the second non-polarizing beamsplitter mixing the output of the half wave plate and the first non-polarizing beamsplitter
    circuit.polarizing_beamsplitter(input_channels_a=(6,7),input_channels_b=(4,5))

    # bring the output to basis ThTbD1hD1vD2hD2vD3hD3v
    circuit.swap(4,2) 
    circuit.swap(5,3)
    circuit.swap(4,6)
    circuit.swap(5,7)

    # We add a half wave plate in front of detector D1 at +22.5 degree 
    detection_1.half_wave_plate(channel_horizontal=2,channel_vertical=3, angle = math.pi/8)
    detection_2.half_wave_plate(channel_horizontal=2,channel_vertical=3, angle = math.pi/8)
    detection_3.half_wave_plate(channel_horizontal=2,channel_vertical=3, angle = 0)
    detection_4.half_wave_plate(channel_horizontal=2,channel_vertical=3, angle = 0)

    # We add a half wave plate in front of detector D2 at -22.5 degree 
    detection_1.half_wave_plate(channel_horizontal=4, channel_vertical=5, angle = -1*math.pi/8)
    detection_2.half_wave_plate(channel_horizontal=4, channel_vertical=5, angle = -1*math.pi/8)
    detection_3.half_wave_plate(channel_horizontal=4, channel_vertical=5, angle = -1*math.pi/8)
    detection_4.half_wave_plate(channel_horizontal=4, channel_vertical=5, angle = -1*math.pi/8)

    # We add a half wave plate in front of detector D3 at +22.5 degree or -22.5 degree
    # for detection_1 with D3 at +45 degree we see 6% occurence of 4 fold correlation
    # for two photons channel a this should lead to a photon on Th and D2h, for two photons in b this should lead to a photon on D1h and D3h
    # for detection_2 with D3 at -45 degree we see no occurence of 4 fold correlation
    detection_1.half_wave_plate(channel_horizontal=6, channel_vertical=7, angle = +1*math.pi/8)
    detection_2.half_wave_plate(channel_horizontal=6, channel_vertical=7, angle = -1*math.pi/8)
    detection_3.half_wave_plate(channel_horizontal=6, channel_vertical=7, angle = +1*math.pi/8)
    detection_4.half_wave_plate(channel_horizontal=6, channel_vertical=7, angle = -1*math.pi/8)

    #limit the number of projections to gain time
    list_of_projections = [
        k for k,v in detection_1._dict_of_valid_component_names.items() if (v[0] == 1 and v[2] ==1 and v[4] ==1 and v[6] == 1)
        ]
    
    # We map the optical channels on the classical channels such that the result in the classical channels in the order Th,Tv,D1h,D1,D2,D2,D3h,D3v
    detection_1.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],
                                            classical_channels_to_be_written=[0,1,2,3],
                                            list_of_projections=list_of_projections)
    detection_2.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],
                                            classical_channels_to_be_written=[0,1,2,3],
                                            list_of_projections=list_of_projections)
    detection_3.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],
                                            classical_channels_to_be_written=[0,1,2,3],
                                            list_of_projections=list_of_projections)
    detection_4.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],
                                            classical_channels_to_be_written=[0,1,2,3],
                                            list_of_projections=list_of_projections)
                      
    initial_collection_of_states_curve = fsc.CollectionOfStates(fock_state_circuit=preparation, input_collection_as_a_dict=dict([]))
    delays = [n/4.0 for n in range(-20,21)]
    for n in delays:
        state1 = fsc.State(collection_of_states=initial_collection_of_states_curve)
        state1.initial_state = 'delay' + str(n)
        state1.optical_components = [('0000', 1)]
        state1.classical_channel_values = [n]
        initial_collection_of_states_curve.add_state(state1)

    circuit_list = [preparation, circuit, detection_1]
    compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)
    result = compound_circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states_curve)

    histo = result.plot(histo_output_instead_of_plot=True)

    probs = []
    for initial_state in initial_collection_of_states_curve.initial_states_as_list():
        outcomes = histo.get(initial_state, None)
        if outcomes is not None:
            for xx in outcomes:
                if xx['output_state'] == '1111':
                    probs.append(xx['probability'])
                    break
        else:
            probs.append(0)

    assert 0.031 < probs[0] < 0.0315 and probs[19] < 0.0015 and 0.020 < probs[25] < 0.021

def test_select_initial_state_for_plot():
    no_error = True
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                no_of_optical_channels = 3,
                                no_of_classical_channels=2,
                                circuit_name = 'test'
                                )
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
    histo = initial_collection_of_states.plot(histo_output_instead_of_plot=True)
    no_error = no_error and len(histo) == len(initial_collection_of_states)
    histo = initial_collection_of_states.plot(initial_states = ['111'],
                                            histo_output_instead_of_plot=True)
    no_error = no_error and len(histo) == 1
    histo = initial_collection_of_states.plot(initial_states = ['000', '122'],
                                            histo_output_instead_of_plot=True)
    no_error = no_error and len(histo) == 2 and '000' in histo.keys() and not '111' in histo.keys()
    assert no_error

def test_print_string_rep_circuit():
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                            no_of_optical_channels = 3,
                            no_of_classical_channels=2,
                            circuit_name = 'test'
                            )

    circuit2 = fsc.FockStateCircuit( length_of_fock_state = 1, 
                                no_of_optical_channels = 2,
                                no_of_classical_channels=2,
                                circuit_name = 'test'
                                )

    circuit3 = fsc.FockStateCircuit( length_of_fock_state = 4, 
                                no_of_optical_channels = 3,
                                no_of_classical_channels=0,
                                circuit_name = 'test'
                                )

    com_circ = fsc.CompoundFockStateCircuit(list_of_circuits=[circuit,circuit2,circuit3])
    result_string = str(com_circ)
    assert 'Optical' in result_string and len(result_string) == 334 and result_string[0:8] == "Compound" and result_string[-2] == '4'

def test_print_string_rep_circuit_with_name():
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                            no_of_optical_channels = 3,
                            no_of_classical_channels=2,
                            circuit_name = 'test'
                            )

    circuit2 = fsc.FockStateCircuit( length_of_fock_state = 1, 
                                no_of_optical_channels = 2,
                                no_of_classical_channels=2,
                                circuit_name = 'test'
                                )

    circuit3 = fsc.FockStateCircuit( length_of_fock_state = 4, 
                                no_of_optical_channels = 3,
                                no_of_classical_channels=0,
                                circuit_name = 'test'
                                )
    com_circ = fsc.CompoundFockStateCircuit(list_of_circuits=[circuit,circuit2,circuit3], compound_circuit_name = 'piet')
    result_string = str(com_circ)
    assert 'piet' in result_string and len(result_string) == 323 and result_string[0:8] == "Compound" and result_string[-2] == '4'

def test_generic_function_on_collection_classical_channels():
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                            no_of_optical_channels = 3,
                            no_of_classical_channels=2,
                            circuit_name = 'test'
                            )
    
    def generic_function(input_collection,parameters):
        for state in input_collection:
            state.classical_channels = parameters[:2]
        return input_collection

    circuit.generic_function_on_collection(function=generic_function, parameters=[1,2])

    input_collection = fsc.CollectionOfStates(fock_state_circuit=circuit)
    result = circuit.evaluate_circuit(collection_of_states_input=input_collection)
    no_error_found = True
    for state in result:
        if state.classical_channels != [1,2]:
            no_error_found = False
    assert no_error_found

def test_generic_function_on_collection_optical_channels():
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                            no_of_optical_channels = 3,
                            no_of_classical_channels=2,
                            circuit_name = 'test'
                            )

    def generic_function(input_collection,parameters):
        control_channel = parameters[0]
        target_channel = parameters[1]
    
        for state in input_collection:
            new_components = dict([])
            old_components = state.optical_components
            for name, amp_prob in old_components.items():
                old_values = state._dict_of_valid_component_names[name].copy()
                old_values[target_channel] = (old_values[target_channel] + old_values[control_channel])%input_collection._length_of_fock_state
                new_name = input_collection._get_state_name_from_list_of_photon_numbers(old_values)
                new_components.update({new_name:amp_prob})
            state.optical_components  = new_components
        return input_collection

    circuit.generic_function_on_collection(function=generic_function,parameters=[0,1])

    input_collection = fsc.CollectionOfStates(fock_state_circuit=circuit)

    result = circuit.evaluate_circuit(collection_of_states_input=input_collection)
    no_error_found = True
    for state in result:
        control_value = int(state.initial_state[0])
        target_value = int(state.initial_state[1])
        new_target = (control_value + target_value)%3
        new_value = str(state.initial_state[0]) + str(new_target) +str(state.initial_state[2])
        if not new_value in state.optical_components.keys():
            no_error_found = False
    assert no_error_found

def test_nonlinear_optical_node():
    ' dummy implementation, just check if no error'
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                            no_of_optical_channels = 3,
                            no_of_classical_channels=2,
                            circuit_name = 'test'
                            )
    circuit.nonlinear_optical_node(operator='1+1-', optical_channels=[0])
    input_collection = fsc.CollectionOfStates(fock_state_circuit=circuit)

    result = circuit.evaluate_circuit(collection_of_states_input=input_collection)
    assert True