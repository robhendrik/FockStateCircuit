import sys  
sys.path.append("./src")
import fock_state_circuit as fsc
import collection_of_states as cos
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

    collection = cos.CollectionOfStates(circuit)
    table = collection.generate_allowed_components_names_as_list_of_strings()
    assert len(table[0]) == 4 and table[1] == '0100' and table.count('0101') == 1 and circuit._channel_0_left_in_state_name == True

def test_generate_state_list_as_wordsfor_double_digit_fock_states_with_reversed_notation():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 11, no_of_optical_channels = 2, channel_0_left_in_state_name=False)

    collection = cos.CollectionOfStates(circuit)
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
        

    selection_of_state_class = cos.CollectionOfStates(circuit,input_collection_as_a_dict=input_collection)
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
        
    
    selection_of_state_class = cos.CollectionOfStates(circuit,input_collection_as_a_dict=input_collection)
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
    collection = cos.CollectionOfStates(fock_state_circuit=circuit)
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
        
    selection_of_state_class = cos.CollectionOfStates(circuit,input_collection_as_a_dict=input_collection)
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

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
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

    selection_of_state_class = cos.CollectionOfStates(circuit,input_collection_as_a_dict=selection_of_states)
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


def test_generate_collection_of_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6 
                                    )
    collection_of_states = cos.CollectionOfStates(circuit)
    text = str(collection_of_states)
    reference_text = "Printing collection of states\nNumber of states in collection: 81"

    assert text[:len(reference_text)] == reference_text

def test_filter_and_copy_collection_of_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6 
                                    )
    collection_of_states = cos.CollectionOfStates(circuit)

    copied_collection = collection_of_states.copy()
    copied_collection.filter_on_initial_state(initial_state_to_filter='2222')
    s1 = copied_collection.get_state(identifier='identifier_80')
    s2 = copied_collection.get_state(identifier='identifier_82')
    s3 = copied_collection.get_state(initial_state='2220')
    assert (s1.initial_state =='2222') and (s2 is None) and (s3 is None)

def test_filter_and_copy_collection_of_states2():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 11, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6 
                                    )
    collection_of_states = cos.CollectionOfStates(circuit)
    
    s3 = collection_of_states.get_state(initial_state='00'*4)
    try:
        s3.classical_channel_values = [9,8,7]
        error = True
    except:
        error = False
    s3.classical_channel_values = [9,8,7,1,2,3]
    collection_of_states.filter_on_classical_channel(classical_channel_numbers=[3,4,5], values_to_filter=[9,8,7])
    assert len(collection_of_states) == 0 and error == False

def test_create_vector_from_state_and_the_other_way_round():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6 
                                    )
    collection_of_states = cos.CollectionOfStates(circuit)
    
    s3 = collection_of_states.get_state(initial_state='1'*4)
    vector, basis = s3.translate_state_components_to_vector()
    error = []
    if len(vector) != len(basis) or len(vector) != 2**4:
        error.append('error1')
    for index in range(len(vector)):
        if vector[index] == np.cdouble(1):
            if basis[index] != '1111':
                error.append('error2')
    s4 = collection_of_states.get_state(initial_state='0011')
    s4.optical_components= {'0011': {'amplitude': math.sqrt(1/2), 'probability': 0.5},
                            '1100': {'amplitude': math.sqrt(1/2), 'probability': 0.5}}
    vector, basis = s4.translate_state_components_to_vector()
    error = []
    if len(vector) != len(basis) or len(vector) != 2**4:
        error.append('error3')
    for index in range(len(vector)):
        if vector[index] == np.cdouble(math.sqrt(1/2)):
            if not (basis[index] == '0011' or basis[index] == '1100'):
                error.append('error4')

    s3.set_state_components_from_vector(state_vector=vector)
    oc = s3.optical_components
    if not ('0011' in oc.keys() and '1100' in oc.keys()):
        error.append('error5')
    amp_prob =  oc.get('0011')
    if not round(abs(amp_prob.get('amplitude'))**2,5) == 1/2:
        error.append('error6')
    
    assert len(error) == 0

def test_iterating_through_collection():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                            no_of_optical_channels = 4,
                            no_of_classical_channels=6 
                                )
    collection_of_states = cos.CollectionOfStates(circuit)

    for index, state in enumerate(collection_of_states):
        if state.initial_state == '1111':
            final_index = index

    target_state = collection_of_states[final_index]
    assert target_state.initial_state == '1111'

def test_iterating_through_collection_with_values():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                            no_of_optical_channels = 4,
                            no_of_classical_channels=6 
                                )
    collection_of_states = cos.CollectionOfStates(circuit)

    for index, state in enumerate(collection_of_states.values()):
        if state.initial_state == '1111':
            final_index = index

    target_state = collection_of_states[final_index]
    assert target_state.initial_state == '1111'

def test_fiter_on_classical_channel_values():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 11, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6 
                                    )
    collection_of_states = cos.CollectionOfStates(circuit)

    s3 = collection_of_states.get_state(initial_state='00'*4)
    s3.classical_channel_values = [1,2,3,9,8,7]
    collection_of_states.filter_on_classical_channel(classical_channel_numbers=[3,4,5], values_to_filter=[9,8,7])
    value1 = len(collection_of_states)
    collection_of_states.initialize_to_default()
    s3 = collection_of_states.get_state(initial_state='00'*4)
    s4 = s3.copy()
    s4.classical_channel_values = [1,2,3,9,8,7]
    collection_of_states.filter_on_classical_channel(classical_channel_numbers=[3,4,5], values_to_filter=[9,8,7])
    value2 = len(collection_of_states)
    assert value1 == 1 and value2 == 0

def test_reversed_name():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 11, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6, 
                                channel_0_left_in_state_name=False
                                    )
    collection_of_states = cos.CollectionOfStates(circuit)
    state1 = collection_of_states.get_state(identifier='identifier_00001')
    name1 = state1.initial_state
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 11, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6, 
                                channel_0_left_in_state_name=True
                                    )
    collection_of_states = cos.CollectionOfStates(circuit2)
    state2 = collection_of_states.get_state(identifier='identifier_00001')
    name2 = state2.initial_state

    assert name1 == '00000001' and name2 == '01000000'

def test_various_methods_for_collection_of_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 2, 
                                    no_of_classical_channels=3, 
                                    )
    collection_of_states = cos.CollectionOfStates(circuit)
    errors = []

    if len(collection_of_states) != 9:
        errors.append('error1')

    collection_as_dict = collection_of_states.collection_as_dict()
    if type(collection_as_dict) != type(dict([])) or len(collection_as_dict) != 9:
        errors.append('error2')

    if collection_of_states.state_identifiers_as_list() != ['identifier_0', 'identifier_1', 'identifier_2', 'identifier_3',
                                                             'identifier_4', 'identifier_5', 'identifier_6', 'identifier_7', 'identifier_8']:
        errors.append('error3')
    if collection_of_states.initial_states_as_list() != ['00', '10', '20', '01', '11', '21', '02', '12', '22']:
        errors.append('error4')

    state_in_col = collection_of_states.get_state(initial_state='00')
    state_in_col.initial_state = 'test123'
    collection_of_states.add_state(state_in_col, identifier = 'id_new')

    if not ('test123' in collection_of_states.initial_states_as_list() and 'id_new' in collection_of_states.state_identifiers_as_list()):
        errors.append('error5')

    if not (bool(collection_of_states) and len(collection_of_states) == 10):
        errors.append('error6')

    collection_of_states.delete_state(identifier='id_new')
    collection_of_states.delete_state(initial_state='20')

    collection_as_dict = collection_of_states.collection_as_dict()
    collection_of_states.clear()
    if not (len(collection_as_dict) == 8 and len(collection_of_states) == 0):
        errors.append('error7')

    collection_of_states.collection_from_dict(collection_of_states_as_dictionary=collection_as_dict)
    if not (len(collection_of_states) == 8):
        errors.append('error8')

    new_collection_of_states = collection_of_states.copy()

    collection_of_states.initialize_to_default()
    if not (len(new_collection_of_states) == 8 and len(collection_of_states) == 9):
        errors.append('error9')


    collection_of_states.filter_on_identifier(identifier_to_filter=['identifier_1', 'identifier_3', 'identifier_5'])
    filtered_collection = collection_of_states.copy()
    filtered_collection_as_dict = filtered_collection.collection_as_dict()
    if not (len(filtered_collection) == 3 and len(filtered_collection_as_dict) == 3):
        errors.append('error10')



    collection_of_states.filter_on_initial_state(['10', '01'])
    filtered_collection_as_dict = collection_of_states.collection_as_dict()
    if not (len(collection_of_states) == 2 and len(filtered_collection_as_dict) == 2):
        errors.append('error11')

    new_collection_of_states = cos.CollectionOfStates(circuit, input_collection_as_a_dict = filtered_collection_as_dict)

    if new_collection_of_states.initial_states_as_list() != ['10', '01']:
        errors.append('error12')
    if new_collection_of_states.state_identifiers_as_list() != ['identifier_1', 'identifier_3']:
        errors.append('error13')

    state = cos.State(collection_of_states)

    new_collection_of_states.add_state(identifier = 'new', state = state)
    new_collection_of_states.delete_state(identifier='identifier_1')
    if new_collection_of_states.state_identifiers_as_list() != ['identifier_3', 'new']:
        errors.append('error14')
    assert len(errors) == 0

def test_state_as_a_class():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 2, 
                                    no_of_classical_channels=3, 
                                    )
    collection = cos.CollectionOfStates(circuit)
    state = cos.State(collection)

    errors = []
    if state.initial_state != '00' or state.cumulative_probability != 1.0 or state.classical_channel_values != [0,0,0]:
        errors.append('error1')
    oc = state.optical_components
    if '00' != list(oc.keys())[0] or list(oc.values())[0]['amplitude'] != np.cdouble(1) or list(oc.values())[0]['probability'] != 1.0:
        errors.append('error2')

    state.initial_state = '11'
    state.cumulative_probability = 0.5
    state.classical_channel_values = [1,2,math.pi]
    state.optical_components = {'11' : {'amplitude' : 0.5, 'probability' : 0.25}}
    state.measurement_results = [{'measurement_results' : [0,0,0], 'probability' : 1}]

    if state.initial_state != '11' or state.cumulative_probability != 0.5 or state.classical_channel_values != [1,2,math.pi]:
        errors.append('error1')
    oc = state.optical_components
    if '11' != list(oc.keys())[0] or list(oc.values())[0]['amplitude'] != np.cdouble(0.5) or list(oc.values())[0]['probability'] != 0.25:
        errors.append('error2')

    if bool(state) and state == 2 and int(state) == 2 and state.photon_number() == (2,True):
        pass
    else:
        errors.append('error3')

    oc = state.optical_components
    oc.update({'01' : {'amplitude' : 0.5, 'probability' : 0.25}})
    if bool(state) and state == 2 and int(state) == 2 and state.photon_number() == (2,False):
        pass
    else:
        errors.append('error4')

    state1 = state.copy()
    if bool(state1) and state1 == 2 and int(state1) == 2 and state1.photon_number() == (2,False):
        pass
    else:
        errors.append('error5')

    state.initialize_this_state()

    if bool(state1) and state1 == 2 and int(state1) == 2 and state1.photon_number() == (2,False):
        pass
    else:
        errors.append('error6')

    if state.initial_state != '00' or state.cumulative_probability != 1.0 or state.classical_channel_values != [0,0,0]:
        errors.append('error7')
    oc = state.optical_components
    if '00' != list(oc.keys())[0] or list(oc.values())[0]['amplitude'] != np.cdouble(1) or list(oc.values())[0]['probability'] != 1.0:
        errors.append('error8')

    assert len(errors) == 0



def test_input_collection_unchanged_by_circuit():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                no_of_optical_channels = 8,
                                no_of_classical_channels=10
                                )
    circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1, angle = -math.pi/8)
    circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5, angle = math.pi/8)
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[6,7,0,1,4,5],classical_channels_to_be_written=[2,3,4,5,6,7])
    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(initial_state_to_filter='00111100')
    result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    state_in = initial_collection_of_states.get_state(initial_state ='00111100')
    error1 = (list(state_in.optical_components.keys()) == ['00111100'])
    result1 = result.copy()
    result1.filter_on_classical_channel(classical_channel_numbers=[6,7], values_to_filter=[2,0])
    state_out1 = result1.get_state(initial_state = '00111100')
    error2 = (list(state_out1.optical_components.keys()) == ['00112000'])
    result.filter_on_classical_channel(classical_channel_numbers=[6,7], values_to_filter=[0,2])
    state_out2 = result.get_state(initial_state = '00111100')
    error3 = (list(state_out2.optical_components.keys()) == ['00110200'])
    assert all([error1, error2, error3])

def test_extend_collection_of_states_class_and_opt():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=2
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state('000000')
    failure_found = no_failure_found and len(initial_collection_of_states) == 1
    new_collection = initial_collection_of_states.copy()
    new_collection.extend(extra_optical_channels=2,extra_classical_channels=1,statistical_distribution=[0.5,0.5])
    failure_found = no_failure_found and initial_collection_of_states == 1
    failure_found = no_failure_found and new_collection == 4
    list_of_component_names = []
    for state in new_collection:
        oc = state.optical_components
        no_failure_found = no_failure_found and len(oc.keys()) == 1
        no_failure_found = no_failure_found and list(oc.keys())[0] in ['00000000','00000001','00000010','00000011']
        list_of_component_names.append(list(oc.keys())[0])
        no_failure_found = no_failure_found and len(state.classical_channel_values) == 3
        no_failure_found = no_failure_found and state.cumulative_probability == 0.25
    no_failure_found = no_failure_found and len(set(list_of_component_names)) == 4

    assert no_failure_found

def test_extend_collection_of_states_class_only():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=2
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state('000000')
    failure_found = no_failure_found and len(initial_collection_of_states) == 1
    new_collection = initial_collection_of_states.copy()
    new_collection.extend(extra_classical_channels=4)
    failure_found = no_failure_found and initial_collection_of_states == 1
    failure_found = no_failure_found and new_collection == 1
    for state in new_collection:
        oc = state.optical_components
        no_failure_found = no_failure_found and len(oc.keys()) == 1
        no_failure_found = no_failure_found and list(oc.keys())[0] == '0'*6
        no_failure_found = no_failure_found and len(state.classical_channel_values) == 6
        no_failure_found = no_failure_found and state.cumulative_probability == 1
    assert no_failure_found


def test_extend_collection_of_states_opt_only():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=2
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state('000000')
    failure_found = no_failure_found and len(initial_collection_of_states) == 1
    new_collection = initial_collection_of_states.copy()
    new_collection.extend(extra_optical_channels=4)
    failure_found = no_failure_found and initial_collection_of_states == 1
    failure_found = no_failure_found and new_collection == 1
    for state in new_collection:
        oc = state.optical_components
        no_failure_found = no_failure_found and len(oc.keys()) == 1
        no_failure_found = no_failure_found and list(oc.keys())[0] == '0'*10
        no_failure_found = no_failure_found and len(state.classical_channel_values) == 2
        no_failure_found = no_failure_found and state.cumulative_probability == 1
    assert no_failure_found

def test_clean_up_collection_of_states_1():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=2,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )
    # test grouping together of states with same optical components
    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state('000000')
    no_failure_found = no_failure_found and len(initial_collection_of_states) == 1
    new_collection = initial_collection_of_states.copy()
    new_collection.clear()
    for state in initial_collection_of_states:
        state.cumulative_probability = 0.25
        state.optical_components = {'001100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2},'101100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}}
        new_collection.add_state(state)
        new_collection.add_state(state)
        new_collection.add_state(state)
        new_collection.add_state(state)
    no_failure_found = no_failure_found and len(new_collection) == 4
    new_collection.clean_up()
    no_failure_found = no_failure_found and len(new_collection) == 1

    # test grouping together of states with same optical components even if probability is low
    new_states = [('001100',0.2),('001100',0.2),('001100',0.2),('001100',0.05),('001100',0.35)]
    for initial_state, cumulative_probability in new_states:
        new_state = initial_collection_of_states.get_state(initial_state='000000').copy()
        new_state.initial_state = initial_state
        new_state.cumulative_probability = cumulative_probability
        new_collection.add_state(new_state)
    no_failure_found = no_failure_found and len(new_collection) == 6
    new_collection.clean_up()
    no_failure_found = no_failure_found and len(new_collection) == 2

    # test discaring states with too low cumulative probability
    new_states = [('001101',0.09),('101101',0.002),('101100',0.09),('101100',0.09)]
    for initial_state, cumulative_probability in new_states:
        new_state = initial_collection_of_states.get_state(initial_state='000000').copy()
        new_state.initial_state = initial_state
        new_state.cumulative_probability = cumulative_probability
        new_collection.add_state(new_state)
    no_failure_found = no_failure_found and len(new_collection) == 6
    new_collection.clean_up()
    no_failure_found = no_failure_found and len(new_collection) == 3

    assert no_failure_found

def test_clean_up_collection_of_states_2():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=2,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )
    # test grouping together of states with same optical components
    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['000000','000001','010000','000100'])
    new_collection = initial_collection_of_states.copy()
    new_collection.clear()
    list_of_ocs = [
        {'001100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2},'101100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}},
        {'001101':{'amplitude': math.sqrt(1/2) , 'probability': 1/2},'101100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}},
        {'001100':{'amplitude': -math.sqrt(1/2) , 'probability': 1/2},'101100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}},
        {'001100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2},'101100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}}
    ]
    for index, state in enumerate(initial_collection_of_states):
        state.initial_state = '000000'
        state.cumulative_probability = 0.25
        state.optical_components = list_of_ocs[index]
        new_collection.add_state(state)
    no_failure_found = no_failure_found and len(new_collection) == 4
    new_collection.clean_up()
    no_failure_found = no_failure_found and len(new_collection) == 3

    # test normalization of optical components
    new_collection = initial_collection_of_states.copy()
    new_collection.clear()
    p1 = 100/100
    p2 = 25/100
    list_of_ocs = [
        {'001100':{'amplitude': 1j*math.sqrt(p1) , 'probability': p1},'101100':{'amplitude': math.sqrt(p2) , 'probability': p2}},
        {'001100':{'amplitude': -1*math.sqrt(p1) , 'probability': p1},'101110':{'amplitude': math.sqrt(p2) , 'probability': p2}},
        {'001100':{'amplitude': math.sqrt(p1) , 'probability': p1},'211100':{'amplitude': math.sqrt(p2) , 'probability': p2}},
        {'001100':{'amplitude': -1j*math.sqrt(p1) , 'probability': p1},'111130':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    ]
    for index, state in enumerate(initial_collection_of_states):
        state.initial_state = '000000'
        state.cumulative_probability = 0.25
        state.optical_components = list_of_ocs[index]
        new_collection.add_state(state)
    no_failure_found = no_failure_found and len(new_collection) == 4
    new_collection.clean_up()
    no_failure_found = no_failure_found and len(new_collection) == 4
    state = new_collection.get_state(initial_state='000000')
    oc = state.optical_components
    no_failure_found = no_failure_found and len(oc) == 2
    no_failure_found = no_failure_found and oc['001100']['probability'] - 0.8 < 0.0001
    no_failure_found = no_failure_found and (np.abs(no_failure_found and oc['001100']['amplitude'] - 1j*math.sqrt(0.8))**2 < 0.0001)

    # test discarding
    new_collection = initial_collection_of_states.copy()
    new_collection.clear()
    p1 = 80/100
    p2 = 1/100
    list_of_ocs = [
        {'021100':{'amplitude': 1j*math.sqrt(p1) , 'probability': p1},'101100':{'amplitude': math.sqrt(p2) , 'probability': p2}},
        {'021100':{'amplitude': -1*math.sqrt(p1) , 'probability': p1},'101110':{'amplitude': math.sqrt(p2) , 'probability': p2}},
        {'021100':{'amplitude': math.sqrt(p1) , 'probability': p1},'112100':{'amplitude': math.sqrt(p2) , 'probability': p2}},
        {'021100':{'amplitude': -1j*math.sqrt(p1) , 'probability': p1},'111110':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    ]
    for index, state in enumerate(initial_collection_of_states):
        state.initial_state = '000000'
        state.cumulative_probability = 0.25
        state.optical_components = list_of_ocs[index]
        new_collection.add_state(state)
    no_failure_found = no_failure_found and len(new_collection) == 4
    new_collection.clean_up()
    no_failure_found = no_failure_found and len(new_collection) == 4
    state = new_collection.get_state(initial_state='000000')
    oc = state.optical_components
    no_failure_found = no_failure_found and len(oc) == 1
    no_failure_found = no_failure_found and oc['021100']['probability'] - 1 < 0.0001
    no_failure_found = no_failure_found and (np.abs(no_failure_found and oc['021100']['amplitude'] - 1j*math.sqrt(1))**2 < 0.0001)


    assert no_failure_found

def test_density_matrix1():

    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=2,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['000000', '000001', '000010', '100000'])

    result1 = initial_collection_of_states.density_matrix(initial_state='000001')
    no_failure_found = no_failure_found and len(result1) == 1

    result = initial_collection_of_states.density_matrix()
    no_failure_found = no_failure_found and len(result) == 4
    no_failure_found = no_failure_found and result['000000']['trace'] ==1 and result['000000']['trace_dm_squared'] == 1

    state = initial_collection_of_states.get_state(initial_state = '000000')
    initial_collection_of_states.clear()
    state.optical_components = {'001100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2},'101100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}}
    initial_collection_of_states.add_state(state)
    result = initial_collection_of_states.density_matrix()
    no_failure_found = no_failure_found and len(result) == 1
    no_failure_found = no_failure_found and result['000000']['trace'] ==1 and result['000000']['trace_dm_squared'] == 1

    # add state with different components
    state = initial_collection_of_states.get_state(initial_state = '000000')
    initial_collection_of_states.clear()
    state.optical_components = {'001100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2},'101100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}}
    state.cumulative_probability = 0.5
    initial_collection_of_states.add_state(state)
    state2 = state.copy()
    state2.optical_components = {'201100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2},'101103':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}}
    initial_collection_of_states.add_state(state2)
    result = initial_collection_of_states.density_matrix()
    no_failure_found = no_failure_found and len(result) == 1
    no_failure_found = no_failure_found and result['000000']['trace'] ==1 and result['000000']['trace_dm_squared'] == 0.5

    # add state with same components
    state = initial_collection_of_states.get_state(initial_state = '000000')
    initial_collection_of_states.clear()
    state.optical_components = {'201100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2},'101103':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}}
    state.cumulative_probability = 0.5
    initial_collection_of_states.add_state(state)
    state2 = state.copy()
    state2.optical_components = {'201100':{'amplitude': math.sqrt(1/2) , 'probability': 1/2},'101103':{'amplitude': math.sqrt(1/2) , 'probability': 1/2}}
    initial_collection_of_states.add_state(state2)
    result = initial_collection_of_states.density_matrix()
    no_failure_found = no_failure_found and len(initial_collection_of_states) == 2
    no_failure_found = no_failure_found and len(result) == 1
    no_failure_found = no_failure_found and result['000000']['trace'] ==1 and result['000000']['trace_dm_squared'] == 1
    
    assert no_failure_found

def test_density_matrix2():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=0,
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['000', '222', '002'])

    result = initial_collection_of_states.density_matrix()
    no_failure_found = no_failure_found and len(result) == 3
    dm = result['000']['density_matrix']
    no_failure_found = no_failure_found and len(dm) == 27
    no_failure_found = no_failure_found and dm[0][0] == 1 and dm[1][1] == 0
    no_failure_found = no_failure_found and dm[0][-1] == 0 and dm[0][-1] == 0

    dm = result['222']['density_matrix']
    no_failure_found = no_failure_found and len(dm) == 27
    no_failure_found = no_failure_found and dm[-1][-1] == 1 and dm[0][0] == 0
    no_failure_found = no_failure_found and dm[0][-1] == 0 and dm[0][-1] == 0

    dm = result['002']['density_matrix']
    basis = initial_collection_of_states.generate_allowed_components_names_as_list_of_strings()
    index = basis.index('002')
    no_failure_found = no_failure_found and len(dm) == 27
    no_failure_found = no_failure_found and dm[index][index] == 1 and dm[1][1] == 0

    state1 = initial_collection_of_states.get_state(initial_state='222')
    state1.cumulative_probability = 0.5
    state1.initial_state = '123'
    state2 = initial_collection_of_states.get_state(initial_state='000')
    state2.cumulative_probability = 0.5
    state2.initial_state = '123'
    initial_collection_of_states.clear()
    initial_collection_of_states.add_state(state1)
    initial_collection_of_states.add_state(state2)
    result = initial_collection_of_states.density_matrix()
    dm = result['123']['density_matrix']
    no_failure_found = no_failure_found and len(result) == 1
    no_failure_found = no_failure_found and dm[-1][-1] == 0.5 and dm[0][0] == 0.5
    no_failure_found = no_failure_found and dm[0][-1] == 0 and dm[0][-1] == 0

    state1 = initial_collection_of_states.get_state(initial_state='123')
    state1.cumulative_probability = 1
    state1.initial_state = '678'
    p1 = 1/2
    p2 = 1/2
    state1.optical_components = {'000':{'amplitude': math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    initial_collection_of_states.clear()
    initial_collection_of_states.add_state(state1)

    result = initial_collection_of_states.density_matrix(decimal_places_for_trace = 2)
    dm = result['678']['density_matrix']
    no_failure_found = no_failure_found and len(result) == 1
    no_failure_found = no_failure_found and abs(dm[-1][-1] - 0.5)<0.00001 and abs(dm[0][0] == 0.5)<0.00001
    
    no_failure_found = no_failure_found and abs(dm[0][-1] - 0.5)<0.00001 and abs(dm[0][-1] - 0.5)<0.00001


    assert no_failure_found


def test_adjust_length_of_fock_state():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=4,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['000000', '100001', '200010', '300000'])
    initial_collection_of_states.adjust_length_of_fock_state(new_length_of_fock_state=2)
    no_failure_found = no_failure_found and initial_collection_of_states._length_of_fock_state == 2

    initial_collection_of_states.get_state(initial_state='300000')
    

def test_reduce():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=4,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['000000', '100001', '200010', '300000'])

    new_collection = initial_collection_of_states.copy()
    new_collection.reduce(optical_channels_to_keep=[0,1,2,3,4,5], classical_channels_to_keep=[0,1,2,3])
    state = new_collection.get_state(initial_state='000000')
    no_failure_found = no_failure_found and len(state.classical_channel_values) == 4

    new_collection = initial_collection_of_states.copy()
    new_collection.reduce()
    state = new_collection.get_state(initial_state='000000')
    no_failure_found = no_failure_found and len(state.classical_channel_values) == 4
    no_failure_found = no_failure_found and len(list(state.optical_components.keys())[0]) == 6


    new_collection = initial_collection_of_states.copy()
    new_collection.reduce(optical_channels_to_keep=[0,1,2,3], classical_channels_to_keep=[0,1,3])
    state = new_collection.get_state(initial_state='000000')

    no_failure_found = no_failure_found and len(state.classical_channel_values) == 3

    no_failure_found = no_failure_found and len(list(state.optical_components.keys())[0]) == 4

    no_failure_found = no_failure_found and np.abs(state.cumulative_probability - 1 ) < 0.0001

    new_collection.clean_up()
    no_failure_found = no_failure_found and len(new_collection) ==4 

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['210000', '220010', '210010', '220000'])
    no_failure_found = no_failure_found and len(new_collection) == 4 
    new_collection = initial_collection_of_states.copy()
    new_collection.reduce(optical_channels_to_keep=[2,3,4,5])
    no_failure_found = no_failure_found and len(new_collection) == 4
    new_collection.clean_up()
    no_failure_found = no_failure_found and len(new_collection) == 4

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    state = initial_collection_of_states.get_state(initial_state='000000')
    initial_collection_of_states.clear()
    p1 = 80/100
    p2 = 20/100
    state.optical_components = {'022100':{'amplitude': 1j*math.sqrt(p1) , 'probability': p1},'021200':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    initial_collection_of_states.add_state(state)
    new_collection = initial_collection_of_states.copy()
    new_collection.reduce(optical_channels_to_keep=[0,1,4,5], classical_channels_to_keep=[0,1,3])
    state = new_collection.get_state(initial_state='000000')
    no_failure_found = no_failure_found and len(state.classical_channel_values) == 3
    no_failure_found = no_failure_found and len(list(state.optical_components.keys())[0]) == 4
    no_failure_found = no_failure_found and np.abs(state.cumulative_probability -0.8 ) < 0.0001
    no_failure_found = no_failure_found and len(new_collection) == 2
    assert no_failure_found

def test_rescale_optical_components():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=4,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    state = initial_collection_of_states.get_state(initial_state='000')
    p1 = 1/2
    p2 = 1/2
    state.optical_components = {'000':{'amplitude': math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    state._rescale_optical_components()
    for amp_prob in state.optical_components.values():
        no_failure_found = no_failure_found and np.abs(amp_prob['probability'] - 0.5) < 0.001

    p1 = 0.2
    p2 = 1
    state.optical_components = {'000':{'amplitude': math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    state._rescale_optical_components()
    no_failure_found = no_failure_found and len(state.optical_components) == 2
    total_prob = 0
    for amp_prob in state.optical_components.values():
        total_prob += amp_prob['probability']
        no_failure_found = no_failure_found and np.abs((amp_prob['amplitude'])**2 - amp_prob['probability']) < 0.001
    no_failure_found = no_failure_found and np.abs(total_prob - 1.0) < 0.001

    p1 = 0.001
    p2 = 0.2
    state.optical_components = {'000':{'amplitude': math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    state._rescale_optical_components()
    no_failure_found = no_failure_found and len(state.optical_components) == 1
    total_prob = 0
    for amp_prob in state.optical_components.values():
        total_prob += amp_prob['probability']
        no_failure_found = no_failure_found and np.abs((amp_prob['amplitude'])**2 - amp_prob['probability']) < 0.001
    no_failure_found = no_failure_found and np.abs(total_prob - 1.0) < 0.001
    assert no_failure_found

def test_identical_optical_components():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=4,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    state1 = initial_collection_of_states.get_state(initial_state='000')
    state2 = initial_collection_of_states.get_state(initial_state='001')
    no_failure_found = no_failure_found and False == state1._identical_optical_components(state2)
    no_failure_found = no_failure_found and False == state2._identical_optical_components(state1)

    p1 = 0.2
    p2 = 1
    state1.optical_components = {'000':{'amplitude': math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    state1._rescale_optical_components()
    p1 = 0.2
    p2 = 1
    state2.optical_components = {'000':{'amplitude': math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    state2._rescale_optical_components()
    no_failure_found = no_failure_found and True == state1._identical_optical_components(state2)
    no_failure_found = no_failure_found and True == state2._identical_optical_components(state1)

    p1 = 0.2
    p2 = 1
    state1.optical_components = {'000':{'amplitude': 1j*math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    state1._rescale_optical_components()
    p1 = 0.2
    p2 = 1
    state2.optical_components = {'000':{'amplitude': math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    state2._rescale_optical_components()
    no_failure_found = no_failure_found and False == state1._identical_optical_components(state2)
    no_failure_found = no_failure_found and False == state2._identical_optical_components(state1)

    p1 = 0.2
    p2 = 0.6
    state1.optical_components = {'000':{'amplitude': math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    state1._rescale_optical_components()
    p1 = 0.1
    p2 = 1
    state2.optical_components = {'000':{'amplitude': math.sqrt(p1) , 'probability': p1},'222':{'amplitude': math.sqrt(p2) , 'probability': p2}}
    state2._rescale_optical_components()
    no_failure_found = no_failure_found and False == state1._identical_optical_components(state2)
    no_failure_found = no_failure_found and False == state2._identical_optical_components(state1)

    state2.optical_components = {'000':{'amplitude': math.sqrt(1) , 'probability': 1}}
    state2._rescale_optical_components()
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 5, 
                                    no_of_classical_channels=4,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states_large = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states_large.filter_on_initial_state('00000')
    state1 = initial_collection_of_states_large.get_state(initial_state='00000')
    no_failure_found = no_failure_found and False == state1._identical_optical_components(state2)
    no_failure_found = no_failure_found and False == state2._identical_optical_components(state1)

    new_collection = initial_collection_of_states_large.copy()
    new_collection.reduce(optical_channels_to_keep=[0,1,2])
    state1 = new_collection.get_state(initial_state='00000')
    no_failure_found = no_failure_found and True == state1._identical_optical_components(state2)
    no_failure_found = no_failure_found and True == state2._identical_optical_components(state1)
    assert no_failure_found

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

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
    state = cos.State(collection_of_states=initial_collection_of_states)
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

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
    state = cos.State(collection_of_states=initial_collection_of_states)
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

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
    state = cos.State(collection_of_states=initial_collection_of_states)
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

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit3, input_collection_as_a_dict=dict([]))
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('0001',1)]
    state.initial_state = '0001'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('1332',1)]
    state.initial_state = '1332'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('2203',1)]
    state.initial_state = '2203'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
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

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit4, input_collection_as_a_dict=dict([]))
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('0001',1)]
    state.initial_state = '0001'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('1332',1)]
    state.initial_state = '1332'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('2203',1)]
    state.initial_state = '2203'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
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

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit5, input_collection_as_a_dict=dict([]))
    state = cos.State(collection_of_states=initial_collection_of_states)
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


def test_adjust_length_of_fock_state():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=4,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['000000', '100001', '200010', '300000'])
    initial_collection_of_states.adjust_length_of_fock_state(new_length_of_fock_state=2)
    no_failure_found = no_failure_found and initial_collection_of_states._length_of_fock_state == 2

    state = initial_collection_of_states.get_state(initial_state='300000')
    no_failure_found = no_failure_found and list(state.optical_components.keys())[0] == '100000'

    state = initial_collection_of_states.get_state(initial_state='200010')
    no_failure_found = no_failure_found and list(state.optical_components.keys())[0] == '000010'

    state = initial_collection_of_states.get_state(initial_state='100001')
    no_failure_found = no_failure_found and list(state.optical_components.keys())[0] == '100001'

    circuit = fsc.FockStateCircuit(length_of_fock_state = 9, 
                                    no_of_optical_channels = 2, 
                                    no_of_classical_channels=4,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['00', '10', '88', '08'])
    initial_collection_of_states.adjust_length_of_fock_state(new_length_of_fock_state=11)
    no_failure_found = no_failure_found and initial_collection_of_states._length_of_fock_state == 11

    state = initial_collection_of_states.get_state(initial_state='00')
    no_failure_found = no_failure_found and list(state.optical_components.keys())[0] == '0000'

    state = initial_collection_of_states.get_state(initial_state='10')
    no_failure_found = no_failure_found and list(state.optical_components.keys())[0] == '0100'

    state = initial_collection_of_states.get_state(initial_state='88')
    no_failure_found = no_failure_found and list(state.optical_components.keys())[0] == '0808'

    assert no_failure_found

def test_reshaping_collection():

    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 5, 
                                    no_of_classical_channels=1,
                                    channel_0_left_in_state_name = False,
                                    threshold_probability_for_setting_to_zero=0.1,
                                    use_full_fock_matrix=True,
                                    )
    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.print_only_last_measurement = False
    initial_collection_of_states.filter_on_initial_state(['22222', '20202'])
    state1 = initial_collection_of_states.get_state(initial_state='20202')
    state1.measurement_results = [{'measurement_results': [1,1,1], 'probability':1}]
    initial_collection_of_states.adjust_length_of_fock_state(new_length_of_fock_state=5)

    initial_collection_of_states.reduce(optical_channels_to_keep=[0,1])
    initial_collection_of_states.extend(extra_classical_channels=4)
    initial_collection_of_states.adjust_length_of_fock_state(new_length_of_fock_state=12)
    state = initial_collection_of_states.get_state(initial_state='22222')
    no_failure_found = no_failure_found and (state._digits_per_optical_channel == 2)
    no_failure_found = no_failure_found and (state._string_format_in_state_as_word == "{:02d}")
    initial_collection_of_states.extend(extra_optical_channels=1)
    initial_collection_of_states.reduce(classical_channels_to_keep=[0,3,4])
    initial_collection_of_states.adjust_length_of_fock_state(new_length_of_fock_state=3)
    state = initial_collection_of_states.get_state(initial_state='20202')

    no_failure_found = no_failure_found and all([
        len(initial_collection_of_states._collection_of_states) == 2,
        initial_collection_of_states._length_of_fock_state == 3,
        initial_collection_of_states._no_of_classical_channels == 3,
        initial_collection_of_states._no_of_optical_channels == 3,
        len(initial_collection_of_states._list_of_fock_states) == 27,
        len(list(initial_collection_of_states._list_of_fock_states)[0]) == 3,
        list(initial_collection_of_states._list_of_fock_states)[-1][-1] == 2,
        state._length_of_fock_state == 3,
        state._no_of_optical_channels == 3,
        state._no_of_classical_channels == 3,      
        state._channel_0_left_in_state_name == False,
        state._digits_per_optical_channel == 1,
        state._dict_of_valid_component_names['000'] == [0,0,0],
        state._string_format_in_state_as_word == "{:01d}",
        state._threshold_probability_for_setting_to_zero == 0.1,
        state._print_only_last_measurement == False,
        True in state._check_valid_state(state_to_check = state),
        state.measurement_results[0]['measurement_results'] == [1,1,1]])

    assert no_failure_found

def test_c_shift():
    no_failure_found = True
    circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                no_of_optical_channels = 3, 
                                no_of_classical_channels=2
                                )
    circuit1.c_shift(control_channels=[0], target_channels=[1])

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('100',1)]
    state.initial_state = '100'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('110',1)]
    state.initial_state = '110'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('000',1)]
    state.initial_state = '000'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
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

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit2, input_collection_as_a_dict=dict([]))
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('0001',1)]
    state.initial_state = '0001'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('1332',1)]
    state.initial_state = '1332'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('2203',1)]
    state.initial_state = '2203'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
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

    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit3, input_collection_as_a_dict=dict([]))
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('0001',1)]
    state.initial_state = '0001'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('1332',1)]
    state.initial_state = '1332'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
    state.optical_components = [('2203',1)]
    state.initial_state = '2203'
    initial_collection_of_states.add_state(state=state)
    state = cos.State(collection_of_states=initial_collection_of_states)
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

def test_simple_draw(mocker):
    def save_drawing():
        plt.savefig("./tests/test_drawings/testdrawing_simple_draw.png")
        return
    with patch("fock_state_circuit.plt.show", wraps=save_drawing) as mock_bar:
        #mocker.patch('fock_state_circuit.plt.show', return_value=None)
        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=2
                                    )
        circuit1.c_shift(control_channels=[0], target_channels=[1])
        circuit1.draw()
        img1 = "./tests/test_drawings/testdrawing_simple_draw.png"
        img2 = img1.replace("testdrawing","reference")
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_all_nodes(mocker):
    def save_drawing():
        plt.savefig("./tests/test_drawings/testdrawing_all_nodes.png")
        return
    with patch("fock_state_circuit.plt.show", wraps=save_drawing) as mock_bar:
        #mocker.patch('fock_state_circuit.plt.show', return_value=None)
        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=3
                                    )
        circuit1.half_wave_plate(channel_horizontal=0,channel_vertical=1, angle=0)
        circuit1.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit1.half_wave_plate_225(channel_horizontal=0,channel_vertical=1)
        circuit1.quarter_wave_plate(channel_horizontal=0,channel_vertical=1, angle=0)
        circuit1.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit1.quarter_wave_plate_225(channel_horizontal=0,channel_vertical=1)
        circuit1.wave_plate_classical_control(optical_channel_horizontal=0,optical_channel_vertical=1, classical_channel_for_orientation=0,classical_channel_for_phase_shift=1)
        circuit1.phase_shift_single_channel(channel_for_shift=0,phase_shift=0)
        circuit1.phase_shift_single_channel_classical_control(optical_channel_to_shift=1,classical_channel_for_phase_shift=0)
        circuit1.polarizing_beamsplitter(input_channels_a=(0,1), input_channels_b=(2,3))
        circuit1.non_polarizing_beamsplitter(input_channels_a=(0,1), input_channels_b=(2,3))
        circuit1.non_polarizing_50_50_beamsplitter(input_channels_a=(0,1), input_channels_b=(2,3))
        circuit1.mix_50_50(first_channel=0,second_channel=1)
        circuit1.mix_generic_refl_transm(first_channel=0,second_channel=1,reflection=0.5,transmission=0.5)
        circuit1.swap(first_channel=0,second_channel=1)
        circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 4, 
                                        no_of_classical_channels=3,
                                        circuit_name='circuit\nfor fun'
                                        )
        circuit1.bridge(next_fock_state_circuit=circuit2)
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2], classical_channels_to_be_written=[0,1,2])
        def test_function(current_values, new_input_values, affected_channels):
            output_list = current_values[::-1] 
            return output_list
        circuit2.classical_channel_function(function=test_function, affected_channels=[0,1])
        circuit2.set_classical_channels(list_of_values_for_classical_channels=[1,1], list_of_classical_channel_numbers=[0,1])
        matrix = np.identity(2, dtype = np.cdouble)
        circuit2.custom_optical_node(matrix_optical=matrix,optical_channels=[0,1])
        matrix = np.identity(len(circuit2.basis()), dtype = np.cdouble)
        circuit2.custom_fock_state_node(custom_fock_matrix=matrix)
        circuit2.channel_coupling(control_channels=[0,1], target_channels=[2,3], coupling_strength=1)
        circuit2.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=1)

        circuit1.draw()
        img1 = "./tests/test_drawings/testdrawing_all_nodes.png"
        img2 = img1.replace("testdrawing","reference")
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_combined_nodes(mocker):
    def save_drawing():
        plt.savefig("./tests/test_drawings/testdrawing_combined_nodes.png")
        #plt.savefig("testdrawing.png")
        return
    with patch("fock_state_circuit.plt.show", wraps=save_drawing) as mock_bar:
        #mocker.patch('fock_state_circuit.plt.show', return_value=None)
        circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 5, 
                                    no_of_classical_channels=3
                                    )
        circuit2.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit2.quarter_wave_plate_45(channel_horizontal=3,channel_vertical=4)
        circuit2.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=1, node_info = {'label': 'combi-gate', 'combined_gate': True})
        circuit2.quarter_wave_plate_45(channel_horizontal=3,channel_vertical=4, node_info = {'label': 'ignore', 'combined_gate': True})
        circuit2.quarter_wave_plate_45(channel_horizontal=3,channel_vertical=4)
        circuit2.quarter_wave_plate_45(channel_horizontal=2,channel_vertical=3)
        circuit2.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=1, node_info = {'label': 'combi-gate', 'combined_gate': True})
        circuit2.quarter_wave_plate_45(channel_horizontal=2,channel_vertical=3, node_info = {'label': 'ignore', 'combined_gate': True})
        circuit2.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=3, node_info = {'label': 'ignore', 'combined_gate': True})
        circuit2.quarter_wave_plate_45(channel_horizontal=3,channel_vertical=4)
        circuit2.quarter_wave_plate_45(channel_horizontal=3,channel_vertical=4)
        circuit2.quarter_wave_plate_45(channel_horizontal=2,channel_vertical=3)
        circuit2.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=1, node_info = {'label': 'combi-gate', 'combined_gate': True})
        circuit2.quarter_wave_plate_45(channel_horizontal=2,channel_vertical=3, node_info = {'label': 'ignore', 'combined_gate': True})
        circuit2.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=3, node_info = {'label': 'ignore', 'combined_gate': True})
        circuit2.quarter_wave_plate_45(channel_horizontal=2,channel_vertical=3, node_info = {'label': 'ignore', 'combined_gate': True})
        circuit2.quarter_wave_plate_45(channel_horizontal=3,channel_vertical=4)
        circuit2.quarter_wave_plate_45(channel_horizontal=2,channel_vertical=3)
        circuit2.set_classical_channels(list_of_values_for_classical_channels=[1,1], list_of_classical_channel_numbers=[0,1])
        circuit2.draw()
        img1 = "./tests/test_drawings/testdrawing_combined_nodes.png"
        img2 = img1.replace("testdrawing","reference")
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_custom_settings(mocker):
    def save_drawing():
        plt.savefig("./tests/test_drawings/testdrawing_custom_settings.png")
        #plt.savefig("testdrawing.png")
        return
    with patch("fock_state_circuit.plt.show", wraps=save_drawing) as mock_bar:
        #mocker.patch('fock_state_circuit.plt.show', return_value=None)
        circuit_draw_settings = {
                    'figure_width_in_inches' : 16,
                    'channel_line_length_as_fraction_of_figure_width' : 0.8,
                    'number_of_nodes_on_a_line': 6,
                    'spacing_between_lines_in_relation_to_spacing_between_nodes' : 2,
                    'compound_circuit_title' : 'New name for this circuit \n and taking two lines',
                    'channel_label_string_max_length': 15,
                    'node_label_string_max_length': 15,
                    'compound_plot_title_font_size' : 25,
                    'circuit_name_font_size': 5,
                    'channel_label_font_size': 20,
                    'node_label_font_size' : 10,
                    'classical_channel_line_color' : 'blue',
                    'classical_channel_line_marker' : '*',
                    'classical_channel_line_marker_size' : 5,
                    'optical_channel_line_color' :'yellow',
                    'optical_channel_line_marker': 's',
                    'optical_channel_line_marker_size' : 5,
                    'bridge_marker' : 'h',
                    'bridge_marker_size' : 20,
                    'bridge_markeredgewidth' : 2,
                    'box_around_node_linestyle' : 'solid',
                    'box_around_node_linewidth': 2,
                    'box_around_node_color' : 'black'
                }
        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 4, 
                                        no_of_classical_channels=3
                                        )
        circuit1.half_wave_plate(channel_horizontal=0,channel_vertical=1, angle=0)
        circuit1.half_wave_plate_45(channel_horizontal=2,channel_vertical=3)
        circuit1.non_polarizing_50_50_beamsplitter(input_channels_a=(0,1),input_channels_b=(2,3))
        circuit1.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit1.quarter_wave_plate_225(channel_horizontal=2,channel_vertical=3)

        circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 4, 
                                        no_of_classical_channels=3
                                        )
        circuit1.bridge(next_fock_state_circuit=circuit2)
        circuit2.swap(first_channel=2, second_channel=3)

        circuit1.draw(settings_for_drawing_circuit=circuit_draw_settings)
        img1 = "./tests/test_drawings/testdrawing_custom_settings.png"
        img2 = img1.replace("testdrawing","reference")
        assert plt_test.compare_images(img1, img2, 0.001) is None


def test_draw_bridges(mocker):
    def save_drawing():
        plt.savefig("./tests/test_drawings/testdrawing_bridges.png")
        return
    with patch("fock_state_circuit.plt.show", wraps=save_drawing) as mock_bar:
    #mocker.patch('fock_state_circuit.plt.show', return_value=None)

        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=3
                                    )
        circuit1.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        def function(current_values, new_input_values, affected_channels):
            for index, channel in enumerate(affected_channels):
                current_values[channel] = new_input_values[index]
            return current_values
        circuit1.classical_channel_function(function = function, affected_channels=[0,1], new_input_values=[10,10] )
        circuit1.wave_plate_classical_control(optical_channel_horizontal=1,optical_channel_vertical=2,classical_channel_for_orientation=1,classical_channel_for_phase_shift=2)
        circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 4, 
                                        no_of_classical_channels=4,
                                        circuit_name="2nd circuit"
                                        )
        circuit1.bridge(next_fock_state_circuit=circuit2)
        circuit2.half_wave_plate_45(channel_horizontal=2,channel_vertical=3)
        circuit2.classical_channel_function(function = function, affected_channels=[0,1], new_input_values=[10,10] )
        circuit2.wave_plate_classical_control(optical_channel_horizontal=1,optical_channel_vertical=2,classical_channel_for_orientation=1,classical_channel_for_phase_shift=2)
        def function(current_values, new_input_values, affected_channels):
            for index, channel in enumerate(affected_channels):
                current_values[channel] = new_input_values[index]
            return current_values
        circuit2.classical_channel_function(function = function, affected_channels=[0,1], new_input_values=[10,10] )
        circuit3 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 5, 
                                        no_of_classical_channels=5,
                                        circuit_name= 'number 3')
        circuit2.bridge(next_fock_state_circuit=circuit3)
        circuit3.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit3.wave_plate_classical_control(optical_channel_horizontal=0,optical_channel_vertical=1,classical_channel_for_orientation=3,classical_channel_for_phase_shift=4)
        circuit3.quarter_wave_plate_45(channel_horizontal=3,channel_vertical=4)
        circuit3.classical_channel_function(function = function, affected_channels=[1,2], new_input_values=[10,10] )
        circuit3.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit4 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 2, 
                                        no_of_classical_channels=0)
        circuit3.bridge(next_fock_state_circuit=circuit4)
        circuit4.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 5, 
                                        no_of_classical_channels=12,
                                        circuit_name= 'last')
        circuit4.bridge(next_fock_state_circuit=circuit5)
        circuit5.classical_channel_function(function = function, affected_channels=[10,11], new_input_values=[10,10] )
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=3,channel_vertical=4)
        circuit5.classical_channel_function(function = function, affected_channels=[8,9], new_input_values=[10,10] )
        circuit5.classical_channel_function(function = function, affected_channels=[6,7], new_input_values=[10,10] )
        circuit5.classical_channel_function(function = function, affected_channels=[0,1], new_input_values=[10,10] )
        circuit5.basis()
        circuit5.channel_coupling(control_channels=[0,1], target_channels=[3,4])
        custom_Fock_state_matrix = np.identity(len(circuit5.basis()),dtype = np.cdouble)
        circuit5.custom_fock_state_node(custom_fock_matrix=custom_Fock_state_matrix)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit6 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 5, 
                                        no_of_classical_channels=12,
                                        circuit_name= 'circuit6')
        circuit5.bridge(next_fock_state_circuit=circuit6)
        circuit6.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit7 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 5, 
                                        no_of_classical_channels=12,
                                        circuit_name= 'circuit7')
        circuit6.bridge(next_fock_state_circuit=circuit7)
        circuit7.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit1.draw()
        img1 = "./tests/test_drawings/testdrawing_bridges.png"
        img2 = img1.replace("testdrawing","reference")
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_compound_circuit(mocker):
    def save_drawing():
        plt.savefig("./tests/test_drawings/testdrawing_compound_circuit.png")
        return
    with patch("fock_state_circuit.plt.show", wraps=save_drawing) as mock_bar:
    #mocker.patch('fock_state_circuit.plt.show', return_value=None)

        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 3, 
                                        no_of_classical_channels=3
                                        )
        circuit1.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        def function(current_values, new_input_values, affected_channels):
            for index, channel in enumerate(affected_channels):
                current_values[channel] = new_input_values[index]
            return current_values
        circuit1.classical_channel_function(function = function, affected_channels=[0,1], new_input_values=[10,10] )
        circuit1.wave_plate_classical_control(optical_channel_horizontal=1,optical_channel_vertical=2,classical_channel_for_orientation=1,classical_channel_for_phase_shift=2)
        circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 4, 
                                        no_of_classical_channels=4,
                                        circuit_name="2nd circuit"
                                        )

        circuit2.half_wave_plate_45(channel_horizontal=2,channel_vertical=3)
        circuit2.classical_channel_function(function = function, affected_channels=[0,1], new_input_values=[10,10] )
        circuit2.wave_plate_classical_control(optical_channel_horizontal=1,optical_channel_vertical=2,classical_channel_for_orientation=1,classical_channel_for_phase_shift=2)
        def function(current_values, new_input_values, affected_channels):
            for index, channel in enumerate(affected_channels):
                current_values[channel] = new_input_values[index]
            return current_values
        circuit2.classical_channel_function(function = function, affected_channels=[0,1], new_input_values=[10,10] )

        circuit3 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 5, 
                                        no_of_classical_channels=5,
                                        circuit_name= 'number 3')
        circuit3.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit3.wave_plate_classical_control(optical_channel_horizontal=0,optical_channel_vertical=1,classical_channel_for_orientation=3,classical_channel_for_phase_shift=4)
        circuit3.quarter_wave_plate_45(channel_horizontal=3,channel_vertical=4)
        circuit3.classical_channel_function(function = function, affected_channels=[1,2], new_input_values=[10,10] )
        circuit3.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)

        circuit4 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 2, 
                                        no_of_classical_channels=0)
        circuit4.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)

        circuit5 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 5, 
                                        no_of_classical_channels=12,
                                        circuit_name= 'last')
        circuit4.bridge(next_fock_state_circuit=circuit5)
        circuit5.classical_channel_function(function = function, affected_channels=[10,11], new_input_values=[10,10] )
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=3,channel_vertical=4)
        circuit5.classical_channel_function(function = function, affected_channels=[8,9], new_input_values=[10,10] )
        circuit5.classical_channel_function(function = function, affected_channels=[6,7], new_input_values=[10,10] )
        circuit5.classical_channel_function(function = function, affected_channels=[0,1], new_input_values=[10,10] )
        circuit5.basis()
        circuit5.channel_coupling(control_channels=[0,1], target_channels=[3,4])
        custom_Fock_state_matrix = np.identity(len(circuit5.basis()),dtype = np.cdouble)
        circuit5.custom_fock_state_node(custom_fock_matrix=custom_Fock_state_matrix)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit5.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)

        circuit6 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 5, 
                                        no_of_classical_channels=12,
                                        circuit_name= 'circuit6')
        circuit6.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit7 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 5, 
                                        no_of_classical_channels=12,
                                        circuit_name= 'circuit7')
        circuit7.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)

        circuit_list = [circuit1,circuit2,circuit3,circuit4,circuit5,circuit6,circuit7]
        compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)
        compound_circuit.draw()
        circuit_list = [circuit1,circuit7]
        compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)
        compound_circuit.draw()

        img1 = "./tests/test_drawings/testdrawing_compound_circuit.png"
        img2 = img1.replace("testdrawing","reference")
        assert plt_test.compare_images(img1, img2, 0.001) is None

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
        initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit1,input_collection_as_a_dict=dict([]))
        state = cos.State(collection_of_states=initial_collection_of_states)
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
        initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit1,input_collection_as_a_dict=dict([]))
        state = cos.State(collection_of_states=initial_collection_of_states)
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
    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
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

    # We map the optical channels on the classical channels such that the result in the classical channels in the order Th,Tv,D1h,D1,D2,D2,D3h,D3v
    detection_1.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],classical_channels_to_be_written=[0,1,2,3])
    detection_2.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],classical_channels_to_be_written=[0,1,2,3])
    detection_3.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],classical_channels_to_be_written=[0,1,2,3])
    detection_4.measure_optical_to_classical(optical_channels_to_be_measured=[0,2,4,6],classical_channels_to_be_written=[0,1,2,3])

    initial_collection_of_states_curve = cos.CollectionOfStates(fock_state_circuit=preparation, input_collection_as_a_dict=dict([]))
    delays = [n/4.0 for n in range(-20,21)]
    for n in delays:
        state1 = cos.State(collection_of_states=initial_collection_of_states_curve)
        state1.initial_state = 'delay' + str(n)
        state1.optical_components = [('0000', 1)]
        state1.classical_channel_values = [n]
        initial_collection_of_states_curve.add_state(state1)

    circuit_list = [preparation, circuit, detection_1]
    compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)
    result = compound_circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states_curve)

    histo = result.plot(histo_output_instead_of_plot=True)

    probs = []
    for key,value in histo.items():
        for xx in value:
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
    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
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