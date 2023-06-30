import sys  
sys.path.append("../src")
import fock_state_circuit as fsc
import collection_of_states as cos
import numpy as np
import math

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

def test_EPR_Bell_inequality():

    circuit = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 4,
                                    no_of_classical_channels=6)

    # orient polatizers before detectors
    circuit.wave_plate_classical_control(optical_channel_horizontal=0,
                                        optical_channel_vertical=1,
                                        classical_channel_for_orientation=0,
                                        classical_channel_for_phase_shift=3)
    circuit.wave_plate_classical_control(optical_channel_horizontal=2,
                                        optical_channel_vertical=3,
                                        classical_channel_for_orientation=1,
                                        classical_channel_for_phase_shift=3)

    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,2],classical_channels_to_be_written=[4,5])
    #circuit.draw()

    # define the angles we need for the half-wave plates in order to rotate polarization over the correct angle
    polarization_settings = {'ab' : {'left': 0  , 'right' : math.pi/16},
                            'a\'b': {'left': 0 , 'right' : 3*math.pi/16}, 
                            'ab\'' : {'left': math.pi/8 , 'right' : math.pi/16}, 
                            'a\'b\'' : {'left': math.pi/8 , 'right' : 3*math.pi/16 }}




        # First create an entangles state where both photons are either both 'H' polarized or both 'V' polarized
    # The optical state is 1/sqrt(2)  ( |HH> + |VV> )
    initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
    entangled_state = initial_collection_of_states.get_state(initial_state='0000').copy()
    initial_collection_of_states.clear()
    entangled_state.initial_state = 'entangled_state'
    entangled_state.optical_components = {'1010' : {'amplitude': math.sqrt(1/2), 'probability': 0.5}, 
                                                '0101' : {'amplitude': math.sqrt(1/2), 'probability': 0.5}}


    S_ent = []
    E = []

    for key, setting in polarization_settings.items():
        initial_collection_of_states.clear()
        state = entangled_state.copy()
        state.classical_channel_values = [setting['left'],setting['right'], 0, math.pi,0,0]
        state.initial_state = 'entangled_state_' + key
        state.cumulative_probability = 1
        initial_collection_of_states.add_state(state)
        result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
        result11 = result.copy()
        result11.filter_on_classical_channel(classical_channel_numbers=[4,5], values_to_filter=[1,1])
        probability11 = sum([state.cumulative_probability for state in result11])
        result00 = result.copy()
        result00.filter_on_classical_channel(classical_channel_numbers=[4,5], values_to_filter=[0,0])
        probability00 = sum([state.cumulative_probability for state in result00])
        E.append(2*(probability00+probability11)-1)
        
    S_ent = E[0] - E[1] + E[2] + E[3]

    assert round(S_ent, 5) == round(2*math.sqrt(2),5)

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

