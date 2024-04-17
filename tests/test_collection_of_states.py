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

def test_generate_collection_of_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6 
                                    )
    collection_of_states = fsc.CollectionOfStates(circuit)
    text = str(collection_of_states)
    reference_text = "Printing collection of states\nNumber of states in collection: 81"

    assert text[:len(reference_text)] == reference_text

def test_filter_and_copy_collection_of_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6 
                                    )
    collection_of_states = fsc.CollectionOfStates(circuit)

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
    collection_of_states = fsc.CollectionOfStates(circuit)
    
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
    collection_of_states = fsc.CollectionOfStates(circuit)
    
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
    collection_of_states = fsc.CollectionOfStates(circuit)

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
    collection_of_states = fsc.CollectionOfStates(circuit)

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
    collection_of_states = fsc.CollectionOfStates(circuit)

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
    collection_of_states = fsc.CollectionOfStates(circuit)
    state1 = collection_of_states.get_state(identifier='identifier_00001')
    name1 = state1.initial_state
    circuit2 = fsc.FockStateCircuit(length_of_fock_state = 11, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=6, 
                                channel_0_left_in_state_name=True
                                    )
    collection_of_states = fsc.CollectionOfStates(circuit2)
    state2 = collection_of_states.get_state(identifier='identifier_00001')
    name2 = state2.initial_state

    assert name1 == '00000001' and name2 == '01000000'

def test_various_methods_for_collection_of_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 2, 
                                    no_of_classical_channels=3, 
                                    )
    collection_of_states = fsc.CollectionOfStates(circuit)
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

    new_collection_of_states = fsc.CollectionOfStates(circuit, input_collection_as_a_dict = filtered_collection_as_dict)

    if new_collection_of_states.initial_states_as_list() != ['10', '01']:
        errors.append('error12')
    if new_collection_of_states.state_identifiers_as_list() != ['identifier_1', 'identifier_3']:
        errors.append('error13')

    state = fsc.State(collection_of_states)

    new_collection_of_states.add_state(identifier = 'new', state = state)
    new_collection_of_states.delete_state(identifier='identifier_1')
    if new_collection_of_states.state_identifiers_as_list() != ['identifier_3', 'new']:
        errors.append('error14')
    assert len(errors) == 0

def test_input_collection_unchanged_by_circuit():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                no_of_optical_channels = 8,
                                no_of_classical_channels=10
                                )
    circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1, angle = -math.pi/8)
    circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5, angle = math.pi/8)
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[6,7,0,1,4,5],classical_channels_to_be_written=[2,3,4,5,6,7])
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
    initial_collection_of_states.filter_on_initial_state(['210000', '220010', '210010', '220000'])
    no_failure_found = no_failure_found and len(new_collection) == 4 
    new_collection = initial_collection_of_states.copy()
    new_collection.reduce(optical_channels_to_keep=[2,3,4,5])
    no_failure_found = no_failure_found and len(new_collection) == 4
    new_collection.clean_up()
    no_failure_found = no_failure_found and len(new_collection) == 4

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

def test_adjust_length_of_fock_state():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 6, 
                                    no_of_classical_channels=4,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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
        state._threshold_probability_for_setting_to_zero == 0.1,
        state._print_only_last_measurement == False,
        True in state._check_valid_state(state_to_check = state),
        state.measurement_results[0]['measurement_results'] == [1,1,1]])

    assert no_failure_found

def test_auxiliary_information():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 5, 
                                    no_of_classical_channels=1,
                                    channel_0_left_in_state_name = False,
                                    threshold_probability_for_setting_to_zero=0.1,
                                    use_full_fock_matrix=True,
                                    )
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
    coll_of_cols= fsc.CollectionOfStateColumns(collection_of_states=initial_collection_of_states)
    for column in coll_of_cols.by_column():
        column.set_photon_information(photon_information = {0:"this is a test"})
    initial_collection_of_states = coll_of_cols.generate_collection_of_states()
    initial_collection_of_states.filter_on_initial_state(['22222', '20202'])
    state1 = initial_collection_of_states.get_state(initial_state='20202')
    state1.measurement_results = [{'measurement_results': [1,1,1], 'probability':1}]
    initial_collection_of_states.adjust_length_of_fock_state(new_length_of_fock_state=5)

    initial_collection_of_states.reduce(optical_channels_to_keep=[0,1])
    initial_collection_of_states.extend(extra_classical_channels=4)
    initial_collection_of_states.adjust_length_of_fock_state(new_length_of_fock_state=12)
    state = initial_collection_of_states.get_state(initial_state='22222')
    no_failure_found = no_failure_found and (state._digits_per_optical_channel == 2)
    initial_collection_of_states.extend(extra_optical_channels=1)
    initial_collection_of_states.reduce(classical_channels_to_keep=[0,3,4])
    initial_collection_of_states.adjust_length_of_fock_state(new_length_of_fock_state=3)
    state = initial_collection_of_states.get_state(initial_state='20202')
    for state in initial_collection_of_states:
        no_failure_found &= state.auxiliary_information['photon_resolution']['photon_information'][0] == "this is a test"
    assert no_failure_found