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
import itertools

def test_column_of_states_default_values():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.optical_components = [('110',1)]
    new_column = fsc.ColumnOfStates(state = state1)
    reference = new_column._DEFAULT_COLUMN_VALUES.copy()

    no_failure_found = True
    no_failure_found &= new_column.column_amplitude == reference['column_amplitude']
    no_failure_found &= new_column.column_boson_factor == reference['column_boson_factor']
    no_failure_found &= new_column.group_cumulative_probability == reference['group_cumulative_probability']
    no_failure_found &= new_column.column_identifier == reference['column_identifier']
    no_failure_found &= new_column.interference_group_identifier == reference['interference_group_identifier']
    assert no_failure_found

def test_column_of_states_set_values():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.optical_components = [('110',1)]
    new_column = fsc.ColumnOfStates(state = state1)
    
    new_column.column_amplitude = 0.5 + 1j * 0.01
    new_column.column_boson_factor = np.sqrt(2)
    new_column.group_cumulative_probability = 0.1
    new_column.interference_group_identifier = 2
    new_column.column_identifier = 4

    no_failure_found = True
    no_failure_found &= new_column.column_amplitude == (0.5 + 1j * 0.01)
    no_failure_found &= new_column.column_boson_factor == np.sqrt(2)
    no_failure_found &= new_column.group_cumulative_probability == 0.1
    no_failure_found &= new_column.column_identifier == 4
    no_failure_found &= new_column.interference_group_identifier == 2
    assert no_failure_found

def test_create_column_from_list():
    # create column in various ways
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    # 1. Create from list that does not contain any column information yet
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['000','212','300','111']
    state_list = []
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        state_list.append(state1)
    column = fsc.ColumnOfStates(list_of_states=state_list)
    correct = True
    correct &= column.column_amplitude == 1 
    correct &= column.column_boson_factor == 1 
    correct &= column.group_cumulative_probability == 1 
    correct &= column.column_identifier == 0 
    correct &= column.interference_group_identifier == 0

    # 2. Create from list that does contain some column information already
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['000','212','300','111']
    state_list = []
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        state1.auxiliary_information= {'photon_resolution':{'column_boson_factor': 3}}
        state_list.append(state1)
    column = fsc.ColumnOfStates(list_of_states=state_list)
    correct &= column.column_amplitude == 1 
    correct &= column.column_boson_factor == 3
    correct &= column.group_cumulative_probability == 1 
    correct &= column.column_identifier == 0 
    correct &= column.interference_group_identifier == 0

    # 3. Create from list and pass information as argument
    info = {'column_boson_factor': 5, 'column_amplitude' : 1/2}
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['000','212','300','111']
    state_list = []
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        state1.auxiliary_information= {'photon_resolution':{'column_boson_factor': 3}}
        state_list.append(state1)
    column = fsc.ColumnOfStates(list_of_states=state_list, column_information=info)
    correct &= column.column_amplitude == 1/2 
    correct &= column.column_boson_factor == 5
    correct &= column.group_cumulative_probability == 1 
    correct &= column.column_identifier == 0 
    correct &= column.interference_group_identifier == 0
    assert correct 

def test_create_column_from_single_states():
    # create column in various ways
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    # 1. Create from single state
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    ocs = ['000','212','300','111']
    state_list = []
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        state_list.append(state1)
    ori_state = state_list[1]
    column = fsc.ColumnOfStates(state=ori_state)
    correct = True
    correct &= column.column_amplitude == 1 
    correct &= column.column_boson_factor == 1 
    correct &= column.group_cumulative_probability == 1 
    correct &= column.column_identifier == 0 
    correct &= column.interference_group_identifier == 0
    print(correct)
    # 2. Create from list that does contain some column information already
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['000','212','300','111']
    state_list = []
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        state1.auxiliary_information= {'photon_resolution':{'column_boson_factor': 3}}
        state_list.append(state1)
    ori_state = state_list[1]
    column = fsc.ColumnOfStates(state=ori_state)
    correct &= column.column_amplitude == 1 
    correct &= column.column_boson_factor == 3
    correct &= column.group_cumulative_probability == 1 
    correct &= column.column_identifier == 0 
    correct &= column.interference_group_identifier == 0
    print(correct)
    # 3. Create from list and pass information as argument
    info = {'column_boson_factor': 5, 'column_amplitude' : 1/2}
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['000','212','300','111']
    state_list = []
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        state1.auxiliary_information= {'photon_resolution':{'column_boson_factor': 3}}
        state_list.append(state1)
    ori_state = state_list[1]
    column = fsc.ColumnOfStates(state=ori_state, column_information=info)
    correct &= column.column_amplitude == 1/2 
    correct &= column.column_boson_factor == 5
    correct &= column.group_cumulative_probability == 1 
    correct &= column.column_identifier == 0 
    correct &= column.interference_group_identifier == 0
    assert correct

def test_create_empty_column():
    # empty column
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    column = fsc.ColumnOfStates()
    correct = True
    correct &= column.column_amplitude == 1 
    correct &= column.column_boson_factor == 1 
    correct &= column.group_cumulative_probability == 1 
    correct &= column.column_identifier == 0 
    correct &= column.interference_group_identifier == 0

    info = {'column_boson_factor': 5, 'column_amplitude' : 1/2}
    column = fsc.ColumnOfStates(column_information=info)
    correct = True
    correct &= column.column_amplitude == 1/2
    correct &= column.column_boson_factor == 5 
    correct &= column.group_cumulative_probability == 1 
    correct &= column.column_identifier == 0 
    correct &= column.interference_group_identifier == 0
    assert correct

def test_column_of_states_set_values_for_state():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.optical_components = [('110',1)]
    new_column = fsc.ColumnOfStates(state = state1)
    
    new_column.column_amplitude = 0.5 + 1j * 0.01
    new_column.column_boson_factor = np.sqrt(2)
    new_column.group_cumulative_probability = 0.1
    new_column.interference_group_identifier = 2
    new_column.column_identifier = 4

    no_failure_found = True
    no_failure_found &= state1.auxiliary_information['photon_resolution']['column_amplitude'] == (0.5 + 1j * 0.01)
    no_failure_found &= state1.auxiliary_information['photon_resolution']['column_boson_factor'] == np.sqrt(2)
    no_failure_found &= state1.auxiliary_information['photon_resolution']['group_cumulative_probability'] == 0.1
    no_failure_found &= state1.auxiliary_information['photon_resolution']['column_identifier'] == 4
    no_failure_found &= state1.auxiliary_information['photon_resolution']['interference_group_identifier'] == 2
    assert no_failure_found

def test_add_state_to_column():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.optical_components = [('110',1)]
    new_column = fsc.ColumnOfStates(state = state1)

    new_column.column_boson_factor = 5

    state2 = state1.copy()
    r = np.sqrt(1/2)
    state2.optical_components =  [('001',r), ('101',r)]
    new_column.add_state(state2)

    assert len(new_column.list_of_states) == 2 and 5 == new_column.list_of_states[-1].auxiliary_information['photon_resolution']['column_boson_factor']

def test_column_copy():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.optical_components = [('110',1)]
    new_column = fsc.ColumnOfStates(state = state1)

    new_column.column_boson_factor = 5

    state2 = state1.copy()
    r = np.sqrt(1/2)
    state2.optical_components =  [('001',r), ('101',r)]
    new_column.add_state(state2)

    copied_column = new_column.copy()
    new_column.column_boson_factor = 1

    correct = len(new_column.list_of_states) == 2 and 1 == new_column.list_of_states[-1].auxiliary_information['photon_resolution']['column_boson_factor']
    correct &=  len(new_column.list_of_states) == 2 and 5 == copied_column.list_of_states[-1].auxiliary_information['photon_resolution']['column_boson_factor']

    new_column.column_boson_factor = 10
    copied_column.column_boson_factor = 15

    correct &= len(new_column.list_of_states) == 2 and 10 == new_column.list_of_states[-1].auxiliary_information['photon_resolution']['column_boson_factor']
    correct &=  len(new_column.list_of_states) == 2 and 15 == copied_column.list_of_states[-1].auxiliary_information['photon_resolution']['column_boson_factor']

    assert correct


def test_column_split():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.optical_components = [('110',1)]
    old_column = fsc.ColumnOfStates(state = state1)

    old_column.column_boson_factor = 5

    new_column = old_column.split()
    correct = new_column is None

    state2 = state1.copy()
    r = np.sqrt(1/2)
    state2.optical_components =  [('001',r), ('101',-r)]
    old_column.add_state(state2)
    new_column = old_column.split()

    correct &= len(new_column.list_of_states) == 2
    correct &= len(old_column.list_of_states) == 2
    correct &= all([len(state.optical_components) == 1 for state in new_column.list_of_states])
    correct &= all([len(state.optical_components) == 1 for state in old_column.list_of_states])
    correct &= new_column.column_amplitude == np.sqrt(1/2)
    correct &= old_column.column_amplitude == -1*np.sqrt(1/2)
    correct &= old_column.column_identifier == 0
    correct &= new_column.column_identifier == -1
    correct &= all([state.auxiliary_information['photon_resolution']['column_amplitude']  == np.sqrt(1/2) for state in new_column.list_of_states])
    correct &= all([state.auxiliary_information['photon_resolution']['column_identifier']  == -1 for state in new_column.list_of_states])
    correct &= all([state.auxiliary_information['photon_resolution']['column_amplitude']  == -1*np.sqrt(1/2) for state in old_column.list_of_states])
    correct &= all([state.auxiliary_information['photon_resolution']['column_identifier']  == 0 for state in old_column.list_of_states])
    
    assert correct

def test_cumulative_probabilities():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.cumulative_probability = 0.5
    r = np.sqrt(1/2)
    state1.optical_components = [('001',r), ('101',-r)]
    old_column = fsc.ColumnOfStates(state = state1)
    old_column.group_cumulative_probability = 0.5

    new_column = old_column.split()

    correct = np.round(new_column.group_cumulative_probability,4) == 0.5
    correct &= np.round(old_column.group_cumulative_probability,4) == 0.5

    assert correct

def test_column_as_iterator():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.cumulative_probability = 1
    r = np.sqrt(1/2)
    state1.optical_components = [('001',r), ('101',-r)]
    column = fsc.ColumnOfStates()

    for number in [0,1,2,3,4,5,6,7,8,9,10]:
        state = state1.copy()
        state.initial_state = 'init_' + str(number)
        column.add_state(state)

    correct = True
    for index,state in enumerate(column):
        correct &= ('init_' + str(index) ) == state.initial_state

    assert correct

def test_check_on_single_component_per_state():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.optical_components = [('110',1)]
    old_column = fsc.ColumnOfStates(state = state1)

    old_column.column_boson_factor = 5

    new_column = old_column.split()
    correct = new_column is None

    state2 = state1.copy()
    r = np.sqrt(1/2)
    state2.optical_components =  [('001',r), ('101',-r)]
    old_column.add_state(state2)

    correct = old_column.all_states_single_optical_component() == False
    new_column = old_column.split()
    correct &= old_column.all_states_single_optical_component() 
    correct &= new_column.all_states_single_optical_component() 

    assert correct

def test_create_single_photon_states_from_single_state():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.cumulative_probability = 1
    r = np.sqrt(1/2)
    state1.optical_components = [('122',1)]
    column = fsc.ColumnOfStates(state= state1)
    column.single_photon_states()
    list_of_names = []
    for state in column:
        list_of_names.append(list(state.optical_components.keys())[0])
    correct = list_of_names.count('100') == 1
    correct &= list_of_names.count('010') == 2
    correct &= list_of_names.count('001') == 2
    correct &= list_of_names.count('122') == 0
    correct &= np.round(column.column_boson_factor,4) == 2
    assert correct

def test_create_single_photon_states_from_multiple_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.cumulative_probability = 1
    state1.optical_components = [('100',1)]
    column = fsc.ColumnOfStates(state= state1)

    state2 = state1.copy()
    state2.optical_components = [('020',1)]
    column.add_state(state2)

    state3 = state1.copy()
    state3.optical_components = [('002',1)]
    column.add_state(state3)


    column.single_photon_states()
    list_of_names = []
    for state in column:
        list_of_names.append(list(state.optical_components.keys())[0])
    correct = list_of_names.count('100') == 1
    correct &= list_of_names.count('010') == 2
    correct &= list_of_names.count('001') == 2
    correct &= list_of_names.count('122') == 0
    correct &= np.round(column.column_boson_factor,4) == 2
    correct &= column.all_states_single_photon_states()
    assert correct

def test_create_photon_list():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.cumulative_probability = 1
    state1.optical_components = [('100',1)]
    column = fsc.ColumnOfStates(state= state1)

    state2 = state1.copy()
    state2.optical_components = [('020',1)]
    column.add_state(state2)

    state3 = state1.copy()
    state3.optical_components = [('002',1)]
    column.add_state(state3)

    column.set_photon_information({'time_stamp':0, 'pulse_width':1})
    first_list = column.list_of_photons()
    column.single_photon_states()
    second_list = column.list_of_photons()

    state4 = fsc.State(collection_of_states=initial_collection_of_states)
    state4.initialize_this_state()
    state4.cumulative_probability = 1
    state4.optical_components = [('122',4)]
    column2 = fsc.ColumnOfStates(state= state4)
    third_list = column2.list_of_photons()

    correct = len(first_list) == 5 and len(second_list) == 5 and len(third_list) == 5

    channels1 = [photon[0] for photon in first_list]
    channels2 = [photon[0] for photon in second_list]
    channels3 = [photon[0] for photon in third_list]

    correct &= channels1.count(0) == 1 and channels2.count(0) == 1 and channels3.count(0) == 1 
    correct &= channels1.count(1) == 2 and channels2.count(1) == 2 and channels3.count(1) == 2 
    correct &= channels1.count(2) == 2 and channels2.count(2) == 2 and channels3.count(2) == 2 

    correct &= first_list[2][1]['time_stamp'] == 0
    correct &= second_list[2][1]['pulse_width'] == 1
    correct &= len(third_list[0][1])== 0
    assert correct
 
def test_column_of_states_set_photon_information():
    # update photon information selectively
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['000','212','300']

    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column0.add_state(state=state1)
    correct = (len(column0)==3)
    correct &= column0.column_boson_factor == 1
    # test 1: Check setting for all states
    column0.set_photon_information({0:'photon information has been set'})
    for state in column0:
        correct &= state.auxiliary_information['photon_resolution']['photon_information'][0] == 'photon information has been set'
    # test2: check setting based on filter and leave rest unchanged
    column0.set_photon_information(photon_information={0:'photon information has been changed'},
                                filter_for_optical_values=[(3,0,0)] )
    for state in column0.list_of_states[0:2]:
        correct &= state.auxiliary_information['photon_resolution']['photon_information'][0] == 'photon information has been set'
    correct &= column0.list_of_states[2].auxiliary_information['photon_resolution']['photon_information'][0] == 'photon information has been changed'
    # test3: check setting based on filter and give rest default value
    column0.set_photon_information(photon_information={0:'some photon information'},
                                default_photon_information={0:'other photon information'},
                                filter_for_optical_values=[[0,0,0],[3,0,0]] )
    correct &= column0.list_of_states[0].auxiliary_information['photon_resolution']['photon_information'][0] == 'some photon information'
    correct &= column0.list_of_states[1].auxiliary_information['photon_resolution']['photon_information'][0] == 'other photon information'
    correct &= column0.list_of_states[2].auxiliary_information['photon_resolution']['photon_information'][0] == 'some photon information'
    assert correct


def test_condense_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.cumulative_probability = 1
    r = np.sqrt(1/2)
    state1.optical_components = [('122',1)]
    column = fsc.ColumnOfStates(state= state1)
    column.single_photon_states()

    correct = len(column) == 5

    column.condense_column_to_single_state()
    correct &= len(column) == 1
    list_of_names = []
    for state in column:
        list_of_names.append(list(state.optical_components.keys())[0])
    correct &= list_of_names.count('100') == 0
    correct &= list_of_names.count('010') == 0
    correct &= list_of_names.count('001') == 0
    correct &= list_of_names.count('122') == 1
    correct &= np.round(column.column_boson_factor,4) == 1
    assert correct

def test_optical_component_from_statecolumn():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.cumulative_probability = 1

    state1.optical_components = [('122',1)]
    column = fsc.ColumnOfStates(state= state1)
    column.single_photon_states()

    oc = column.generate_optical_component_corresponding_to_column()

    assert oc == '122'


def test_column_of_states_complete():
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    state.optical_components = [("2321",np.sqrt(1/6)), ("2772",np.sqrt(1/6)), ("0110",np.sqrt(1/3)), ("3002",np.sqrt(1/3))]
    initial_collection_of_states.add_state(state)
    column0 = fsc.ColumnOfStates(state=state)
    correct=(column0.all_states_single_optical_component() == False)
    column1 = column0.split()
    correct&=(column0.all_states_single_optical_component() == False)
    column2 = column0.split()
    correct&=(column0.all_states_single_optical_component() == False)
    column3 = column0.split()


    # check the check
    correct&=(column0.all_states_single_photon_states() == False)
    column0.single_photon_states()
    correct&=(column0.all_states_single_photon_states() == True)
    column1.single_photon_states()
    correct&=(column1.all_states_single_photon_states() == True)
    column2.single_photon_states()
    correct&=(column2.all_states_single_photon_states() == True)
    column3.single_photon_states()
    # check if column amplitude is same as component amplitude in original state
    correct&=(compare(column0.column_amplitude**2,1/3))
    correct&=(compare(column1.column_amplitude**2,1/6))
    correct&=(compare(column2.column_amplitude**2,1/6))
    correct&=(compare(column3.column_amplitude**2,1/3))
    # check boson factor for expansion
    correct&=(compare(column0.column_boson_factor**2,3*2*2))
    correct&=(compare(column1.column_boson_factor**2,2*3*2*1*2))
    correct&=(compare(column2.column_boson_factor**2,(2*7*6*5*4*3*2)**2))
    correct&=(compare(column3.column_boson_factor**2,1))
    # check cumulative probabilities
    correct&=(compare(column0.group_cumulative_probability,1))
    correct&=(compare(column1.group_cumulative_probability,1))
    correct&=(compare(column2.group_cumulative_probability,1))
    correct&=(compare(column3.group_cumulative_probability,1))
    # check cumulative identifier
    correct&=(compare(column0.column_identifier,0)) # for original column id = 0
    correct&=(compare(column1.column_identifier,-1)) # split of column has id = -1
    correct&=(compare(column2.column_identifier,-1)) # split of column has id = -1
    correct&=(compare(column3.column_identifier,-1)) # split of column has id = -1
    # generation optical components from column
    correct&=(column0.generate_optical_component_corresponding_to_column() == '3002') 
    correct&=(column3.generate_optical_component_corresponding_to_column() == '0110') 
    # condense to single state
    column1.condense_column_to_single_state()
    correct&=(len(column1) == 1)
    correct&=(compare(column1.column_boson_factor**2,1))
    for state in column1:
        correct&=(list(state.optical_components.keys()) == ['2321'])
    # create a new state
    new_state = column2.generate_single_state_from_column(use_group_cumulative_probability_for_state=False)
    reference_state = state.copy()
    reference_state.optical_components = [('2772',1)]
    correct&=(compare(np.abs(new_state.inner(reference_state))**2,1))
    correct&=(compare(new_state.cumulative_probability,1))
    # check photon list and set photon info
    column2.set_photon_information(photon_information={0:"this is a test"})
    photon_list = column2.list_of_photons()
    channel_list = [photon[0] for photon in photon_list]
    correct&=(channel_list == [0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3])
    correct&=(all([photon[1]=="this is a test"] for photon in photon_list))
    assert correct

def test_boson_factor_examples():
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    """
        Example 1:
            - If the column consists of states with components '001', '100' and '100' the resulting state will have optical component
            '201'. The amplitude of this component will be 1 and the column_boson_factor will be sqrt(1/2)
    """
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['001','100','100']
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column0.add_state(state=state1)
    correct = (len(column0)==3)
    correct &= column0.column_boson_factor == 1
    column0.condense_column_to_single_state()
    correct &= (compare(column0.column_boson_factor, np.sqrt(1/2)))

    new_state = column0.generate_single_state_from_column()
    reference_state = state.copy()
    reference_state.optical_components = [('201',1)]
    correct &= compare(reference_state.inner(new_state),1)
    """
        Example 2: 
            - We start with state '22' with column_boson_factor 1
            - We expand into 4 single photon states '01','01','10','10' in a column with column_boson_factor 2
            - after condense_column_to_single_state the column_boson_factor is again 1
    """
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=2,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['22']
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column0.add_state(state=state1)
    correct &= (len(column0)==1)
    correct &= column0.column_boson_factor == 1
    column0.single_photon_states()
    correct &= (compare(column0.column_boson_factor, 2))

    column0.condense_column_to_single_state()
    correct &= (compare(column0.column_boson_factor, 1))
    new_state = column0.generate_single_state_from_column()
    reference_state = state.copy()
    reference_state.optical_components = [('22',1)]
    correct &= compare(reference_state.inner(new_state),1)

    """
        Example 3: 
            - We start with a column with states '20' and '02' with column_boson_factor 1
            - After condense_column_to_single_state the column_boson_factor is 1 and the state is '22'
    """
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=2,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['02','20']
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column0.add_state(state=state1)
    correct &= (len(column0)==2)
    correct &= column0.column_boson_factor == 1
    column0.condense_column_to_single_state()
    correct &= (compare(column0.column_boson_factor, 1))
    new_state = column0.generate_single_state_from_column()
    reference_state = state.copy()
    reference_state.optical_components = [('22',1)]
    correct &= compare(reference_state.inner(new_state),1)

    """
        Example 4: 
            - We start with a column with states '11' and '11' with column_boson_factor 1
            - After condense_column_to_single_state the column_boson_factor is 1/2 and the state is '22'
    """
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=2,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['11','11']
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column0.add_state(state=state1)
    correct &= (len(column0)==2)
    correct &= column0.column_boson_factor == 1
    column0.condense_column_to_single_state()
    correct &= (compare(column0.column_boson_factor, 1/2))
    new_state = column0.generate_single_state_from_column()
    reference_state = state.copy()
    reference_state.optical_components = [('22',1)]
    correct &= compare(reference_state.inner(new_state),1)
    assert correct

def test_edge_cases_vacuum_state_and_multi_photon_states_in_column():
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    """
    edge case 1: Vacuum state in the column. Should work 'normally'. When we generate a photon list the 
    vacuum state does not contribute (as it has not photons)
    edge case 2: Condense to single state from states that already have multiple photons
    """
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['000','212','300']

    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column0.add_state(state=state1)
    correct = (len(column0)==3)
    correct &= column0.column_boson_factor == 1
    correct &= column0.all_states_single_optical_component()
    # edge case 2, we condense from multi-photon states. 
    # in edge case 1 we condense with vacuum state in the colum
    column0.condense_column_to_single_state()
    # check if boson factor works in edge case 2
    correct &= (compare(column0.column_boson_factor, np.sqrt((2*2*3*2)/(5*4*3*2*2))))
    new_state = column0.generate_single_state_from_column()
    # and check if we generate the right state in edge case 2
    reference_state = state.copy()
    reference_state.optical_components = [('512',1)]
    correct &= compare(reference_state.inner(new_state),1)
    list_of_photons = column0.list_of_photons()
    correct &= len(list_of_photons) == 8
    column0.single_photon_states()
    correct &= len(column0) == 8

    # Edge case 1: if we do not condense but immediately split the state 
    # we should have 9 states in the column (also vacuum state should be there)
    column1= fsc.ColumnOfStates()
    ocs = ['000','212','300']
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column1.add_state(state=state1)
    correct = (len(column1)==3)
    correct &= column1.column_boson_factor == 1

    column1.single_photon_states()
    correct &= (compare(column1.column_boson_factor, np.sqrt(2*2*3*2)))
    correct &= len(column1) == 9 # vacuum state is also in the column
    # Edge case 1: vacuum state should not contribute to photon list
    list_of_photons = column1.list_of_photons()
    correct &= len(list_of_photons) == 8 # 8 photons in the column
    assert correct

def test_edge_case_multiple_optical_components_to_be_ignored():
    # column_edge_cases with multiple optical components
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    """
    edge case 1: Vacuum state in the column. Should work 'normally'. When we generate a photon list the 
    vacuum state does not contribute (as it has not photons)
    edge case 2: Condense to single state from states that already have multiple photons
    edge case 3: Multiple optical components, only take first
    """
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['000','212','300']
    ocs_to_be_ignored = ['656', '101', '000']
    r = np.sqrt(1/2)
    for oc, oc_ignore in zip(ocs,ocs_to_be_ignored):
        state1 = state.copy()
        state1.optical_components = [(oc,r), (oc_ignore,r)]
        column0.add_state(state=state1)
    correct = (len(column0)==3)
    correct &= column0.column_boson_factor == 1
    correct &= not column0.all_states_single_optical_component()
    # edge case 2, we condense from multi-photon states. 
    # in edge case 1 we condense with vacuum state in the colum
    column0.condense_column_to_single_state()
    # check if boson factor works in edge case 2
    correct &= (compare(column0.column_boson_factor, np.sqrt((2*2*3*2)/(5*4*3*2*2))))
    new_state = column0.generate_single_state_from_column()
    # and check if we generate the right state in edge case 2
    reference_state = state.copy()
    reference_state.optical_components = [('512',1)]
    correct &= compare(reference_state.inner(new_state),1)
    list_of_photons = column0.list_of_photons()
    correct &= len(list_of_photons) == 8
    column0.single_photon_states()
    correct &= len(column0) == 8

    # Edge case 1: if we do not condense but immediately split the state 
    # we should have 9 states in the column (also vacuum state should be there)
    column1= fsc.ColumnOfStates()
    ocs = ['000','212','300']
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column1.add_state(state=state1)
    correct = (len(column1)==3)
    correct &= column1.column_boson_factor == 1

    column1.single_photon_states()
    correct &= (compare(column1.column_boson_factor, np.sqrt(2*2*3*2)))
    correct &= len(column1) == 9 # vacuum state is also in the column
    # Edge case 1: vacuum state should not contribute to photon list
    list_of_photons = column1.list_of_photons()
    correct &= len(list_of_photons) == 8 # 8 photons in the column
    assert correct

def test_create_interference_group():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.optical_components = [('110',1)]
    interference_group = fsc.InterferenceGroup(state = state1)
    correct = len(interference_group) == 1
    for column in interference_group:
        correct &= column.interference_group_identifier == interference_group.interference_group_identifier
        correct &= column.column_identifier == 0
    
    assert correct

    
def test_split_interference_group():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('001',r)]
    interference_group = fsc.InterferenceGroup(state = state1)
    correct = len(interference_group) == 1
    interference_group.split_columns()
    correct &= len(interference_group) == 2
    for column in interference_group:
        correct &= column.all_states_single_optical_component()
    
    assert correct
 


def test_create_single_photon_states():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',r)]
    interference_group = fsc.InterferenceGroup(state = state1)
    correct = len(interference_group) == 1
    interference_group.split_columns()
    correct &= len(interference_group) == 2
    for column in interference_group:
        correct &= column.all_states_single_optical_component()
        correct &= len(column) == 1
    interference_group.single_photon_states()
    correct &= len(interference_group) == 2
    for column in interference_group:
        correct &= column.all_states_single_optical_component()
        correct &= len(column) == 2
    
    assert correct

def test_interference_group_identifier():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',r)]
    interference_group = fsc.InterferenceGroup(state = state1)
    correct = len(interference_group) == 1
    interference_group.split_columns()
    correct &= len(interference_group) == 2
    for column in interference_group:
        correct &= column.all_states_single_optical_component()
        correct &= len(column) == 1
    interference_group.single_photon_states()
    correct &= len(interference_group) == 2
    for column in interference_group:
        correct &= column.all_states_single_optical_component()
        correct &= len(column) == 2

    interference_group.interference_group_identifier = 4
    for column in interference_group:
        correct &= column.interference_group_identifier == 4
        for state in column:
            correct &= state.auxiliary_information['photon_resolution']['interference_group_identifier'] == 4

    assert correct

def test_regenerate_state_from_column():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    state1.optical_components = [('210',1)]
    correct = state1.inner(state1) == 1

    column = fsc.ColumnOfStates(state = state1.copy())
    column.single_photon_states

    new_state = column.generate_single_state_from_column()

    correct &= state1.inner(new_state) == 1
    assert correct
    correct &= new_state.auxiliary_information.get('photon_resolution',None) is None

    assert correct

def test_various_ways_to_create_group():
    # create group in various ways
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    # 1. Create from single state
    """
    1) Provide a state and do not provide a 'list_of_state_columns' (or provide empty list)
        - We create a new group containing just this 'state'
        - If the 'state' has an 'interference_group_identifier' that is used as identifier for the group,
            otherwise we choose 0 as identifier
        - If interference_group_identifier is passed as argument to this constructor then that value is used
            as identifier for the group, overriding the information contained in the 'state'
        - For 'group_cumulative_probability' we use the value in 'state.cumulative_probability' for the given state.
    """
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initial_state = 'init_1'
    state1.optical_components = [('210',1/2),('000',1/2),('111',1/2),('210',1/2)]
    state1.cumulative_probability = 0.7
    group = fsc.InterferenceGroup(state=state1)
    correct = group.interference_group_identifier == 0
    correct &= group.group_cumulative_probability == 0.7
    state1.auxiliary_information['photon_resolution']['interference_group_identifier'] = 5
    group2 = fsc.InterferenceGroup(state=state1)
    correct &= group2.interference_group_identifier == 5
    group3 = fsc.InterferenceGroup(state=state1, interference_group_identifier=2)
    correct &= group3.interference_group_identifier == 2
    print(correct)
    """
    2) Provide a non-empty 'list_of_state_columns' and no 'state'
        - We create a new group containing the columns in the list
        - If the first 'column' has an 'interference_group_identifier' that is used as identifier for the group,
            otherwise we choose 0 as identifier.
        - If interference_group_identifier is passed as argument to this constructor then that value is used
            as identifier for the group, overriding the information contained in the 'list_of_columns'
        - The 'group_cumulative_probability' is set to the FIRST COLUMN IN 'column.group_cumulative_probability' 
        
    """
    circuit = fsc.FockStateCircuit(length_of_fock_state=8,no_of_optical_channels=3,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states= initial_collection_of_states)
    column0 = fsc.ColumnOfStates()
    ocs = ['000','212','300']
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column0.add_state(state=state1)
        column0.group_cumulative_probability = 0.5
    column1 = fsc.ColumnOfStates()
    ocs = ['111','212','343']
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column1.add_state(state=state1)
    ocs = ['341','412','222']
    column2 = fsc.ColumnOfStates()
    for oc in ocs:
        state1 = state.copy()
        state1.optical_components = [(oc,1)]
        column2.add_state(state=state1)
    list_of_columns = [column0, column1, column2]
    group = fsc.InterferenceGroup(list_of_state_columns=list_of_columns)
    correct &=(group.interference_group_identifier == 0)
    correct &=(group.group_cumulative_probability == 0.5)
    group = fsc.InterferenceGroup(list_of_state_columns=list_of_columns, interference_group_identifier=2)
    correct &=(group.interference_group_identifier == 2)
    correct &=(group.group_cumulative_probability == 0.5)
    for column in group.by_column():
        correct &= column.interference_group_identifier == 2
    """
                3) If an empty 'list_of_columns' is provided as well as no 'state' the constructor creates an empty group.
                    - An empty group is created, containing no columns
                    - The interference_group_identifier is zero, or equal to the value for interference_group_identifier which
                        is passed as argument to this constructor.
                    - The 'group_cumulative_probability' is set 1
    """
    group = fsc.InterferenceGroup()
    correct &=(group.interference_group_identifier == 0)
    correct &=(group.group_cumulative_probability == 1) 
    group = fsc.InterferenceGroup(interference_group_identifier=8)
    correct &=(group.interference_group_identifier == 8)
    assert correct

def test_regenerate_state_from_interference_group_1():
    # create group from
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initial_state = 'init_1'
    state1.optical_components = [('210',np.sqrt(1/8)),('000',1/2),('111',1/2),('210',np.sqrt(1/8))]
    state1.cumulative_probability = 0.7
    original_state = state1.copy()
    group = fsc.InterferenceGroup(state=state1)
    group.split_columns()
    group.single_photon_states()
    # Rule 1: The amplitude-square for columns in the group add up to 1
    s = 0
    for column in group.by_column():
        s += np.abs(column.column_amplitude)**2
    correct = compare(s,1)
    # Rule 2: The group_cum_prob for group and all columns are same as state.cumulative_probability for the original state
    for column in group.by_column():
        correct &= column.group_cumulative_probability == 0.7
    # Rule 3. Length is no of photons + vac state if orginally present
    correct &= len(group) == 3
    # Rule 4: We can retreive the original state
    new_state = group.generate_single_state_from_interference_group()
    correct &= (compare(np.abs(new_state.inner(original_state))**2,1))
    correct &= (compare(new_state.cumulative_probability, 0.7))
    assert correct


def test_regenerate_state_from_interference_group_2():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',1/2), ('101',1/2)]
    correct = state1.inner(state1) == 1

    interference_group = fsc.InterferenceGroup(state = state1.copy())
    interference_group.split_columns()
    interference_group.single_photon_states()

    new_state = interference_group.generate_single_state_from_interference_group()

    correct &= state1.inner(new_state) == 1

    assert correct

def test_regenerate_state_from_interference_group_2():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',1/2), ('101',1/2)]
    original_state = state1.copy()
    interference_group = fsc.InterferenceGroup(state = state1)
    
    new_state = interference_group.generate_single_state_from_interference_group()

    assert np.round(np.abs(new_state.inner(original_state))**2,4) == 1

def test_create_empty_group_and_add_column():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',1/2), ('101',1/2)]
    correct = state1.inner(state1) == 1

    interference_group = fsc.InterferenceGroup(state = state1.copy())
    interference_group.split_columns()
    interference_group.single_photon_states()

    new_group = fsc.InterferenceGroup()
    for column in interference_group:
        new_group.add_column(column)
    new_state = new_group.generate_single_state_from_interference_group()

    correct &= state1.inner(new_state) == 1


    assert correct

def test_create_empty_group_and_add_state():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',1/2), ('101',1/2)]
    correct = state1.inner(state1) == 1

    interference_group = fsc.InterferenceGroup(state = state1.copy())
    interference_group.split_columns()
    interference_group.single_photon_states()

    new_group = fsc.InterferenceGroup()
    new_group.add_state(state1)

    for column in new_group:
        correct &= column.column_identifier ==0
        correct &= column.interference_group_identifier==0
        for state in column:
            correct &= np.round(np.abs(state.inner(state1))**2,4) == 1


    assert correct

def test_create_group_from_column():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',1/2), ('101',1/2)]
    correct = state1.inner(state1) == 1

    interference_group = fsc.InterferenceGroup(state = state1.copy())
    interference_group.split_columns()
    interference_group.single_photon_states()

    list_of_columns = []
    for column in interference_group:
        list_of_columns.append(column)
    another_new_group = fsc.InterferenceGroup(list_of_state_columns=list_of_columns)

    another_new_state = another_new_group.generate_single_state_from_interference_group()

    correct &= state1.inner(another_new_state) == 1

    assert correct


def test_add_state_without_photon_resolution_information():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',1/2), ('101',1/2)]

    new_state = state1.copy()

    correct = new_state.inner(state1) == 1

    interference_group = fsc.InterferenceGroup(state = state1.copy())
    interference_group.split_columns()
    interference_group.single_photon_states()
    correct &= len(interference_group) == 3
    for column in interference_group:
        correct &= len(column) == 2

    # add new state without any photon_resolution information to the interference group in new column
    interference_group.add_state(state = new_state.copy())
    correct &= len(interference_group) == 4

    # add new state without any photon_resolution information to the interference group in existing column
    interference_group.add_state(state = new_state.copy(), column_identifier=0)
    correct &= len(interference_group) == 4
    for column in interference_group:
        if column.column_identifier == 0:
            correct &= len(column) == 3
        elif column.column_identifier == 3:
            correct &= len(column) == 1
        else:
            correct &= len(column) == 2
    assert correct

def test_add_state_with_photon_resolution_information():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',1/2), ('101',1/2)]

    new_state = state1.copy()
    new_state.auxiliary_information['photon_resolution'] = {
                            'column_amplitude' : 1, 
                            'column_boson_factor' : 1, 
                            'group_cumulative_probability' : 1, 
                            'column_identifier' :0, 
                            'interference_group_identifier' : 0
                            }

    correct = new_state.inner(state1) == 1

    interference_group = fsc.InterferenceGroup(state = state1.copy())
    interference_group.split_columns()
    interference_group.single_photon_states()
    correct &= len(interference_group) == 3
    for column in interference_group:
        correct &= len(column) == 2

    # add new state with photon_resolution information to the interference group in existing column
    interference_group.add_state(state = new_state.copy())
    correct &= len(interference_group) == 3
    for column in interference_group:
        if column.column_identifier == 0:
            correct &= len(column) == 3
        else:
            correct &= len(column) == 2

    # add new state with photon_resolution information to the interference group new existing column
    new_state.auxiliary_information['photon_resolution']['column_identifier'] = 4
    interference_group.add_state(state = new_state.copy())
    correct &= len(interference_group) == 4
    for column in interference_group:
        if column.column_identifier == 0:
            correct &= len(column) == 3
            pass
        elif column.column_identifier == 4:
            correct &= len(column) == 1
        else:
            correct &= len(column) == 2
    assert correct

def test_collection_of_state_columns():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initialize_this_state()
    state1.initial_state = 'init_1'
    r = np.sqrt(1/2)
    state1.optical_components = [('110',r),('011',1/2), ('101',1/2)]
    initial_collection_of_states.add_state(state1.copy())

    collection_by_column = fsc.CollectionOfStateColumns(collection_of_states=initial_collection_of_states)

    column = collection_by_column.as_dictionary()['init_1'][0][0]
    correct = np.round(np.abs(state1.inner(column.list_of_states[0]))**2,4) == 1

    collection_by_column.split()
    collection_by_column.single_photon_states()

    for state in collection_by_column.by_state():
        correct &= int(state) == 1

    new_collection_of_states = collection_by_column.generate_collection_of_states()

    for state in new_collection_of_states:
        correct &= int(state) == 1

    new_collection_by_column = fsc.CollectionOfStateColumns(collection_of_states=new_collection_of_states)

    for interference_group in new_collection_by_column.by_group():
        interference_group.generate_single_state_from_interference_group()

    another_collection_of_states = collection_by_column.generate_collection_of_states()

    correct &= len(another_collection_of_states) == 6

    assert correct

def test_collection_of_state_columns_2():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    group_ids = [9,9,9,10,10,11,11,11,11]
    column_ids = [0,0,0,0,1,4,4,5,5]
    for group_id,column_id in zip(group_ids,column_ids):
        state1 = fsc.State(collection_of_states=initial_collection_of_states)
        state1.initialize_this_state()
        state1.initial_state = 'init_1'
        r = np.sqrt(1/2)
        state1.optical_components = [('110',r),('011',1/2), ('101',1/2)]
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.1, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :column_id, 
                                                        'interference_group_identifier' : group_id
                                                        }
        initial_collection_of_states.add_state(state1.copy())
    correct = True
    collection_by_column = fsc.CollectionOfStateColumns(collection_of_states=initial_collection_of_states)
    groups_for_init = collection_by_column.collection_by_column['init_1']
    correct &= len(groups_for_init) == 3
    correct &= 9 in groups_for_init.keys()
    correct &= 10 in groups_for_init.keys()
    correct &= 11 in groups_for_init.keys()
    for group in collection_by_column.by_group():
        if group.interference_group_identifier == 9:
            correct &= len(group) == 1
            for column in group.by_column():
                correct &= column.interference_group_identifier == 9

        if group.interference_group_identifier == 10:
            correct &= len(group) == 2
            for column in group.by_column():
                correct &= column.interference_group_identifier == 10
        if group.interference_group_identifier == 11:
            correct &= len(group) == 2
            for column in group.by_column():
                correct &= column.interference_group_identifier == 11
    assert correct

def test_projection():
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)
    correct = True
    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    group = fsc.InterferenceGroup(state=state)
    photon_list, boson = group._measurement_projection(optical_component='1111')
    correct &= photon_list == [0, 1, 2, 3]
    correct &= compare(boson, 1)

    photon_list, boson = group._measurement_projection(optical_component='2222')
    correct &= photon_list == [0, 0, 1, 1, 2, 2, 3, 3]
    correct &= compare(boson, 4)

    photon_list, boson = group._measurement_projection(optical_component='2002')
    correct &= photon_list == [0, 0, 3, 3]
    correct &= compare(boson, 2)

    photon_list, boson = group._measurement_projection(optical_component='3000') 
    correct &= photon_list == [0, 0, 0]
    correct &= compare(boson, np.sqrt(6))

    photon_list, boson = group._measurement_projection(optical_component='3003')
    correct &= photon_list == [0, 0, 0, 3, 3, 3]
    correct &= compare(boson, 6)

    photon_list, boson = group._measurement_projection(target_values_and_channels= {'channels': [0,1,2], 'values': [2,2,2]}) 
    correct &= photon_list == [0, 0, 1, 1, 2, 2]
    correct &= compare(boson, np.sqrt(8))
    assert correct

def test_valid_sequences():
    """
            column id:12
            column amplitude:(0.24999998716429697-2.6264515646646206e-18j)
            column boson factor:1.4142135623730951
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'0010': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
            -----
            column id:13
            column amplitude:(0.24999998716429697-2.626451564664622e-18j)
            column boson factor:1.4142135623730951
                {'0010': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
    """
    """ 
            Photon list column 12: [0,2,1]. The numbers are the channels for the photons
            Photon list column 13: [2,0,1]
            Projection list on '1110': [0,1,2]
            For column 12 the valid order of the photons is (0,2,1) 
             ==> (on projection index 0 we map photon 0, in projection index 1 we map photon 2 and on projection index 2 we map photon 1)
            For column 13 the valid order of the photons is (1,2,0)
             ==> (on projection index 0 we map photon 1, in projection index 1 we map photon 2 and on projection index 2 we map photon 0)
            Sequence should be ((0,1),(1,0),(2,2)) 
            (This means:    Photon 0 column 12 with photon 1 column 13, 
                            Photon 1 column 12 with photon 0 column 13,
                            Photon 2 column 12 with photon 2 column 13)
    """

    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','0010','0100','0010','1000','0100']
    column_identifier = [12,12,12,13,13,13]
    column_delay = [5,5,0,5,5,0]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.25, 
                                                        'column_boson_factor' : np.sqrt(2), 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)


    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    
    group = coll_of_col.collection_by_column[initial_state][interference_group]

 
    correct = True
    measurement_projection, projection_boson_factor = group._measurement_projection(optical_component ='1110')
    correct &= (measurement_projection == [0,1,2])
    correct &= (np.round(projection_boson_factor,4) == 1)

    column_12 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(12)
    column_13 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(13)

    valid_1 = group._valid_combinations(measurement_projection,column_12)
    valid_2 = group._valid_combinations(measurement_projection,column_13)
    correct &= (valid_1 == [(0,2,1)])
    correct &= (valid_2 == [(1,2,0)])
    valid_bra_ket_pairs = group._find_valid_photon_pairs_for_projection(measurement_projection, column_12, column_13)
    sequence = valid_bra_ket_pairs[0]
    correct &= (len(sequence) == 3)
    correct &= ((0,1) in sequence and (1,0) in sequence  and (2,2) in sequence )
    assert correct

def test_valid_sequences_2():
    """
            column id:12
            column amplitude:(0.1)
            column boson factor:1
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
            -----
            column id:13
            column amplitude:(0.1)
            column boson factor:1
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
    """
    """ 
            Photon list column 12: [0,0,1]. The numbers are the channels for the photons
            Photon list column 13: [1,0,0]
            Projection list on '2100': [0,0,1]
            For column 12 the valid orders of the photons is (0,1,2) or (1,0,2)
            For column 13 the valid order of the photons is (2,1,0) or (1,2,0)
            Sequences are: 
            ((0,1),(1,2),(2,0)) 
            ((0,2),(1,1),(2,0)) 

    """


    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','1000','0100','0100','1000','1000']
    column_identifier = [12,12,12,13,13,13]
    column_delay = [5,5,0,5,5,0]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.1, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)


    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    group = coll_of_col.collection_by_column[initial_state][interference_group]

    correct = True
    measurement_projection, projection_boson_factor = group._measurement_projection(optical_component ='2100')
    correct &= (measurement_projection == [0,0,1])
    correct &= (np.round(projection_boson_factor,4) == np.round(np.sqrt(2),4))

    column_12 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(12)
    column_13 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(13)

    valid_1 = group._valid_combinations(measurement_projection,column_12)
    valid_2 = group._valid_combinations(measurement_projection,column_13)
    correct &= (valid_1 == [(0,1,2),(1,0,2)]) or (valid_1 == [(1,0,2),(0,1,2)])
    correct &= (valid_2 == [(1,2,0) ,(2,1,0)]) or (valid_2 == [(2,1,0),(1,2,0)])

    valid_bra_ket_pairs = group._find_valid_photon_pairs_for_projection(measurement_projection, column_12, column_13)
    sequence_1 = ((0, 1), (1, 2), (2, 0))
    sequence_2 = ((0, 2), (1, 1), (2, 0))
    correct &= (sequence_1 in valid_bra_ket_pairs and sequence_2 in valid_bra_ket_pairs)
    assert correct

def test_valid_sequences_3():
    """
            column id:12
            column amplitude:(0.1)
            column boson factor:1
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
            -----
            column id:13
            column amplitude:(0.1)
            column boson factor:1
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
                
    """
    """ 
            Photon list column 12: [0,0,0,1]. The numbers are the channels for the photons
            Photon list column 13: [1,0,0,0]
            Projection list on '3100': [0,0,0,1]
            For column 12 there are 6 valid orders of the photons: (0,1,2,3), (1,0,2,3) etc 
            ==>(always 3 last and all permutations of first three numbers)
            
            For column 13 there are 6 valid orders of the photons: (3,2,1,0) or (3,2,1,0) etc
            ==>(always 0 last and all permutations of first three numbers)
            
            There ar 36 sequences including:
            ((0,1),(1,2),(2,3),(3,0))
            ((0,2),(1,1),(2,3),(3,0))	
            ((0,2),(1,3),(2,1),(3,0))

    """


    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','1000','1000','0100','0100','1000','1000','1000']
    column_identifier = [12,12,12,12,13,13,13,13]
    column_delay = [5,5,0,0,0,5,5,0]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.1, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)


    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    group = coll_of_col.collection_by_column[initial_state][interference_group]

    correct = True
    measurement_projection, projection_boson_factor = group._measurement_projection(optical_component = '3100')
    correct &= (measurement_projection == [0,0,0,1])
    correct &= (np.round(projection_boson_factor,4) == np.round(np.sqrt(6),4))

    column_12 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(12)
    column_13 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(13)

    valid_1 = group._valid_combinations(measurement_projection,column_12)
    valid_2 = group._valid_combinations(measurement_projection,column_13)

    for p in itertools.permutations([0,1,2]):
        correct &= tuple([n for n in p] + [3]) in valid_1
    for p in itertools.permutations([1,2,3]):
        correct &= tuple([n for n in p] + [0]) in valid_2


    valid_bra_ket_pairs = group._find_valid_photon_pairs_for_projection(measurement_projection, column_12, column_13)
    correct &= len(valid_bra_ket_pairs) == 36

    sequence_1 = ((0,1),(1,2),(2,3),(3,0))
    sequence_2 = ((0,2),(1,1),(2,3),(3,0))	
    sequence_3 = ((0,2),(1,3),(2,1),(3,0))
    correct &= (sequence_1 in valid_bra_ket_pairs and sequence_2 in valid_bra_ket_pairs and sequence_3 in valid_bra_ket_pairs)

    assert(correct) 

def test_valid_sequences_4_partial_measurement():
    """
            column id:12
            column amplitude:(0.24999998716429697-2.6264515646646206e-18j)
            column boson factor:1.4142135623730951
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'0010': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
            -----
            column id:13
            column amplitude:(0.24999998716429697-2.626451564664622e-18j)
            column boson factor:1.4142135623730951
                {'0010': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
    """
    """ 
            Photon list column 12: [0,2,1]. The numbers are the channels for the photons
            Photon list column 13: [2,0,1]
            Projection list on '11xx': [0,1]
            For column 12 the valid order of the photons is (0,2)
             ==> (on projection index 0 we map photon 0, on projection index 1 we map photon 2)
            For column 13 the valid order of the photons is (1,2)
             ==> (on projection index 0 we map photon 1, on projection index 1 we map photon 2)
            Sequence should be ((0,1),(2,2)) 
            (This means:    Photon 0 column 12 with photon 1 column 13, 
                            Photon 2 column 12 with photon 2 column 13)
    """

    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','0010','0100','0010','1000','0100']
    column_identifier = [12,12,12,13,13,13]
    column_delay = [5,5,0,5,5,0]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.25, 
                                                        'column_boson_factor' : np.sqrt(2), 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)


    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    
    group = coll_of_col.collection_by_column[initial_state][interference_group]

 
    correct = True
    measurement_projection, projection_boson_factor = group._measurement_projection(target_values_and_channels={
                                                                        'channels' : [0,1],
                                                                        'values' : [1,1]
                                                                        })
    correct &= (measurement_projection == [0,1])
    correct &= (np.round(projection_boson_factor,4) == 1)

    column_12 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(12)
    column_13 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(13)

    valid_1 = group._valid_combinations(measurement_projection,column_12)
    valid_2 = group._valid_combinations(measurement_projection,column_13)
    correct &= (valid_1 == [(0,2)])
    correct &= (valid_2 == [(1,2)])
    valid_bra_ket_pairs = group._find_valid_photon_pairs_for_projection(measurement_projection, column_12, column_13)
    sequence = valid_bra_ket_pairs[0]
    correct &= (len(sequence) == 2)
    correct &= ((0,1) in sequence and (2,2) in sequence )
    assert correct

def test_probabilities_1():
    """
            column id:12
            column amplitude:(0.1)
            column boson factor:1
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
            -----
            column id:13
            column amplitude:(0.1)
            column boson factor:1
                {'0100': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 5, 'pulse_width': 1}
                {'1000': {'amplitude': 1, 'probability': 1}}{'time_delay': 0, 'pulse_width': 1}
                
    """
    """ 
            Photon list column 12: [0,0,0,1]. The numbers are the channels for the photons
            Photon list column 13: [1,0,0,0]
            Projection list on '3100': [0,0,0,1]
            For column 12 there are 6 valid orders of the photons: (0,1,2,3), (1,0,2,3) etc 
            ==>(always 3 last and all permutations of first three numbers)
            
            For column 13 there are 6 valid orders of the photons: (3,0,1,2) or (3,2,0,1) etc
            ==>(always 3 first and all permutations of last three numbers)
            !! new definition: (1,2,3,0) or (2,3,1,0) etc. Always 0 last
            
            There ar 36 sequences including:
            ((0,1),(1,2),(2,3),(3,0))
            ((0,2),(1,1),(2,3),(3,0))	
            ((0,2),(1,3),(2,1),(3,0))

    """
    from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import photon_probability_function

    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','1000','1000','0100','0100','1000','1000','1000']
    column_identifier = [12,12,12,12,13,13,13,13]
    column_delay = [0,0,0,0,0,0,0,0]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.1, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)


    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    coll_of_col.photon_probability_function = photon_probability_function
    group = coll_of_col.collection_by_column[initial_state][interference_group]
    correct = True
    measurement_projection, projection_boson_factor = group._measurement_projection('3100')
    correct &= (measurement_projection == [0,0,0,1])
    correct &= (np.round(projection_boson_factor,4) == np.round(np.sqrt(6),4))

    column_12 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(12)
    column_13 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(13)

    calculated = group._calculate_probability_for_this_projection_for_these_state_columns(
                                                                            measurement_projection,
                                                                            projection_boson_factor,
                                                                            column_12, 
                                                                            column_13)
    # probability should be amplitude_bra x amplitude_ket x overlap_factor / boson_fac_bra x boson_fac_ket x boson_proj x boson_proj

    # if the photons all have same photon_information their wavefunctions fully overlap
    expected = 36 * column_12.column_amplitude * column_13.column_amplitude / (projection_boson_factor**2 * column_12.column_boson_factor * column_12.column_boson_factor)
    correct &= np.round(calculated,4) == np.round(expected,4)

    # if we fully separate the photons for these columns the probability should drop to zero
    for state in coll_of_col.by_state():
        if state.auxiliary_information['photon_resolution']['column_identifier'] == 12:
            state.auxiliary_information['photon_resolution']['photon_information'] = {'time_delay': 10, 'pulse_width': 1}
    calculated = group._calculate_probability_for_this_projection_for_these_state_columns(
                                                                            measurement_projection,
                                                                            projection_boson_factor,
                                                                            column_12, 
                                                                            column_13)
    correct &= np.round(calculated,4) == 0

    # for same photon_information but different interference_groups probability should also be zero,
    # but we have to recreate the collection_of_columns from the state information
    for state in coll_of_col.by_state():
        if state.auxiliary_information['photon_resolution']['column_identifier'] == 12:
            state.auxiliary_information['photon_resolution']['photon_information'] = {'time_delay': 5, 'pulse_width': 1}
            state.auxiliary_information['photon_resolution']['interference_group_identifier'] = 1
    coll_of_col = coll_of_col.new_instance_from_state_information()
    coll_of_col.photon_probability_function = photon_probability_function
    calculated = group._calculate_probability_for_this_projection_for_these_state_columns(
                                                                            measurement_projection,
                                                                            projection_boson_factor,
                                                                            column_12, 
                                                                            column_13)
    correct &= np.round(calculated,4) == 0
    assert correct

def test_probabilities_2():
    # same test as in test_probabilities_1 but play with photon overlap
    # if we make every photon unique (so only overlaps with itself in same channel) the overlap factor should 
    # reduce from 36 to 6
    from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import photon_probability_function

    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','1000','1000','0100','0100','1000','1000','1000']
    column_identifier = [12,12,12,12,13,13,13,13]
    column_delay = [-15,-10,-5,0,0,-5,-10,-15]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.1, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)


    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    group = coll_of_col.collection_by_column[initial_state][interference_group]
    coll_of_col.photon_probability_function = photon_probability_function
    
    correct = True
    measurement_projection, projection_boson_factor = group._measurement_projection('3100')
    correct &= (measurement_projection == [0,0,0,1])
    correct &= (np.round(projection_boson_factor,4) == np.round(np.sqrt(6),4))

    column_12 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(12)
    column_13 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(13)

    calculated = group._calculate_probability_for_this_projection_for_these_state_columns(
                                                                            measurement_projection,
                                                                            projection_boson_factor,
                                                                            column_12, 
                                                                            column_13)
    # probability should be amplitude_bra x amplitude_ket x overlap_factor / boson_fac_bra x boson_fac_ket x boson_proj x boson_proj

    # if the photons all have same photon_information their wavefunctions fully overlap
    # The photons at the bra-side have 6 options to pair with the project operator (all permutations for the photons in channel 0)
    # The photons at the key-side then have only one option to match, to multiplicity is 6
    expected = 6 * column_12.column_amplitude * column_13.column_amplitude / (projection_boson_factor**2 * column_12.column_boson_factor * column_12.column_boson_factor)
    correct &= np.round(calculated,4) == np.round(expected,4)
    assert correct

def test_probabilities_3():
    # same test as in test_probabilities_1 but play with two photons overlapping
    # if we make every photon unique (so only overlaps with itself in same channel) the overlap factor should 
    # reduce from 36 to 12 (6 for the photons at bra-side to match with projection, and two options for ket-side)
    # we also test negative amplitude for one column here
    from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import photon_probability_function

    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','1000','1000','0100','0100','1000','1000','1000']
    column_identifier = [12,12,12,12,13,13,13,13]
    column_delay = [-10,-10,-5,0,0,-5,-10,-10]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.1, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)

    for state in initial_collection_of_states:
        if state.auxiliary_information['photon_resolution']['column_identifier'] == 12:
            state.auxiliary_information['photon_resolution']['column_amplitude'] = -0.1

    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    coll_of_col.photon_probability_function = photon_probability_function
    correct = True
    group = coll_of_col.collection_by_column[initial_state][interference_group]
    measurement_projection, projection_boson_factor = group._measurement_projection('3100')
    
    correct &= (measurement_projection == [0,0,0,1])
    correct &= (np.round(projection_boson_factor,4) == np.round(np.sqrt(6),4))

    column_12 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(12)
    column_13 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(13)

    calculated = group._calculate_probability_for_this_projection_for_these_state_columns(
                                                                            measurement_projection,
                                                                            projection_boson_factor,
                                                                            column_12, 
                                                                            column_13)
    # probability should be amplitude_bra x amplitude_ket x overlap_factor / boson_fac_bra x boson_fac_ket x boson_proj x boson_proj

    # if the photons all have same photon_information their wavefunctions fully overlap
    # The photons at the bra-side have 6 options to pair with the project operator (all permutations for the photons in channel 0)
    # The photons at the key-side then have only one option to match, to multiplicity is 6
    expected = 12 * column_12.column_amplitude * column_13.column_amplitude / (projection_boson_factor**2 * column_12.column_boson_factor * column_12.column_boson_factor)
    correct &= np.round(calculated,4) == np.round(expected,4)
    correct &= calculated < 0
    assert correct

def test_probabilities_4():
    """ Test if columns have states without photons"""
    """
    group id:2
    group probability: 0.94
        -----
        column id:0
        column amplitude:(0.4999999828857291-1.4782794711999372e-16j)
        column boson factor:1
            Component: '01', Amplitude: (1.00 - 0.00i).  , time_delay: -100.00 , pulse_width: 1.00
            Component: '00', Amplitude: (1.00 - -0.00i).  , time_delay: 100.00 , pulse_width: 1.00
        -----
        column id:1
        column amplitude:(0.4999999828857291-6.123233919984101e-17j)
        column boson factor:1
            Component: '00', Amplitude: (1.00 - -0.00i).  , time_delay: -100.00 , pulse_width: 1.00
            Component: '01', Amplitude: (1.00 - 0.00i).  , time_delay: 100.00 , pulse_width: 1.00
        -----
        column id:2
        column amplitude:(-0.4999999828857291+1.0453014315991737e-16j)
        column boson factor:1
            Component: '10', Amplitude: (1.00 - 0.00i).  , time_delay: -100.00 , pulse_width: 1.00
            Component: '00', Amplitude: (1.00 - -0.00i).  , time_delay: 100.00 , pulse_width: 1.00
        -----
        column id:3
        column amplitude:(0.4999999828857291-1.7934536409574678e-17j)
        column boson factor:1
            Component: '00', Amplitude: (1.00 - -0.00i).  , time_delay: -100.00 , pulse_width: 1.00
            Component: '10', Amplitude: (1.00 - 0.00i).  , time_delay: 100.00 , pulse_width: 1.00_summary_
    """
    """ 
            Photon list column 2: [0]. The numbers are the channels for the photons
            Photon list column 3: [0]
            Projection list on '10': [0]
            For column 2 there is 1 valid order of the photons: (0)
            
            There is one sequence
            ((0,0))

            The probability should be zero becuase the overlap is zero


    """
    from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import photon_probability_function

    circuit = fsc.FockStateCircuit(length_of_fock_state=2,no_of_optical_channels=2,no_of_classical_channels=2)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['01','00','00','01','10','00','00','10']
    column_identifier = [0,0,1,1,2,2,3,3]
    column_delay = [-100,100,-100,100,-100,100,-100,100]
    column_amplitude = [0.25,0.25,0.25,0.25,-0.25,-0.25,0.25,0.25]
    initial_state = "init_state"
    interference_group = 2
    for name,id,delay,amp in zip(column_component,column_identifier,column_delay,column_amplitude):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : amp, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)


    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    group = coll_of_col.collection_by_column[initial_state][interference_group]
    coll_of_col.photon_probability_function = photon_probability_function
    correct = True
    measurement_projection, projection_boson_factor = group._measurement_projection('10')
    correct &= (measurement_projection == [0])
    correct &= (np.round(projection_boson_factor,4) == np.round(np.sqrt(1),4))

    column_2 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(2)
    column_3 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(3)
    
    valid_2 = group._valid_combinations(measurement_projection,column_2)
    valid_3 = group._valid_combinations(measurement_projection,column_3)

    correct &= valid_2 == [(0,)]
    correct &= valid_3 == [(0,)]

    valid_bra_ket_pairs = group._find_valid_photon_pairs_for_projection(measurement_projection, column_2, column_3)
    correct &= len(valid_bra_ket_pairs) == 1
  
    correct &= (valid_bra_ket_pairs == [((0,0),) ])
    
    calculated = group._calculate_probability_for_this_projection_for_these_state_columns(
                                                                            measurement_projection,
                                                                            projection_boson_factor,
                                                                            column_2, 
                                                                            column_3)
    expected = 0 #(due to wave function overlap)
    correct &= np.round(calculated,4) == np.round(expected,4)
    assert correct

def test_probabilities_5_partial_measurement():
    # partial measurement on 1st channel with three photon and 36 combinations combination
    from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import photon_probability_function

    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','1000','1000','0100','0100','1000','1000','1000']
    column_identifier = [12,12,12,12,13,13,13,13]
    column_delay = [0,0,0,0,0,0,0,0]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.1, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)

    for state in initial_collection_of_states:
        if state.auxiliary_information['photon_resolution']['column_identifier'] == 12:
            state.auxiliary_information['photon_resolution']['column_amplitude'] = -0.1

    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    coll_of_col.photon_probability_function = photon_probability_function
    correct = True
    group = coll_of_col.collection_by_column[initial_state][interference_group]
    # project on '3xxx'
    measurement_projection, projection_boson_factor = group._measurement_projection(target_values_and_channels={
                                                                        'channels' : [0],
                                                                        'values' : [3]
                                                                        })

    correct &= (measurement_projection == [0,0,0])
    correct &= (np.round(projection_boson_factor,4) == np.round(np.sqrt(6),4))

    column_12 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(12)
    column_13 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(13)

    calculated = group._calculate_probability_for_this_projection_for_these_state_columns(
                                                                            measurement_projection,
                                                                            projection_boson_factor,
                                                                            column_12, 
                                                                            column_13)
    # probability should be amplitude_bra x amplitude_ket x overlap_factor / boson_fac_bra x boson_fac_ket x boson_proj x boson_proj

    # if the photons all have same photon_information their wavefunctions fully overlap
    # The photons at the bra-side have 6 options to pair with the project operator (all permutations for the photons in channel 0)
    # The photons at the key-side then have only one option to match, to multiplicity is 6

    expected = 36 * column_12.column_amplitude * column_13.column_amplitude / (projection_boson_factor**2 * column_12.column_boson_factor * column_12.column_boson_factor)
    correct &= np.round(calculated,4) == np.round(expected,4)

    correct &= calculated < 0
    assert correct

def test_probabilities_6_partial_measurement():
    # partial measurement on 2nd channel with one photon and only one combination
    from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import photon_probability_function

    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','1000','1000','0100','0100','1000','1000','1000']
    column_identifier = [12,12,12,12,13,13,13,13]
    column_delay = [0,0,0,0,0,0,0,0]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : 0.1, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)

    for state in initial_collection_of_states:
        if state.auxiliary_information['photon_resolution']['column_identifier'] == 12:
            state.auxiliary_information['photon_resolution']['column_amplitude'] = -0.1

    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    coll_of_col.photon_probability_function = photon_probability_function
    correct = True
    group = coll_of_col.collection_by_column[initial_state][interference_group]
    # project on 'x1xx'
    measurement_projection, projection_boson_factor = group._measurement_projection(target_values_and_channels={
                                                                        'channels' : [1],
                                                                        'values' : [1]
                                                                        })

    correct &= (measurement_projection == [1])
    correct &= (np.round(projection_boson_factor,4) == np.round(np.sqrt(1),4))
    column_12 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(12)
    column_13 = coll_of_col.collection_by_column[initial_state][interference_group].extract_column_by_identifier(13)

    calculated = group._calculate_probability_for_this_projection_for_these_state_columns(
                                                                            measurement_projection,
                                                                            projection_boson_factor,
                                                                            column_12, 
                                                                            column_13)
    # probability should be amplitude_bra x amplitude_ket x overlap_factor / boson_fac_bra x boson_fac_ket x boson_proj x boson_proj

    # if the photons all have same photon_information their wavefunctions fully overlap
    # The photons at the bra-side have 6 options to pair with the project operator (all permutations for the photons in channel 0)
    # The photons at the key-side then have only one option to match, to multiplicity is 6
    expected = 1 * column_12.column_amplitude * column_13.column_amplitude / (projection_boson_factor**2 * column_12.column_boson_factor * column_12.column_boson_factor)
    correct &= np.round(calculated,4) == np.round(expected,4)
    correct &= calculated < 0
    assert correct

def test_regenerate_collection_of_columns_after_changing_state_information():
    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))

    column_component = ['1000','1000','1000','0100','0100','1000','1000','1000']
    column_identifier = [12,12,12,12,13,13,13,13]
    column_delay = [-15,-10,-5,0,0,-5,-10,-15]
    initial_state = "init_state"
    interference_group = 9
    for name,id,delay in zip(column_component,column_identifier,column_delay):
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = initial_state
        state1.optical_components = {name: {'amplitude': 1, 'probability': 1}}
        photon_information_dict = {'time_delay': delay, 'pulse_width': 1}
        state1.auxiliary_information['photon_resolution'] = {'column_amplitude' : -0.2, 
                                                        'column_boson_factor' : 1, 
                                                        'group_cumulative_probability' : 1, 
                                                        'column_identifier' :id, 
                                                        'interference_group_identifier' : interference_group,
                                                        'photon_information':photon_information_dict}
        initial_collection_of_states.add_state(state1)


    coll_of_col = fsc.CollectionOfStateColumns(initial_collection_of_states)
    correct = True
    for state in coll_of_col.by_state():
        if state.auxiliary_information['photon_resolution']['column_identifier'] == 12:
            state.auxiliary_information['photon_resolution']['column_amplitude'] = -0.1
            state.auxiliary_information['photon_resolution']['column_identifier'] = 15

    coll_of_col = coll_of_col.new_instance_from_state_information()
    for column in coll_of_col.by_column():
        for state in column:
            correct &= state.auxiliary_information['photon_resolution']['column_amplitude'] == column.column_amplitude
    for group in coll_of_col.by_group():
        for column in group.by_column():
            correct &= group.interference_group_identifier == column.interference_group_identifier
    assert correct

def test_set_pulse_width_all_channels():
    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    for oc in ['1100', '1001', '0011', '0110']:
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = oc
        state1.optical_components = [(oc,1)]
        initial_collection_of_states.add_state(state=state1)
    circuit.set_pulse_width(pulse_width=4)
    result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    for state in result:
        if not state.auxiliary_information['photon_resolution']['photon_information']['pulse_width'] == 4:
            assert False
    assert True

def test_set_pulse_width_no_channel():
    # first check on 'virgin' collection
    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    for oc in ['1100', '1001', '0011', '0110']:
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = oc
        state1.optical_components = [(oc,1)]
        initial_collection_of_states.add_state(state=state1)
    circuit.set_pulse_width(affected_channels=[],pulse_width=4)
    result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    for state in result:
        ai = state.auxiliary_information
        if 'photon_resolution' in ai.keys(): 
            assert False
    # then check collection that already has pulse width
    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    for oc in ['1100', '1001', '0011', '0110']:
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = oc
        state1.optical_components = [(oc,1)]
        initial_collection_of_states.add_state(state=state1)
    circuit.set_pulse_width(pulse_width=2)
    circuit.set_pulse_width(affected_channels=[],pulse_width=4)
    result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    for state in result:
        ai = state.auxiliary_information
        if 'photon_resolution' in ai.keys() and 'photon_information' in ai['photon_resolution'].keys() and ai['photon_resolution']['photon_information']['pulse_width'] == 4:
            assert False

    assert True

def test_set_pulse_width_some_channels():
    # first check on 'virgin' collection
    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    for oc in ['1100', '1001', '0011', '0110']:
        state1 = fsc.State(initial_collection_of_states)
        state1.initial_state = oc
        state1.optical_components = [(oc,1)]
        initial_collection_of_states.add_state(state=state1)
    circuit.set_pulse_width(affected_channels=[0,1],pulse_width=4)
    result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    for state in result:
        # have to be single photon states with single optical components
        if int(state) != 1 or len(state.optical_components) != 1:
            assert False
        # photon_information needs to be present
        ai = state.auxiliary_information
        if not ('photon_resolution' in ai.keys() and 'photon_information' in ai['photon_resolution'].keys()):
            assert False
        # check channels that should be changed
        oc = state.optical_components
        values = state._dict_of_valid_component_names[list(oc.keys())[0]]
        if values[0] ==1 or values[1] ==1:
            if ai['photon_resolution']['photon_information']['pulse_width'] != 4:
                assert False
        # check other channels have default value
        if values[2] ==1 or values[3] ==1:
            if ai['photon_resolution']['photon_information']['pulse_width'] != 1:
                assert False
    assert True

def test_partial_measurement_1():
    # first measure two channels, show that for those two channels you have right statistics. Then measure remaining
    # two channels and show that they can still interfere
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.initial_state = 'init'
    state.optical_components = [ ('1110', np.sqrt(1/2)),('1101', np.sqrt(1/2))]
    initial_collection_of_states.add_state(state=state)
    circuit.time_delay_full(affected_channels=[0],delay=10,pulse_width=1)
    circuit.mix_50_50(first_channel=0,second_channel=1)
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1], classical_channels_to_be_written=[0,1])
    # we expect HOM effect depending on delay between channel 0 and 1. For channels 2 and 3 this should still be pure state
    # which can be rotated to align with detection
    circuit.half_wave_plate_225(2,3)
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[2,3], classical_channels_to_be_written=[2,3])
    result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    histo = result.plot(histo_output_instead_of_plot=True)
    correct = True
    for res in histo['init']:
        if res['output_state'] == '0201':
            correct &= compare(res['probability'],0) 
        if res['output_state'] == '2001':
            correct &= compare(res['probability'],0) 
        if res['output_state'] == '1101':
            correct &= compare(res['probability'],0)
    assert correct

def test_partial_measurement_2():
    # Start with state that is a tensor product, so first two channels 
    # If you detect all channels then channels cannot interfere anymore
    def compare(a,b):
        return np.round(a,4) == np.round(b,4)

    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=4,no_of_classical_channels=4)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=initial_collection_of_states)
    state.initial_state = 'init'
    state.optical_components = [ ('1110', np.sqrt(1/2)),('1101', np.sqrt(1/2))]
    initial_collection_of_states.add_state(state=state)
    circuit.time_delay_full(affected_channels=[0],delay=10,pulse_width=1)
    circuit.mix_50_50(first_channel=0,second_channel=1)
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], classical_channels_to_be_written=[0,1,2,3])
    # we expect HOM effect depending on delay between channel 0 and 1. For channels 2 and 3 this shoudl be mixture
    circuit.half_wave_plate_225(2,3)
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[2,3], classical_channels_to_be_written=[2,3])
    result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    histo = result.plot(histo_output_instead_of_plot=True)
    correct = True
    for res in histo['init']:
        if res['output_state'] == '0201':
            correct &= compare(res['probability'],0.125) 
        if res['output_state'] == '2001':
            correct &= compare(res['probability'],0.125) 
        if res['output_state'] == '1101':
            correct &= compare(res['probability'],0.25) 
    assert correct