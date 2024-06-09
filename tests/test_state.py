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

def test_state_as_a_class():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                    no_of_optical_channels = 2, 
                                    no_of_classical_channels=3, 
                                    )
    collection = fsc.CollectionOfStates(circuit)
    state = fsc.State(collection)

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


def test_rescale_optical_components():
    no_failure_found = True
    circuit = fsc.FockStateCircuit(length_of_fock_state = 4, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=4,
                                    threshold_probability_for_setting_to_zero=0.1
                                    )

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

    initial_collection_of_states_large = fsc.CollectionOfStates(fock_state_circuit=circuit)
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

def test_state_auxiliary_information():
    no_error_found = True

    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                            no_of_optical_channels = 3,
                            no_of_classical_channels=2,
                            circuit_name = 'test'
                            )

    input_collection = fsc.CollectionOfStates(fock_state_circuit=circuit)

    state = fsc.State(collection_of_states=input_collection)

    no_error_found = no_error_found and 'auxiliary_information' not in state.state.keys()
    state.measurement_results = [{'measurement_results': [1, 3, 3.14], 'probability': 0.5}]
    state.auxiliary_information = {'spectral_information': "all kinds of data", 'NS box': ' a lot of other data'}

    state2 = state.copy()
    no_error_found = no_error_found and 'auxiliary_information' in state2.state.keys()
    no_error_found = no_error_found and 'spectral_information' in state2.state['auxiliary_information'].keys()
    no_error_found = no_error_found and 'NS box' in state2.state['auxiliary_information'].keys()


    state2.auxiliary_information = {'spectral_information': ' a lot of other data'}
    no_error_found = no_error_found and 'NS box' not in state2.state['auxiliary_information'].keys()
    no_error_found = no_error_found and 'spectral_information' in state2.state['auxiliary_information'].keys()
    assert no_error_found


def test_inner():
    no_error_found = True
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                        no_of_optical_channels = 3,
                        no_of_classical_channels=2,
                        circuit_name = 'test'
                        )

    input_collection = fsc.CollectionOfStates(fock_state_circuit=circuit)

    state1 = fsc.State(collection_of_states=input_collection)
    state1.optical_components = [('200',1)]

    state2 = fsc.State(collection_of_states=input_collection)
    state2.optical_components = [('220',1)]

    state3 = fsc.State(collection_of_states=input_collection)
    state3.optical_components = [('222',1)]

    state4 = fsc.State(collection_of_states=input_collection)
    state4.optical_components = [('200',1j*np.sqrt(1/2)),('020',1/2),('002',1/2)]

    state5 = fsc.State(collection_of_states=input_collection)
    state5.optical_components = [('200',np.sqrt(1/2)),('220',1/2),('222',1/2)]

    no_error_found = no_error_found and np.round(state1.inner(state2),6) == np.round(0,6)

    no_error_found = no_error_found and np.round(state2.inner(state3),6) == np.round(0,6)

    no_error_found = no_error_found and np.round(state3.inner(state3),6) == np.round(1,6)

    no_error_found = no_error_found and np.round(state5.inner(state3),6) == np.round(1/2,6)
    
    no_error_found = no_error_found and np.round(state4.inner(state5),6) == np.round(1/2,6) * -1 * 1j

    assert no_error_found
    
def test_outer():
    no_error_found = True
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                        no_of_optical_channels = 3,
                        no_of_classical_channels=2,
                        circuit_name = 'test'
                        )
    
    input_collection = fsc.CollectionOfStates(fock_state_circuit=circuit)
    
    state1 = fsc.State(collection_of_states=input_collection)
    state1.optical_components = [('200',np.sqrt(1/2)),('220',1/2),('222',1/2)]

    state2 = fsc.State(collection_of_states=input_collection)
    state2.optical_components = [('100',np.sqrt(1/2)),('110',1/2),('111',1/2)]
    
    bss = circuit.basis()
    out = state1.outer(state2)

    no_error_found = no_error_found and len(out) == len(bss)

    comp1,comp2 = '200','100'
    index1 = list(bss.keys()).index(comp1)
    index2 = list(bss.keys()).index(comp2)
    no_error_found = no_error_found and np.round(out[index1,index2],6) == np.round(state1.optical_components[comp1]['amplitude']*state2.optical_components[comp2]['amplitude'],6)

    assert no_error_found

def test_tensor():
    no_error_found = True
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                        no_of_optical_channels = 3,
                        no_of_classical_channels=2,
                        circuit_name = 'test'
                        )

    input_collection = fsc.CollectionOfStates(fock_state_circuit=circuit)

    state1 = fsc.State(collection_of_states=input_collection)
    state1.optical_components = [('200',np.sqrt(1/2)),('220',1/2),('222',1/2)]

    state2 = fsc.State(collection_of_states=input_collection)
    state2.optical_components = [('100',np.sqrt(1/2)),('110',1/2),('111',1/2)]

    oc = state1.tensor(state2)

    no_error_found = no_error_found and '200100' in oc.keys()
    no_error_found = no_error_found and '220111' in oc.keys()
    no_error_found = no_error_found and not '100222' in oc.keys()
    no_error_found = no_error_found and len(oc) == 9
    no_error_found = no_error_found and np.round(oc['222111']['amplitude'],6) == np.round(1/4,6)
    
    oc2 = state2.optical_components
    oc = state1.tensor(oc2)

    no_error_found = no_error_found and '200100' in oc.keys()
    no_error_found = no_error_found and '220111' in oc.keys()
    no_error_found = no_error_found and not '100222' in oc.keys()
    no_error_found = no_error_found and len(oc) == 9
    no_error_found = no_error_found and np.round(oc['220100']['amplitude'],6) == np.round(np.sqrt(1/8),6)

    oc = state1.tensor([('100',np.sqrt(1/2)),('110',1/2),('111',1/2)])

    no_error_found = no_error_found and '200100' in oc.keys()
    no_error_found = no_error_found and '220111' in oc.keys()
    no_error_found = no_error_found and not '100222' in oc.keys()
    no_error_found = no_error_found and len(oc) == 9
    no_error_found = no_error_found and np.round(oc['200100']['amplitude'],6) == np.round(1/2,6)

    assert no_error_found

def test_tensor_rev_state_notation():
    no_error_found = True
    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                        no_of_optical_channels = 3,
                        no_of_classical_channels=2,
                        circuit_name = 'test',
                        channel_0_left_in_state_name=False
                        )

    input_collection = fsc.CollectionOfStates(fock_state_circuit=circuit)

    state1 = fsc.State(collection_of_states=input_collection)
    state1.optical_components = [('200',np.sqrt(1/2)),('220',1/2),('222',1/2)]

    state2 = fsc.State(collection_of_states=input_collection)
    state2.optical_components = [('100',np.sqrt(1/2)),('110',1/2),('111',1/2)]

    oc = state1.tensor(state2)

    no_error_found = no_error_found and '100200' in oc.keys()
    no_error_found = no_error_found and '111220' in oc.keys()
    no_error_found = no_error_found and not '222100' in oc.keys()
    no_error_found = no_error_found and len(oc) == 9
    no_error_found = no_error_found and np.round(oc['111222']['amplitude'],6) == np.round(1/4,6)

    oc2 = state2.optical_components
    oc = state1.tensor(oc2)

    no_error_found = no_error_found and '100200' in oc.keys()
    no_error_found = no_error_found and '111220' in oc.keys()
    no_error_found = no_error_found and not '222100' in oc.keys()
    no_error_found = no_error_found and len(oc) == 9
    no_error_found = no_error_found and np.round(oc['100220']['amplitude'],6) == np.round(np.sqrt(1/8),6)

    oc = state1.tensor([('100',np.sqrt(1/2)),('110',1/2),('111',1/2)])

    no_error_found = no_error_found and '100200' in oc.keys()
    no_error_found = no_error_found and '111220' in oc.keys()
    no_error_found = no_error_found and not '222100' in oc.keys()
    no_error_found = no_error_found and len(oc) == 9
    no_error_found = no_error_found and np.round(oc['100200']['amplitude'],6) == np.round(1/2,6)

    assert no_error_found


def test_create_optical_components():
    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=3,no_of_classical_channels=3)
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
    state1 = fsc.State(collection_of_states=initial_collection_of_states)
    state1.initial_state = 'init_1'
    state1.optical_components = [('210',np.sqrt(1/8)),('000',1/2),('111',1/2),('210',np.sqrt(1/8))]
    oc = state1.optical_components
    correct = len(oc) == 3
    correct &= '000' in oc.keys()
    correct &= np.round(oc['210']['probability'],4) == 0.5
    assert correct