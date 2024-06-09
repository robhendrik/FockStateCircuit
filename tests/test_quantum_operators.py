import sys  
sys.path.append("./src")
import fock_state_circuit as fsc
import numpy as np
import math
import pytest

def test_wave_plate_from_hamiltonian():
    length_of_fock_state=2
    channels = (0,1)

    theta = np.pi/4 #theta is used in teh rotation/mixing operator to mix channels with amplitudes cos(theta) and sin(theta)

    no_optical_channels = max(channels)+1
    circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
    circuit.quarter_wave_plate(channel_horizontal=channels[0], channel_vertical=channels[1], angle = theta)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    component = ['0'] * no_optical_channels
    component[channels[0]] = '1'
    state.optical_components = [(''.join(component),1)]
    collection.add_state(state)
    result1 = circuit.evaluate_circuit(collection)


    no_optical_channels = max(channels)+1
    circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
    circuit.wave_plate_from_hamiltonian(channel_horizontal=channels[0], channel_vertical=channels[1], theta = theta, phi=np.pi/2)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    component = ['0'] * no_optical_channels
    component[channels[0]] = '1'
    state.optical_components = [(''.join(component),1)]
    collection.add_state(state)
    result2 = circuit.evaluate_circuit(collection)

    for state1 in result1:
        for state2 in result2:
            oc1 = state1.optical_components
            oc2 = state2.optical_components
            for comp, amp_prob in oc1.items():
                if not np.round(amp_prob['amplitude'],2) == np.round(oc2[comp]['amplitude'],2):
                    assert False
    assert True

def test_wave_plate_from_hamiltonian_incl_classical_control():
    length_of_fock_state=4
    channels = (2,3)

    theta = np.pi/4 #theta is used in teh rotation/mixing operator to mix channels with amplitudes cos(theta) and sin(theta)

    no_optical_channels = max(channels)+1
    circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
    circuit.quarter_wave_plate(channel_horizontal=channels[0], channel_vertical=channels[1], angle = theta)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    component = ['0'] * no_optical_channels
    component[channels[0]] = '3'
    state.optical_components = [(''.join(component),1)]
    collection.add_state(state)
    result1 = circuit.evaluate_circuit(collection)

    no_optical_channels = max(channels)+1
    circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
    circuit.wave_plate_from_hamiltonian(channel_horizontal=channels[0], channel_vertical=channels[1], theta = theta, phi=np.pi/2)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    component = ['0'] * no_optical_channels
    component[channels[0]] = '3'
    state.optical_components = [(''.join(component),1)]
    collection.add_state(state)
    result2 = circuit.evaluate_circuit(collection)

    for state1 in result1:
        for state2 in result2:
            oc1 = state1.optical_components
            oc2 = state2.optical_components
            for comp, amp_prob in oc1.items():
                if not np.round(amp_prob['amplitude'],2) == np.round(oc2[comp]['amplitude'],2):
                    assert False
    assert True


    no_optical_channels = max(channels)+1
    circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=2)
    circuit.wave_plate_from_hamiltonian_classical_control(  optical_channel_horizontal=channels[0],
                                                            optical_channel_vertical=channels[1],
                                                            classical_channel_for_orientation=0,
                                                            classical_channel_for_phase_shift=1)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    component = ['0'] * no_optical_channels
    component[channels[0]] = '3'
    state.optical_components = [(''.join(component),1)]
    state.classical_channel_values = [theta, np.pi/2]
    collection.add_state(state)
    result2 = circuit.evaluate_circuit(collection)

    for state1 in result1:
        for state2 in result2:
            oc1 = state1.optical_components
            oc2 = state2.optical_components
            for comp, amp_prob in oc1.items():
                if not np.round(amp_prob['amplitude'],2) == np.round(oc2[comp]['amplitude'],2):
                    assert False
    assert True

def test_kerr_gate():
    length_of_fock_state=4
    channels = (0,1)

    a = 0.1


    operator_channels = [0,0,1]
    ham_kerr = fsc.Hamiltonian(operators=[('-++0', a), ('+--0', -a)], operator_channels=operator_channels)

    no_optical_channels = max(channels)+1
    circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
    circuit.hamiltonian_operator_gate(operators=[ham_kerr],channels=channels)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    for n in [1]:
        state = fsc.State(collection_of_states=collection)
        component = ['0'] * no_optical_channels
        component[channels[0]] = str(n)
        state.optical_components = [(''.join(component),1)]
        collection.add_state(state)
    result= circuit.evaluate_circuit(collection)
    for state in result:
        oc = state.optical_components
        assert '11' in oc.keys() and np.round(oc['11']['amplitude'],2) == 0.1
    return

def test_double_digit_photon_number_notation():
 
    theta = np.pi/4 #theta is used in teh rotation/mixing operator to mix channels with amplitudes cos(theta) and sin(theta)

    circuit = fsc.FockStateCircuit(length_of_fock_state=11,no_of_optical_channels=2,no_of_classical_channels=0)
    circuit.wave_plate_from_hamiltonian(channel_horizontal=0, channel_vertical=1, theta = theta, phi=np.pi/4)
    circuit.wave_plate_from_hamiltonian(channel_horizontal=0, channel_vertical=1, theta = theta, phi=np.pi/4)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    component = '0001'
    state.optical_components = [(''.join(component),1)]
    collection.add_state(state)
    result = circuit.evaluate_circuit(collection)
    state = result.get_state(initial_state='0000')
    oc = state.optical_components
    errors = []
    errors.append('0100' in oc.keys() and '0001' in oc.keys())
    errors.append(np.round(oc['0100']['probability'],4) == 0.5)
    errors.append(np.round(oc['0001']['probability'],4) == 0.5)
    assert all(errors)
    
def test_reverse_state_notation():
  
    theta = np.pi/4 #theta is used in teh rotation/mixing operator to mix channels with amplitudes cos(theta) and sin(theta)

    circuit = fsc.FockStateCircuit(length_of_fock_state=4,no_of_optical_channels=3,no_of_classical_channels=0, channel_0_left_in_state_name=False)
    circuit.wave_plate_from_hamiltonian(channel_horizontal=0, channel_vertical=1, theta = theta, phi=np.pi/4)
    circuit.wave_plate_from_hamiltonian(channel_horizontal=0, channel_vertical=1, theta = theta, phi=np.pi/4)
    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
    state = fsc.State(collection_of_states=collection)
    component = '301'
    state.optical_components = [(''.join(component),1)]
    collection.add_state(state)
    result = circuit.evaluate_circuit(collection)
    state = result.get_state(initial_state='000')
    oc = state.optical_components
    errors = []
    errors.append('310' in oc.keys() and '301' in oc.keys())
    errors.append(np.round(oc['310']['probability'],4) == 0.5)
    errors.append(np.round(oc['301']['probability'],4) == 0.5)
    assert all(errors)

def test_full_consistency_with_general_wave_plate_optical_node():
    # check consistency if you build the quantum operator yourself
    length_of_fock_state=2
    phi = np.pi
    theta = np.pi/4 #theta is used in teh rotation/mixing operator to mix channels with amplitudes cos(theta) and sin(theta)
    channels = (0,1)

    for theta in [0,np.pi/16,np.pi/8,np.pi/4,np.pi/2,np.pi,3.5*np.pi]:
        for phi in [0,np.pi/8,np.pi/4,np.pi/2,np.pi,2*np.pi,1.5*np.pi]:
            theta = theta%(2*np.pi)
            phi = phi%(2*np.pi)
            no_optical_channels = max(channels)+1
            circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
            circuit.wave_plate(channel_horizontal=channels[0],channel_vertical=channels[1],theta=theta,phi=phi)
            collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
            state = fsc.State(collection_of_states=collection)
            component = ['0'] * no_optical_channels
            component[channels[0]] = '1'
            state.optical_components = [(''.join(component),1)]
            collection.add_state(state)
            result1 = circuit.evaluate_circuit(collection)

            # same with Hamiltonian operator
            # add additional phase shift as very last step
            chi = -phi/2 

            operator_channels = [0,0,1,1]
            ham_phi = fsc.Hamiltonian([('+-00', -1j*phi/2),('00+-',+1j*phi/2 )], operator_channels)
            ham_theta_plus = fsc.Hamiltonian([('+0-0',theta ),('-0+0',-1*theta )], operator_channels)
            ham_theta_minus = fsc.Hamiltonian([('+0-0',-1*theta ),('-0+0',theta )], operator_channels)
            ham_chi = fsc.Hamiltonian([('-+00', -1j*chi),('00-+',-1j*chi )], operator_channels)

            operator_generic_wave_plate = [ham_theta_minus,ham_phi,ham_theta_plus,ham_chi]

            no_optical_channels = max(channels)+1
            circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
            circuit.hamiltonian_operator_gate(channels=(0,1), operators=operator_generic_wave_plate)
            collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
            state = fsc.State(collection_of_states=collection)
            component = ['0'] * no_optical_channels
            component[channels[0]] = '1'
            state.optical_components = [(''.join(component),1)]
            collection.add_state(state)
            result2 = circuit.evaluate_circuit(collection)
        
            errors = []
            for state1, state2 in zip(result1, result2):
                for comp in state1.optical_components.keys():
                    oc1, oc2 = state1.optical_components, state2.optical_components
                    amp1, amp2 = oc1[comp]['amplitude'], oc2[comp]['amplitude']
                    errors.append(np.round(np.abs(amp1),2) == np.round(np.abs(amp2),2))
                    if np.round(np.abs(amp2),2) > 0:
                        angle_difference = np.round((np.angle(amp1) -np.angle(amp2))%(np.pi *2),2)
                        errors.append(angle_difference == 0 or angle_difference == np.round(np.pi*2,2))

    assert all(errors)

def test_full_consistency_with_general_wave_plate_optical_node_2():
    # check consistency if you buiuse the build in wave plate from hamiltonian
    length_of_fock_state=2
    phi = np.pi
    theta = np.pi/4 #theta is used in teh rotation/mixing operator to mix channels with amplitudes cos(theta) and sin(theta)
    channels = (0,1)

    for theta in [0,np.pi/16,np.pi/8,np.pi/4,np.pi/2,np.pi,3.5*np.pi]:
        for phi in [0,np.pi/8,np.pi/4,np.pi/2,np.pi,2*np.pi,1.5*np.pi]:
            theta = theta%(2*np.pi)
            phi = phi%(2*np.pi)
            no_optical_channels = max(channels)+1
            circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
            circuit.wave_plate(channel_horizontal=channels[0],channel_vertical=channels[1],theta=theta,phi=phi)
            collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
            state = fsc.State(collection_of_states=collection)
            component = ['0'] * no_optical_channels
            component[channels[0]] = '1'
            state.optical_components = [(''.join(component),1)]
            collection.add_state(state)
            result1 = circuit.evaluate_circuit(collection)

            # same with Hamiltonian operator

            no_optical_channels = max(channels)+1
            circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
            circuit.wave_plate_from_hamiltonian(channel_horizontal=channels[0],channel_vertical=channels[1],theta=theta,phi=phi)
            collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
            state = fsc.State(collection_of_states=collection)
            component = ['0'] * no_optical_channels
            component[channels[0]] = '1'
            state.optical_components = [(''.join(component),1)]
            collection.add_state(state)
            result2 = circuit.evaluate_circuit(collection)
        
            errors = []
            for state1, state2 in zip(result1, result2):
                for comp in state1.optical_components.keys():
                    oc1, oc2 = state1.optical_components, state2.optical_components
                    amp1, amp2 = oc1[comp]['amplitude'], oc2[comp]['amplitude']
                    errors.append(np.round(np.abs(amp1),2) == np.round(np.abs(amp2),2))
                    if np.round(np.abs(amp2),2) > 0:
                        angle_difference = np.round((np.angle(amp1) -np.angle(amp2))%(np.pi *2),2)
                        errors.append(angle_difference == 0 or angle_difference == np.round(np.pi*2,2))
    assert all(errors)

def test_full_consistency_with_general_wave_plate_optical_node_3():
    # check consistency if you buiuse the build in wave plate from hamiltonian with classical control
    length_of_fock_state=2
    phi = np.pi
    theta = np.pi/4 #theta is used in teh rotation/mixing operator to mix channels with amplitudes cos(theta) and sin(theta)
    channels = (0,1)

    for theta in [0,np.pi/16,np.pi/8,np.pi/4,np.pi/2,np.pi,3.5*np.pi]:
        for phi in [0,np.pi/8,np.pi/4,np.pi/2,np.pi,2*np.pi,1.5*np.pi]:
            theta = theta%(2*np.pi)
            phi = phi%(2*np.pi)
            no_optical_channels = max(channels)+1
            circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=0)
            circuit.wave_plate(channel_horizontal=channels[0],channel_vertical=channels[1],theta=theta,phi=phi)
            collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
            state = fsc.State(collection_of_states=collection)
            component = ['0'] * no_optical_channels
            component[channels[0]] = '1'
            state.optical_components = [(''.join(component),1)]
            collection.add_state(state)
            result1 = circuit.evaluate_circuit(collection)

            # same with Hamiltonian operator

            no_optical_channels = max(channels)+1
            circuit = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,no_of_optical_channels=no_optical_channels,no_of_classical_channels=2)
            circuit.wave_plate_from_hamiltonian_classical_control(optical_channel_horizontal=channels[0],
                                                                optical_channel_vertical=channels[1],
                                                                classical_channel_for_orientation=0,
                                                                classical_channel_for_phase_shift=1)
            collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
            state = fsc.State(collection_of_states=collection)
            component = ['0'] * no_optical_channels
            component[channels[0]] = '1'
            state.optical_components = [(''.join(component),1)]
            state.classical_channel_values = [theta,phi]
            collection.add_state(state)
            result2 = circuit.evaluate_circuit(collection)
        
            errors = []
            for state1, state2 in zip(result1, result2):
                for comp in state1.optical_components.keys():
                    oc1, oc2 = state1.optical_components, state2.optical_components
                    amp1, amp2 = oc1[comp]['amplitude'], oc2[comp]['amplitude']
                    errors.append(np.round(np.abs(amp1),2) == np.round(np.abs(amp2),2))
                    if np.round(np.abs(amp2),2) > 0:
                        angle_difference = np.round((np.angle(amp1) -np.angle(amp2))%(np.pi *2),2)
                        errors.append(angle_difference == 0 or angle_difference == np.round(np.pi*2,2))
    assert all(errors)