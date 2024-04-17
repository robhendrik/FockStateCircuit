import sys  
sys.path.append("./src")
import fock_state_circuit as fsc

from fock_state_circuit.no_signalling_boxes import _stokes_vector_from_amplitudes,_group_configurations_per_outcome,probability_from_stokes_vectors,_repair_amplitudes_for_photon_pair,_stokes_vector_for_pair_from_amplitudes_hh_hv_vh_vv
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


def test_stokes_vector_from_amplitudes():
    no_error_found = True
    angles_s = [
        tuple([np.pi/2,0,0,0]),
        tuple([0,0,0,0]),
        tuple([np.pi/2,0,np.pi/2,0]),
        tuple([0,np.pi/2,0,np.pi/2]),
        tuple([np.pi/2,np.pi/2,np.pi/2,np.pi/2]),
        tuple([np.pi/4,0,0,1.1*np.pi]),
        tuple([np.pi/4,np.pi/2,0,0]),
        tuple([np.pi/5,np.pi/3,np.pi/6,np.pi/8]),
        tuple([0,np.pi/3,np.pi/6,np.pi/8]),
        tuple([np.pi/5,np.pi/3,0,np.pi/8]),
        tuple([np.pi/5,np.pi/3,np.pi/6,0]),
        tuple([np.pi/5,np.pi/16,1.9*np.pi,0.1]),
        tuple([10*np.pi/25,8*np.pi/17,0,0]),
        tuple([np.pi/5,10*np.pi/33,0,1.5]),
        tuple([np.pi/18,2*np.pi/3,0,0])
        ]
    lengths = [0.1,0.001, 3.14, np.pi, np.sqrt(2)]
    for index, angles in enumerate(angles_s):
        for indices in [(0,1),(2,3),(1,2),(0,3)]:
            if angles[indices[0]] > np.pi/2 or angles[indices[1]] > np.pi:
                      continue
            length = lengths[index%len(lengths)]
            phase = np.cos(angles[indices[1]]) + 1j*np.sin(angles[indices[1]])
            amplitude_north = length*np.cos(angles[indices[0]])*phase
            amplitude_south = length*np.sin(angles[indices[0]])
 
            vector, norm = _stokes_vector_from_amplitudes(amplitude_north, amplitude_south)
            no_error_found = no_error_found and np.round(sum([c**2 for c in vector]),4) == 1
            no_error_found = no_error_found and np.round(length,4) == np.round(norm,4)
            angle_psi = np.arccos(vector[2])
            no_error_found = no_error_found and np.round(angle_psi,4) == np.round(2*angles[indices[0]],4)
            if np.round(np.cos(angles[indices[0]])) != 0 and np.round(np.sin(angles[indices[0]])) != 0:
                angle_phi = np.arctan2(vector[0],vector[1])
                no_error_found = no_error_found and np.round(angle_phi,4) == np.round(angles[indices[1]],4)
            else:
                angle_phi = 0
            if not no_error_found:
                print(angles,[angle/np.pi for angle in angles], indices)
                print(np.round(length,2),np.round(norm,2),'-',np.round(angle_psi,4),np.round(2*angles[indices[0]],4),'-',np.round(angle_phi,2),np.round(angles[indices[1]],2))
                print(vector)
    assert no_error_found

def test_repair_amplitudes():
    no_error_found = True
    no_error_found = no_error_found and _repair_amplitudes_for_photon_pair([1,0,0,0]) == _repair_amplitudes_for_photon_pair([1,0,0,0])
    no_error_found = no_error_found and _repair_amplitudes_for_photon_pair([1,0,1,0]) == _repair_amplitudes_for_photon_pair([1,0,1,0])
    no_error_found = no_error_found and _repair_amplitudes_for_photon_pair([0,1,0,1]) == _repair_amplitudes_for_photon_pair([0,1,0,1])
    no_error_found = no_error_found and _repair_amplitudes_for_photon_pair([1,1,2,1]) == _repair_amplitudes_for_photon_pair([1,1,2,1])
    no_error_found = no_error_found and _repair_amplitudes_for_photon_pair([0,0,0,0]) == _repair_amplitudes_for_photon_pair([0,0,0,0])
    no_error_found = no_error_found and _repair_amplitudes_for_photon_pair([1,0,0,0]) == _repair_amplitudes_for_photon_pair([1,0,0,0])
    no_error_found = no_error_found and _repair_amplitudes_for_photon_pair([0,3,0,0]) == _repair_amplitudes_for_photon_pair([0,3,0,0])
    no_error_found = no_error_found and _repair_amplitudes_for_photon_pair([0,0,0,4]) == _repair_amplitudes_for_photon_pair([0,0,0,4])
    no_error_found = no_error_found and all(np.round(_repair_amplitudes_for_photon_pair([100,10,10,0]),4) == np.round(_repair_amplitudes_for_photon_pair([100,10,10,1]),4))
    no_error_found = no_error_found and _repair_amplitudes_for_photon_pair([1,0,0,0]) == _repair_amplitudes_for_photon_pair([1,0,0,0])
    assert no_error_found

def test_amplitudes_for_photon_from_photon_pairs_hh_hv_vh_vv():
    no_error_found = True
    angles_s = [
        tuple([np.pi/4,0,0,0]),
        tuple([np.pi/2,0,0,0]),
        tuple([0,0,0,0]),
        tuple([np.pi/2,0,np.pi/2,0]),
        tuple([0,np.pi/2,0,np.pi/2]),
        tuple([np.pi/2,np.pi/2,np.pi/2,np.pi/2]),
        tuple([np.pi/4,0,0,1.1*np.pi]),
        tuple([np.pi/4,np.pi/2,0,0]),
        tuple([np.pi/5,np.pi/3,np.pi/6,np.pi/8]),
        tuple([0,np.pi/3,np.pi/6,np.pi/8]),
        tuple([np.pi/5,np.pi/3,0,np.pi/8]),
        tuple([np.pi/5,np.pi/3,np.pi/6,0]),
        tuple([np.pi/5,np.pi/16,1.9*np.pi,0.1]),
        tuple([10*np.pi/25,8*np.pi/17,0,0]),
        tuple([np.pi/5,10*np.pi/33,0,1.5]),
        tuple([np.pi/18,2*np.pi/3,0,0])
        ]    
    lengths = [0.1,0.001, 3.14, np.pi, np.sqrt(2)]
    for index, angles in enumerate(angles_s):
        if angles[0] > np.pi/2 or angles[1] > np.pi:
            continue
        if angles[2] > np.pi/2 or angles[3] > np.pi:
            continue
        for indices in [(0,1),(2,3)]:

            length_1 = lengths[index%len(lengths)]
            length_2 = lengths[index%len(lengths)]

            
            if indices == (0,1):
                phase_1 = np.cos(angles[indices[1]]) + 1j*np.sin(angles[indices[1]])
                amplitude_north_1 = length_1*np.cos(angles[indices[0]])*phase_1
                amplitude_south_1 = length_1*np.sin(angles[indices[0]])
                vector_1, norm_1 = _stokes_vector_from_amplitudes(amplitude_north_1, amplitude_south_1)
            else:
                phase_2 = np.cos(angles[indices[1]]) + 1j*np.sin(angles[indices[1]])
                amplitude_north_2 = length_2*np.cos(angles[indices[0]])*phase_2
                amplitude_south_2 = length_2*np.sin(angles[indices[0]])
                vector_2, norm_2 = _stokes_vector_from_amplitudes(amplitude_north_2, amplitude_south_2)
        hh = amplitude_north_1 * amplitude_north_2
        hv = amplitude_north_1 * amplitude_south_2
        vh = amplitude_south_1 * amplitude_north_2
        vv = amplitude_south_1 * amplitude_south_2
        amplitudes = (hh, hv, vh, vv)
        vector_1_from_pair, vector_2_from_pair, length_of_amplitudes = _stokes_vector_for_pair_from_amplitudes_hh_hv_vh_vv(amplitudes)
   
        no_error_found = no_error_found and np.round(length_1*length_2,4) == np.round(length_of_amplitudes,4)
        no_error_found = no_error_found and all([np.round(a,4) == np.round(b,4) for a,b in zip(vector_1,vector_1_from_pair)])
        no_error_found = no_error_found and all([np.round(a,4) == np.round(b,4) for a,b in zip(vector_2,vector_2_from_pair)])

        if not no_error_found:
            print(angles,np.round([angle/np.pi for angle in angles],2), indices)
            print(amplitude_north_1, amplitude_south_1, amplitude_north_2, amplitude_south_2)
            print(hh, hv, vh, vv)
            print(length_1,length_2,norm_1, norm_2,np.round(length_1*length_2,4),np.round(length_of_amplitudes,4))
            print(np.round(vector_1,4),np.round(vector_1_from_pair,4))
            print(np.round(vector_2,4),np.round(vector_2_from_pair,4))
    assert no_error_found

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
        res = probability_from_stokes_vectors(vector1, vector2)
        if not np.round(res,4) == np.round(expectation,4):
            no_error_found = False
            print(vector1, vector2, expectation )
    assert no_error_found

def test_create_collection_with_NS_boxes():
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
    initial_states = collection_with_boxes.initial_states_as_list()
    no_error_found = no_error_found and ('00' in initial_states and '33' in initial_states and '23' in initial_states and '11' in initial_states)
    test_state = collection_with_boxes.get_state(initial_state='33')
    oc = test_state.optical_components
    for name,  amp_rep in oc.items():
        no_error_found = no_error_found and (name == '01010101')
        no_error_found = no_error_found and (amp_rep['amplitude'] == 1)
    test_state = collection_with_boxes.get_state(initial_state='02')
    oc = test_state.optical_components
    for name,  amp_rep in oc.items():
        no_error_found = no_error_found and (name == '10100110')
        no_error_found = no_error_found and (amp_rep['amplitude'] == 1)
    super_info = test_state.auxiliary_information['no_signalling_box']
    ms = test_state.measurement_results
    no_error_found = no_error_found and super_info['quantumness_indicator'] == [1,2]
    no_error_found = no_error_found and (ms[0]['results'] == [1,4,3,2])
    for test_state in collection_with_boxes:
        init_state = test_state.initial_state
        configuration = test_state.auxiliary_information['no_signalling_box']['configuration']
        no_error_found = no_error_found and init_state == configuration

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
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1
        if '01011010' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1
        if '10100101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1
        if '01010101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1
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
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1
        if '10010110' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
        if '10010101' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 1
        if '10010110' == res_as_dict['output_state']:
            no_error_found = no_error_found and np.round(res_as_dict['probability'],4) == 0
    assert no_error_found

def test_no_signalling_box_application():
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