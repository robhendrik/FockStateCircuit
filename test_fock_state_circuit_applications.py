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

def test_GHZ_state_curves():
    preparation = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 4,
                                    no_of_classical_channels=1,
                                    circuit_name = 'Preparation'
                                    )

    circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                    no_of_optical_channels = 8,
                                    no_of_classical_channels=4,
                                    circuit_name = 'GHZ creation'
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


    fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,10))
    fig.suptitle('Simulation of GHZ state creation')
    ax1.set_title("D1 at +45 degree, D2 at -45 degree")
    ax1.set_xlabel('time delay channel a')
    ax1.set_ylabel('4 fold correlation')
    ax1.set_ylim(0,0.08)

    ax2.set_title("D1 at +0 degree, D2 at -45 degree")
    ax2.set_xlabel('time delay channel a')
    ax2.set_ylabel('4 fold correlation')
    ax2.set_ylim(0,0.08)

    circuit_list = [preparation, circuit, detection_1]
    compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)
    result = compound_circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states_curve)
    histo = result.plot(histo_output_instead_of_plot=True)

    probs_detection = []
    for n in delays:
        if 'delay' + str(n) in histo.keys():
            value = histo['delay' + str(n)]
            for datapoint in value:
                if datapoint['output_state'] == '1111':
                    probs_detection.append(datapoint['probability'])
        else:
            probs_detection.append(0)
    probs_detection1 = probs_detection

    circuit_list = [preparation, circuit, detection_2]
    compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)
    result = compound_circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states_curve)
    histo = result.plot(histo_output_instead_of_plot=True)


    probs_detection = []
    for n in delays:
        if 'delay' + str(n) in histo.keys():
            value = histo['delay' + str(n)]
            for datapoint in value:
                if datapoint['output_state'] == '1111':
                    probs_detection.append(datapoint['probability'])
        else:
            probs_detection.append(0)
    probs_detection2 = probs_detection

    circuit_list = [preparation, circuit, detection_3]
    compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)
    result = compound_circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states_curve)
    histo = result.plot(histo_output_instead_of_plot=True)

    probs_detection = []
    for n in delays:
        if 'delay' + str(n) in histo.keys():
            value = histo['delay' + str(n)]
            for datapoint in value:
                if datapoint['output_state'] == '1111':
                    probs_detection.append(datapoint['probability'])
        else:
            probs_detection.append(0)
    probs_detection3 = probs_detection

    circuit_list = [preparation, circuit, detection_4]
    compound_circuit = fsc.CompoundFockStateCircuit(circuit_list)
    result = compound_circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states_curve)
    histo = result.plot(histo_output_instead_of_plot=True)

    probs_detection = []
    for n in delays:
        if 'delay' + str(n) in histo.keys():
            value = histo['delay' + str(n)]
            for datapoint in value:
                if datapoint['output_state'] == '1111':
                    probs_detection.append(datapoint['probability'])
        else:
            probs_detection.append(0)
    probs_detection4 = probs_detection

    ax1.plot(delays,probs_detection1, label = "detector D3 at +45 deg")
    ax1.plot(delays,probs_detection2, label = "detector D3 at -45 deg")
    ax2.plot(delays,probs_detection3, label = "detector D3 at +45 deg")
    ax2.plot(delays,probs_detection4, label = "detector D3 at -45 deg")
    ax2.legend()
    ax1.legend()
    img1 = "./tests/test_drawings/testdrawing_GHZ_curves.png"
    plt.savefig(img1)
    plt.close()
    img2 = img1.replace("testdrawing","reference")
    assert plt_test.compare_images(img1, img2, 0.001) is None

def test_HOM_with_delay_curves():
    circuit = fsc.FockStateCircuit(length_of_fock_state = 5, 
                                    no_of_optical_channels = 4, 
                                    no_of_classical_channels=4,
                                    circuit_name="HOM Effect"
                                    )
    circuit.time_delay_classical_control(affected_channels=[0,1], classical_channel_for_delay=0, bandwidth=1)

    circuit.non_polarizing_50_50_beamsplitter(input_channels_a=[0,1],
                                            input_channels_b=[2,3]
                                            )
    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                        classical_channels_to_be_written=[0,1,2,3]
                                        )

    initial_collection_of_states = fsc.CollectionOfStates(
                                    fock_state_circuit=circuit, 
                                    input_collection_as_a_dict=dict([])
                                    )
    delays = [ (number-50)/10.0 for number in range(100)]
    for delay in delays:
        input_two_photons = fsc.State(collection_of_states=initial_collection_of_states)
        input_two_photons.optical_components = [('1010', 1)]
        input_two_photons.classical_channel_values = [delay,0,0,0]
        input_two_photons.initial_state = "\'1010\'\ndelay = " + str(delay)
        initial_collection_of_states.add_state(state=input_two_photons)

    result = circuit.evaluate_circuit(initial_collection_of_states)

    histo = result.plot(histo_output_instead_of_plot=True)

    delays, bunched, anti_bunched  = [], [], []
    for key, values in histo.items():
        delays.append(float(key.split("=")[-1]))
        bunched.append(0)
        anti_bunched.append(0)
        for value in values:
                if '2' in value['output_state']:
                    bunched[-1] += value['probability']
                else:
                    anti_bunched[-1] += value['probability']
    anti_bunched = [i/(i+j) for i,j in zip(anti_bunched, bunched)]
    fig, ax = plt.subplots(figsize = (16,10))
    ax.plot(delays, anti_bunched)

    ax.set(xlabel='time delay / coherence length', ylabel='count',
        title='HOM dip: reduced observation of photon in separate output ports when wavepackets overlap')
    ax.grid()

    img1 = "./tests/test_drawings/testdrawing_HOM_curves.png"
    plt.savefig(img1)
    plt.close()
    img2 = img1.replace("testdrawing","reference")
    assert plt_test.compare_images(img1, img2, 0.001) is None


def test_decoherence_time():
    trace_horver = []
    trace_diagonal = []
    trace_circular = []
    gs = [i/10 for i in range(11)]
    outcomes = dict([])
    for coupling in ['circular', 'diagonal', 'horver']:
        trace_horver = []
        trace_diagonal = []
        trace_circular = []
        for g in [i/10 for i in range(11)]:
            circuit_preparation = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                            no_of_optical_channels = 2,
                                            no_of_classical_channels=0,
                                            circuit_name = "prep"
                                            )

            circuit_preparation.half_wave_plate(channel_horizontal=0, channel_vertical=1, angle = 0)
            circuit_decoherence = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                            no_of_optical_channels = 4,
                                            no_of_classical_channels=0,
                                            circuit_name = "decoherence"
                                            )
            circuit_preparation.bridge(next_fock_state_circuit=circuit_decoherence)
            if coupling == 'circular':
                circuit_decoherence.quarter_wave_plate_45(channel_horizontal=0, channel_vertical=1)
                circuit_decoherence.channel_coupling(control_channels=[0,1], target_channels=[2,3], coupling_strength=g)
                circuit_decoherence.quarter_wave_plate_45(channel_horizontal=1, channel_vertical=0)
            elif coupling == 'diagonal':
                circuit_decoherence.half_wave_plate_225(channel_horizontal=0, channel_vertical=1)
                circuit_decoherence.channel_coupling(control_channels=[0,1], target_channels=[2,3], coupling_strength=g)
                circuit_decoherence.half_wave_plate_225(channel_horizontal=1, channel_vertical=0)
            else:
                circuit_decoherence.channel_coupling(control_channels=[0,1], target_channels=[2,3], coupling_strength=g)
            circuit_analysis = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                            no_of_optical_channels = 2,
                                            no_of_classical_channels=0,
                                            circuit_name = "analyze"
                                            )
            circuit_decoherence.bridge(next_fock_state_circuit=circuit_analysis)
            circuit_analysis.half_wave_plate(channel_horizontal=0, channel_vertical=1, angle = 0)
 
            amp = 1/np.sqrt(2)
            initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit_preparation, input_collection_as_a_dict=dict([]))
            state = fsc.State(collection_of_states=initial_collection_of_states)
            state.initial_state = 'horizontal'
            state.optical_components= [('10',1)]
            initial_collection_of_states.add_state(state=state)

            state = fsc.State(collection_of_states=initial_collection_of_states)
            state.initial_state = 'diagonal'
            state.optical_components= [('10',amp), ('01',-1*amp)]
            initial_collection_of_states.add_state(state=state)

            state = fsc.State(collection_of_states=initial_collection_of_states)
            state.initial_state = 'circular'
            state.optical_components= [('10',np.cdouble(amp)), ('01',np.cdouble(1j*amp))]
            initial_collection_of_states.add_state(state=state)

            result = circuit_preparation.evaluate_circuit(collection_of_states_input=initial_collection_of_states)

            dm = result.density_matrix()
            trace_horver.append(dm['horizontal']['trace_dm_squared'])
            trace_diagonal.append(dm['diagonal']['trace_dm_squared'])
            trace_circular.append(dm['circular']['trace_dm_squared'])
        traces = dict([])
        traces.update({'horver': trace_horver})
        traces.update({'diagonal':trace_diagonal})
        traces.update({'circular':trace_circular})
        outcomes.update({coupling:traces})

    plt.rcParams['figure.figsize'] = (20, 20)
    fig, ax = plt.subplots(3,1,figsize = (16,10))
    titles = ['c-shift in circular basis', 'c-shift on diagonal basis', 'c-shift on horizontal/vertical basis']
    for index,coupling in enumerate(['circular', 'diagonal', 'horver']):
        ax[index].plot(gs,outcomes[coupling]['horver'], marker = 'o', label = 'horizontal/vertical polarization')
        ax[index].plot(gs,outcomes[coupling]['diagonal'], marker = 'o', label = 'diagonal polarization')
        ax[index].plot(gs,outcomes[coupling]['circular'], marker = 'o', label = 'circular polarization')
        ax[index].set(xlabel="Coupling paramater g", ylabel="trace of dm-squared", title = titles[index])
        ax[index].legend()
    
    img1 = "./tests/test_drawings/testdrawing_decoherence_curves.png"
    plt.savefig(img1)
    plt.close()
    img2 = img1.replace("testdrawing","reference")
    assert plt_test.compare_images(img1, img2, 0.001) is None

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
    initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
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