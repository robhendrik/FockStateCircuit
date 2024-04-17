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

generate_reference_images = False

def test_quick_delay_between_wave_plates(mocker):
    img1 = "./tests/test_drawings/testdrawing_quick_delay_wave_plates.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:
        circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
        overall_collection = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))

        for delay in range(-15,16):
            circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
            circuit1.half_wave_plate_225(0,1)
            circuit1.time_delay(affected_channels=[0], delay = delay,pulse_width = 5)
            circuit1.half_wave_plate_225(0,1)
            circuit1.measure_optical_to_classical([0,1],[0,1])

            initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
            state1 = fsc.State(collection_of_states=initial_collection_of_states)
            state1.initial_state = 'init:' + str(delay)
            state1.optical_components = [('01',1)]
            initial_collection_of_states.add_state(state1)

            interim_collection_of_states = circuit1.evaluate_circuit(initial_collection_of_states)


            for state in interim_collection_of_states:
                overall_collection.add_state(state)
        overall_collection.plot()
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_full_delay_between_wave_plates(mocker):
    img1 = "./tests/test_drawings/testdrawing_full_delay_wave_plates.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:
        circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
        overall_collection = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))

        for delay in range(-15,16):
            circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
            circuit1.half_wave_plate_225(0,1)
            circuit1.time_delay_full(affected_channels=[0], delay = delay,pulse_width=5)
            circuit1.half_wave_plate_225(0,1)
            circuit1.measure_optical_to_classical([0,1],[0,1])

            initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
            state1 = fsc.State(collection_of_states=initial_collection_of_states)
            state1.initial_state = 'init:' + str(delay)
            state1.optical_components = [('01',1)]
            initial_collection_of_states.add_state(state1)

            interim_collection_of_states = circuit1.evaluate_circuit(initial_collection_of_states)

            for state in interim_collection_of_states:
                overall_collection.add_state(state)
        overall_collection.plot()
        assert plt_test.compare_images(img1, img2, 0.001) is None
        

def test_HOM_quick_delay_classical_control(mocker):
    img1 = "./tests/test_drawings/testdrawing_HOM_quick_delay_classical_control.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:
        circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                        no_of_optical_channels = 4, 
                                        no_of_classical_channels=4,
                                        circuit_name="HOM Effect"
                                        )
        circuit.time_delay_classical_control(affected_channels=[0,1], classical_channel_for_delay=0, bandwidth=1)

        circuit.non_polarizing_50_50_beamsplitter(input_channels_a=[0,1],
                                                input_channels_b=[2,3]
                                                )
        circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], 
                                            classical_channels_to_be_written=[0,1,2,3],
                                            list_of_projections=['1010','2000','0020']
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
        plt.show()
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_HOM_full_delay_classical_control(mocker):
    img1 = "./tests/test_drawings/testdrawing_HOM_full_delay_classical_control.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:
        circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=4,no_of_classical_channels=4)
        circuit.time_delay_classical_control_full(affected_channels=[0],classical_channel_for_delay=0, pulse_width=1)
        circuit.non_polarizing_50_50_beamsplitter(input_channels_a=[0,1],input_channels_b=[2,3])
        circuit.measure_optical_to_classical([0,2],[0,1])


        initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit, input_collection_as_a_dict=dict([]))
        for delay in range(-5,6):
            state1 = fsc.State(collection_of_states=initial_collection_of_states)
            state1.initialize_this_state()
            state1.initial_state = 'init:' + str(delay)
            state1.classical_channel_values[0] = delay
            r = np.sqrt(1/2)
            state1.optical_components = [('2100',r),('0120',-r)]
            state1.cumulative_probability = 1
            initial_collection_of_states.add_state(state1)

        result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
        result.plot()
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_HOM_full_delay_multiple_gates(mocker):
    img1 = "./tests/test_drawings/testdrawing_HOM_full_delay_multiple_gates.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:
        circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
        overall_collection = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))

        for delay in range(-15,16):
            circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
            # delay before first half wave plate should not matter
            circuit1.time_delay_full(affected_channels=[1], delay = 20,pulse_width=10)
            circuit1.half_wave_plate_225(0,1)
            circuit1.time_delay_full(affected_channels=[0], delay = delay,pulse_width=5)
            # delay channel 1 with fixed amount to shift the curve
            circuit1.time_delay_full(affected_channels=[1], delay = 2,pulse_width=10)
            circuit1.half_wave_plate_225(0,1)
            # delay after last half wave plate does not matter
            circuit1.time_delay_full(affected_channels=[1], delay = 12,pulse_width=10)
            circuit1.set_pulse_width(pulse_width=1)
            circuit1.measure_optical_to_classical([0,1],[0,1])

            initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
            state1 = fsc.State(collection_of_states=initial_collection_of_states)
            state1.initial_state = 'init:' + str(delay)
            state1.optical_components = [('01',1)]
            initial_collection_of_states.add_state(state1)

            interim_collection_of_states = circuit1.evaluate_circuit(initial_collection_of_states)

            for state in interim_collection_of_states:
                overall_collection.add_state(state)
        overall_collection.plot()
        assert plt_test.compare_images(img1, img2, 0.001) is None


def test_HOM_full_delay_pulse_width_per_channel(mocker):
    img1 = "./tests/test_drawings/testdrawing_HOM_full_delay_pulse_width_per_channel.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:
        circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
        overall_collection = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
        for delay in range(-15,16):
            circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
            circuit1.half_wave_plate_225(0,1)
            circuit1.time_delay_full(affected_channels=[0], delay = delay,pulse_width=5)
            circuit1.time_delay_full(affected_channels=[1], delay = -3,pulse_width=5)
            circuit1.set_pulse_width(affected_channels=[1],pulse_width=1)
            circuit1.half_wave_plate_225(0,1)
            circuit1.measure_optical_to_classical([0,1],[0,1])

            initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
            state1 = fsc.State(collection_of_states=initial_collection_of_states)
            state1.initial_state = 'init:' + str(delay)
            state1.optical_components = [('01',1)]
            initial_collection_of_states.add_state(state1)

            interim_collection_of_states = circuit1.evaluate_circuit(initial_collection_of_states)

            for state in interim_collection_of_states:
                overall_collection.add_state(state)
        overall_collection.plot()
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_non_physical_time_delay_single_channel(mocker):
    img1 = "./tests/test_drawings/testdrawing_non_physical_time_delay_single_channel.png"
    img2 = img1.replace("testdrawing","reference")
    if not generate_reference_images:
        def save_drawing():
            plt.savefig(img1)
            return
    else:
        def save_drawing():
            plt.savefig(img2)
            return
    with patch("fock_state_circuit.circuit.plt.show", wraps=save_drawing) as mock_bar:

        circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
        overall_collection = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))

        for delay in range(-15,16):
            circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2)
            circuit1.half_wave_plate_225(0,1)
            circuit1.time_delay_full(affected_channels=[0], delay = delay,pulse_width=5)
            circuit1.half_wave_plate_225(0,1)
            circuit1.set_pulse_width(affected_channels=[1],pulse_width=1)
            circuit1.measure_optical_to_classical([0,1],[0,1])
            
            initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit1, input_collection_as_a_dict=dict([]))
            state1 = fsc.State(collection_of_states=initial_collection_of_states)
            state1.initial_state = 'init:' + str(delay)
            state1.optical_components = [('01',1)]
            initial_collection_of_states.add_state(state1)

            interim_collection_of_states = circuit1.evaluate_circuit(initial_collection_of_states)

            for state in interim_collection_of_states:
                overall_collection.add_state(state)
        overall_collection.plot()
        assert plt_test.compare_images(img1, img2, 0.001) is None