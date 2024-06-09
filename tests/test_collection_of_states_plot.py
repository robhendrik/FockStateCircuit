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

def test_plot_bars_1():
    img1 = "./tests/test_drawings/testdrawing_plot_full_bar_graph.png"
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
                                no_of_classical_channels=4
                                )
        circuit.non_polarizing_50_50_beamsplitter(input_channels_a=(0,1), input_channels_b=(2,3))
        circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], classical_channels_to_be_written=[0,1,2,3])

        initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
        initial_collection_of_states.filter_on_initial_state(initial_state_to_filter=['1000', '0010', '1010'])

        result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)

        result.plot()

        result.plot()
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_plot_bars_2():
    img1 = "./tests/test_drawings/testdrawing_plot_selected_bar_graph.png"
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
                                no_of_classical_channels=4
                                )
        circuit.non_polarizing_50_50_beamsplitter(input_channels_a=(0,1), input_channels_b=(2,3))
        circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], classical_channels_to_be_written=[0,1,2,3])

        initial_collection_of_states = fsc.CollectionOfStates(fock_state_circuit=circuit)
        initial_collection_of_states.filter_on_initial_state(initial_state_to_filter=['1000', '0010', '1010'])

        result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)

        result.plot()

        result.plot(classical_channels=[0,2])
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_correlations():
    no_error_found = True
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

    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3],classical_channels_to_be_written=[2,3,4,5])

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

    initial_collection_of_states.clear()
    for key, setting in polarization_settings.items():
        state = entangled_state.copy()
        state.classical_channel_values = [setting['left'],setting['right'], 0, math.pi,0,0]
        state.initial_state = 'entangled_state_' + key
        state.cumulative_probability = 1
        initial_collection_of_states.add_state(state)

    result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
    channel_combis_for_correlation = [(2,3),(4,5),(2,4),(3,5)]
    correlations = result.plot_correlations(channel_combis_for_correlation, 
                                            correlation_output_instead_of_plot=True,
                                            initial_states_to_assess=['entangled_state_ab'])
    no_error_found = no_error_found and len(correlations) == 1
    values = correlations['entangled_state_ab']
    no_error_found = no_error_found and np.round(values[0],4) == -1
    no_error_found = no_error_found and np.round(values[1],4) == -1
    no_error_found = no_error_found and np.round(values[2],4) == np.round(np.sqrt(1/2),4)
    no_error_found = no_error_found and np.round(values[3],4) == np.round(np.sqrt(1/2),4)

    correlations = result.plot_correlations(channel_combis_for_correlation, 
                                            correlation_output_instead_of_plot=True)
    no_error_found = no_error_found and len(correlations) == 4
    values = correlations["entangled_state_a'b"]
    no_error_found = no_error_found and np.round(values[0],4) == -1
    no_error_found = no_error_found and np.round(values[1],4) == -1
    no_error_found = no_error_found and np.round(values[2],4) == -1*np.round(np.sqrt(1/2),4)
    no_error_found = no_error_found and np.round(values[3],4) == -1*np.round(np.sqrt(1/2),4)

    assert no_error_found

def test_plot_correlations():
    img1 = "./tests/test_drawings/testdrawing_plot_correlations.png"
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

        circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3],classical_channels_to_be_written=[2,3,4,5])

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

        initial_collection_of_states.clear()
        for key, setting in polarization_settings.items():
            state = entangled_state.copy()
            state.classical_channel_values = [setting['left'],setting['right'], 0, math.pi,0,0]
            state.initial_state = 'entangled_state_' + key
            state.cumulative_probability = 1
            initial_collection_of_states.add_state(state)

        result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)
        channel_combis_for_correlation = [(2,3),(4,5),(2,4),(3,5)]
        result.plot_correlations(channel_combis_for_correlation)
        assert plt_test.compare_images(img1, img2, 0.001) is None