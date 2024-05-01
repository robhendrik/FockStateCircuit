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

def test_simple_draw(mocker):
    img1 = "./tests/test_drawings/testdrawing_simple_draw.png"
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
        #mocker.patch('fock_state_circuit.circuit.plt.show', return_value=None)
        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                    no_of_optical_channels = 3, 
                                    no_of_classical_channels=2
                                    )
        circuit1.c_shift(control_channels=[0], target_channels=[1])
        circuit1.draw()
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_all_nodes(mocker):
    img1 = "./tests/test_drawings/testdrawing_all_nodes.png"
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
        #mocker.patch('fock_state_circuit.circuit.plt.show', return_value=None)
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

        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_all_nodes_custom_appearance(mocker):
    img1 = "./tests/test_drawings/testdrawing_all_nodes_custom_appearance.png"
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
        #mocker.patch('fock_state_circuit.circuit.plt.show', return_value=None)
        circuit1 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                            no_of_optical_channels = 4, 
                            no_of_classical_channels=3
                            )
        node_info = {
            'markers' : [r'$?$'],
            'markercolor' : ['yellow'],
            'marker_text_fontsize': [5],
            'marker_text_color': ['white'],
            'markersize' : [30],
            'markeredgewidth' : 1,
            'fillstyle' : ['full'],
            'classical_marker_color' : ['purple'],
            'classical_marker' : [r'$!$'],
            'classical_marker_size' : ['10'],
            'classical_marker_text' : [''],
            'classical_marker_text_color' : ['white'],
            'classical_marker_text_fontsize': [25],
            'combined_gate': 'single'
        }
        circuit1.half_wave_plate(channel_horizontal=0,channel_vertical=1, angle=0)
        circuit1.half_wave_plate(channel_horizontal=0,channel_vertical=1, angle=0, node_info=node_info)
        circuit1.half_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit1.half_wave_plate_45(channel_horizontal=0,channel_vertical=1, node_info=node_info)
        circuit1.half_wave_plate_225(channel_horizontal=0,channel_vertical=1)
        circuit1.half_wave_plate_225(channel_horizontal=0,channel_vertical=1, node_info=node_info)
        circuit1.quarter_wave_plate(channel_horizontal=0,channel_vertical=1, angle=0)
        circuit1.quarter_wave_plate(channel_horizontal=0,channel_vertical=1, angle=0, node_info=node_info)
        circuit1.wave_plate_classical_control(optical_channel_horizontal=0,optical_channel_vertical=1, classical_channel_for_orientation=0,classical_channel_for_phase_shift=1)
        circuit1.wave_plate_classical_control(optical_channel_horizontal=0,optical_channel_vertical=1, classical_channel_for_orientation=0,classical_channel_for_phase_shift=1,node_info=node_info)
        circuit1.phase_shift_single_channel(channel_for_shift=0,phase_shift=0)
        circuit1.phase_shift_single_channel(channel_for_shift=0,phase_shift=0, node_info=node_info)
        circuit1.phase_shift_single_channel_classical_control(optical_channel_to_shift=1,classical_channel_for_phase_shift=0)
        circuit1.phase_shift_single_channel_classical_control(optical_channel_to_shift=1,classical_channel_for_phase_shift=0, node_info=node_info)
        circuit1.polarizing_beamsplitter(input_channels_a=(0,1), input_channels_b=(2,3))
        circuit1.polarizing_beamsplitter(input_channels_a=(0,1), input_channels_b=(2,3), node_info=node_info)
        circuit1.mix_generic_refl_transm(first_channel=0,second_channel=1,reflection=0.5,transmission=0.5)
        circuit1.mix_generic_refl_transm(first_channel=0,second_channel=1,reflection=0.5,transmission=0.5, node_info=node_info)
        circuit2 = fsc.FockStateCircuit(length_of_fock_state = 2, 
                                        no_of_optical_channels = 4, 
                                        no_of_classical_channels=3,
                                        circuit_name='circuit\nfor fun'
                                        )
        circuit1.bridge(next_fock_state_circuit=circuit2)
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2], classical_channels_to_be_written=[0,1,2])
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2], classical_channels_to_be_written=[0,1,2], node_info=node_info)
        def test_function(current_values, new_input_values, affected_channels):
            output_list = current_values[::-1] 
            return output_list
        circuit2.classical_channel_function(function=test_function, affected_channels=[0,1])
        circuit2.classical_channel_function(function=test_function, affected_channels=[0,1], node_info=node_info)
        circuit2.set_classical_channels(list_of_values_for_classical_channels=[1,1], list_of_classical_channel_numbers=[0,1])
        circuit2.set_classical_channels(list_of_values_for_classical_channels=[1,1], list_of_classical_channel_numbers=[0,1], node_info=node_info)
        matrix = np.identity(2, dtype = np.cdouble)
        circuit2.custom_optical_node(matrix_optical=matrix,optical_channels=[0,1])
        circuit2.custom_optical_node(matrix_optical=matrix,optical_channels=[0,1], node_info=node_info)
        matrix = np.identity(len(circuit2.basis()), dtype = np.cdouble)
        circuit2.custom_fock_state_node(custom_fock_matrix=matrix)
        circuit2.custom_fock_state_node(custom_fock_matrix=matrix, node_info=node_info)
        circuit2.channel_coupling(control_channels=[0,1], target_channels=[2,3], coupling_strength=1)
        circuit2.channel_coupling(control_channels=[0,1], target_channels=[2,3], coupling_strength=1, node_info=node_info)
        circuit2.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=1)
        circuit2.quarter_wave_plate_45(channel_horizontal=0,channel_vertical=1, node_info=node_info)
        circuit2.time_delay(affected_channels=[0])
        circuit2.time_delay(affected_channels=[0],node_info=node_info)
        circuit2.time_delay_classical_control(affected_channels=[0],classical_channel_for_delay=0)
        circuit2.time_delay_classical_control(affected_channels=[0],classical_channel_for_delay=0,node_info=node_info)
        def generic_function(input_collection,parameters):
            for state in input_collection:
                state.classical_channels = parameters[:2]
            return input_collection

        circuit2.generic_function_on_collection(function=generic_function, parameters=[1,2])
        circuit2.generic_function_on_collection(function=generic_function, parameters=[1,2],node_info=node_info)

        circuit1.draw()

        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_combined_nodes(mocker):
    img1 = "./tests/test_drawings/testdrawing_combined_nodes.png"
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
        #mocker.patch('fock_state_circuit.circuit.plt.show', return_value=None)
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

        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_custom_settings(mocker):
    img1 = "./tests/test_drawings/testdrawing_custom_settings.png"
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
        #mocker.patch('fock_state_circuit.circuit.plt.show', return_value=None)
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

        assert plt_test.compare_images(img1, img2, 0.001) is None


def test_draw_bridges(mocker):
    img1 = "./tests/test_drawings/testdrawing_bridges.png"
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
    #mocker.patch('fock_state_circuit.circuit.plt.show', return_value=None)

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

        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_compound_circuit(mocker):
    img1 = "./tests/test_drawings/testdrawing_compound_circuit.png"
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
    #mocker.patch('fock_state_circuit.circuit.plt.show', return_value=None)

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


        assert plt_test.compare_images(img1, img2, 0.001) is None


def test_draw_complete_circuit_with_stations(mocker):
    img1 = "./tests/test_drawings/testdrawing_complete_circuit_with_stations.png"
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
        circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                            no_of_optical_channels = 6,
                            no_of_classical_channels=2,
                            circuit_name = 'test'
                            )
        for n in range(11):
            circuit.half_wave_plate_225(channel_horizontal=0,channel_vertical=1)
        circuit.phase_shift_single_channel_classical_control(optical_channel_to_shift=2,classical_channel_for_phase_shift=1)
        circuit.set_classical_channels(list_of_values_for_classical_channels=[1,1],list_of_classical_channel_numbers=[0,1])
        circuit.half_wave_plate(0,1)

        circuit2 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                no_of_optical_channels = 7,
                                no_of_classical_channels=4,
                                circuit_name = 'test2'
                                )
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0], classical_channels_to_be_written=[0])
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[3], classical_channels_to_be_written=[3])
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0,3], classical_channels_to_be_written=[0,3])
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0,1], classical_channels_to_be_written=[2,3])

        node_info = {'label' : "test1",'channels_optical': [0,1,2,3,4,5,6], 'channels_classical': [0,1,2,3] }
        circuit2.custom_fock_state_node(custom_fock_matrix=None,node_info = node_info)

        circuit2.half_wave_plate_225(channel_horizontal=2,channel_vertical=3)
        circuit2.phase_shift_single_channel_classical_control(optical_channel_to_shift=2,classical_channel_for_phase_shift=1)
        circuit2.set_classical_channels(list_of_values_for_classical_channels=[2,2],list_of_classical_channel_numbers=[2,3])
        node_info = {'label' : "test2",'channels_optical': [0,1,2,3,4,5,6], 'channels_classical': [0,1,2,3], 'combined_gate': 'NPBS' }

        circuit2.custom_fock_state_node(custom_fock_matrix=None,node_info = node_info)

        node_info = {'label' : "test2",'combined_gate': 'NPBS' }
        circuit2.half_wave_plate_225(channel_horizontal=2,channel_vertical=3, node_info=node_info)
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], classical_channels_to_be_written=[0,1,2,3])
        circuit2.phase_shift_single_channel_classical_control(optical_channel_to_shift=6,classical_channel_for_phase_shift=0)
        circuit.bridge(next_fock_state_circuit=circuit2)
        station_channels = {'bob': {'optical_channels': [0,1], 'classical_channels': [0,1]}, 
                        'alice': {'optical_channels': [2,3], 'classical_channels': [2,3]},
                        'charlie':{'optical_channels': [6]}}
        settings = circuit.create_settings_for_drawing_stations(stations={'station_channels':station_channels})['total_circuit']

        circuit.draw(settings_for_drawing_circuit=settings)
        assert plt_test.compare_images(img1, img2, 0.001) is None

def test_draw_circuit_with_single_station(mocker):
    img1 = "./tests/test_drawings/testdrawing_circuit_with_single_station.png"
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
        circuit = fsc.FockStateCircuit( length_of_fock_state = 3, 
                            no_of_optical_channels = 6,
                            no_of_classical_channels=2,
                            circuit_name = 'test'
                            )
        for n in range(11):
            circuit.half_wave_plate_225(channel_horizontal=0,channel_vertical=1)
        circuit.half_wave_plate_225(channel_horizontal=1,channel_vertical=2)
        circuit.phase_shift_single_channel_classical_control(optical_channel_to_shift=2,classical_channel_for_phase_shift=1)
        circuit.set_classical_channels(list_of_values_for_classical_channels=[1,1],list_of_classical_channel_numbers=[0,1])
        circuit.half_wave_plate(0,1)

        circuit2 = fsc.FockStateCircuit( length_of_fock_state = 3, 
                                no_of_optical_channels = 7,
                                no_of_classical_channels=4,
                                circuit_name = 'test2'
                                )
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0], classical_channels_to_be_written=[0])
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[3], classical_channels_to_be_written=[3])
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0,3], classical_channels_to_be_written=[0,3])
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0,1], classical_channels_to_be_written=[2,3])

        node_info = {'label' : "test1",'channels_optical': [0,1,2,3,4,5,6], 'channels_classical': [0,1,2,3] }
        circuit2.custom_fock_state_node(custom_fock_matrix=None,node_info = node_info)

        circuit2.half_wave_plate_225(channel_horizontal=2,channel_vertical=3)
        circuit2.phase_shift_single_channel_classical_control(optical_channel_to_shift=2,classical_channel_for_phase_shift=1)
        circuit2.set_classical_channels(list_of_values_for_classical_channels=[2,2],list_of_classical_channel_numbers=[2,3])
        node_info = {'label' : "test2",'channels_optical': [0,1,2,3,4,5,6], 'channels_classical': [0,1,2,3], 'combined_gate': 'NPBS' }

        circuit2.custom_fock_state_node(custom_fock_matrix=None,node_info = node_info)

        node_info = {'label' : "test2",'combined_gate': 'NPBS' }
        circuit2.half_wave_plate_225(channel_horizontal=2,channel_vertical=3, node_info=node_info)
        circuit2.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2,3], classical_channels_to_be_written=[0,1,2,3])
        circuit2.phase_shift_single_channel_classical_control(optical_channel_to_shift=6,classical_channel_for_phase_shift=0)
        circuit.bridge(next_fock_state_circuit=circuit2)
        station_channels = {'bob': {'optical_channels': [0,1], 'classical_channels': [0,1]}, 
                        'alice': {'optical_channels': [2,3], 'classical_channels': [2,3]},
                        'charlie':{'optical_channels': [6]}}
        settings = circuit.create_settings_for_drawing_stations(stations={'station_channels':station_channels})['total_circuit']

        stations = {'station_to_draw': 'alice','station_channels': station_channels}
        circuit.draw_station(stations=stations)
        assert plt_test.compare_images(img1, img2, 0.001) is None


def test_draw_circuit_with_timedelay_and_pulsewidth(mocker):
    img1 = "./tests/test_drawings/testdrawing_circuit_with_timedelay_and_pulsewidth.png"
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
        circuit1 = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=2,no_of_classical_channels=2,circuit_name='circuit with time delays')
        # delay before first half wave plate should not matter
        circuit1.time_delay_classical_control(affected_channels=[1], classical_channel_for_delay=0,pulse_width=10)
        circuit1.half_wave_plate_225(0,1)
        circuit1.time_delay_full(affected_channels=[0], delay = 2,pulse_width=5)
        # delay channel 1 with fixed amount to shift the curve
        circuit1.time_delay_classical_control_full(affected_channels=[1],classical_channel_for_delay=1,pulse_width=10)
        circuit1.half_wave_plate_225(0,1)
        # delay after last half wave plate does not matter
        circuit1.time_delay_full(affected_channels=[1], delay = 12,pulse_width=10)
        circuit1.set_pulse_width(pulse_width=1)
        circuit1.set_pulse_width(affected_channels=[1],pulse_width=1)
        circuit1.set_pulse_width(affected_channels=[],pulse_width=1)
        circuit1.measure_optical_to_classical([0,1],[0,1])
        circuit1.draw()
        assert plt_test.compare_images(img1, img2, 0.001) is None