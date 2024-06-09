from __future__ import annotations
from fock_state_circuit.nodes.nodelist.node_list import NodeList
import math

class ControlledNodes(NodeList):  
    """Method(s) to create an optical node which reads settings from the classical channels

            phase_shift_single_channel_classical_control(self, 
                    optical_channel_to_shift: int = 0, 
                    classical_channel_for_phase_shift: int = 0, 
                    node_info: dict = None
                    ) -> None

                    Apply phase shift to a specified channel of the fock state circuit. The phase shift (in radians) is read from 
                    the given classical channel.
                    Example: circuit.phase_shift_single_channel_classical_control(optical_channel_to_shift = 2, classical_channel_for_phase_shift = 1) 
                    would read classical channel '1'. If that channel has value 'pi' a phase shift of 'pi' is applied to optical channel 2

            wave_plate_classical_control(self, optical_channel_horizontal: int = 0, 
                                     optical_channel_vertical: int = 1, 
                                     classical_channel_for_orientation: int = 0, 
                                     classical_channel_for_phase_shift: int = 0,
                                     node_info: dict = None) -> None:   
                                     
                    Apply a wave plate to specified channels of the fock state circuit getting the phase shift and plate orientation 
                    from a the classical channels. 
                    Example: wave_plate_classical_control(self, channel_horizontal = 3, 
                                channel_vertical = 2, 
                                classical_channel_for_orientation = 0, 
                                classical_channel_for_phase_shift = 2)
                    would read a orientation angel 'theta' from classical channel 0 and a phase shift 'phi' from classical channel 2. This
                    would be applied as a wave plate to optical channels 3 and 2).

        Last modified: April 16th, 2024
    """
    _VERSION = '1.0.0'
    def phase_shift_single_channel_classical_control(self, 
                                                     optical_channel_to_shift: int = 0, 
                                                     classical_channel_for_phase_shift: int = 0, 
                                                     node_info: dict = None) -> None:   
        """ Apply phase shift to a specified channel of the fock state circuit. The phase shift (in radians) is read from 
            the given classical channel.
            Example: circuit.phase_shift_single_channel_classical_control(optical_channel_to_shift = 2, classical_channel_for_phase_shift = 1) 
            would read classical channel '1'. If that channel has value 'pi' a phase shift of 'pi' is applied to optical channel 2
                        
        Args:
            optical_channel_to_shift: channel to apply phase shift to
            classical_channel_for_phase_shift: classical control channel to read phase shift from
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            exception if the requested channels are not available
        """ 
        if classical_channel_for_phase_shift > self._no_of_classical_channels:
            raise Exception('the classical channel is a not defined channel in the circuit')
        if optical_channel_to_shift > self._no_of_optical_channels:
            raise Exception('the optical channel is a not defined channel in the circuit')

        if node_info == None: node_info = {}
        node_info = {
                'label' : "phase-shift(c)",
                'channels_optical' : [optical_channel_to_shift],
                'channels_classical' : [classical_channel_for_phase_shift],
                'markers' : ['o'],
                'markercolor' : ['pink'],
                'markerfacecolor' : ['pink'],
                'marker_text' : [r"$\phi$"],
                'markersize' : 20,
                'fillstyle' : 'full',
                'classical_marker_text' : [r"$c$"],
                'classical_marker_text_fontsize' : [5]
            }|node_info
        # we use a generic rotation with theta is zero, so phase shift is applied to the second ('the vertical') channel. 
        # so channel_to_be_shifted has to be the vertical channel. Another option would be to rotate the phase plate over 
        # 90 degrees with theta and apply phase shift to horizontal channel
        # channel_for_shift has to become the 'channel_vertical'. This is the channel to which the phase shift is applied.
        if optical_channel_to_shift == 0:
            channel_horizontal, channel_vertical = 1,0
        else:
            channel_horizontal, channel_vertical = 0,optical_channel_to_shift

        channel_numbers = [channel_horizontal, channel_vertical]
        tensor_list = self._translate_channels_numbers_to_tensor_list(channel_numbers = channel_numbers)

        control_parameters =    {'variables_to_be_controlled' : ['phi'],
                                'function_parameters'  : {'theta': 0, 'phi': 0},
                                'classical_control_channels' : [classical_channel_for_phase_shift]
                                } 

        node_to_be_added = {'function' : '__generic_optical_matrix(theta, phi)',
                                'control_parameters': control_parameters,
                                'tensor_list' : tensor_list,
                                'node_type' : 'controlled',
                                'node_info' : node_info
                                }
        #self.__apply_generic_rotation_2_channels(channel_horizontal, channel_for_shift, theta = 0, phi = phase_shift, node_info = node_info)

        self._update_list_of_nodes(node_to_be_added)

        return
    
    def wave_plate_classical_control(self, optical_channel_horizontal: int = 0, 
                                     optical_channel_vertical: int = 1, 
                                     classical_channel_for_orientation: int = 0, 
                                     classical_channel_for_phase_shift: int = 0,
                                     node_info: dict = None) -> None:   
        """ Apply a wave plate to specified channels of the fock state circuit getting the phase shift and plate orientation 
            from a the classical channels. 
            Example: wave_plate_classical_control(self, channel_horizontal = 3, 
                        channel_vertical = 2, 
                        classical_channel_for_orientation = 0, 
                        classical_channel_for_phase_shift = 2)
            would read a orientation angel 'theta' from classical channel 0 and a phase shift 'phi' from classical channel 2. This
            would be applied as a wave plate to optical channels 3 and 2).
                        
        Args:
            optical_channel_horizontal: channel number for horizontal polarization
            optical_channel_vertical: channel number for vertical polarization
            classical_channel_for_orientation: classical control channel to read orientation angle 'theta' (radians) from
            classical_channel_for_phase_shift: classical control channel to read phase shift 'phi' (radians) from
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            exception if the requested channels are not available
        """ 
        if max(classical_channel_for_phase_shift, classical_channel_for_orientation) > self._no_of_classical_channels:
            raise Exception('the classical channel is a not defined channel in the circuit')
        if max(optical_channel_horizontal, optical_channel_vertical) > self._no_of_optical_channels:
            raise Exception('the optical channel is a not defined channel in the circuit')
        
        if node_info == None: node_info = {}
        node_info = {
                'label' : "wave_plate(c)",
                'channels_optical' : [optical_channel_horizontal, optical_channel_vertical],
                'channels_classical' : [classical_channel_for_phase_shift, classical_channel_for_orientation],
                'markers' : ['o'],
                'markercolor' : ['pink'],
                'markerfacecolor' : ['pink'],
                'marker_text' : [r"$\theta$"],
                'markersize' : 20,
                'fillstyle' : 'full',
                'classical_marker_text' : [r"$c$"],
                'classical_marker_text_fontsize' : [5]
            }|node_info
        
        # add to list of nodes
        channel_numbers = [optical_channel_horizontal, optical_channel_vertical]
        tensor_list = self._translate_channels_numbers_to_tensor_list(channel_numbers = channel_numbers)

        control_parameters =    {'variables_to_be_controlled' : ['theta', 'phi'],
                                'function_parameters'  : {'theta': math.pi/2, 'phi': 0},
                                'classical_control_channels' : [classical_channel_for_orientation, classical_channel_for_phase_shift]
                                } 

        node_to_be_added = {'function' : '__generic_optical_matrix(theta, phi)',
                                'control_parameters': control_parameters,
                                'tensor_list' : tensor_list,
                                'node_type' : 'controlled',
                                'node_info' : node_info
                                }

        self._update_list_of_nodes(node_to_be_added = node_to_be_added)

        return