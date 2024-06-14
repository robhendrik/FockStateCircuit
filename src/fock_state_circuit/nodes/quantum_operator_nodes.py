from __future__ import annotations
from fock_state_circuit.nodes.nodelist.node_list import NodeList
from fock_state_circuit.quantum_operator_functionality.quantum_operators import QuantumOperator, Hamiltonian
import numpy as np

class QuantumOperatorNodes(NodeList):
    """ Method(s) to generate nonlinear optical nodes which represent physical processes
     
        Last modified: May 31st, 2024      
        """
    _VERSION = '1.0.0'

    def hamiltonian_operator_gate(self,
                    channels: list[int],
                    operators: list[Hamiltonian],
                    node_type: str = 'generic function on collection',
                    node_info: dict = None
                    ) -> None:  
        
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "Hamiltonian",
                    'channels_optical' : [channels],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\hat(H)$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
        
        if not isinstance(operators, list):
            operators = [operators]

        def generic_function(collection,parameters):
            channels = parameters[0]
            operators = parameters[1]

            hamiltonian_operator  = QuantumOperator(operators, 
                                                    length_of_fock_state=collection._length_of_fock_state,
                                                    power_for_exponent_taylor_expansion=25)
            return hamiltonian_operator.apply_operator_to_collection(collection,optical_channels=channels)

        node_to_be_added = {
            'generic_function' : generic_function,
            'generic_function_parameters' : [channels, operators],
            'node_type' : node_type,
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
        return
    
    def wave_plate_from_hamiltonian(self,
                    channel_horizontal: int = 0,
                    channel_vertical: int = 1,
                    theta: float = 0, 
                    phi: float = 0,
                    node_type: str = 'generic function on collection',
                    node_info: dict = None
                    ) -> None:  
        """ Apply a wave plate with phase delay between fast and slow axis of Phi, and a rotation of the axis over Theta to the two channels. 
            If Phi is equal to pi this will be a half wave plate, if Phi is equal to pi/2 this is a quarter wave plate.

        Args:
            channel_horizontal: channel number for horizontal polarization
            channel_vertical: channel number for vertical polarization
            theta: rotation of the axis in radians
            phi: phase difference between fast and slow axis in radians
        
        Returns:
            nothing
        
        Raises:
            nothing
        """
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "wave plate",
                    'channels_optical' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\theta\phi$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
        
        def generic_function(collection,parameters):
            channels = (parameters[0], parameters[1])
            theta = parameters[2]
            phi = parameters[3]

            # we apply an overall phase shift and a sign swap to make the outcome of this operation identical to the outcome of the 'regular' gates
            # for wave plates. This is for consistency in the package.
            # add additional phase shift as very last step
            chi = -phi/2 

            operator_channels = [0,0,1,1]
            ham_phi = Hamiltonian([('+-00', -1j*phi/2),('00+-',+1j*phi/2 )], operator_channels)
            ham_theta_plus = Hamiltonian([('+0-0',theta ),('-0+0',-1*theta )], operator_channels)
            ham_theta_minus = Hamiltonian([('+0-0',-1*theta ),('-0+0',theta )], operator_channels)
            ham_chi = Hamiltonian([('-+00', -1j*chi),('00-+',-1j*chi )], operator_channels)

            operator_generic_wave_plate_list= [ham_theta_minus,ham_phi,ham_theta_plus,ham_chi]

            operator_generic_wave_plate = QuantumOperator(operator_generic_wave_plate_list,
                                                          length_of_fock_state=collection._length_of_fock_state)
            return operator_generic_wave_plate.apply_operator_to_collection(collection,optical_channels=channels)

        node_to_be_added = {
            'generic_function' : generic_function,
            'generic_function_parameters' : [channel_horizontal, channel_vertical, theta, phi],
            'node_type' : 'generic function on collection', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
        return
    
    def wave_plate_from_hamiltonian_classical_control(self, optical_channel_horizontal: int = 0, 
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
        
        def generic_function(collection,parameters):
            channels = (parameters[0], parameters[1])
            classical_channel_for_theta, classical_channel_for_phi = parameters[2], parameters[3]
            output_collection = collection.copy(empty_template = True)
            for state in collection:
                # we apply an overall phase shift and a sign swap to make the outcome of this operation identical to the outcome of the 'regular' gates
                # for wave plates. This is for consistency in the package.
                phi = state.classical_channel_values[classical_channel_for_phi]
                theta = state.classical_channel_values[classical_channel_for_theta]


                # add additional phase shift as very last step
                chi = -phi/2 

                operator_channels = [0,0,1,1]
                ham_phi = Hamiltonian([('+-00', -1j*phi/2),('00+-',+1j*phi/2 )], operator_channels)
                ham_theta_plus = Hamiltonian([('+0-0',theta ),('-0+0',-1*theta )], operator_channels)
                ham_theta_minus = Hamiltonian([('+0-0',-1*theta ),('-0+0',theta )], operator_channels)
                ham_chi = Hamiltonian([('-+00', -1j*chi),('00-+',-1j*chi )], operator_channels)

                operator_generic_wave_plate_list= [ham_theta_minus,ham_phi,ham_theta_plus,ham_chi]
        

                operator_generic_wave_plate = QuantumOperator( operator_generic_wave_plate_list, 
                                                            length_of_fock_state=collection._length_of_fock_state,
                                                            power_for_exponent_taylor_expansion=25)
                new_state = operator_generic_wave_plate.apply_operator_to_state(state,optical_channels = channels)
                output_collection.add_state(state=new_state)
       
            return output_collection

        node_to_be_added = {
            'generic_function' : generic_function,
            'generic_function_parameters' : [optical_channel_horizontal, optical_channel_vertical, classical_channel_for_orientation,classical_channel_for_phase_shift],
            'node_type' : 'generic function on collection', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
        return