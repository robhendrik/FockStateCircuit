from __future__ import annotations
from fock_state_circuit.nodes.nodelist.node_list import NodeList
from fock_state_circuit.popescu_rohrlich_correlation_functionality.popescu_rohrlich_correlations import _create_collection_with_popescu_rohrlich_correlations
from fock_state_circuit.popescu_rohrlich_correlation_functionality.popescu_rohrlich_correlations import execute_popescu_rohrlich_gate
from fock_state_circuit.popescu_rohrlich_correlation_functionality.popescu_rohrlich_correlations import perform_measurement_popescu_rohrlich_correlation
import numpy as np

class SuperQuantumNodes(NodeList):  
    """
    Last modified: June 1st, 2024
    """
    _VERSION = '1.0.1'

    

    def popescu_rohrlich_correlation_gate( self, 
                                    pr_correlation: list, 
                                    node_info: dict = None
                                    ) -> None:
        """ 
        Creates (a set of) with superquantum correlations. The collection at input of this gate is completely replaced, so what happens before this gate
        does not affect the outcome (so this should be first gate in any circuit). When a measurement node is applied later in the circuit
        all optical states are measured, classical channels are written as specified in the measurement node. The 'initial_state' for the 
        resulting collection (after measurement) is the state.initial_state for the input state.

        See example code a bit below.

        Arguments:
            pr_correlation: List of form [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                                        {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used
        
        Example:
            pr_correlation = [  {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                                        {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
            dict_for_plotting = dict([])
            for angle in [a*np.pi/20 for a in range(10)]:
                circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
                circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=0)
                circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=0)
                circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(8)],classical_channels_to_be_written=[n for n in range(8)])

                collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                state = fsc.State(collection_of_states=collection)
                state.initial_state = 'popescu_rohrlich_correlation'
                collection.add_state(state)
                output_collection = circuit.evaluate_circuit(collection_of_states_input=collection)

                channel_combis_for_correlation = [(0,2),(4,6)]
                label_for_channel_combinations = ['first pair, q = 1','second pair, q = 5']
                
                correlations = output_collection.plot_correlations(channel_combis_for_correlation=channel_combis_for_correlation,
                                                                correlation_output_instead_of_plot=True)['popescu_rohrlich_correlation']

                lst = []
                lst.append({'output_state': 'first pair, q = 1', 'probability': correlations[0]})
                lst.append({'output_state': 'second pair, q = 5', 'probability': correlations[1]})
                dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})

            fsc.plot_correlations_from_dict(dict_for_plotting)
        """
        color_list = ['blue','cyan','red','green','magenta','yellow','purple','darkred','lime','crimson','pink']
        target_channels = []
        marker_colors = []
        for set_index, set_of_boxes in enumerate(pr_correlation):
            target_channels += set_of_boxes['channels_Ah_Av_Bh_Bv']
            color_index = set_index%len(color_list)
            marker_colors += [color_list[color_index]] * 4

        if node_info == None: node_info = {}
        node_info = {
            'label' : "PR photons",
            'channels_optical' : target_channels ,
            'channels_classical' : [],
            'markers' : ['s']*len(target_channels),
            'markercolor' : marker_colors,
            'markerfacecolor' : marker_colors,
            'marker_text' : [r'$PR$']*len(target_channels),
            'marker_text_fontsize' : [10],
            'marker_text_color' : ['black']*len(target_channels),
            'markersize' : 20,
            'fillstyle' : 'full'
        }|node_info
        
        def generic_function(input_collection,parameters):
            pr_correlation = parameters[0]
            output_collection = execute_popescu_rohrlich_gate(collection = input_collection,pr_correlation=pr_correlation)
            return output_collection
        
        node_to_be_added = {
            'generic_function' : generic_function,
            'generic_function_parameters' : [pr_correlation],
            'node_type' : 'generic function on collection', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
        return
    
    def perform_measurement_pr_correlation(     self,
                                                collection_of_states: CollectionOfStates, # type: ignore
                                                optical_channels_to_measure: list,
                                                classical_channels_to_write_to: list) -> CollectionOfStates: # type: ignore
        """ There is no need to include this function as a separate gate in a FockStateCircuit. This function will
            be called from a general measurement node if the for the input CollectionOfStates the function
            collection_of_states.has_pr_correlations() returns 'True'.

            This function 'perform_measurement_pr_correlation' performs a measurement on a collection of states with
            'popescu_rohrlich_correlation'. This means the states should have state.auxiliary_information['popescu_rohrlich_correlation']. 

            NOTE: This gate will always perform a complete measurement(i.e., collapse the total wave function for all optical channels. 
                This argument only determines which optical channels are measured to the corresponding classical channels). As example consider 
                this code snippet:
                            pr_correlation = [  {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1}]

                            circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=6,no_of_classical_channels=4)
                            circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
                            circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=0)
                            circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=0)
                            circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(4)],
                                                                classical_channels_to_be_written=[n for n in range(4)])

                            collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                            state = fsc.State(collection_of_states=collection)
                            state.optical_components = [('000010', np.sqrt(1/2)),('000001', np.sqrt(1/2))]
                            state.initial_state = 'popescu_rohrlich_correlation'
                            collection.add_state(state)
                            output_collection = circuit.evaluate_circuit(collection_of_states_input=collection)
                            output_collection.clean_up()
                            print(len(output_collection))
                    We have one set of PR-photons in channels 0,1,2, and 3. We measure both photons in the same orientation. 
                    So we expect to see either both photons in horizontal polarization ('0101xx') or both in vertical polarization ('1010xx'). 
                    We should not see the photons in different polarization (so states with optical component ('0110xx') or ('1001xx') we should
                    not see.). Since we start with the photons in channels 4 and 5 in a superposition between '10' and '01' we would expect
                    to find 2 states in the output, where each state has two optical components. We find however 4 states with a single optical 
                    component. The reason is that the system does a complete measurement for all optical channels once it sees that the collection
                    is generated with PR-correlation, so also the last two channels are measured and collapse to a single component photon number 
                    state. 

        Args:
            collection_of_states (CollectionOfStates): 
            optical_channels_to_measure (list): Optical channels to be measured. 
            
            NOTE: This gate will always perform a complete measurement(i.e.,
                collapse the total wave function for all optical channels. This argument only determines which optical channels are measured to 
                the corresponding classical channels.)
            classical_channels_to_write_to (list): Classical channels to be written
        Returns:
            CollectionOfStates: CollectionOfStates after measurement with collapsed wave function for optical channels and classical channels
                        written with the measurement results
        """
        return perform_measurement_popescu_rohrlich_correlation(collection_of_states=collection_of_states, 
                                                                optical_channels_to_measure=optical_channels_to_measure, 
                                                                classical_channels_to_write_to=classical_channels_to_write_to)
    
    # deprecates nodes left in the code for backward compatibility
    
    def create_no_signalling_boxes( self, 
                                    ns_boxes: list, 
                                    node_info: dict = None
                                    ) -> None:  
        
        """ DEPRECATED: Use popescu_rohrlich_correlation_gate instead !

        Creates (a set of) no-signalling boxes. The collection at input of this gate is completely replaced, so what happens before this gate
        does not affect the outcome (so this should be first gate in any circuit). When a measurement node is applied later in the circuit
        all optical states are measured, classical channels are written as specified in the measurement node. The 'initial_state' for the 
        resulting collection (after measurement) is 'no_signalling_boxes'

        See example code a bit below.

        Arguments:
            ns_boxes: List of form [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                                        {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used
        
        Example:
            pr_correlation = [  {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                                        {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
            dict_for_plotting = dict([])
            for angle in [a*np.pi/20 for a in range(10)]:
                circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                circuit.popescu_rohrlich_correlation_gate(pr_correlation=pr_correlation)
                circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=0)
                circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=0)
                circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(8)],classical_channels_to_be_written=[n for n in range(8)])

                collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                state = fsc.State(collection_of_states=collection)
                state.initial_state = 'popescu_rohrlich_correlation'
                collection.add_state(state)
                output_collection = circuit.evaluate_circuit(collection_of_states_input=collection)

                channel_combis_for_correlation = [(0,2),(4,6)]
                label_for_channel_combinations = ['first pair, q = 1','second pair, q = 5']
                
                correlations = output_collection.plot_correlations(channel_combis_for_correlation=channel_combis_for_correlation,
                                                                correlation_output_instead_of_plot=True)['popescu_rohrlich_correlation']

                lst = []
                lst.append({'output_state': 'first pair, q = 1', 'probability': correlations[0]})
                lst.append({'output_state': 'second pair, q = 5', 'probability': correlations[1]})
                dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})

            fsc.plot_correlations_from_dict(dict_for_plotting)
        """
        color_list = ['blue','green','red','cyan','magenta','yellow','purple','darkred','lime','crimson','pink']
        target_channels = []
        marker_colors = []
        for set_index, set_of_boxes in enumerate(ns_boxes):
            target_channels += set_of_boxes['channels_Ah_Av_Bh_Bv']
            color_index = set_index%len(color_list)
            marker_colors += [color_list[color_index]] * 4

        if node_info == None: node_info = {}
        node_info = {
            'label' : "NS box",
            'channels_optical' : target_channels ,
            'channels_classical' : [],
            'markers' : ['s']*len(target_channels),
            'markercolor' : marker_colors,
            'markerfacecolor' : marker_colors,
            'marker_text' : [r'$NS$']*len(target_channels),
            'marker_text_fontsize' : [6],
            'marker_text_color' : ['black']*len(target_channels),
            'markersize' : 15,
            'fillstyle' : 'full'
        }|node_info
        def generic_function(input_collection,parameters):
            """ Create a CollectionOfStates which can be processed in a FockStateCircuit. This collection
                represents  set of 'Popescu-Rohrlich correlations' which can have 'superquantum' correlations.

                The argument 'state' is a state matching the circuit on which the returned collection will
                be evaluated. The exact content of the state is not important, it is used as a 'template' 
                to create the collection from.

                The argument ns_boxes should be of form:
                    [   {'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                        {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
                Here the 'channels' are indices for the optical channels. The quantumness indicator determines
                how strong the 'superquantum' correlation will be. 
            """
            ns_boxes = parameters[0]
            for state in input_collection:
                template_state = state
                break
            output_collection = _create_collection_with_popescu_rohrlich_correlations(state=template_state, pr_correlation=ns_boxes)
            return output_collection
        
        node_to_be_added = {
            'generic_function' : generic_function,
            'generic_function_parameters' : [ns_boxes],
            'node_type' : 'generic function on collection', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
        return
    
