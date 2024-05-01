from fock_state_circuit.nodes.nodelist.node_list import NodeList
from fock_state_circuit.no_signalling_boxes import create_collection_with_NS_boxes
import numpy as np

class SuperQuantumNodes(NodeList):  
    """

    Args:
        NodeList (_type_): _description_

    Last modified: April 28th, 2024
    """
    _VERSION = '1.0.0'

    def create_no_signalling_boxes( self, 
                                    ns_boxes: list, 
                                    node_info: dict = None
                                    ) -> None:  
        
        """ 
        Creates (a set of) no-signalling boxes. The collection at input of this gate is completely replaced, so what happens before this gate
        does not affect the outcome (so this should be first gate in any circuit). When a measurement node is applied later in the circuit
        all optical states are measured, classical channels are written as specified in the measurement node. The 'initial_state' for the 
        resulting collection (after measurement) is 'no_signalling_boxes'

        See example code a bit below.

        Arguments:
            ns_boxes: List of form [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                                        {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used
        
        Example code:
            N = 10
            angles = [0]
            circuit_is_already_drawn = False

            ns_boxes = [{'channels_Ah_Av_Bh_Bv':[0,1,2,3],'quantumness_indicator':1},
                        {'channels_Ah_Av_Bh_Bv':[4,5,6,7],'quantumness_indicator':5}]
            for angle_first in angles:
                dict_for_plotting = dict([])
                for angle in [(a/N)*np.pi/2 for a in range(N)]:
                    circuit = fsc.FockStateCircuit(length_of_fock_state=3,no_of_optical_channels=8,no_of_classical_channels=8)
                    circuit.create_no_signalling_boxes(ns_boxes=ns_boxes)
                    circuit.half_wave_plate(channel_horizontal=2,channel_vertical=3,angle=angle)
                    circuit.half_wave_plate(channel_horizontal=6,channel_vertical=7,angle=angle)
                    circuit.half_wave_plate(channel_horizontal=0,channel_vertical=1,angle=angle_first)
                    circuit.half_wave_plate(channel_horizontal=4,channel_vertical=5,angle=angle_first)
                    circuit.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(8)],classical_channels_to_be_written=[n for n in range(8)])

                    if not circuit_is_already_drawn:
                        circuit.draw()
                        circuit_is_already_drawn = True

                    collection = fsc.CollectionOfStates(fock_state_circuit=circuit,input_collection_as_a_dict=dict([]))
                    state = fsc.State(collection_of_states=collection)
                    collection.add_state(state)
                    output_collection = circuit.evaluate_circuit(collection_of_states_input=collection)

                    channel_combis_for_correlation = [(0,2),(4,6)]
                    label_for_channel_combinations = ['first pair, K = 1','second pair, K = 5']
                    
                    correlations = output_collection.plot_correlations(channel_combis_for_correlation=channel_combis_for_correlation,
                                                                    correlation_output_instead_of_plot=True)['no_signalling_boxes']
                    lst = []
                    lst.append({'output_state': 'first pair, K = 1', 'probability': correlations[0]})
                    lst.append({'output_state': 'second pair, K = 5', 'probability': correlations[1]})
                    dict_for_plotting.update({str(np.round(angle/np.pi,3)) + ' pi': lst})
                # plot the resulting correlations for the no-signalling boxes
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
            """Create a CollectionOfStates which can be processed in a FockStateCircuit. This collection
                represents  set of 'No signalling boxes' which can have 'superquantum' correlations.

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
            output_collection = create_collection_with_NS_boxes(state=template_state,ns_boxes=ns_boxes)
            return output_collection
        
        node_to_be_added = {
            'generic_function' : generic_function,
            'generic_function_parameters' : [ns_boxes],
            'node_type' : 'generic function on collection', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
        return
