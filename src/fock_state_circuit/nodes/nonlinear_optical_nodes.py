from fock_state_circuit.nodes.nodelist.node_list import NodeList
import numpy as np

class NonlinearOpticalNodes(NodeList):
    """ Method(s) to generate nonlinear optical nodes which represent physical processes
     
        Last modified: April 16th, 2024      
        """
    _VERSION = '1.0.0'
    def nonlinear_optical_node(self, 
                            operator: str,
                            optical_channels: list[int],  
                            node_info: dict = None
                            ) -> None: 
        """ Dummy implementation """
        if node_info == None: node_info = {}
        node_info = {
                'label' : "non-lin",
                'channels_optical' : optical_channels,
                'markers' : ['h'],
                'markercolor' : ['red'],
                'markerfacecolor' : ['red'],
                'marker_text' : [r'$nl$'],
                'marker_text_fontsize' : [8],
                'marker_text_color' : ['white'],
                'markersize' : 20,
                'fillstyle' : 'full'
            }|node_info
        
        
        def function_representing_quantum_operator(input_collection,parameters):
            quantum_operator = parameters[0]
            output_collection = input_collection
            return output_collection
        
        node_to_be_added = {
            'generic_function' : function_representing_quantum_operator,
            'generic_function_parameters' : [optical_channels, operator],
            'node_type' : 'generic function on collection', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)