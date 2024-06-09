from __future__ import annotations
from fock_state_circuit.nodes.nodelist.node_list import NodeList

class BridgeNodes(NodeList):  
    """Method(s) to create bridge nodes

            def bridge(self,
                    next_fock_state_circuit,
                    node_type: str = 'bridge', 
                    node_info: dict = None
                    ) -> None:
                    Apply a bridge node to the circuit to transfer the collection of states from one circuit to another. Used when the characteristics
                    of the circuit change (i.e., change number of optical/classical channels). 

        Last modified: April 16th, 2024
    """
    _VERSION = '1.0.0'
    def bridge(self,
            next_fock_state_circuit,
            node_type: str = 'bridge', 
            node_info: dict = None
            ) -> None:
        """ Apply a bridge node to the circuit to transfer the collection of states from one circuit to another. Used when the characteristics
            of the circuit change (i.e., change number of optical/classical channels). 
        """
        if node_info == None: node_info = {}
        node_info = {
                'label' : "bridge",
                'channels_optical' : [],
                'channels_classical' : [],
                'markers' : ['o'],
                'markercolor' : ['blue'],
                'markerfacecolor' : ['white'],
                'marker_text' : [''],
                'markersize' : 40,
                'fillstyle' : 'none'
            }|node_info
        self._update_list_of_nodes({'next_fock_state_circuit':next_fock_state_circuit,'node_type': node_type, 'node_info': node_info})

   