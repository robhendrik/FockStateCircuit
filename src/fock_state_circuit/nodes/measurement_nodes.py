from __future__ import annotations
from fock_state_circuit.nodes.nodelist.node_list import NodeList

class MeasurementNodes(NodeList):
    """Method(s) to perform a measurement:   
            
            measure_optical_to_classical(self, 
                    optical_channels_to_be_measured: list[int] = [0], 
                    classical_channels_to_be_written: list[int] = [0], 
                    node_info: dict = None
                    ) -> None

                    Read photon number in a optical channel and write to classical channel. 

        Last modified: April 16th, 2024
    """
    _VERSION = '1.0.0'
    def measure_optical_to_classical(self, 
                                     optical_channels_to_be_measured: list[int] = [0], 
                                     classical_channels_to_be_written: list[int] = [0], 
                                     list_of_projections: list[str] = None,
                                     node_info: dict = None) -> None: 
        """ Read photon number in a optical channel and write to classical channel.
                        
        Args:
            optical_channels_to_be_measured (list[int]): list of of optical channel numbers to be measured
            classical_channels_to_be_written (list[int]): list of classical channel numbers to write the measurement result to
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Raises:
            Exception: when list of optical channels and list of classical channels are not of equal length
        """ 
        optical_channels_allowed = all([( 0 <= c < self._no_of_optical_channels) for c in optical_channels_to_be_measured])
        classical_channels_allowed = all([( 0 <= c < self._no_of_classical_channels) for c in classical_channels_to_be_written])
        if len(classical_channels_to_be_written) != len(optical_channels_to_be_measured) or not optical_channels_allowed or not classical_channels_allowed:
            raise Exception('error in defining measurement from optical to classical')
    
        if node_info == None: node_info = {}
        node_info = {
                'label' : "measurement",
                'channels_optical' : optical_channels_to_be_measured,
                'channels_classical'  : classical_channels_to_be_written,
                'markercolor' : ['black'],
                'markerfacecolor' : ['black'],
                'marker_text' : [r"$M$"],
                'markersize' : 20,
                'fillstyle' : 'full',
                'classical_marker_text' : [r"$t$"],
                'classical_marker_text_fontsize' : [5]
            }|node_info
        
        node_to_be_added = {
            'optical_channels_to_be_read' : optical_channels_to_be_measured,
            'classical_channels_to_be_written' : classical_channels_to_be_written,
            'list_of_projections' : list_of_projections,
            'node_type' : 'measurement', 
            'node_info' : node_info
        }

        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
        return
