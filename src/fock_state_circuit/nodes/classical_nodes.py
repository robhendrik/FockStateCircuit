from __future__ import annotations
from fock_state_circuit.nodes.nodelist.node_list import NodeList

class ClassicalNodes(NodeList):  
    """Method(s) to control the values in the classical channels:                   

            classical_channel_function(self, 
                    function, 
                    affected_channels: list[int] = None, 
                    new_input_values: list[int] = None, 
                    node_info: dict = None) -> None

                    Perform a calculation/operation on the classical channels only. Function should use the full list of channels 
                    as input and as output.

            set_classical_channels(self, 
                    list_of_values_for_classical_channels: list[int] = None, 
                    list_of_classical_channel_numbers: list[int] = None, 
                    node_info: dict = None
                    ) -> None

                    Sets values for classical channels. Input is a list of channels and a list of values to be written to the classical channel.
                    The two lists should have same length. First value in the list of values is written to the first channel in list of channels.

        Last modified: April 16th, 2024
    
    """
    _VERSION = '1.0.0'
    def classical_channel_function(self, 
                                   function, 
                                   affected_channels: list[int] = None, 
                                   new_input_values: list[int] = None, 
                                   node_info: dict = None) -> None: 
        """ Perform a calculation/operation on the classical channels only. Function should use the full list of channels 
            as input and as output.
            
            Example function to inverse the order of classical channels:
                
                def test_function(current_values, new_input_values, affected_channels):
                    output_list = current_values[::-1]
                    return output_list
            
            Example function to set values in the classical channels to fixed values
                
                def function(current_values, new_input_values, affected_channels):
                    for index, channel in enumerate(affected_channels):
                        current_values[channel] = new_input_values[index]
                    return current_values
                        
        Args:
            function: function using list of classical channels as input and as output
            affected_channels (list[int]): = None, 
            new_input_values (list[int]): = None, 
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            nothing
        """ 
        if affected_channels == None:
            affected_channels = []
        
        if new_input_values == None:
            new_input_values = []

        if node_info == None: node_info = {}
        node_info = {
                'label' : "class. function",
                'channels_optical' : [],
                'channels_classical' : affected_channels,
                'markercolor' : ['black'],
                'markerfacecolor' : ['black'],
                'marker_text' : [],
                'markersize' : 20,
                'fillstyle' : 'full'
            }|node_info

        
        node_to_be_added = {
            'function' : function,
            'affected_channels' : affected_channels,
            'new_input_values' : new_input_values,
            'node_type' : 'classical', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
  
        return
    
    def set_classical_channels(self, 
                               list_of_values_for_classical_channels: list[int] = None, 
                               list_of_classical_channel_numbers: list[int] = None, 
                               node_info: dict = None) -> None:
        """ Sets values for classical channels. Input is a list of channels and a list of values to be written to the classical channel.
            The two lists should have same length. First value in the list of values is written to the first channel in list of channels.
    
        Args:
            list_of_values_for_classical_channels (list[int]): list of values
            list_of_classical_channel_numbers (list[int]): list of channel numbers
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used
        
        Returns:
            nothing

        Raises:
            exception is channel number is/are invalid or list of values does not match channels
        """
        conditions = [
            len(list_of_values_for_classical_channels) != len(list_of_classical_channel_numbers),
            max(list_of_classical_channel_numbers) >= self._no_of_classical_channels
            ]
        if any(conditions):
            raise Exception('Incorrect input in function set_classical_channels')
        
        def function(classical_channel_values, list_of_values_for_classical_channels, list_of_classical_channel_numbers ):
            for index, channel in enumerate(list_of_classical_channel_numbers):
                classical_channel_values[channel] = list_of_values_for_classical_channels[index]
            return classical_channel_values

        if node_info == None: node_info = {}
        node_info = {
                'label' : "set classic",
                'channels_optical' : [],
                'channels_classical' : list_of_classical_channel_numbers,
                'markercolor' : ['black'],
                'markerfacecolor' : ['black'],
                'marker_text' : [],
                'markersize' : 20,
                'fillstyle' : 'full'
            }|node_info
        self.classical_channel_function(function, 
                                        affected_channels=list_of_classical_channel_numbers, 
                                        new_input_values=list_of_values_for_classical_channels,
                                        node_info = node_info)
        
        return
 