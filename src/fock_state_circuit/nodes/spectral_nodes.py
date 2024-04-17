from fock_state_circuit.nodes.nodelist.node_list import NodeList
import numpy as np
from fock_state_circuit.temporal_and_spectral_gate_functionality.column_of_states import ColumnOfStates
from fock_state_circuit.temporal_and_spectral_gate_functionality.interference_group import InterferenceGroup
from fock_state_circuit.temporal_and_spectral_gate_functionality.collection_of_state_columns import CollectionOfStateColumns
from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import add_time_delay
from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import add_time_delay_classical_control
from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import photon_probability_function
from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import add_quick_time_delay_classical_control
from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import add_quick_time_delay
from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import set_pulse_width


class SpectralNodes(NodeList):  
    """ Method(s) to create spectral nodes, affecting time or spectral properties of photons. 

            time_delay_full(self,
                    affected_channels: list[int] = None,
                    delay: float = 0,
                    pulse_width: float = 1,
                    node_type: str = 'spectrum', 
                    node_info: dict = None
                    ) -> None:

                    Apply a time delay for a given pulse_width. The node will turn a pure state into a statistical mixture where states
                    obtain a time shift that depends on the time delay. 

                    This is the FULL implementation allowing successive timing gates in the circuit. 
        
            time_delay_classical_control_full(self,
                    affected_channels: list[int] = None,
                    classical_channel_for_delay: int = 0,
                    pulse_width: float = 1,
                    node_type: str = 'spectrum', 
                    node_info: dict = None
                    ) -> None:

                    Apply a time delay for a given pulse_width. The delay is read from the provided classical channel.
                    The node will turn a pure state into a statistical mixture where states
                    obtain a time shift that depends on the time delay. 

                    This is the FULL implementation allowing successive timing gates in the circuit. 
                       
            time_delay(self,
                    affected_channels: list[int] = None,
                    delay: float = 0,
                    pulse_width: float = None,
                    bandwidth: float = None,
                    node_type: str = 'spectrum', 
                    node_info: dict = None
                    ) -> None:

                    Apply a time delay for a given pulse_width. The node will turn a pure state into a statistical mixture where states
                    obtain a time shift that depends on the time delay. 

                    This is the QUICK implementation allowing only a single time gates in the circuit. 

                    The use of argument bandwidth is deprecated an only provided for backward compatibility. Please use pulse_width
                
            time_delay_classical_control(self,
                    affected_channels: list[int] = None,
                    classical_channel_for_delay: int = 0,
                    pulse_width: float = None,
                    bandwidth: float = None,
                    node_type: str = 'spectrum', 
                    node_info: dict = None
                    ) -> None:
                    
                    Apply a time delay for a given pulse_width. The delay is read from the provided classical channel.
                    The node will turn a pure state into a statistical mixture where states
                    obtain a time shift that depends on the time delay. 

                    This is the QUICK implementation allowing only a single time gates in the circuit.
                    
                    The use of argument bandwidth is deprecated an only provided for backward compatibility. Please use pulse_width
        
        
            set_pulse_width(self,
                    affected_channels: list[int] = None,
                    pulse_width: float = None,
                    node_type: str = 'spectrum', 
                    node_info: dict = None
                    ) -> None:

                    Set pulse width for requested optical channels. If affected channels is None (default) all channels will be set. 
                    If affected_channels is an empty list the node will have no effect.

        Last modified: April 16th, 2024

    """
    _VERSION = '1.0.0'
    def time_delay_full(self,
            affected_channels: list[int] = None,
            delay: float = 0,
            pulse_width: float = 1,
            node_type: str = 'spectrum', 
            node_info: dict = None
            ) -> None:
        """ Apply a time delay for a given pulse_width. The node will turn a pure state into a statistical mixture where states
            obtain a time shift that depends on the time delay. 

            This is the FULL implementation allowing successive timing gates in the circuit. 
        """
        if node_info == None: node_info = {}
        node_info = {
                'label' : "delay",
                'channels_optical' : affected_channels,
                'channels_classical' : [],
                'markers' : ['s'],
                'markercolor' : ['darkblue'],
                'markerfacecolor' : ['blue'],
                'marker_text' : [r"$\tau$"],
                'markersize' : 20,
                'fillstyle' : 'full'
            }|node_info
        
        function = add_time_delay

        self.generic_function_on_collection(
                                   function, 
                                   affected_optical_channels= affected_channels, 
                                   affected_classical_channels = [], 
                                   parameters = [affected_channels,delay,pulse_width],
                                   node_info= node_info) 
        
    def time_delay_classical_control_full(self,
            affected_channels: list[int] = None,
            classical_channel_for_delay: int = 0,
            pulse_width: float = 1,
            node_type: str = 'spectrum', 
            node_info: dict = None
            ) -> None:
        """ Apply a time delay for a given pulse_width. The delay is read from the provided classical channel.
            The node will turn a pure state into a statistical mixture where states
            obtain a time shift that depends on the time delay. 

            This is the FULL implementation allowing successive timing gates in the circuit. 
        """
        if node_info == None: node_info = {}
        node_info = {
                'label' : "delay(c)",
                'channels_optical' : affected_channels,
                'channels_classical' : [classical_channel_for_delay],
                'markers' : ['s'],
                'markercolor' : ['darkblue'],
                'markerfacecolor' : ['blue'],
                'marker_text' : [r"$\tau$"],
                'markersize' : 20,
                'fillstyle' : 'full'
            }|node_info
        
        function = add_time_delay_classical_control

        self.generic_function_on_collection(
                                   function, 
                                   affected_optical_channels= affected_channels, 
                                   affected_classical_channels = [], 
                                   parameters = [affected_channels,classical_channel_for_delay,pulse_width],
                                   node_info= node_info) 
        
    def time_delay(self,
            affected_channels: list[int] = None,
            delay: float = 0,
            pulse_width: float = None,
            bandwidth: float = None,
            node_type: str = 'spectrum', 
            node_info: dict = None
            ) -> None:
        """ Apply a time delay for a given pulse_width. The node will turn a pure state into a statistical mixture where states
            obtain a time shift that depends on the time delay. 

            This is the QUICK implementation allowing only a single time gates in the circuit. 

            The use of argument bandwidth is deprecated an only provided for backward compatibility. Please use pulse_width
        """
        if pulse_width is None and bandwidth is None:
            pulse_width = 1
        elif pulse_width is None and bandwidth is not None:
            pulse_width = 1/bandwidth
        
        if node_info == None: node_info = {}
        node_info = {
                'label' : "delay",
                'channels_optical' : affected_channels,
                'channels_classical' : [],
                'markers' : ['s'],
                'markercolor' : ['darkblue'],
                'markerfacecolor' : ['blue'],
                'marker_text' : [r"$\tau$"],
                'markersize' : 20,
                'fillstyle' : 'full'
            }|node_info
        
        function = add_quick_time_delay
        self.generic_function_on_collection(
                                   function, 
                                   affected_optical_channels= affected_channels, 
                                   affected_classical_channels = [], 
                                   parameters = [affected_channels,delay,pulse_width],
                                   node_info= node_info) 
        
    def time_delay_classical_control(self,
            affected_channels: list[int] = None,
            classical_channel_for_delay: int = 0,
            pulse_width: float = None,
            bandwidth: float = None,
            node_type: str = 'spectrum', 
            node_info: dict = None
            ) -> None:
        """ Apply a time delay for a given pulse_width. The delay is read from the provided classical channel.
            The node will turn a pure state into a statistical mixture where states
            obtain a time shift that depends on the time delay. 

            This is the QUICK implementation allowing only a single time gates in the circuit.
            
            The use of argument bandwidth is deprecated an only provided for backward compatibility. Please use pulse_width
        """

        if pulse_width is None and bandwidth is None:
            pulse_width = 1
        elif pulse_width is None and bandwidth is not None:
            pulse_width = 1/bandwidth


        if node_info == None: node_info = {}
        node_info = {
                'label' : "delay(c)",
                'channels_optical' : affected_channels,
                'channels_classical' : [classical_channel_for_delay],
                'markers' : ['s'],
                'markercolor' : ['darkblue'],
                'markerfacecolor' : ['blue'],
                'marker_text' : [r"$\tau$"],
                'markersize' : 20,
                'fillstyle' : 'full'
            }|node_info
        
        function = add_quick_time_delay_classical_control

        self.generic_function_on_collection(
                                   function, 
                                   affected_optical_channels= affected_channels, 
                                   affected_classical_channels = [], 
                                   parameters = [affected_channels,classical_channel_for_delay,pulse_width],
                                   node_info= node_info) 
        
    def set_pulse_width(self,
            affected_channels: list[int] = None,
            pulse_width: float = None,
            node_type: str = 'spectrum', 
            node_info: dict = None
            ) -> None:
        """ Set pulse width for requested optical channels. If affected channels is None (default) all channels will be set. 
            If affected_channels is an empty list the node will have no effect.

            NOTE: Changing pulse width in some channels can lead to non-physical results. We do not take into account absorption 
            in spectral filters or 'chirp' in stretched pulses. It is safer to keep pulse-width the same for all photons and all 
            channels.
        """
        if affected_channels is None:
            affected_channels = [n for n in range(self._no_of_optical_channels)]

        if node_info == None: node_info = {}
        node_info = {
                'label' : "pulse width",
                'channels_optical' : affected_channels,
                'channels_classical' : [],
                'markers' : ['s'],
                'markercolor' : ['darkblue'],
                'markerfacecolor' : ['blue'],
                'marker_text' : [r"$\delta$"],
                'markersize' : 20,
                'fillstyle' : 'full'
            }|node_info
        
        function = set_pulse_width

        self.generic_function_on_collection(
                                   function, 
                                   affected_optical_channels= affected_channels, 
                                   affected_classical_channels = [], 
                                   parameters = [affected_channels,pulse_width],
                                   node_info= node_info) 
        
# This function does not belong to class SpectralNodes
def perform_measurement_photon_resolved(collection_of_states, 
                            optical_channels_to_measure, 
                            classical_channels_to_write_to, 
                            list_of_projections: list = None):
    """ This function performs a measurement on a collection of states with 'spectral information'. This means
        the states should have state.auxiliary_information['photon_resolution']. Before calling this function
        check if 'collection_of_states.is_photon_resolved()' returns 'True'.

        list_of_projections can be provided in form ['1011','2134'] to limit the 'search'. Only outcomes in 
        list_of_projections will be considered as valid outcomes.

    Args:
        collection_of_states (CollectionOfStates): (photon resolved) collection of states
        optical_channels_to_measure (list): list of of optical channel numbers to be measured
        classical_channels_to_write_to (list): list of classical channel numbers to write the measurement result to
        list_of_projections (list, optional): List of projections to limit the 'search' and speed up measurement. Defaults to None.

    Returns:
        CollectionOfStates: CollectionOfStates after measurement
    """

    coll_of_cols = CollectionOfStateColumns(collection_of_states=collection_of_states)
    coll_of_cols.split()
    coll_of_cols.single_photon_states()
    coll_of_cols.photon_probability_function = photon_probability_function
    return coll_of_cols.perform_measurement(    optical_channels_to_measure,
                                                classical_channels_to_write_to,
                                                list_of_projections
                                                )
 
