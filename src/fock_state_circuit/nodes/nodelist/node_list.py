
from __future__ import annotations
from fock_state_circuit.nodes.nodelist.optical_values_component_names import OpticalValues,ComponentNames

class NodeList():    
    """
        __init__(self, length_of_fock_state: int = 2, 
                no_of_optical_channels: int = 2, 
                no_of_classical_channels: int = 0, 
                channel_0_left_in_state_name: bool = True,
                threshold_probability_for_setting_to_zero: float = 0.0001,
                use_full_fock_matrix:bool = False,
                circuit_name : str = None
                ):
                Constructor for an instance of class NodeList. In a nodelist the characteristics for a FockStateCircuit are defined. 'Nodes' in the
                    rcuit inherit these characteristics, as does the 'FockStateCircuit' in which these nodes are used.
                    
                    Args:
                        length_of_fock_state (int, optional): 
                            If 'length_of_fock_state' is for instance 3 the possible values for any optical channel are 0, 1 or 2 photons. So the
                            maximum photon number in this case is 2. In general the maximum photon number is equal to 'length_of_fock_state'-1.. Defaults to 2. 
                        no_of_optical_channels (int, optional): 
                            Number of 'quantum' channels that can carry photons. Defaults to 2.
                        no_of_classical_channels (int, optional): 
                            Number of classical channels to store measurement results or to carry control date. Defaults to 0.
                        channel_0_left_in_state_name (bool, optional): 
                            Parameters for how to 'write' an optical state ('channel_0_left_in_state_name'). Of channel 0 has
                            one photon and the other channels zero photon then default the state is written as '1000'. With 
                            'channel_0_left_in_state_name' set to False this state would be written as '0001'. Teh value only affects the string value 
                            to indicate the state. All underlying mathematics and ordering of matrices and vectors is unaffected.. Defaults to True.
                        threshold_probability_for_setting_to_zero (float, optional): 
                            Forces rounding to zero for probabilities below the given level. Defaults to 0.0001.
                        use_full_fock_matrix (bool, optional): 
                            Default the class optimizes calculations by reducing the size of the Fock state matrix to match the photon population
                            (i.e., if states are empty they are discarded from the fock matrix). When the bool 'use_full_fock_matrix' is set to True 
                            the system will always use the full size Fock matrix and skip the optimization. Defaults to False.
                        circuit_name (str, optional): Name of the circuit as shown when drawing the circuit. Defaults to None.
    
        _update_list_of_nodes(self, node_to_be_added: dict = None) -> None:

                Update the list of nodes for node_list. This is the single function to update the list of nodes which serves as
                as a gatekeeper. node to be added is a dictionary describing the node.

        _translate_channels_numbers_to_tensor_list(self, channel_numbers: list[int] = []) -> list[int]:

                Generate a tensor list. The tensor list indicates to which optical channel a gate should be applied.
                Default the gates are applied to the optical channels with the lowest index (e.g., channel 0 for single channel
                gate, channels 0 and 1 for two channel gates). When we generate transition matrices we have to swap the columns
                and rows to make them apply to right channel. This information is contained in the 'tensor_list'.
                Example: If for a 4 channel circuit a gate should apply to channel 3 the tensor list is [3,1,2,0,4]
                Example: If for a 3 channel circuit a gate should apply to channel 2 and channel 1 the list is [2,1,0]
                (so if the gate operates on channel 0 by default it will operate on channel_numbers[0] by action of the tensor_list,
                if the gate operates on channel 1 by default it will operate on channel_numbers[1] by action of the tensor_list)
        
        _get_state_name_from_list_of_photon_numbers(self, state: list) -> str:
                For a list of photon numbers generate a string which can serve as name for the state or a component in the state.
                Example: state [0,1,3] would become '310' with 0 representing the photon number in channel 0. If we use reversed
                state notation state [0,1,3] would become '013' (reversed or regular state notation is set in initialization of the 
                FockStateCircuit). If we allow per channel photon numbers which require more digits (e.g., 10) the format of the string will be adjusted.
                Example [10,1,3] would become '100103'
                
        Last modified: April 16th, 2024                
    """
    _TYPES_OF_NODES = ['optical','custom fock matrix','controlled','classical', 'measurement', 'generic function on collection', 'spectrum', 'bridge']
        
    def __init__(self, length_of_fock_state: int = 2, 
                 no_of_optical_channels: int = 2, 
                 no_of_classical_channels: int = 0, 
                 channel_0_left_in_state_name: bool = True,
                 threshold_probability_for_setting_to_zero: float = 0.0001,
                 use_full_fock_matrix:bool = False,
                 circuit_name : str = None
                 ):
        """ 
        """
        # the length of the fock state is the number of possible photon numbers. So if the 'length' is 2 the maximum
        # number of photons is 1 (either 0 or 1 photons). If the length is 4 we can have 0,1,2 of 3 photons per channel.
        self._length_of_fock_state = length_of_fock_state

        # the number of channels defining the circuit.
        self._no_of_optical_channels = no_of_optical_channels
        self._no_of_classical_channels = no_of_classical_channels
        
        # we need at least a fock states with length 2 (0 or 1 photon) and two optical channels. Anything else is a 
        # trivial circuit with either zero photons or one channel without interaction.
        if self._length_of_fock_state < 1 or self._no_of_optical_channels < 2:
            raise Exception('length_of_fock_state minimal value is 1, no_of_optical_channels minimum value is 2')

        # for naming the states we need a convention. if 'channel_0_left_in_state_name' is set to 'True' we
        # write a state with 2 photons in channel 0 and 5 photons in channel 1 as '05'. With this value set
        # to 'False' we would write this same state as '50'. 
        self._channel_0_left_in_state_name = channel_0_left_in_state_name

        # '_digits_per_optical_channel' defines the number of digits used when 
        # writing a fock state as word. For more than 10 <= photons <100 per channel we 
        # need 2 digits per channel. For 100 or more need 3 digits.
        self._digits_per_optical_channel = len(str(self._length_of_fock_state-1))

        # list of nodes
        self.node_list = [] 

        # probabilities below this threshold are set to zero. 
        # this means that if the probability to find an outcome is lower than this threshold the outcome is discarded.
        self._threshold_probability_for_setting_to_zero = threshold_probability_for_setting_to_zero
       
        # self.use_full_fock_matrix = True means the system always uses the full fock matrix. This means matrix only has to 
        # be calculated once. For small circuits with low 'length_of_fock_state' this is optimal. 
        # When self.use_full_fock_matrix = False the system reduces fock matrix to relevant states (i.e.,
        # states with lower photon number are discarded). This reduces operations with large matrices. 
        # This is optimal for system with large 'length_of_fock_state' and many optical channels, but only a few photons running 
        # through the system.
        self._use_full_fock_matrix = use_full_fock_matrix
        
        # name for the circuit
        self._circuit_name = circuit_name

        if False:
            string_format_in_state_as_word = "{:0"+str(len(str(self._length_of_fock_state-1)))+ "d}"
            tuple_of_strings = tuple(string_format_in_state_as_word.format(number) for number in range(self._length_of_fock_state))
            self._list_of_fock_states = [[(n//(self._length_of_fock_state**c))%self._length_of_fock_state for c in range(self._no_of_optical_channels)] for n in range(self._length_of_fock_state**self._no_of_optical_channels)]
            if self._channel_0_left_in_state_name == True:
                self._dict_of_valid_component_names = {''.join([tuple_of_strings[number] for number in optical_state]):optical_state for optical_state in self._list_of_fock_states}
            else: #self.state_least_significant_digit_left == False:
                self._dict_of_valid_component_names = {''.join([tuple_of_strings[number] for number in optical_state[::-1]]):optical_state for optical_state in self._list_of_fock_states}       
            self._dict_of_optical_values = {tuple(v):k for k,v in self._dict_of_valid_component_names.items()}
        else:
            self._dict_of_valid_component_names = ComponentNames(length_of_fock_state=self._length_of_fock_state,
                                             no_of_optical_channels=self._no_of_optical_channels,
                                             channel_0_left_in_state_name = self._channel_0_left_in_state_name)
            self._dict_of_optical_values = OpticalValues(self._dict_of_valid_component_names)
        
        # create template state
        #name = ''.join([tuple_of_strings[0]] *self._no_of_optical_channels)
        name = self._dict_of_optical_values[tuple(0 for n in range(self._no_of_optical_channels))]
        self._template_state_dict = {   'initial_state' : name,
                            'cumulative_probability' : 1,
                            'optical_components' : { name: {'amplitude':1,'probability': 1} }, 
                            'classical_channel_values' : [0]*self._no_of_classical_channels,
                            'measurement_results' : [],
                            'auxiliary_information' : {}
                            }

    def _update_list_of_nodes(self, node_to_be_added: dict = None) -> None:

        """ Update the list of nodes for node_list. This is the single function to update the list of nodes which serves as
            as a gatekeeper. node to be added is a dictionary describing the node.

        Args:
            node_to_be_added (dict, optional): dictionary containing node to be added to the node list. Defaults to None.

        Raises:
            Exception: If invalid node in 'node_to_be_added
        """        

        if (node_to_be_added is None or
            'node_type' not in node_to_be_added.keys() or
            node_to_be_added['node_type'] not in NodeList._TYPES_OF_NODES):
            raise Exception('Error when updating the list of nodes for the fock state circuit')

        self.node_list.append(node_to_be_added)

        return

    def _translate_channels_numbers_to_tensor_list(self, channel_numbers: list[int] = []) -> list[int]:

        """ Generate a tensor list. The tensor list indicates to which optical channel a gate should be applied.
            Default the gates are applied to the optical channels with the lowest index (e.g., channel 0 for single channel
            gate, channels 0 and 1 for two channel gates). When we generate transition matrices we have to swap the columns
            and rows to make them apply to right channel. This information is contained in the 'tensor_list'.
            Example: If for a 4 channel circuit a gate should apply to channel 3 the tensor list is [3,1,2,0,4]
            Example: If for a 3 channel circuit a gate should apply to channel 2 and channel 1 the list is [2,1,0]
            (so if the gate operates on channel 0 by default it will operate on channel_numbers[0] by action of the tensor_list,
            if the gate operates on channel 1 by default it will operate on channel_numbers[1] by action of the tensor_list)

        Args:
            channel_numbers (list, optional): List of channel numbers to apply the gate to Defaults to [].

        Raises:
            Exception: if the gate requires two channels and the give channels in channel_numbers are the same
            Exception: if too many channels are given in channel_numbers

        Returns:
            list: tensor_list describing how to re-order the rows and columns in the fock matrix describing the node
        """        
        # create a tensor list as regularly ordered list of channel numbers [0,1,2,3..]
        tensor_list = [n for n in range(self._no_of_optical_channels)]
        if not channel_numbers or len(channel_numbers) == 0:
            # if no channel numbers are given return the default list [0,1,2,..]
            return tensor_list
        elif len(channel_numbers) == 1:
            # if there is only one channel number we swap channel number '0' and the given channel number
            channel_for_shift = channel_numbers[0]
            if channel_for_shift != 0:
                tensor_list[0], tensor_list[channel_for_shift] = tensor_list[channel_for_shift], tensor_list[0] 
        
        elif len(channel_numbers) == 2:
            channel_horizontal, channel_vertical = channel_numbers[0], channel_numbers[1]
            if channel_horizontal == channel_vertical:
                raise Exception('channel numbers identical where two different channel numbers are needed')
            
            if (channel_horizontal, channel_vertical) == (0,1):
                pass # tensor_list is already ordered correctly
            elif (channel_horizontal, channel_vertical) == (1,0):
                tensor_list[0], tensor_list[1] = tensor_list[1], tensor_list[0] #swap
            else:
                tensor_list[0], tensor_list[channel_horizontal] = tensor_list[channel_horizontal],  tensor_list[0]
                tensor_list[1], tensor_list[channel_vertical] = tensor_list[channel_vertical],  tensor_list[1]  
        
        else:
            raise Exception('error creating tensor list, more than two channels provided')
        
        return tensor_list