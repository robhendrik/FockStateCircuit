from fock_state_circuit.nodes.nodelist.node_list import NodeList
import numpy as np

class CustomNodes(NodeList):  
    """ Method(s) to create custom nodes, beyond what is provided in the standard set

            custom_optical_node(self, 
                    matrix_optical, 
                    optical_channels: list[int], 
                    node_type: str = 'optical', 
                    node_info: dict = None
                    ) -> None

                    Apply a custom optical matrix to the circuit. The matrix has to be a 2x2 numpy array with numpy cdouble entries. The function does
                    NOT check whether the matrix is physically possible (i.e.,invertible, unitary).

            custom_fock_state_node(self,
                    custom_fock_matrix,
                    node_type: str = 'custom fock matrix', 
                    node_info: dict = None
                    ) -> None

                    Apply a custom Fock state matrix to the circuit. The matrix has to be an LxL numpy array with numpy cdouble entries. L is the total size
                    of the Fock state basis (which can be retrieved via FockStateCircuit.basis() )The function does NOT check whether the matrix is physically 
                    possible (i.e.,invertible, unitary). 

            def generic_function_on_collection(self, 
                    function, 
                    affected_optical_channels: list[int] = None, 
                    affected_classical_channels: list[int] = None, 
                    parameters: list[int] = None, 
                    node_info: dict = None
                    ) -> None: 

                    Perform a calculation/operation on the full collection of states. The function should return a transformed collection matching the layout of the circuit.
                    This is the most generic node giving full freedom for a transformation on the collection_of_states. The function that is passed as argument will be called as:
                    collection_of_states_output = generic_function(collection_of_states_input,generic_function_parameters)

                        channel_coupling(self, 
                    control_channels: list[int] = [0],
                    target_channels: list[int] = [1],
                    coupling_strength: float = 0,
                    node_info: dict = None
                    ) -> None:  
                    Apply a node to the circuit to couple channels with the given 'coupling_strength'. The node will effectively 
                    apply a controlled shift from control channels to target channels.

            shift(  self, 
                    target_channels: list[int] = [1],
                    shift_per_channel: list[int] = [0],
                    node_info: dict = None
                    ) -> None:  
                    Apply a shift node to the circuit. The photon value in the target channel(s) is increased by the value in 
                    the list 'shift_per_channel' modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, 
                    shift value is 2 and target channel value is 1 the result is that target becomes (2+1)%3 = 0. This is not a 
                    linear optical operation (photons are created in this process). 

            c_shift(self, 
                    control_channels: list[int] = [0],
                    target_channels: list[int] = [1],
                    node_info: dict = None
                    ) -> None:  
                    Apply a controlled-shift (c-shift) node to the circuit. The photon value in the target channel(s) is increased by the value in 
                    the control channels(s) modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, control channel value is 2 and 
                    target channel valie is 1 the result is that control value remains 2 and target becomes (2+1)%3 = 0

            shift_direct(   self, 
                            target_channels: list[int] = [1],
                            shift_per_channel: list[int] = [0],
                            node_info: dict = None
                            ) -> None:  
                    Apply a shift node to the circuit. The photon value in the target channel(s) is increased by the value in 
                    the list 'shift_per_channel' modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, 
                    shift value is 2 and target channel value is 1 the result is that target becomes (2+1)%3 = 0. This is not a 
                    linear optical operation (photons are created in this process). 
                    
                    NOTE: This 'direct' node does not create a FockState matrix but directly changes the 'optical_components' of the states in the collection. 
                    functionality is the same as for the regular 'shift' node but the implementation can be faster, depending on number of states in the collection 
                    and the number of channels in the circuit.

            c_shift_direct( self, 
                            control_channels: list[int] = [0],
                            target_channels: list[int] = [1],
                            node_info: dict = None
                            ) -> None:  
                    Apply a controlled-shift (c-shift) node to the circuit. The photon value in the target channel(s) is increased by the value in 
                    the control channels(s) modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, control channel value is 2 and 
                    target channel valie is 1 the result is that control value remains 2 and target becomes (2+1)%3 = 0

                    NOTE: This 'direct' node does not create a FockState matrix but directly changes the 'optical_components' of the states in the collection. 
                    functionality is the same as for the c_shift node but the implementation can be faster, depending on number of states in the collection 
                    and the number of channels in the circuit.

        Last modified: April 16th, 2024
    """
    _VERSION = '1.0.0'
    def custom_optical_node(self, 
                            matrix_optical, #: np.array[np.cdouble], 
                            optical_channels: list[int],  
                            node_type: str = 'optical', 
                            node_info: dict = None
                            ) -> None:  
        
        """ Apply a custom optical matrix to the circuit. The matris has to be a 2x2 numpy array with numpy cdouble entries. The function does
            NOT check whether the matrix is physically possible (i.e.,invertible, unitary). 

        Args:
            matrix_optical (np.array[np.cdouble]): optical matrix to be applied
            optical_channels (list[int]): channels to apply the optical matrix to
            node_type (str): type of optical node ('optical' or 'optical non-linear' )
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.

        Returns:
            nothing
        
        Raises:
            Exception: If the matrix is not 2x2
        """
        if node_info == None: node_info = {}
        node_info = {
                'label' : "custom",
                'channels_optical' : optical_channels,
                'channels_classical' : [],
                'markers' : ['*'],
                'markercolor' : ['blue'],
                'markerfacecolor' : ['lightblue'],
                'marker_text' : [r"$c$"],
                'marker_text_fontsize' : [7],
                'markersize' : 25,
                'fillstyle' : 'full'
            }|node_info
 
        if len(matrix_optical) != 2 and len(matrix_optical[0]) != 2:
            raise Exception('Only optical nodes with 2-channel interaction are implemented')
        channel_numbers = optical_channels
        tensor_list = self._translate_channels_numbers_to_tensor_list(channel_numbers = channel_numbers)
        matrix_optical = matrix_optical
        self._update_list_of_nodes({'matrix_optical': matrix_optical, 'tensor_list':tensor_list, 'node_type': node_type, 'node_info': node_info})

        return

    def custom_fock_state_node(self,
            custom_fock_matrix,
            node_type: str = 'custom fock matrix', 
            node_info: dict = None
            ) -> None:
        """ Apply a custom Fock state matrix to the circuit. The matrix has to be an LxL numpy array with numpy cdouble entries. L is the total size
            of the Fock state basis (which can be retrieved via FockStateCircuit.basis() )The function does NOT check whether the matrix is physically 
            possible (i.e.,invertible, unitary). 
        """
        if node_info == None: node_info = {}
        node_info = {
                'label' : "custom",
                'channels_optical' : [channel for channel in range(self._no_of_optical_channels)],
                'channels_classical' : [],
                'markers' : ['*'],
                'markercolor' : ['blue'],
                'markerfacecolor' : ['lightblue'],
                'marker_text' : [r"$c$"],
                'marker_text_fontsize' : [7],
                'markersize' : 25,
                'fillstyle' : 'full'
            }|node_info
        self._update_list_of_nodes({'custom_fock_matrix':custom_fock_matrix,'node_type': node_type, 'node_info': node_info})

        return
    
    def shift(  self, 
                target_channels: list[int] = [1],
                shift_per_channel: list[int] = [0],
                node_info: dict = None
                ) -> None:  
        """ Apply a shift node to the circuit. The photon value in the target channel(s) is increased by the value in 
            the list 'shift_per_channel' modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, 
            shift value is 2 and target channel value is 1 the result is that target becomes (2+1)%3 = 0. This is not a 
            linear optical operation (photons are created in this process). 

            Args: 
                target_channels (list[int]): Channels that change value based on the values in 'shift_per_channel'
                shift_per_channel (list[int]): Shift value per channel
                node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
            
            Raises: 
                Exception when the input channel numbers do not match with the circuit.
                
        """
        # check if lists with channel numbers are ok, otherwise throw exception
        error = False
        if len(shift_per_channel) == len(target_channels):
            for shift, target in zip(shift_per_channel, target_channels):
                if min(shift, target) < 0 or target > self._no_of_optical_channels:
                    error = True
        else:
            error = True
        if error:
            Exception('Error in channel input for function FockStateCircuit.shift')

        this_is_first_round = True  
        # for each channel create a transition matrix
        # multiply the matrices to get to the final overall matrix
        for shift, channel_to_shift in zip(shift_per_channel,target_channels):
            # create an identity matrix for the given combination of one target and one control channel
            coupling_matrix_one_target_control_combination = np.zeros((len(self._list_of_fock_states),len(self._list_of_fock_states)), dtype = np.ubyte)
            for input_index, input in enumerate(self._list_of_fock_states):
                for output_index, output in enumerate(self._list_of_fock_states):
                    # except for target channel values should be equal
                    if all([input[channel] == output[channel] for channel in range(len(output)) if channel != channel_to_shift]):
                    # for target channel the value has to be (shift + original target value) % length. So shift with control value modulus length
                        if (output[channel_to_shift] == (shift + input[channel_to_shift])%self._length_of_fock_state):
                                coupling_matrix_one_target_control_combination[output_index][input_index] = np.ubyte(1)
            if this_is_first_round:
                coupling_matrix = coupling_matrix_one_target_control_combination
                this_is_first_round = False
            else:
                coupling_matrix  = np.matmul(coupling_matrix, coupling_matrix_one_target_control_combination)

        if node_info == None: node_info = {}
        node_info = {
            'label' : "shift",
            'channels_optical' : target_channels ,
            'channels_classical' : [],
            'markers' : ['h']*len(target_channels),
            'markercolor' : ['blue']*len(target_channels),
            'markerfacecolor' : ['lightblue']*len(target_channels),
            'marker_text' : [r'$s$']*len(target_channels),
            'marker_text_fontsize' : [8],
            'marker_text_color' : ['black']*len(target_channels),
            'markersize' : 15,
            'fillstyle' : 'full'
        }|node_info
    
        self.custom_fock_state_node(custom_fock_matrix = coupling_matrix, 
                                    node_type='custom fock matrix', 
                                    node_info = node_info)
        return
    
    def shift_direct(   self, 
                        target_channels: list[int] = [1],
                        shift_per_channel: list[int] = [0],
                        node_info: dict = None
                        ) -> None: 
        """ Apply a shift node to the circuit. The photon value in the target channel(s) is increased by the value in 
            the list 'shift_per_channel' modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, 
            shift value is 2 and target channel value is 1 the result is that target becomes (2+1)%3 = 0. This is not a 
            linear optical operation (photons are created in this process). 

            NOTE: This 'direct' node does not create a FockState matrix but directly changes the 'optical_components' of the states in the collection. 
            functionality is the same as for the regular 'shift' node but the implementation can be faster, depending on number of states in the collection 
            and the number of channels in the circuit.

            Args: 
                target_channels (list[int]): Channels that change value based on the values in 'shift_per_channel'
                shift_per_channel (list[int]): Shift value per channel
                node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
            
            Raises: 
                Exception when the input channel numbers do not match with the circuit.
                
        """
        if node_info == None: node_info = {}
        node_info = {
            'label' : "shift",
            'channels_optical' : target_channels ,
            'channels_classical' : [],
            'markers' : ['h']*len(target_channels),
            'markercolor' : ['blue']*len(target_channels),
            'markerfacecolor' : ['lightblue']*len(target_channels),
            'marker_text' : [r'$s$']*len(target_channels),
            'marker_text_fontsize' : [8],
            'marker_text_color' : ['black']*len(target_channels),
            'markersize' : 15,
            'fillstyle' : 'full'
        }|node_info
        def generic_function(input_collection,parameters):
            shift_per_channel = parameters[0]
            target_channels = parameters[1]
            output_collection = input_collection.copy()
            output_collection.clear()
            for state in input_collection:
                new_state = state.copy()
                old_oc = state.optical_components.copy()
                new_oc = dict([])
                for component, amp_prob in old_oc.items():
                    values = state._dict_of_valid_component_names[component].copy()
                    for shift, target in zip(shift_per_channel, target_channels):
                        values[target] = (values[target] + shift)%state._length_of_fock_state
                    new_component = state._dict_of_optical_values[tuple(values)]
                    new_oc.update({new_component:amp_prob})
                new_state.optical_components = new_oc
                output_collection.add_state(new_state)
            return output_collection
        
        node_to_be_added = {
            'generic_function' : generic_function,
            'generic_function_parameters' : [shift_per_channel,target_channels],
            'node_type' : 'generic function on collection', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)


    def c_shift_direct(self, 
                control_channels: list[int] = [0],
                target_channels: list[int] = [1],
                node_info: dict = None
                ) -> None: 
        """ Apply a controlled-shift (c-shift) node to the circuit. The photon value in the target channel(s) is increased by the value in 
            the control channels(s) modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, control channel value is 2 and 
            target channel valie is 1 the result is that control value remains 2 and target becomes (2+1)%3 = 0The control channel(s) remain unaffected. 
            This is not a linear optical operation (photons are created in this process). The node can create entanglement and allows 'coupling' 
            between optical channels.

            NOTE: This 'direct' node does not create a FockState matrix but directly changes the 'optical_components' of the states in the collection. 
            functionality is the same as for the c_shift node but the implementation can be faster, depending on number of states in the collection 
            and the number of channels in the circuit.

            Args: 
                control_channels (list[int]): Channels that 'control' the target channel and remain unchanged themselves
                target_channels (list[int]): Channels that change value based on the values in the control channels
                node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
            
            Raises: 
                Exception when the input channel numbers do not match with the circuit.
                
        """
        if node_info == None: node_info = {}
        node_info = {
                'label' : "c-shift",
                'channels_optical' : control_channels + target_channels ,
                'channels_classical' : [],
                'markers' : ['o']*len(control_channels) + ['h']*len(target_channels),
                'markercolor' : ['black']*len(control_channels) + ['blue']*len(target_channels),
                'markerfacecolor' : ['black']*len(control_channels) + ['lightblue']*len(target_channels),
                'marker_text' : [r'$c$']*len(control_channels) + [r'$t$']*len(target_channels),
                'marker_text_fontsize' : [8],
                'marker_text_color' : ['white']*len(control_channels) + ['black']*len(target_channels),
                'markersize' : 15,
                'fillstyle' : 'full'
            }|node_info
        
        def generic_function(input_collection,parameters):
            control_channels = parameters[0]
            target_channels = parameters[1]
            output_collection = input_collection.copy()
            output_collection.clear()
            for state in input_collection:
                new_state = state.copy()
                old_oc = state.optical_components.copy()
                new_oc = dict([])
                for component, amp_prob in old_oc.items():
                    values = state._dict_of_valid_component_names[component].copy()
                    for control, target in zip(control_channels, target_channels):
                        values[target] = (values[target] + values[control])%state._length_of_fock_state
                    new_component = state._dict_of_optical_values[tuple(values)]
                    new_oc.update({new_component:amp_prob})
                new_state.optical_components = new_oc
                output_collection.add_state(new_state)
            return output_collection
        
        node_to_be_added = {
            'generic_function' : generic_function,
            'generic_function_parameters' : [control_channels,target_channels],
            'node_type' : 'generic function on collection', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
    
    def c_shift(self, 
                control_channels: list[int] = [0],
                target_channels: list[int] = [1],
                node_info: dict = None
                ) -> None:  
        """ Apply a controlled-shift (c-shift) node to the circuit. The photon value in the target channel(s) is increased by the value in 
            the control channels(s) modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, control channel value is 2 and 
            target channel valie is 1 the result is that control value remains 2 and target becomes (2+1)%3 = 0The control channel(s) remain unaffected. 
            This is not a linear optical operation (photons are created in this process). The node can create entanglement and allows 'coupling' 
            between optical channels.

            Args: 
                control_channels (list[int]): Channels that 'control' the target channel and remain unchanged themselves
                target_channels (list[int]): Channels that change value based on the values in the control channels
                node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
            
            Raises: 
                Exception when the input channel numbers do not match with the circuit.
                
        """
               # if channel input is a channel number (integer) and not a list make it a list
        if type(control_channels) == type(1):
            control_channels = [control_channels]
        if type(target_channels) == type(1):
            target_channels = [target_channels]

        if node_info == None: node_info = {}
        node_info = {
                'label' : "c-shift",
                'channels_optical' : control_channels + target_channels ,
                'channels_classical' : [],
                'markers' : ['o']*len(control_channels) + ['h']*len(target_channels),
                'markercolor' : ['black']*len(control_channels) + ['blue']*len(target_channels),
                'markerfacecolor' : ['black']*len(control_channels) + ['lightblue']*len(target_channels),
                'marker_text' : [r'$c$']*len(control_channels) + [r'$t$']*len(target_channels),
                'marker_text_fontsize' : [8],
                'marker_text_color' : ['white']*len(control_channels) + ['black']*len(target_channels),
                'markersize' : 15,
                'fillstyle' : 'full'
            }|node_info


        # check if lists with channel numbers are ok, otherwise throw exception
        error = False
        if len(control_channels) == len(target_channels):
            for control, target in zip(control_channels,target_channels):
                if min(control, target) < 0 or max(control, target) > self._no_of_optical_channels:
                    error = True
                if control == target:
                    error = True
        else:
            error = True
        if error:
            Exception('Error in channel input for function FockStateCircuit.c_shift')
            
        
        this_is_first_round = True  
        # run through the combinations of control and target 
        # for each combination create a transition matrix
        # multiply the matrices to get to the final overall matrix
        for control,target in zip(control_channels,target_channels):
            # create an identity matrix for the given combination of one target and one control channel
            coupling_matrix_one_target_control_combination = np.zeros((len(self._list_of_fock_states),len(self._list_of_fock_states)), dtype = np.ubyte)
            for input_index, input in enumerate(self._list_of_fock_states):
                for output_index, output in enumerate(self._list_of_fock_states):
                    # except for target channel values should be equal
                    if all([input[channel] == output[channel] for channel in range(self._no_of_optical_channels) if channel != target]):
                    # for target channel the value has to be (control + original target value) % length. So shift with control value modulus length
                        if (output[target] == (input[control] + input[target])%self._length_of_fock_state):
                                coupling_matrix_one_target_control_combination[output_index][input_index] = np.ubyte(1)
            if this_is_first_round:
                coupling_matrix = coupling_matrix_one_target_control_combination
                this_is_first_round = False
            else:
                coupling_matrix  = np.matmul(coupling_matrix, coupling_matrix_one_target_control_combination)

        self.custom_fock_state_node(custom_fock_matrix = coupling_matrix, node_type='custom fock matrix', node_info = node_info)

        return

    def channel_coupling(self, 
                        control_channels: list[int] = [0],
                        target_channels: list[int] = [1],
                        coupling_strength: float = 0,
                        node_info: dict = None
                        ) -> None:  
        """ Apply a node to the circuit to couple channels with the given 'coupling_strength'. The node will effectively 
            apply a controlled shift from control channels to target channels.

            Args: 
                control_channels (list[int]): Channels that 'control' the target channel and remain unchanged themselves
                target_channels (list[int]): Channels that change value based on the values in the control channels
                coupling_strength (float): Set coupling strength between 0 (no coupling, node does nothing) and 1 (full coupling)
                node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
            
            Raises: 
                Exception when the input channel numbers do not match with the circuit.
                
        """
        # if channel input is a channel number (integer) and not a list make it a list
        if type(control_channels) == type(1):
            control_channels = [control_channels]
        if type(target_channels) == type(1):
            target_channels = [target_channels]

        if node_info == None: node_info = {}
        node_info = {
                'label' : "coupling",
                'channels_optical' : control_channels + target_channels ,
                'channels_classical' : [],
                'markers' : ['o']*len(control_channels) + ['h']*len(target_channels),
                'markercolor' : ['black']*len(control_channels) + ['blue']*len(target_channels),
                'markerfacecolor' : ['black']*len(control_channels) + ['lightblue']*len(target_channels),
                'marker_text' : [r'$c$']*len(control_channels) + [r'$t$']*len(target_channels),
                'marker_text_fontsize' : [8],
                'marker_text_color' : ['white']*len(control_channels) + ['black']*len(target_channels),
                'markersize' : 15,
                'fillstyle' : 'full'
            }|node_info


        # check if lists with channel numbers are ok, otherwise throw exception
        error = False
        if len(control_channels) == len(target_channels):
            for control, target in zip(control_channels,target_channels):
                if min(control, target) < 0 or max(control, target) > self._no_of_optical_channels:
                    error = True
                if control == target:
                    error = True
        else:
            error = True
        if error:
            Exception('Error in channel input for function FockStateCircuit.channel_coupling')
            
        
        # the coupling matrix is pre-filled as identity matrix
        coupling_matrix = np.identity(len(self._list_of_fock_states), dtype = np.csingle)
        
        # run through the combinations of control and target 
        # for each combination create a transition matrix
        # multiply the matrices to get to the final overall matrix
        for control,target in zip(control_channels,target_channels):
            # create an identity matrix for the given combination of one target and one control channel
            coupling_matrix_one_target_control_combintation = np.identity(len(self._list_of_fock_states), dtype = np.csingle)
            for input_index, input in enumerate(self._list_of_fock_states):
                for output_index, output in enumerate(self._list_of_fock_states):
                    # except for target channel values should be equal
                    valid_transition = all([input[channel] == output[channel] for channel in range(self._no_of_optical_channels) if channel != target])
                    # for target channel the value has to be (control + original target value) % length. So shift with control value modulus length
                    valid_transition = valid_transition and (output[target] == (input[control] + input[target])%self._length_of_fock_state)
                    # check if on diagonal
                    on_diagonal = (input_index == output_index)
                    if valid_transition and not on_diagonal:
                        coupling_matrix_one_target_control_combintation[output_index][input_index] = np.sqrt(np.cdouble(coupling_strength)*(1 + 0*1j))
                    elif valid_transition and on_diagonal:
                        coupling_matrix_one_target_control_combintation[output_index][input_index] = np.cdouble(1)
                    elif (not valid_transition) and on_diagonal:
                        coupling_matrix_one_target_control_combintation[output_index][input_index] = np.sqrt(np.cdouble((1-coupling_strength))*(1 + 0*1j))
                    else:
                        coupling_matrix_one_target_control_combintation[output_index][input_index] = np.cdouble(0)
            coupling_matrix  = np.matmul(coupling_matrix, coupling_matrix_one_target_control_combintation)

        self.custom_fock_state_node(custom_fock_matrix = coupling_matrix, node_type='custom fock matrix', node_info = node_info)

        return
    
    def generic_function_on_collection(self, 
                                   function, 
                                   affected_optical_channels: list[int] = None, 
                                   affected_classical_channels: list[int] = None, 
                                   parameters: list[int] = None, 
                                   node_info: dict = None) -> None: 
        """ Perform a calculation/operation on the full collection of states. The function should return a transformed collection matching the layout of the circuit.
            This is the most generic node giving full freedom for a transformation on the 
            
            The function that is passed as argument will be called as:
                collection_of_states_output = generic_function(collection_of_states_input,generic_function_parameters)

            Example 1, fill classical channels:
            def generic_function(input_collection,parameters):
                for state in input_collection:
                    state.classical_channel_values = parameters[:2]
                return input_collection

            Example 2, perform controlled shift operation:
            def generic_function(input_collection,parameters):
                control_channel = parameters[0]
                target_channel = parameters[1]
            
                for state in input_collection:
                    new_components = dict([])
                    old_components = state.optical_components
                    for name, amp_prob in old_components.items():
                        old_values = state._dict_of_valid_component_names[name].copy()
                        old_values[target_channel] = (old_values[target_channel] + old_values[control_channel])%input_collection._length_of_fock_state
                        new_name = input_collection._get_state_name_from_list_of_photon_numbers(old_values)
                        new_components.update({new_name:amp_prob})
                    state.optical_components  = new_components
                return input_collection
        
        Args:
            function (_type_): function to be executed on the full collection of states
            affected_optical_channels (list[int], optional): optical channels (only used in drawing circuit). Defaults to None.
            affected_classical_channels (list[int], optional): classical channels (only used in drawing circuit). Defaults to None.
            parameters (list[int], optional): list of parameters passed to the function. Defaults to None.
            node_info (dict, optional): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used. Defaults to None.       
 
        Returns:
            nothing
        
        Raises:
            nothing
        """ 
        if affected_optical_channels == None:
            affected_optical_channels = [channel for channel in range(self._no_of_optical_channels)]

        if affected_classical_channels == None:
            affected_classical_channels = [channel for channel in range(self._no_of_classical_channels)]
        
        if parameters == None:
            parameters = []

        if node_info == None: node_info = {}
        node_info = {
                'label' : "gen. function",
                'channels_optical' : affected_optical_channels,
                'channels_classical' : affected_classical_channels,
                'markers' : ['s'],
                'marker_text' : [''],
                'markercolor' : ['black'],
                'markerfacecolor' : ['black'],
                'markersize' : 20,
                'fillstyle' : 'full'
            }|node_info

        node_to_be_added = {
            'generic_function' : function,
            'generic_function_parameters' : parameters,
            'node_type' : 'generic function on collection', 
            'node_info' : node_info
        }
        self._update_list_of_nodes(node_to_be_added = node_to_be_added)
  
        return