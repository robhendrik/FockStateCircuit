import random
import string
import math
import numpy as np
from fock_state_circuit.state import State
from fock_state_circuit.visualization.plot import Plot
from fock_state_circuit.nodes.nodelist.node_list import NodeList
from fock_state_circuit.nodes.nodelist.optical_values_component_names import ComponentNames,OpticalValues

class CollectionOfStates(Plot):
    """ CollectionOfStates is a class containing a collection of instances of the class State. A State describes the values in the optical
        and classical channels of an instance of class FockStateCircuit. The collection of states is evaluated by a circuit to model and 
        describe the evolution of the states through the circuit. 
        
        An instance of CollectionOfStates can be generate by passing a FockStateCircuit as argument to the class constructor. The resulting
        instance then takes parameters (like the number of optical channels, the number of classical channels, maximum number of photons) from 
        the FockStateCircuit. 

        The CollectionOfStates can be default populated with all pure photon numbers states for the circuit, or can be created with specific states
        by also passing a input_collection_as_a_dict as an argument to the constructor. If this argument is an empty dictionary (dict([])) the collection
        will be created without state (so an empty collection).

        After evaluation the FockStateCircuit will return a Collection of States. 

        The collection_of_states is an iterable. We can run though all states with 'for state in collection_of_states'.

        We can evaluate the CollectionOfStates by reviewing the states in the collection, but can also directly visualize the collection by calling
        CollectionOfStates.plot() or CollectionOfStates.plot_correlations(). These methods are described in the class 
        fock_state_circuit.visualization.plot.Plot

        States are stored in a dictionary. The key is an 'identifier'. Data structure (including the structure of the underlying class State) 
        is given below.

        {'identifier_1': {
                    'initial_state': 'state_name1',
                    'cumulative_probability' : 1.0,
                    'optical_components': { '1011':{'amplitude': 0.71 + 0j, 'probability': 0.5},
                                            '1110':{'amplitude': 0.71 + 0j, 'probability': 0.5}
                                            }
                    'classical_channel_values': [0,3,3.14]
                    'measurement_results': [{'classical_channel_values': [0,3,0],'probability': 0.5},
                                            {'classical_channel_values': [0,3,0],'probability': 0.5}]
                    }
        'identifier_2:   {
                    'initial_state': 'state_name2',
                    'cumulative_probability' : 1.0,
                    'optical_components': { '101':{'amplitude': 1.0 + 0j, 'probability': 1.0}
                                            }
                    'classical_channel_values': [0,0,0]
                    'measurement_results': [{'classical_channel_values': [0,0,0],'probability': 1.0}]
                    }


        Methods:
        __init__(self, fock_state_circuit: any, input_collection_as_a_dict: dict[str,"State"] = None):
            Constructor for an instance of the class CollectionOfStates. The instance takes its parameters from the FockStateCircuit in 
                the arguments. If input_collection_as_a_dict is passed together with a circuit that dictionary will be used 
                to populate the states. In that case the dictionary needs to have values of type 'State'.
                If no input_collection_as_a_dict is passed the collection will be populated by all
                basis states supported by the circuit. If an empty dict is passed (input_collection_as_a_dict = dict([])) the collection of states
                will be empty.
        
        __str__(self) -> str:
                Function to print a collection of states in a digestible format. 
                collection_of_states.print_only_last_measurement = True/False determines whether the full measurement history is printed.               
        
        __len__(self) -> int:
                Function to return the number of states in the collection.
                        
        collection_as_dict(self) -> dict:
                Function to return the content of the collection of states as a dictionary. The keys will be the identifiers.
                The values will be the state (and will be of type "State"). To also get the state in type dict further conversion is needed.

        state_identifiers_as_list(self) -> list:
                Returns a list of all state identifiers in the circuit. The identifiers are the 'keys' use in the dict in which the data
                for the collection is stored.

        initial_states_as_list(self) -> list:
                Return the initial states in the collection as a list. For each state the initial state is returned, so one initial state can appear
                multiple times in the list.
        
        add_state(self, state: "State", identifier = None) -> str:
                Add a new state to the circuit. If identifier is given that will be used, otherwise random new identifier will be generated. Take care
                when using existing identifier, the function will overwrite the previous state with that identifier.

        get_state(self, identifier = None, initial_state = None) -> State:
                Function returns a state from the collection. The collection will remain unchanged (so the state is NOT removed).
                If an identifier is given as argument the function will search for the state with that identifier. If not found the
                function will look for a state with the initial_state that is passed as argument. If that is also not found the function
                returns 'None'.

        set_collection_as_statistical_mixture(self, list_of_tuples_state_weight: list) -> None:
                Sets the collection of states to a statistical mixture. The function replaces the original content of the collection.
                The input is a list of tuples in the form (state, weight). The collection is populated by all states in the input list, each
                with cumulative probability set to 'weight'. So, input list [(state1, 0.5), (state2, 0.5)] would lead to a 
                collection of states consisting of two states (state1 and state2), each with cumulative_probability set 0.5 and the same value
                for initial_state. The identifiers for the states are randomly generated. The name for the initial_state is take from the first state
                in the input list.

        delete_state(self, identifier: str = None, initial_state: str = None) -> None:
                Delete (remove) a state from the collection of states. Either the identifier or the initial_state is used to identify to state to remove.
                If more states exist with the same initial_state the first one will be removed. If identier or initial_states given as argument
                are not present in the collection the function will do nothing and leave the collection as it was. 

        collection_from_dict(self, collection_of_states_as_dictionary)-> None:
                Function to load the collection_of_states with values from the dictionary given as argument. All content in the orginal
                collection of states will be removed and replaced.
        
        __bool__(self) -> bool:
            Private function to check whether the collection of states is a valid collection.
            
        __check_valid_dictionary(self, collection_of_states_as_dictionary) -> bool:
                Private function to check whether a dictionary has valid keys and values to create a collection of states. The function
                checks whether all states in the dictionary are valid states. This function checks the dictionary given as argument and
                not the collection_of_states itself.
        
        filter_on_initial_state(self, initial_state_to_filter) -> None:
                Removes all states from the collection which do not match the filter criterium. In this case the state's initial_state needs to be present in the
                list "initial_state_to_filter". If no state in the collection has any of the initial_states given as input an empty collection will result.
            
        filter_on_identifier(self, identifier_to_filter: list) -> None:
                Removes all states from the collection which do not match the filter criterium. In this case the state identifier needs to be present in the list
                list "identifiers to filter". The identifier is the 'key' used in the dictionary '_collection_of_states' where the states are stored as values. So for a valid
                the collection should be reduced to one single state. If the identifier is not present in the collection and empty collection will result.
            
        filter_on_classical_channel(self, classical_channel_numbers: list, values_to_filter: list) -> None:
                Removes all states from the collection which do not match the filter criterium. In this case the classical channels need to have the right values.
                if classical_channel_numbers = [0,3] and values_to_filter = [9,10] only states with classical channel [0] value 9 and channel[3] value
                10 wil remain in the collection. If no state has the correct values in the classical channels an empty collection will result.

        copy(self, empty_template: bool = False) -> 'CollectionOfStates':
                Function to create a deep copy of the collection of states. Changing the copy will not affect the original state.
                Note 1: copy function will create new collection for "self._fock_state_circuit". Ensure this is set to the right circuit.
                Note 2: setting the bool 'empty_template' to true will return a collection with zero states. All supporting arguments will 
                    be set to the same value as the original.            
        
        generate_allowed_components_names_as_list_of_strings(self) -> list:
                Function to return a list of all allowed names for optical components for the given fock state circuit.

        initialize_to_default(self) -> None:
                Function initializes collection of states to default for the given fock state circuit. For each basis state in the circuit 
                a state in the collection will be generated (so a state for '100', for '101', for '111', for '001' etc for a 3 channel circuit).
                The states all have as identifier 'state_x', where x is a number. The initial state is always equal to name the of teh optical components
                (so '000' or '2132'). This optical component has amplitude 1 and probability 1. Classical channels are initialized to 0 for each channel.

                {'identifier_1': {
                            'initial_state': 'state_name1',
                            'cumulative_probability' : 1.0,
                            'optical_components': { '1011':{'amplitude': 0.71 + 0j, 'probability': 0.5},
                                                    '1110':{'amplitude': 0.71 + 0j, 'probability': 0.5}
                                                    }
                            'classical_channel_values': [0,3,3.14]
                            'measurement_results': [{'classical_channel_values': [0,3,0],'probability': 0.5},
                                                    {'classical_channel_values': [0,3,0],'probability': 0.5}]
                            }
                'identifier_2:   {
                            'initial_state': 'state_name2',
                            'cumulative_probability' : 1.0,
                            'optical_components': { '101':{'amplitude': 1.0 + 0j, 'probability': 1.0}
                                                    }
                            'classical_channel_values': [0,0,0]
                            'measurement_results': [{'classical_channel_values': [0,0,0],'probability': 1.0}]
                            }
        
        clear(self) -> None:
                Function to completely clear the collection of states.
                        
        clean_up(self, initial_state: str = '') -> None:
                Cleans up the collection_of_states. All optical components with low probability are removed and the total set 
                of optical components is renormalized. Then all states with the same 'initial_state' and the same 'optical_components'
                are grouped together (i.e., replaced by one state with as 'cumulative_probability' the sum of probabilities of the
                identical states). Finally, states with a low 'cumulative_probability' are removed. The threshold for removing is defined 
                by the parameter 'threshold_probability_for_setting_to_zero' which can be passed to the FockStateCircuit for which this 
                collection of states is valid.
                
                Warning: Function only checks for same 'initial_state' and same 'optical_components' if any other parameter differs between 
                the states will be discarded.
        
        _group_states_together_by_photon_number(self)-> dict[str,list[str]]:
                Group states together based on the total number of photons in the components of the state. The function
                returns a dictionary where the key indicates photon number and the value is a list of state identifiers. The 
                function groups states with components that all have the same photon number 
                Example: if components in the state are '005', '221' and '230' the key will be '=5'
                Example: if components in the state are '005', '211' and '200' the key will be '<=5'

        density_matrix(self, initial_state: str = '', decimal_places_for_trace: int = 2) -> dict:
                Create density matrix as well as trace of the density matrix and trace of density matrix squared. If no initial state 
                is given in the arguments all initial states are used (function returns densitry matices and traces per initial state). For trace values 
                the precision can be set with 'decimal_places_for_trace'.

                The function returns a dictionary: 
                {initial_state : {'density_matrix' : dm_mixed_state, 'trace' : trace_dm, 'trace_dm_squared' : trace_dm_squared}}
            
        reduce(self, optical_channels_to_keep: list[int] = [],
                            classical_channels_to_keep: list[int] = []
                            ) -> None:
                Reduces the number of optical and/or classical channels. 

                Optical channels are removed by 'tracing out'. Pure states will become statistical mixtures 
                since the channels that are traced out are effectively 'measured'. Classical channels are simply 
                removed irrespective of data. Note that measurement_results for states is not affected.

                This function will modify the current collection 'in_place'. For a new collection with different
                settings first create a copy and then modify the copy.
        
        adjust_length_of_fock_state(self, new_length_of_fock_state: int = 0):
                Adjusts the 'length of Fock state' for the collection. This determines the maximum
                number of photons per channel. If 'length of Fock state' is for instance 4 the allowed photon
                numbers are 0,1,2 and 3. 

                This function will modify the current collection 'in_place'. For a new collection with different
                settings first create a copy and then modify the copy.

        extend( self, extra_optical_channels: int = 0,
                    extra_classical_channels: int = 0,
                    statistical_distribution: list[int] = []
                    ) -> any:
                Extends the original collection of states with new optical and/or classical channels. 
                The optical channels will be filled according to the parameter 'statistical_distribution'. 
                If the 'length_of_fock_state' for the circuit is for instance 3 the allowed photon numbers are 0,1 and 2. 
                'statistical_distribution' should then be a list of length three. 
                - The probability for 0 photons in the new channels will be statistical_distribution[0]
                - The probability for 1 photon  in the new channels will be statistical_distribution[1]
                - etc
                So to fill new channels with value 0 photons 'statistical_distribution' should be [1,0,0, ..]
                (which will be created as default when an empty list [] is passed as parameter )

                Number of classical channels can also be extended. These will always be filled with value 0.

                This function will modify the current collection 'in_place'. For a new collection with different
                settings first create a copy and then modify the copy. 
        
        is_photon_resolved(self):
                Return False if not all states have data on 'photon_resolution' in their state.auxiliary_information.
                If this function returns True ALL states have this information.

        has_no_signalling_boxes(self):
                Return False if not all states have data on 'no_signalling_box' in their state.auxiliary_information.
                If this function returns True ALL states have this information.


        plot(self, classical_channels = [], initial_states = [], info_for_bar_plot = dict([]), histo_output_instead_of_plot = False):
                Function to plot a bar graph for a collection of states. The bars indicate the probability to go from an initial state
                to an outcome in the classical channels. The circuit has to include measurement to the classical channels. 
                Optionally the classical channels to use can be specified as well as a selection of initial states. The 

                The histogram returned will be of the form {'initial_state1': [  {'output_state': '1010', 'probability': 1.0},
                                                                                {'output_state': '0101', 'probability': 0.5}
                                                                                ]}

                If classical_channels is the empty list all channels will be used (this is the default value)
                If initial_states is the empty list all initial states will be used (this is the default value)

        plot_correlations(self, 
                          channel_combis_for_correlation : list, 
                          info_for_bar_plot = dict([]), 
                          correlation_output_instead_of_plot = False,
                          initial_states_to_assess = []):    
                Determine correlations between channels. If correlation is 1, the outcomes in the channels are always the same,
                if correlation is -1 the outcome is always different.

                Format for channel combinations is like: [(2,3),(4,5),(2,4),(3,5)]

                If correlation_output_instead_of_plot is set to True function will return correlation values instead of 
                creating a plot. Format for returned correlations is:
                Results in format: {'initial_state1': [0.7,1.0,0.7,-0.7], 
                                    'initial_state2': [0.5,0.5,0.5,-0.5]}

        Last modified: June 3rd, 2024
        
    """
    _VERSION = '1.0.1'
   
    def __init__(self, fock_state_circuit: any, input_collection_as_a_dict: dict[str,"State"] = None):
        """ Constructor for an instance of the class CollectionOfStates. The instance takes its parameters from the FockStateCircuit in 
            the arguments. If input_collection_as_a_dict is passed together with a circuit that dictionary will be used 
            to populate the states. In that case the dictionary needs to have values of type 'State'.
            If no input_collection_as_a_dict is passed the collection will be populated by all
            basis states supported by the circuit. If an empty dict is passed (input_collection_as_a_dict = dict([])) the collection of states
            will be empty.

        Args:
            fock_state_circuit ("FockStateCircuit"): FockStateCircuit to take the parameters from to form the collection of states. 
            input_collection_as_a_dict (dict, optional): Content to populate the collection. Dicationary needs to be in right format. Defaults to None.

        Raises:
            Exception: If input_collection_as_a_dict is not in the right format
        """        

        # the collection takes its paramaters from the fock state circuit on which it can be evaluated
        self._fock_state_circuit = fock_state_circuit
        self._input_collection_as_a_dict = input_collection_as_a_dict

        # the length of the fock state is the number of possible photon numbers. So if the 'length' is 2 the maximum
        # number of photons is 1 (either 0 or 1 photons). If the length is 4 we can have 0,1,2 of 3 photons per channel.
        self._length_of_fock_state = fock_state_circuit._length_of_fock_state

        # the number of channels defining the circuit.
        self._no_of_optical_channels = fock_state_circuit._no_of_optical_channels
        self._no_of_classical_channels = fock_state_circuit._no_of_classical_channels

        # for naming the states we need a convention. if 'channel_0_left_in_state_name' is set to 'True' we
        # write a state with 2 photons in channel 0 and 5 photons in channel 1 as '05'. With this value set
        # to 'False' we would write this same state as '50'. 
        self._channel_0_left_in_state_name = fock_state_circuit._channel_0_left_in_state_name

        # probabilities below this threshold are set to zero. 
        # this means that if the probability to find an outcome is lower than this threshold the outcome is discarded.
        self._threshold_probability_for_setting_to_zero = fock_state_circuit._threshold_probability_for_setting_to_zero

        # parameter use for __str__. With print_only_last_measurement set to True only last measurement is printed. With False the
        # complete history is printed.
        self.print_only_last_measurement = True
        
        # create a dict with as keys the valid state names as strings and the list of photon numbers as value
        self._dict_of_valid_component_names = fock_state_circuit._dict_of_valid_component_names

        # create a dict with as keys the tuple of the list of photon numbers as value, 
        # and valid state names as strings as value
        self._dict_of_optical_values = fock_state_circuit._dict_of_optical_values 

        # either load the collection with default data, or populate from the collection used in the input of __init__
        if input_collection_as_a_dict is None:
            self.initialize_to_default()
        elif input_collection_as_a_dict == dict([]):
            self.clear()
        else:
            if not self.__check_valid_dictionary(input_collection_as_a_dict):
                raise Exception('Invalid dictionary used to create a new collection of states')
            self._collection_of_states = dict([])
            for identifier, state in input_collection_as_a_dict.items():
                if isinstance(state, State):
                    self._collection_of_states.update({identifier : state})
                if isinstance(state, dict):
                    self._collection_of_states.update({identifier : State(self,input_state_as_a_dict=state)})

    def __repr__(self) -> str:
        text = f'CollectionOfStates("{self._fock_state_circuit}","{self._input_collection_as_a_dict}")'
        return text

    def __str__(self) -> str:
        """ Function to print a collection of states in a digestible format. 
            collection_of_states.print_only_last_measurement = True/False determines whether the full measurement history is printed.
        """        
        text = 'Printing collection of states\n'
        text += 'Number of states in collection: {}\n'.format(len(self._collection_of_states))
        for name,state in self._collection_of_states.items():
            text += 'Identifier: {!r}\n'.format(name)
            text += str(state)
        return text
    
    def __len__(self) -> int:
        """ Function to return the number of states in the collection.
        """        
        return len(self._collection_of_states)
    
    def __getitem__(self, identifier):
        if identifier in self._collection_of_states.keys():
            return self._collection_of_states[identifier]
        elif isinstance(identifier, int):
            return list(self._collection_of_states.values())[identifier]
        else:
            return None

    def items(self):
        return self._collection_of_states.items()

    def keys(self):
        return self._collection_of_states.keys()

    def values(self):
        return self._collection_of_states.values()
        
    def collection_as_dict(self) -> dict:
        """ Function to return the content of the collection of states as a dictionary. The keys will be the identifiers.
            The values will be the state (and will be of type "State"). To also get the state in type dict further conversion is needed.

        Returns:
            dict: dictionary with the content from the collection of states.
        """        
        return self._collection_of_states
    
    def state_identifiers_as_list(self) -> list:
        """ Returns a list of all state identifiers in the circuit. The identifiers are the 'keys' use in the dict in which the data
            for the collection is stored.

        Returns:
            list: list of state identifiers
        """        
        return list(self._collection_of_states.keys())

    def initial_states_as_list(self) -> list:
        """ Return the initial states in the collection as a list. For each state the initial state is returned, so one initial state can appear
            multiple times in the list.

        Returns:
            list: all  initial states in the collection of states.
        """        
        initial_states_as_list = []
        for identifier, state in self._collection_of_states.items():
            initial_states_as_list.append(state.initial_state)
        return initial_states_as_list
    
    def add_state(self, state: "State", identifier = None) -> str:
        """ Add a new state to the circuit. If identifier is given that will be used, otherwise random new identifier will be generated. Take care
        when using existing identifier, the function will overwrite the previous state with that identifier.

        Returns: Nothing
        """    
    
        if identifier is None:
            alphabet = list(string.ascii_lowercase) + [str(i) for i in range(10)]
            identifier = 'identifier_' + alphabet[random.randint(0,len(alphabet)-1)]
            while identifier in self._collection_of_states.keys():
                identifier += alphabet[random.randint(0,len(alphabet)-1)]
            
        self._collection_of_states.update({identifier:state})
        
        return

    def get_state(self, identifier = None, initial_state = None) -> State:
        """ Function returns a state from the collection. The collection will remain unchanged (so the state is NOT removed).
            If an identifier is given as argument the function will search for the state with that identifier. If not found the
            function will look for a state with the initial_state that is passed as argument. If that is also not found the function
            returns 'None'.

        Args:
            identifier (_type_, optional): 
            . Defaults to None.
            initial_state (_type_, optional): _description_. Defaults to None.

        Returns:
            State: _description_
        """        
        if identifier is not None and identifier in self._collection_of_states.keys():
            return self._collection_of_states[identifier]
        elif initial_state is not None:
            for identifier, state in self._collection_of_states.items():
                if state.initial_state == initial_state:
                    return state
        else:
            return None
        
    def set_collection_as_statistical_mixture(self, list_of_tuples_state_weight: list) -> None:
        """ Sets the collection of states to a statistical mixture. The function replaces the original content of the collection.
            The input is a list of tuples in the form (state, weight). The collection is populated by all states in the input list, each
            with cumulative probability set to 'weight'. So, input list [(state1, 0.5), (state2, 0.5)] would lead to a 
            collection of states consisting of two states (state1 and state2), each with cumulative_probability set 0.5 and the same value
            for initial_state. The identifiers for the states are randomly generated. The name for the initial_state is take from the first state
            in the input list.

        Args:
            list_of_tuples_state_weight (list): List of tuples in the form [(state1, weight), (state2, weight)]
        """        
        mixture = dict([])
        alphabet = list(string.ascii_lowercase) + [str(i) for i in range(10)]
        initial_state = 'mixture-'
        for _ in range(5):
            initial_state += alphabet[random.randint(0,len(alphabet)-1)]

        for state, weight in list_of_tuples_state_weight:
            # make random identifier which is not yet used in self._collection_of_states.keys()
            identifier = 'identifier_' + alphabet[random.randint(0,len(alphabet)-1)]
            while identifier in self._collection_of_states.keys():
                identifier += alphabet[random.randint(0,len(alphabet)-1)]

            # for initial state use initial state name in first state and copy to all in the mixture

            if bool(state):
                state.cumulative_probability = weight
                state.initial_state = initial_state
                mixture.update({identifier : state})
        self._collection_of_states = mixture

        return

    def delete_state(self, identifier: str = None, initial_state: str = None) -> None:
        """ Delete (remove) a state from the collection of states. Either the identifier or the initial_state is used to identify to state to remove.
            If more states exist with the same initial_state the first one will be removed. If identier or initial_states given as argument
            are not present in the collection the function will do nothing and leave the collection as it was. 

        Args:
            identifier (str, optional): identifier for the state to be removed. Defaults to None.
            initial_state (str, optional): initial_state for the state to be removed. Defaults to None.

        """        
        if identifier is not None and identifier in self._collection_of_states.keys():
            del self._collection_of_states[identifier]
            return
        elif initial_state is not None:
            for identifier, state in self._collection_of_states.items():
                if state.initial_state == initial_state:
                    del self._collection_of_states[identifier]
                    return  
        return
    
    def collection_from_dict(self, collection_of_states_as_dictionary)-> None:
        """ Function to load the collection_of_states with values from the dictionary given as argument. All content in the orginal
            collection of states will be removed and replaced.
        
        Returns:
            bool: True if dictionary is valid, otherwise false.
        """
        if self.__check_valid_dictionary(collection_of_states_as_dictionary):
            self._collection_of_states = collection_of_states_as_dictionary
        return
    
    def __bool__(self) -> bool:
        """ Private function to check whether the collection of states is a valid collection.
        
        Returns:
            bool: True if dictionary is valid, otherwise false.
        """ 
        return self.__check_valid_dictionary(self._collection_of_states)
    
    
    def __check_valid_dictionary(self, collection_of_states_as_dictionary) -> bool:
        """ Private function to check whether a dictionary has valid keys and values to create a collection of states. The function
            checks whether all states in the dictionary are valid states. This function checks the dictionary given as argument and
            not the collection_of_states itself.

        Args:
            collection_of_states_as_dictionary (_type_): _description_

        Returns:
            bool: True if dictionary is valid, otherwise false.
        """        
        if not type(collection_of_states_as_dictionary) == type(dict([])):
            return False     
        for identifier, state in collection_of_states_as_dictionary.items():
            if not type(identifier) == type('name') and isinstance(state, State) and bool(state):
                return False
        return True
    
    def filter_on_initial_state(self, initial_state_to_filter) -> None:
        """ Removes all states from the collection which do not match the filter criterium. In this case the state's initial_state needs to be present in the
            list "initial_state_to_filter". If no state in the collection has any of the initial_states given as input an empty collection will result.

        Args:
            initial_state_to_filter (list of strings): list of initial states to keep in resulting collection. All other states are removed.
        """    
        if initial_state_to_filter is not None and type(initial_state_to_filter) != type([]):
            initial_state_to_filter = [initial_state_to_filter]

        new_selection_of_states = dict([])
        for identifier,state in self._collection_of_states.items():
            if initial_state_to_filter is not None and state.initial_state in initial_state_to_filter:
                new_state_identifier = "{:s}".format(identifier)
                new_state = state.copy()
                new_selection_of_states.update({new_state_identifier : new_state})

        self._collection_of_states = new_selection_of_states
        return 
        
    def filter_on_identifier(self, identifier_to_filter: list) -> None:
        """ Removes all states from the collection which do not match the filter criterium. In this case the state identifier needs to be present in the list
            list "identifiers to filter". The identifier is the 'key' used in the dictionary '_collection_of_states' where the states are stored as values. So for a valid
            the collection should be reduced to one single state. If the identifier is not present in the collection and empty collection will result.

        Args:
            identifier_to_filter (list of strings): list of identifiers to keep in resulting collection. All other states are removed.
        """        
        if identifier_to_filter is not None and type(identifier_to_filter) != type([]):
            identifier_to_filter = [identifier_to_filter]

        new_selection_of_states = dict([])
        for identifier,state in self._collection_of_states.items():
            if identifier_to_filter is not None and identifier in identifier_to_filter:
                new_state_identifier = "{:s}".format(identifier)
                new_state = state.copy()
                new_selection_of_states.update({new_state_identifier : new_state})

        self._collection_of_states = new_selection_of_states
        return 
        
    def filter_on_classical_channel(self, classical_channel_numbers: list, values_to_filter: list) -> None:
        """ Removes all states from the collection which do not match the filter criterium. In this case the classical channels need to have the right values.
            if classical_channel_numbers = [0,3] and values_to_filter = [9,10] only states with classical channel [0] value 9 and channel[3] value
            10 wil remain in the collection. If no state has the correct values in the classical channels an empty collection will result.

        Args:
            classical_channel_numbers (list): list of classical channel numbers
            values_to_filter (list): lis of values on which the given classical channels will be filtered.
        """
        new_selection_of_states = dict([])
        list_of_tuples = [(classical_channel_numbers[index], values_to_filter[index]) for index in range(min(len(classical_channel_numbers), len(values_to_filter)))]
        for identifier,state in self._collection_of_states.items():              
            if all([(state.classical_channel_values[channel] == value) for channel, value in list_of_tuples]):
                    new_state_identifier = identifier
                    new_state = state.copy()
                    new_selection_of_states.update({new_state_identifier : new_state})

        self._collection_of_states = new_selection_of_states
        return 

    def copy(self, empty_template: bool = False) -> 'CollectionOfStates':
        """ Function to create a deep copy of the collection of states. Changing the copy will not affect the original state.
            Note 1: copy function will create new collection for "self._fock_state_circuit". Ensure this is set to the right circuit.
            Note 2: setting the bool 'empty_template' to true will return a collection with zero states. All supporting arguments will 
                be set to the same value as the original.

        Returns:
            CollectionOfStates: new collection of states with identical values and parameters as the original
        """        
        if empty_template:
            return CollectionOfStates(self._fock_state_circuit, dict([]))
        else:
            new_selection_of_states = dict([])
            for identifier,state in self._collection_of_states.items():
                new_state_identifier = "{:s}".format(identifier)
                new_state = state.copy()
                new_selection_of_states.update({new_state_identifier : new_state})

            return CollectionOfStates(self._fock_state_circuit, new_selection_of_states)
              
    
    def generate_allowed_components_names_as_list_of_strings(self) -> list:
        """ Function to return a list of all allowed names for optical components for the given fock state circuit.

        Returns:
            list: list of strings describing photon numbers in the optical channels
        """        

        return list(self._dict_of_valid_component_names.keys())

    def initialize_to_default(self) -> None:
        """ Function initializes collection of states to default for the given fock state circuit. For each basis state in the circuit 
            a state in the collection will be generated (so a state for '100', for '101', for '111', for '001' etc for a 3 channel circuit).
            The states all have as identifier 'state_x', where x is a number. The initial state is always equal to name the of teh optical components
            (so '000' or '2132'). This optical component has amplitude 1 and probability 1. Classical channels are initialized to 0 for each channel.

            {'identifier_1': {
                        'initial_state': 'state_name1',
                        'cumulative_probability' : 1.0,
                        'optical_components': { '1011':{'amplitude': 0.71 + 0j, 'probability': 0.5},
                                                '1110':{'amplitude': 0.71 + 0j, 'probability': 0.5}
                                                }
                        'classical_channel_values': [0,3,3.14]
                        'measurement_results': [{'classical_channel_values': [0,3,0],'probability': 0.5},
                                                {'classical_channel_values': [0,3,0],'probability': 0.5}]
                        }
            'identifier_2:   {
                        'initial_state': 'state_name2',
                        'cumulative_probability' : 1.0,
                        'optical_components': { '101':{'amplitude': 1.0 + 0j, 'probability': 1.0}
                                                }
                        'classical_channel_values': [0,0,0]
                        'measurement_results': [{'classical_channel_values': [0,0,0],'probability': 1.0}]
                        }
        """   
        string_format_in_state_as_word = "{:0"+str(len(str(len(self.generate_allowed_components_names_as_list_of_strings()))))+ "d}"
        collection = dict([])
        for index, component in enumerate(self.generate_allowed_components_names_as_list_of_strings()):
            identifier = 'identifier_' + string_format_in_state_as_word.format(index)
            new_state = State(self)
            new_state.initial_state = component
            new_state.optical_components = {component: {'amplitude' : 1.0, 'probability' : 1.0}}
            collection.update({identifier : new_state})

        self._collection_of_states = collection
        
        return
    
    def clear(self) -> None:
        """ Function to completely clear the collection of states.
        """        

        self._collection_of_states = dict([])
        
        return
    
    def _group_states_together_by_photon_number(self)-> dict[str,list[str]]:
        """ Group states together based on the total number of photons in the components of the state. The function
            returns a dictionary where the key indicates photon number and the value is a list of state identifiers. The 
            function groups states with components that all have the same photon number 
            Example: if components in the state are '005', '221' and '230' the key will be '=5'
            Example: if components in the state are '005', '211' and '200' the key will be '<=5'

        Returns:
            dict: dictionary of with as value a list of state identifiers grouped by the photon numbers in their components, as a key
                the photon number plus and identifier whether all compononents have the same photon number or not.
        """
        states_grouped_by_photon_number = dict([])
        for identifier,state in self._collection_of_states.items():
            photon_number, equal = state.photon_number()
            if equal:
                key = '=' +  str(photon_number)
            else: 
                key = '<=' + str(photon_number)
            # add the state under the right key
            if key in states_grouped_by_photon_number.keys():
                # if key exists append to the list of names
                states_grouped_by_photon_number[key].append(identifier)
            else:
                # if new key create a new list and add to dictionary
                states_grouped_by_photon_number.update({key:[identifier]})
        return states_grouped_by_photon_number

    def density_matrix(self, initial_state: str = '', decimal_places_for_trace: int = 2) -> dict:
        """ Create density matrix as well as trace of the density matrix and trace of density matrix squared. If no initial state 
            is given in the arguments all initial states are used (function returns densitry matices and traces per initial state). For trace values 
            the precision can be set with 'decimal_places_for_trace'.

            The function returns a dictionary: 
            {initial_state : {'density_matrix' : dm_mixed_state, 'trace' : trace_dm, 'trace_dm_squared' : trace_dm_squared}}

            Args:
                initial_state (str): initial state for which to generate density matrix. If non given all initial_states are used.
                decimal_places_for_trace (int) : decimal places (precision) for the returned traces

            Returns:
                dict: dictionary with initial state as key. dict contains density matrix and values for the two traces
                
        """
        # either take the initial_state from the arguments, or if non is given make a set of all initial_states in the collection
        if initial_state != '':
            set_of_initial_states = set([initial_state])
        else:
            set_of_initial_states = set([state['initial_state'] for state in self._collection_of_states.values()])

        dictionary_with_return_values = dict([])
        for initial_state in set_of_initial_states:
            len_of_basis = len(self._dict_of_valid_component_names)
            dm_mixed_state = np.zeros((len_of_basis, len_of_basis), dtype=np.cdouble)
            for state in self._collection_of_states.values():
                if state.initial_state == initial_state:
                    vector, basis = state.translate_state_components_to_vector()
                    dm_single_state = np.outer(np.conjugate(vector),vector)
                    dm_mixed_state += state.cumulative_probability * dm_single_state

            trace_dm = np.round(np.abs(np.trace(dm_mixed_state)),decimals=decimal_places_for_trace)
            trace_dm_squared = np.round(np.abs(np.trace(np.matmul(dm_mixed_state,dm_mixed_state))),decimals=decimal_places_for_trace)
            dictionary_with_return_values.update({initial_state : {'density_matrix' : dm_mixed_state, 'trace' : trace_dm, 'trace_dm_squared' : trace_dm_squared}})

        return dictionary_with_return_values
        
    def reduce(self, optical_channels_to_keep: list[int] = [],
                classical_channels_to_keep: list[int] = []
                ) -> None:
        """ Reduces the number of optical and/or classical channels. 

            Optical channels are removed by 'tracing out'. Pure states will become statistical mixtures 
            since the channels that are traced out are effectively 'measured'. Classical channels are simply 
            removed irrespective of data. Note that measurement_results for states is not affected.

            This function will modify the current collection 'in_place'. For a new collection with different
            settings first create a copy and then modify the copy.

            Args:
                optical_channels_to_keep (list): list of optical channel numbers that should remain
                classical_channels_to_keep (list): list of classical channel numbers that should remain
           
            Returns:
                Nothing. The 'collection of states' will be modified in-place.
                
        """        
        def new_component_name_from_list(values):
            """ function to create string label from list of values"""
            new_component_names = ComponentNames(length_of_fock_state=self._length_of_fock_state,
                                                 no_of_optical_channels=len(values),
                                                 channel_0_left_in_state_name=self._channel_0_left_in_state_name)
            new_optical_values = OpticalValues(new_component_names)
        
            return new_optical_values[tuple(values)]
        
        if optical_channels_to_keep == []:
            optical_channels_to_keep  = [*range(self._no_of_optical_channels)]
        if classical_channels_to_keep ==  []:
            classical_channels_to_keep  = [*range(self._no_of_classical_channels)]

        if not all([channel in [*range(self._no_of_optical_channels)] for channel in optical_channels_to_keep]):
            raise Exception('error in channel selection in CollectionOfStates.reduce()')
        if not all([channel in [*range(self._no_of_classical_channels)] for channel in classical_channels_to_keep]):
            raise Exception('error in channel selection in CollectionOfStates.reduce()')
        
        # create empty collection_of_states with reduced number of channels
        updated_circuit = NodeList(length_of_fock_state = self._length_of_fock_state, 
                                no_of_optical_channels = len(optical_channels_to_keep), 
                                no_of_classical_channels= len(classical_channels_to_keep),
                                channel_0_left_in_state_name = self._channel_0_left_in_state_name
                                )
        
        new_collection_of_states = CollectionOfStates(fock_state_circuit=updated_circuit)
        new_collection_of_states.clear()
        new_collection_of_states._channel_0_left_in_state_name = self._channel_0_left_in_state_name
        new_collection_of_states._threshold_probability_for_setting_to_zero = self._threshold_probability_for_setting_to_zero
        new_collection_of_states.print_only_last_measurement = self.print_only_last_measurement

        # create a lookup table from original component names to reduced component names. 
        # for each original name there will be multiple names in the resulting collection
        lookup_table_component_names = dict([])
        string_format_in_state_as_word = "{:0"+str(len(str(self._length_of_fock_state-1)))+ "d}"
        for name, list in self._dict_of_valid_component_names.items():
            new_list = [list[channel_number] for channel_number in optical_channels_to_keep]
            new_name = new_component_name_from_list(new_list)
            traced_out_values = [list[number] for number in range(self._no_of_optical_channels) if number not in optical_channels_to_keep]
            traced_out_name = ''.join([string_format_in_state_as_word.format(number) for number in traced_out_values])
            lookup_table_component_names.update({name: {'new_name' : new_name, 'traced_out_values': traced_out_name }})

        for identifier, old_state in self._collection_of_states.items():
            # for each state in original collection group together the optical components that lead to the same 
            # optical component in the resulting state
            dict_by_traced_out_values = dict([])
            for component_name, amplitude_probability in old_state['optical_components'].items():
                new_component_name = lookup_table_component_names[component_name]['new_name']
                traced_out_value = lookup_table_component_names[component_name]['traced_out_values']
                if traced_out_value in dict_by_traced_out_values.keys():
                    dict_by_traced_out_values[traced_out_value].update({new_component_name:amplitude_probability})
                else:
                    dict_by_traced_out_values.update({traced_out_value : {new_component_name:amplitude_probability}})
            
            # calculate for the resulting optical component the amplitude and probability
            for traced_out_value, new_optical_components in dict_by_traced_out_values.items():
                cumulative_probability = 0
                for amplitude_probability in new_optical_components.values():
                    cumulative_probability += np.abs(amplitude_probability['amplitude'])**2
                scale_factor = math.sqrt((1/cumulative_probability))
                for component in new_optical_components.keys():
                    new_amplitude = new_optical_components[component]['amplitude'] * scale_factor
                    new_optical_components[component] = {'amplitude': new_amplitude, 'probability': np.abs(new_amplitude)**2}
                
                new_classical_values = [old_state['classical_channel_values'][index] for index in classical_channels_to_keep]
                new_state_as_a_dict = {   'initial_state' : old_state['initial_state'],
                    'cumulative_probability' :  old_state['cumulative_probability'] *cumulative_probability,
                    'optical_components' : new_optical_components, 
                    'classical_channel_values' : new_classical_values,
                    'measurement_results' : old_state['measurement_results'],
                    'auxiliary_information' : old_state['auxiliary_information']
                    }
                new_state = State(collection_of_states=new_collection_of_states, input_state_as_a_dict=new_state_as_a_dict)
                new_collection_of_states.add_state(state=new_state,identifier=identifier+traced_out_value)


            #self = new_collection_of_states.copy()
            self._fock_state_circuit = new_collection_of_states._fock_state_circuit
            self._collection_of_states = new_collection_of_states.collection_as_dict()
            self._no_of_classical_channels = new_collection_of_states._no_of_classical_channels
            self._no_of_optical_channels = new_collection_of_states._no_of_optical_channels
            self._dict_of_valid_component_names = new_collection_of_states._dict_of_valid_component_names
            self._dict_of_optical_values = new_collection_of_states._dict_of_optical_values

        return 
    
    def adjust_length_of_fock_state(self, new_length_of_fock_state: int = 0):
        """ Adjusts the 'length of Fock state' for the collection. This determines the maximum
            number of photons per channel. If 'length of Fock state' is for instance 4 the allowed photon
            numbers are 0,1,2 and 3. 

        This function will modify the current collection 'in_place'. For a new collection with different
        settings first create a copy and then modify the copy.

        Args:
            new_length_of_fock_state (int): Should be integer larger than 1. 
       
        Returns:
            Nothing. The 'collection of states' will be modified in-place.
                
        """

        if new_length_of_fock_state == 0:
            new_length_of_fock_state = self._length_of_fock_state
        if new_length_of_fock_state < 1:
            raise Exception('error in setting length_of_fock_state in CollectionOfStates._adjust_length_of_fock_state()')
        
        # create empty collection_of_states with new length of fock states
        updated_circuit = NodeList(length_of_fock_state = new_length_of_fock_state, 
                                no_of_optical_channels = self._no_of_optical_channels, 
                                no_of_classical_channels= self._no_of_classical_channels,
                                channel_0_left_in_state_name = self._channel_0_left_in_state_name
                                )
        new_collection_of_states = CollectionOfStates(fock_state_circuit= updated_circuit )
        new_collection_of_states.clear()
        new_collection_of_states._channel_0_left_in_state_name = self._channel_0_left_in_state_name
        new_collection_of_states._threshold_probability_for_setting_to_zero = self._threshold_probability_for_setting_to_zero
        new_collection_of_states.print_only_last_measurement = self.print_only_last_measurement
        
        for identifier, state in self._collection_of_states.items():
            new_optical_components = dict([])
            for name, amp_prob in state.optical_components.items():
                values = self._dict_of_valid_component_names[name]
                new_values = []
                for value in values:
                    new_values.append(value%new_length_of_fock_state)
                new_name = updated_circuit._dict_of_optical_values[tuple(new_values)]
                new_optical_components.update({new_name:amp_prob})
            new_state_as_a_dict = { 'initial_state' : state.initial_state,
                                    'cumulative_probability' : state.cumulative_probability,
                                    'optical_components' : new_optical_components,
                                    'classical_channel_values' : state.classical_channel_values,
                                    'measurement_results' : state.measurement_results,
                                    'auxiliary_information' : state.auxiliary_information
                                    }
            new_state = State(collection_of_states=new_collection_of_states,input_state_as_a_dict=new_state_as_a_dict)
            new_collection_of_states.add_state(state = new_state, identifier=identifier)
        
        # self = new_collection_of_states.copy()
        self._fock_state_circuit = new_collection_of_states._fock_state_circuit
        self._collection_of_states = new_collection_of_states.collection_as_dict()
        self._length_of_fock_state = new_collection_of_states._length_of_fock_state
        self._dict_of_valid_component_names = new_collection_of_states._dict_of_valid_component_names
        self._dict_of_optical_values = new_collection_of_states._dict_of_optical_values
    
        return

    
    def clean_up(self, initial_state: str = '') -> None:
        """ Cleans up the collection_of_states. All optical components with low probability are removed and the total set 
            of optical components is renormalized. Then all states with the same 'initial_state' and the same 'optical_components'
            are grouped together (i.e., replaced by one state with as 'cumulative_probability' the sum of probabilities of the
            identical states). Finally, states with a low 'cumulative_probability' are removed. The threshold for removing is defined 
            by the parameter 'threshold_probability_for_setting_to_zero' which can be passed to the FockStateCircuit for which this 
            collection of states is valid.
            
            Warning: Function only checks for same 'initial_state' and same 'optical_components' if any other parameter differs between 
            the states will be discarded.

            Args:
                initial_state (str): initial state to clean_up. If none is given all initial_states will be 'cleaned' sequentially
        
            Returns:
                Nothing. The 'collection of states' will be modified in-place.
        """
        # either take the initial_state from the arguments, or if non is given make a set of all initial_states in the collection
        if initial_state != '':
            set_of_initial_states = set([initial_state])
        else:
            set_of_initial_states = set([state['initial_state'] for state in self._collection_of_states.values()])

        for initial_state in set_of_initial_states:
            # check for any component in optical_components with too low probability and remove this component
            group_identifiers_with_same_optical_components = dict([])
            for identifier, state in self._collection_of_states.items():
                if state['initial_state'] == initial_state:
                    state._rescale_optical_components()

                    for other_identifier in group_identifiers_with_same_optical_components.keys():
                       if state._identical_optical_components(self._collection_of_states[other_identifier]):
                            group_identifiers_with_same_optical_components[other_identifier].append(identifier)
                            break
                    else: # for-loop ended without break, so no state with identical opt components found.
                        group_identifiers_with_same_optical_components.update({identifier:[]})

            for identifier, list_of_same_oc in group_identifiers_with_same_optical_components.items():
                overall_cumulative_probability = self._collection_of_states[identifier]['cumulative_probability']
                for other_identifier in list_of_same_oc:
                    overall_cumulative_probability = overall_cumulative_probability + self._collection_of_states[other_identifier]['cumulative_probability']
                    del self._collection_of_states[other_identifier]
                if overall_cumulative_probability >= self._threshold_probability_for_setting_to_zero:
                    self._collection_of_states[identifier]['cumulative_probability'] = overall_cumulative_probability
                else:
                    del self._collection_of_states[identifier]
        
        return

    def extend( self, extra_optical_channels: int = 0,
                extra_classical_channels: int = 0,
                statistical_distribution: list[int] = []
                ) -> any:
        """ Extends the original collection of states with new optical and/or classical channels. 
            The optical channels will be filled according to the parameter 'statistical_distribution'. 
            If the 'length_of_fock_state' for the circuit is for instance 3 the allowed photon numbers are 0,1 and 2. 
            'statistical_distribution' should then be a list of length three. 
            - The probability for 0 photons in the new channels will be statistical_distribution[0]
            - The probability for 1 photon  in the new channels will be statistical_distribution[1]
            - etc
            So to fill new channels with value 0 photons 'statistical_distribution' should be [1,0,0, ..]
            (which will be created as default when an empty list [] is passed as parameter )

            Number of classical channels can also be extended. These will always be filled with value 0.

            This function will modify the current collection 'in_place'. For a new collection with different
            settings first create a copy and then modify the copy.

            Args:
                extra_optical_channels (int): number of optical channels to be added
                extra_classical_channels (int): number of classical channels to be added
                statistical_distribution (list[int]): probabilities for photon numbers in new channels. Default 
                    new channels will be filled with 0 photons.
           
            Returns:
                Nothing. The 'collection of states' will be modified in-place.
        """
        def new_component_name_from_list(values):
            """ function to create string label from list of values"""
            new_component_names = ComponentNames(length_of_fock_state=self._length_of_fock_state,
                                                 no_of_optical_channels=len(values),
                                                 channel_0_left_in_state_name=self._channel_0_left_in_state_name)
            new_optical_values = OpticalValues(new_component_names)
        
            return new_optical_values[tuple(values)]
        
        # create default value if statistical_distribution is empty list
        if statistical_distribution == []:
            statistical_distribution = [0]*self._length_of_fock_state
            statistical_distribution[0] = 1
        elif len(statistical_distribution) != self._length_of_fock_state:
            raise Exception('error with channel numbers in function collection_of_states.extend')  
    
        # check if 'extra_optical_channels' is positive number
        if extra_optical_channels < 0 or extra_classical_channels < 0:
            raise Exception('error with channel numbers in function collection_of_states.extend')
        if extra_optical_channels == 0 and extra_classical_channels == 0:
            return self
        
        # create a new (still empty) collection_of_states with the extra optical channels
        updated_circuit  = NodeList(length_of_fock_state = self._length_of_fock_state, 
                        no_of_optical_channels = self._no_of_optical_channels+extra_optical_channels,
                        no_of_classical_channels= self._no_of_classical_channels+extra_classical_channels,
                        channel_0_left_in_state_name = self._channel_0_left_in_state_name
                        )
        new_collection_of_states = CollectionOfStates(fock_state_circuit= updated_circuit )
        new_collection_of_states.clear()
        new_collection_of_states._channel_0_left_in_state_name = self._channel_0_left_in_state_name
        new_collection_of_states._threshold_probability_for_setting_to_zero = self._threshold_probability_for_setting_to_zero
        new_collection_of_states.print_only_last_measurement = self.print_only_last_measurement

        # create a dict with values for extra channels and their probabilities

        extra_channel_values = dict({'initial_value' : (1,[])})
        for _ in range(extra_optical_channels):
            new_extra_channel_values = dict([])
            for old_probability, values in extra_channel_values.values():
                for index, fock_state_value in enumerate(range(self._length_of_fock_state)):
                    new_probability = statistical_distribution[index]*old_probability
                    new_values = values + [fock_state_value]
                    new_label = new_component_name_from_list(new_values)
                    new_extra_channel_values.update({new_label: (new_probability,new_values)})
            extra_channel_values = new_extra_channel_values

        # for each existing state create a statistical mixture for possible values in new channels
        for probability, values in extra_channel_values.values():
            for old_state in self._collection_of_states.values():
                optical_components = old_state['optical_components']
                new_optical_components = dict([])
                for component_name, amplitude_probability in optical_components.items():
   
                    old_values = list(self._dict_of_valid_component_names[component_name])
                    new_list = old_values + values
   
                    new_component_name = new_component_name_from_list(new_list)
       
                    new_optical_components.update({new_component_name:amplitude_probability})

                new_state_as_a_dict = {   'initial_state' : old_state['initial_state'],
                    'cumulative_probability' :  old_state['cumulative_probability']*probability,
                    'optical_components' : new_optical_components, 
                    'classical_channel_values' : old_state['classical_channel_values'] + [0]*extra_classical_channels,
                    'measurement_results' : old_state['measurement_results'],
                    'auxiliary_information' : old_state['auxiliary_information']
                    }
                if new_state_as_a_dict['cumulative_probability'] >= self._threshold_probability_for_setting_to_zero:
                    new_state = State(collection_of_states=new_collection_of_states, input_state_as_a_dict=new_state_as_a_dict)
                    new_collection_of_states.add_state(state=new_state)
            
        # self = new_collection_of_states.copy()
        self._fock_state_circuit = new_collection_of_states._fock_state_circuit
        self._collection_of_states = new_collection_of_states.collection_as_dict()
        self._no_of_classical_channels = new_collection_of_states._no_of_classical_channels
        self._no_of_optical_channels = new_collection_of_states._no_of_optical_channels
        self._dict_of_valid_component_names = new_collection_of_states._dict_of_valid_component_names
        self._dict_of_optical_values = new_collection_of_states._dict_of_optical_values
       
        return
    
    def is_photon_resolved(self):
        """ Return False if not all states have data on 'photon_resolution' in their state.auxiliary_information.
            If this function returns True ALL states have this information.
        """
        for state in self:
            if not 'photon_resolution' in state.auxiliary_information.keys():
                return False
        return True
    
    def has_pr_correlations(self):
        """ Return False if not all states have data on 'popescu_rohrlich_correlation' in their state.auxiliary_information.
            If this function returns True ALL states have this information.
        """
        for state in self:
            if not 'popescu_rohrlich_correlation' in state.auxiliary_information.keys():
                return False
        return True