from __future__ import annotations
from fock_state_circuit.state import State
import numpy as np

class ColumnOfStates():
    """ Class for a group of states (instances of FockStateCircuit.State). This class is used in situations where the 
        photons that create a 'state' have properties that need to tracked individually. A key example is when the timing
        of the photons is not the same, for instance because one photon had to travel a longer route from the moment of creation
        as another photon. The 'ColumnOfState' groups together the single photon states and allows to keep track of the 
        timing (arrival time, pulse width) of this single photon. Alternatively we could want to track spectral information, or 
        specific spatial information.

        The basic use case is when the original state consists of a single optical component ('121'). A column of single photon states 
        would then carry 4 states with optical components '100', '010', '010' and '001'.

        Attributes:
            column_amplitude (np.cdouble):
                The amplitude of the column. If we start with a state with two optical components
                (e.g., '100' with amplitude 0.71 and '010' with amplitude 0.71) then the two columns will have states with optical 
                components ('100', amplitude 1) for one column and ('010, amplitude 1) for the other. The colums will have 
                'column_amplitude' 0.71.

            column_boson_factor (float):
                This is the 'boson factor' related to expansion of the original state into single photons states. If a state
                |n> is decomposed the boson factor is sqrt(n!). So for states |3> the boson factor is sqrt(3 x 2 x 1) = sqrt(6). 

            column_identifier (int): 
                This is the index the column has in the InterferenceGroup.

            group_cumulative_probability (float):
                This is the group_cumulative_probability of the interference group to which the column belongs

            interference_group_identifier (int):
                This is the index of the InterferenceGroup to which the column belongs has in the CollectionOfStateColumns.

        Methods:
            __getitem__(self, identifier) -> State:
                return self.list_of_states[identifier]
            
            __len__(self) -> int:
                return len(self.list_of_states)
            
            __str__(self)-> str:
                Return the string for 'pretty printing' the interference column.

            add_state(self, state: State) -> None:
                    Add a new state to the column. The new state will adjusted to fit the column (i.e., cumulative probability will be
                    adjusted, as well as state.auxiliary_information['photon_resolution'][..] ).

                    If the state already has state.auxiliary_information['photon_resolution'] this will be overwritten.
                
            copy(self) -> ColumnOfStates:
                    Return new instance with copied contents. All states and information are copied to new instance.
                
        

            split(self, new_column_identifier: int = -1) -> ColumnOfStates:
                    This function searches the states in the column. If a state is found with more than one optical component the state is split
                    and a new time column is created. The first optical component will go to the new time column, the remaining components stay 
                    in the current column. All other states are copied.

                    If no state with more than one optical component is found in the time column the function returns 'None', otherwise the 
                    a new instance of ColumnOfStates is returned.

            single_photon_states(self) -> None:
                    Turn this column into a column with only single photon states and vacuum states.

                    If the vacuum state (state without photons) is part of the original column this state 
                    will be added to/remain in the column after applying this function. 

                    The attribute 'column_boson_factor' will reflect the fact that multi-photon states are split. Splitting a 
                    state with n photons in one channel and m in another "nm" will generate column_boson_factor sqrt(n! m!)

                    Example 1 : A state with optical component '201' will be expanded in three states with
                        optical components '100', '100' and '001' with column_boson_factor of sqrt(2)

                    Example 2 : A state with optical component '230' will be expanded in three states with
                        optical components '100', '100', '010', '010' and '010' with column_boson_factor of sqrt(12) = sqrt(2) x sqrt(3 x 2)
                
            list_of_photons(self) -> list:
                    Returns a list of photons. This is a list with one element per photon in the state. The elements in the 
                    list are tuples. The first element of the tuple is the channel for this photon, and the second element 
                    in the tuple is the dictionary 'photon_information' (taken from 
                    state.auxiliary_information['photon_resolution']['photon_information'])

                    If states in the column contain more than one optical component only the first optical component if considered.
            
            set_photon_information(self,photon_information: dict = dict([]), 
                                    filter_for_optical_values: list = None, 
                                    ault_photon_information: dict = None) -> None:
                    Update states in the column with 'photon information'. This is the information used to calculate 
                    the overlap between photons. The function calculating this overlap has to be able to use the dict
                    in the form that it is added here.

                    If filter_for_optical_values is used only states with optical values in this filter will be updated.
                    Other states will either get ault_photon_information is that is available in the arguments, or
                    will remain unchanged. 

                    filter_for_optical_values if a list of lists, or list of tuples. If we want to update only states
                    with one single photon in the first optical channel  filter_for_optical_values would be 
                    [[1,0,0]] or [(1,0,0)]

                    For adding timing information per photon the dict would be {'time_stamp': 0, 'pulse_width': 1}
                    For adding spectral information the dict would be {'frequency band' : 0, 'bandwidth' : 1}

                    If the key 'photon_information' is not present in state.auxiliary_information['photon_resolution']
                    it will be added, otherwise it will be updated.
                
            all_states_single_photon_states(self) -> bool:
                    Return True if all states in the column have exactly one or zero photons. If a state has more optical
                    components the function will check all components in that state. 
                
        
            all_states_single_optical_component(self) -> bool:
                    Return True if all states in the column have exactly one optical components 
        
            condense_column_to_single_state(self, use_group_cumulative_probability_for_state: bool = True) -> None:
                    Combine states in the column into a single states with a single optical component.
                    So states '001', '100' and '100' will become state '201'. After calling this function 
                    the column will contain just one state.

                    The original column is modified by this function.

                    If states in the column contain more than one optical component only the first optical component is considered.

                    For the boson factor the operation is the reverse of splitting a state in single components.
                    NOTE: if the 'photons' in the column are not fully identical care needs to be taken in setting the right boson factor. This function
                    'assumes' the photons are identical. For correcting column boson factors in an interference group call InterferenceGroup.rescale_boson_factors()

                    If use_group_cumulative_probability_for_state is True (default) the cumulative_probability of the state will be 
                    the group_cumulative_probability x |column_amplitude|**2

                    If the photon numbers for the states add up to a number that is larger or equal to circuit.length_of_fock_state the function
                    will throw an error.
                    
                    Example 1:
                        - If the column consists of states with components '001', '100' and '100' the resulting state will have optical component
                            '201'. The amplitude of this component will be 1 and the column_boson_factor will be sqrt(1/2)
                    Example 2: 
                        - We start with state '22' with column_boson_factor 1
                        - We expand into 4 single photon states '01','01','10','10' in a column with column_boson_factor 2
                        - after condense_column_to_single_state the column_boson_factor is again 1
                    Example 3: 
                        - We start with a column with states '20' and '02' with column_boson_factor 1
                        - After condense_column_to_single_state the column_boson_factor is 1 and the state is '22'
                    Example 4: 
                        - We start with a column with states '11' and '11' with column_boson_factor 1
                        - After condense_column_to_single_state the column_boson_factor is 1/2 and the state is '22'
    
            generate_optical_component_corresponding_to_column(self, 
                                                            return_values_instead_of_component_name_as_string: bool = False,
                                                            use_stored_values: bool = False) -> str:
                    Return the optical component as string (like '122' or '212') from a column. For states in the column
                    only the first components are used. The function is intended for columns where the states only have
                    a single optical component.

                    If the bool 'return_values_instead_of_component_name_as_string' is set to 'True' the function will return a tuple representing
                    the photon numbers per channel instead of a string representing the component name.

                    If the bool 'use_stored_values' is set to 'True' function will return a values from earlier calculation. This is faster and can be used
                    if column is not changed since last time function was called.

                    If states in the column contain more than one optical component only the first optical component is considered.

                    The original column remains unchanged.

                    If the photon numbers in the states add up to an invalid photon number in one channel the function will not return the
                    component as a string, but a tuple of photon values in the channels. To avoid this increase 'circuit.length_of_fock_state' to 
                    a higher number (if you want to model channels with 4 photons, the length_of_fock_state has to be 5. For n photons be channels set
                    length_of_fock_state to n+1).
                                        
            generate_single_state_from_column(self, use_group_cumulative_probability_for_state: bool = True) -> State:

                    This function first condenses the column to a single state and then returns a copy of that state. The photons from the 
                    various states in the column are added without taking into account and photon properties. The returned state will not 
                    contain any information under 'photon_resolution' in auxiliary_information. The cumulative probability for the returned state will be the 
                    cumulative probability for the states in the original column.

                    For the boson factor the operation is the reverse of splitting a state in single components.
                    NOTE: if the 'photons' in the column are not fully identical care needs to be taken in setting the right boson factor. This function
                    'assumes' the photons are identical. For correcting column boson factors in an interference group call InterferenceGroup.rescale_boson_factors()
                    
                    If states in the column contain more than one optical component only the first optical component is considered.

                    If use_group_cumulative_probability_for_state is True (default) the cumulative_probability of the state will be 
                    the group_cumulative_probability x |column_amplitude|**2

                    The original column remains unchanged.

                    If the photon numbers for the states add up to a number that is larger or equal to circuit.length_of_fock_state the function
                    will throw an error.

                    Example 1:
                        - If the column consists of states with components '001', '100' and '100' the resulting state will have optical component
                            '201'. The amplitude of this component will be 1 and the column_boson_factor will be sqrt(1/2)
                    Example 2: 
                        - We start with state '22' with column_boson_factor 1
                        - We expand into 4 single photon states '01','01','10','10' in a column with column_boson_factor 2
                        - after condense_column_to_single_state the column_boson_factor is again 1
                    Example 3: 
                        - We start with a column with states '20' and '02' with column_boson_factor 1
                        - After condense_column_to_single_state the column_boson_factor is 1 and the state is '22'
                    Example 4: 
                        - We start with a column with states '11' and '11' with column_boson_factor 1
                        - After condense_column_to_single_state the column_boson_factor is 1/2 and the state is '22'

        Last modified: April 16th, 2024                                 

"""
    _VERSION = '1.0.0'
    _DEFAULT_COLUMN_VALUES = {'column_amplitude' : 1, 
                              'column_boson_factor' : 1, 
                              'column_identifier' :0, 
                              'group_cumulative_probability' : 1, 
                              'interference_group_identifier' : 0}

    def __init__(self, list_of_states: list = [], state: any = None, column_information: dict = dict([])):
        """ Constructor for an instance of the class ColumnOfStates.

            There are two options to create a new column:
            Option 1: Provide a state as argument for the constructor:
                - A new column will be created with a single state. So len(column) will be 1
                - If column_information is provided that information is used for teh column. The
                    information already in the original state is overwritten
                - If column_information is not provided information in state.auxiliary_information
                    is used to create the column.
                - If state.auxiliary_information does not contain the required information default
                    values are used (ColumnOfStates._DEFAULT_COLUMN_VALUES)
                NOTE: State.cumulative_probability is not used to set column.group_cumulative_probability
                If you want to use this value do so explicitely after creation of the instance.

            Option 2: Provide a list of states as argument:
                - A new column is created containing all states in the list
                - If column_information is provided that information is used for teh column. The
                    information already in the original state is overwritten
                - If column_information is not provided information in the first state from the list
                    (list_of_states[0].auxiliary_information) is used to create the column.
                    NOTE: Only the first state in the list is evaluated for column_information.
                - If the first state in the list does not contain the required information default
                    values are used (ColumnOfStates._DEFAULT_COLUMN_VALUES)
                NOTE: State.cumulative_probability from states is not used to set column.group_cumulative_probability
                If you want to use this value do so explicitely after creation of the instance.

            Option 3: If no list (or an empty list) is provided:
                An empty column is created based in information in (ColumnOfStates._DEFAULT_COLUMN_VALUES)
            
            Both a single state and a list_of_states is provided an exception will be raised.

        Args:
            list_of_states (list, optional): List of states that will form the column. Defaults to [].
            state (any, optional): Single state which will form the column. Defaults to None.
            column_information (dict, optional): Column information which will overwrite information already
                present in the state(s) passed as argument. Defaults to dict([]).

        Raises:
            Exception: If both list_of_states and state are provided
        """


        self.column = self._DEFAULT_COLUMN_VALUES.copy()
        if len(list_of_states) != 0 and state is None:
            # Create the column based on the states in the list that is passed as argument.
            # Parameters come first from 'column_information' passed as paramater in the instance creation,
            # Secondly (of column info is not in arguments) the info is taken from the first state in the list.
            # Thirdly, no information is given at instance creation, and the first state also does not contain the information default
            # information will be used

            # populate the list of states for the column
            self.list_of_states = list_of_states

            # if there is no photon_resolution in state.auxiliary_information add it as an empty dictionary
            for state in self.list_of_states:
                if 'photon_resolution' not in state.auxiliary_information.keys():
                    state.auxiliary_information.update({'photon_resolution': dict([])})

            # prepare column information to the state
            used_column_information = self._DEFAULT_COLUMN_VALUES.copy()
            # try to read from first state, if data available with the right key override default values
            for k,v in self._DEFAULT_COLUMN_VALUES.items():
                if k in self.list_of_states[0].auxiliary_information['photon_resolution'].keys():
                    used_column_information.update({k:self.list_of_states[0].auxiliary_information['photon_resolution'][k]})
            # finally update the column information with what is passed as argument
            used_column_information =  used_column_information | column_information

            # By setting the column parameters also the states in the list are updated (See the 'setter' functions)
            self.column_identifier = used_column_information['column_identifier']
            self.column_amplitude = used_column_information['column_amplitude']
            self.column_boson_factor = used_column_information['column_boson_factor']
            self.interference_group_identifier = used_column_information['interference_group_identifier'] 
            self.group_cumulative_probability = used_column_information['group_cumulative_probability'] 
        
        elif len(list_of_states) == 0 and state is not None:
            # Create the column based on the single state that is passed as argument in the constructor.
            # Parameters come first from 'column_information' passed as paramater in the instance creation,
            # Secondly (of column info is not in arguments) the info is taken from the provided state
            # Thirdly, no information is given at instance creation, and the state also does not contain the information default
            # information will be used

            # create list with just the one initial state
            self.list_of_states = [state]

            # if there is no photon_resolution in state.auxiliary_information add it as an empty dictionary
            if 'photon_resolution' not in state.auxiliary_information.keys():
                state.auxiliary_information.update({'photon_resolution': dict([])})

            # prepare column information to the state
            used_column_information = self._DEFAULT_COLUMN_VALUES.copy()
            # try to read from first state, if data available with the right key override default values
            for k,v in self._DEFAULT_COLUMN_VALUES.items():
                if k in self.list_of_states[0].auxiliary_information['photon_resolution'].keys():
                    used_column_information.update({k:self.list_of_states[0].auxiliary_information['photon_resolution'][k]})
            # finally update the column information with what is passed as argument
            used_column_information =  used_column_information | column_information

            # By setting the column parameters also the states in the list are updated (See the 'setter' functions)
            self.column_identifier = used_column_information['column_identifier']
            self.column_amplitude = used_column_information['column_amplitude']
            self.column_boson_factor = used_column_information['column_boson_factor']
            self.interference_group_identifier = used_column_information['interference_group_identifier'] 
            self.group_cumulative_probability = used_column_information['group_cumulative_probability'] 

        elif len(list_of_states) == 0 and state is None:
            # Neither a list nor a single states are pass as argument for the constructor. Create an empty column.
            # Create empty list of states
            self.list_of_states = []

            # load default values
            default_column_information = self._DEFAULT_COLUMN_VALUES.copy()
            used_column_information =  default_column_information | column_information
            self.column_identifier = used_column_information['column_identifier']
            self.column_amplitude = used_column_information['column_amplitude']
            self.column_boson_factor = used_column_information['column_boson_factor']
            self.interference_group_identifier = used_column_information['interference_group_identifier'] 
            self.group_cumulative_probability = used_column_information['group_cumulative_probability']
            
        else:
            raise Exception('To create ColumnOfStates either a list of single photons states, or a single component Fock state is required')

        return

    @property
    def column_identifier(self):
        return self.column['column_identifier']
       
    @column_identifier.setter
    def column_identifier(self,index: int):
        self.column['column_identifier'] = index
        for state in self.list_of_states:
            state.auxiliary_information['photon_resolution']['column_identifier'] = index
    
    @property
    def interference_group_identifier(self):
        return self.column['interference_group_identifier']
       
    @interference_group_identifier.setter
    def interference_group_identifier(self,index: int):
        self.column['interference_group_identifier']  = index
        for state in self.list_of_states:
            state.auxiliary_information['photon_resolution']['interference_group_identifier'] = index

    @property
    def column_amplitude(self):
        return self.column['column_amplitude']
       
    @column_amplitude.setter
    def column_amplitude(self, amplitude: np.cdouble):
        self.column['column_amplitude']  = amplitude
        for state in self.list_of_states:
            state.auxiliary_information['photon_resolution']['column_amplitude'] = amplitude

    @property
    def column_boson_factor(self):
        return self.column['column_boson_factor']
       
    @column_boson_factor.setter
    def column_boson_factor(self, factor: float):
        self.column['column_boson_factor'] = factor
        for state in self.list_of_states:
            state.auxiliary_information['photon_resolution']['column_boson_factor'] = factor

    @property
    def group_cumulative_probability(self):
        return self.column['group_cumulative_probability']
       
    @group_cumulative_probability.setter
    def group_cumulative_probability(self, probability: float):
        self.column['group_cumulative_probability'] = probability
        for state in self.list_of_states:
            state.auxiliary_information['photon_resolution']['group_cumulative_probability'] = probability

    def __getitem__(self, identifier) -> State:
        return self.list_of_states[identifier]
    
    def __len__(self) -> int:
        return len(self.list_of_states)
    
    def __str__(self)-> str:
        """ Return the string for 'pretty printing' the interference column.

        Returns:
            str: string describing the column
        """ 
        text = ''
        column = self
        text += '\t\t-----' + '\n'
        text += "\t\tcolumn id:" +str(column.column_identifier) + '\n'
        text += "\t\tcolumn amplitude:" + str(column.column_amplitude) + '\n'
        text += "\t\tcolumn boson factor:" +str(column.column_boson_factor) + '\n'
        for state in column:
            text += "\t\t\t" +str(state.optical_components) + '\n'
        return text

    def add_state(self, state: State) -> None:
        """ Add a new state to the column. The new state will adjusted to fit the column (i.e., cumulative probability will be
            adjusted, as well as state.auxiliary_information['photon_resolution'][..] ).

            If the state already has state.auxiliary_information['photon_resolution'] this will be overwritten.

        Args:
            state (State): State to be added to the column
        """
        self.list_of_states.append(state)
        if not 'photon_resolution' in self.list_of_states[-1].auxiliary_information:
            self.list_of_states[-1].auxiliary_information['photon_resolution'] = dict([])
        self.list_of_states[-1].auxiliary_information['photon_resolution']['column_identifier'] = self.column_identifier 
        self.list_of_states[-1].auxiliary_information['photon_resolution']['column_amplitude'] = self.column_amplitude 
        self.list_of_states[-1].auxiliary_information['photon_resolution']['column_boson_factor'] = self.column_boson_factor
        self.list_of_states[-1].auxiliary_information['photon_resolution']['interference_group_identifier']  = self.interference_group_identifier 
        self.list_of_states[-1].auxiliary_information['photon_resolution']['group_cumulative_probability'] = self.group_cumulative_probability
        self.list_of_states
        return
    
    def copy(self) -> ColumnOfStates:
        """ Return new instance with copied contents. All states and information are copied to new instance.
        """
        new_column = ColumnOfStates()
        new_column.column = self.column.copy()
        for state in self.list_of_states:
            new_column.add_state(state.copy())
        return new_column

    def split(self, new_column_identifier: int = -1) -> ColumnOfStates:
        """ This function searches the states in the column. If a state is found with more than one optical component the state is split
            and a new time column is created. The first optical component will go to the new time column, the remaining components stay 
            in the current column. All other states are copied.

            If no state with more than one optical component is found in the time column the function returns 'None', otherwise the 
            a new instance of ColumnOfStates is returned.

        Args:
            new_column_identifier (int, optional): Column_identifier to be used for the new column. Default value is -1

        Returns:
            ColumnOfStates: New column that is split of from the original.
        """
        new_column = None
        for index, state in enumerate(self.list_of_states):
            if len(state.optical_components) == 1:
                continue
            else:
                # We identified a state which has to be 
                # We create  'remaining' column and a 'new' column
                # Optical components in this state are for instance '100','010' and '001' with amplitudes 0.71,0.5 and 0.5
                # Step 1. Split the column (i.e., create a new one)
                index_of_split_state = index

                new_column = self.copy()
                new_column.column_identifier = new_column_identifier

                # Step 2. Split the state
                # the first component will go to the duplicate (new) state, the rest remains in the original (old) state
                # so in our example:
                # Duplicate state (in new column) has components '100', amplitude 1. The 'column_amplitude' will 
                # be multiplied by 0.71 and the cumulative probability by 0.71**2
                # Remaining state (in old column) will have components '010' and '001' with amplitudes 0.71 and 0.71
                # cumulative probability will be multipied by (1 - 0.71**2)
                original_optical_components = state.optical_components.copy()
                list_of_components = list(original_optical_components.keys())
                split_of_component = list_of_components[0]
                split_of_amplitude = state.optical_components[split_of_component]['amplitude']
                split_of_probability = np.abs(split_of_amplitude)**2
                
                # new state receives the first component with amplitude 1
                new_state = new_column.list_of_states[index_of_split_state]
                new_state.optical_components = [(split_of_component,1)]

                # remaining state keeps the other components, amplitudes are normalized
                remaining_state = self.list_of_states[index_of_split_state]
                if len(list_of_components) == 2:
                    remaining_component = list_of_components[1]
                    remaining_amplitude = remaining_state.optical_components[remaining_component]['amplitude']
                    remaining_probability = np.abs(remaining_amplitude)**2
                    remaining_state.optical_components = [(remaining_component,1)]
                else:
                    remaining_probability = 1-split_of_probability
                    remaining_amplitude = np.sqrt(remaining_probability)
                    rem_oc = [(comp, original_optical_components[comp]['amplitude']/remaining_amplitude) for comp in list_of_components[1:]]
                    remaining_state.optical_components = rem_oc

                # Step 3: Adjust the column amplitudes 
                # first the 'remaining' states
                self.column_amplitude *= remaining_amplitude

                # adjust the 'new' states
                new_column.column_amplitude *= split_of_amplitude

                break
        return new_column


    def single_photon_states(self) -> None:
        """ Turn this column into a column with only single photon states and vacuum states.

            If the vacuum state (state without photons) is part of the original column this state 
            will be added to/remain in the column after applying this function. 

            The attribute 'column_boson_factor' will reflect the fact that multi-photon states are split. Splitting a 
            state with n photons in one channel and m in another "nm" will generate column_boson_factor sqrt(n! m!)

            Example 1 : A state with optical component '201' will be expanded in three states with
                optical components '100', '100' and '001' with column_boson_factor of sqrt(2)

            Example 2 : A state with optical component '230' will be expanded in three states with
                optical components '100', '100', '010', '010' and '010' with column_boson_factor of sqrt(12) = sqrt(2) x sqrt(3 x 2)

        Raises:
            Exception: if not all states are  single component states

        """
        # check is all states have a single optical component
        if not self.all_states_single_optical_component:
            raise Exception('in function split_in_single_photon_states the input collection should only have states with a single optical component')
        
        # this list will contain the single photon states
        new_list_of_states = []

        # iterate through states in the original column
        boson_factor = self.column_boson_factor
        for state in self:
            oc = state.optical_components
 
            component_name = list(oc.keys())[0]
            component_amplitude = oc[component_name]['amplitude']
            component_values = state._dict_of_valid_component_names[component_name]  
            if sum(component_values) > 0:
                # we have one or more photons in this state
                for channel_index, channel_value in enumerate(component_values):
                    for photon_count in range(channel_value): 
                        new_state = state.copy()
                        values = [0] * len(component_values)
                        values[channel_index] = 1
                        single_photon_component = new_state._dict_of_optical_values[tuple(values)]
                        new_state.optical_components = [(single_photon_component, component_amplitude)]
                        boson_factor *= np.sqrt(photon_count+1)
                        new_list_of_states.append(new_state)
            else: 
                # if no photons in the original state add the vacuum state to the time column
                new_state = state.copy()
                values = [0] * len(component_values)
                zero_photon_component = new_state._dict_of_optical_values[tuple(values)]
                new_state.optical_components = [(zero_photon_component, component_amplitude)]
                new_list_of_states.append(new_state)

        self.list_of_states = new_list_of_states          
        self.column_boson_factor = boson_factor

        return 

    def list_of_photons(self) -> list:
        """ Returns a list of photons. This is a list with one element per photon in the state. The elements in the 
            list are tuples. The first element of the tuple is the channel for this photon, and the second element 
            in the tuple is the dictionary 'photon_information' (taken from 
            state.auxiliary_information['photon_resolution']['photon_information'])

            If states in the column contain more than one optical component only the first optical component if considered.

        Returns:
            list [tuple]: list of tuple [(channel photon0, photon info photon0), (channel photon1, photon info photon1), ...]
        """
        photon_list = []
        for state in self:
            oc = state.optical_components
            component_name = list(oc.keys())[0]
            component_values = state._dict_of_valid_component_names[component_name]    
            photon_information = state.auxiliary_information['photon_resolution'].get('photon_information', dict([]))
            for channel_index, channel_value in enumerate(component_values):
                for photon_count in range(channel_value):
                    photon_list.append((channel_index, photon_information))
        return photon_list
    
    def set_photon_information(self,photon_information: dict = dict([]), 
                               filter_for_optical_values: list = None, 
                               default_photon_information: dict = None) -> None:
        """ Update states in the column with 'photon information'. This is the information used to calculate 
            the overlap between photons. The function calculating this overlap has to be able to use the dict
            in the form that it is added here.

            If filter_for_optical_values is used only states with optical values in this filter will be updated.
            Other states will either get default_photon_information is that is available in the arguments, or
            will remain unchanged. 

            filter_for_optical_values if a list of lists, or list of tuples. If we want to update only states
            with one single photon in the first optical channel  filter_for_optical_values would be 
            [[1,0,0]] or [(1,0,0)]

            For adding timing information per photon the dict would be {'time_stamp': 0, 'pulse_width': 1}
            For adding spectral information the dict would be {'frequency band' : 0, 'bandwidth' : 1}

            If the key 'photon_information' is not present in state.auxiliary_information['photon_resolution']
            it will be added, otherwise it will be updated.

        Args:
            photon_information (dict, optional): information to be added to each photon. Defaults to dict([]).
            filter_for_optical_values (list, optional): filter to select which states/photons should receive photon_information.
                                                        format [[0,1,2],[2,1,0]] or [(0,1,2),(2,1,0)]
            default_photon_information (dict, optional): information to be added to states that do not match the filter.
        """
        if filter_for_optical_values is None:
            # update all states
            for state in self:
                state.auxiliary_information['photon_resolution'].update({'photon_information':photon_information})
            return
        else:
            # update states with matching values in filters, 
            # if default_photon_information is given add that to the states that do not match with filter
            filter_for_optical_values = [tuple(values) for values in filter_for_optical_values]
            for state in self:
                values = tuple(state._dict_of_valid_component_names[list(state.optical_components.keys())[0]])
                if values in filter_for_optical_values:
                    state.auxiliary_information['photon_resolution'].update({'photon_information':photon_information})
                else:
                    if default_photon_information is not None:
                        state.auxiliary_information['photon_resolution'].update({'photon_information':default_photon_information})
            return

    def all_states_single_photon_states(self) -> bool:
        """ Return True if all states in the column have exactly one or zero photons. If a state has more optical
            components the function will check all components in that state. 
        """
        for state in self:
            oc = state.optical_components
            for component_name in oc.keys():
                component_values = state._dict_of_valid_component_names[component_name]          
                if not (sum(component_values) == 1 or sum(component_values) == 0):
                    return False
        return True
    
    def all_states_single_optical_component(self) -> bool:
        """ Return True if all states in the column have exactly one optical components """
        for state in self:
            oc = state.optical_components
            if len(oc) != 1:
                return False
            component_name = list(oc.keys())[0]
            amplitude = oc[component_name]['amplitude']
            if np.round(amplitude,4) != 1:
                return False
        return True
    
    def condense_column_to_single_state(self, use_group_cumulative_probability_for_state: bool = True) -> None:
        """ Combine states in the column into a single states with a single optical component.
            So states '001', '100' and '100' will become state '201'. After calling this function 
            the column will contain just one state.

            The original column is modified by this function.

            If states in the column contain more than one optical component only the first optical component is considered.

            For the boson factor the operation is the reverse of splitting a state in single components.
            NOTE: if the 'photons' in the column are not fully identical care needs to be taken in setting the right boson factor. This function
            'assumes' the photons are identical. For correcting column boson factors in an interference group call InterferenceGroup.rescale_boson_factors()

            If use_group_cumulative_probability_for_state is True (default) the cumulative_probability of the state will be 
            the group_cumulative_probability x |column_amplitude|**2

            If the photon numbers for the states add up to a number that is larger or equal to circuit.length_of_fock_state the function
            will throw an error.
            
            Example 1:
                - If the column consists of states with components '001', '100' and '100' the resulting state will have optical component
                    '201'. The amplitude of this component will be 1 and the column_boson_factor will be sqrt(1/2)
            Example 2: 
                - We start with state '22' with column_boson_factor 1
                - We expand into 4 single photon states '01','01','10','10' in a column with column_boson_factor 2
                - after condense_column_to_single_state the column_boson_factor is again 1
            Example 3: 
                - We start with a column with states '20' and '02' with column_boson_factor 1
                - After condense_column_to_single_state the column_boson_factor is 1 and the state is '22'
            Example 4: 
                - We start with a column with states '11' and '11' with column_boson_factor 1
                - After condense_column_to_single_state the column_boson_factor is 1/2 and the state is '22'

        Args:
            use_group_cumulative_probability_for_state (bool, optional): If this is True the formula
                                column.group_cumulative_probability x |column_amplitude|**2 
                                is used to set state.cumulative_probability. Otherwise the state.cumulative_probability is not
                                changed.

        """
        # store the original column_boson_factor for later use
        original_column_boson_factor = self.column_boson_factor
        list_of_states_boson_factors = []
        component_values = None
        for state in self:
            oc = state.optical_components
            component_name = list(oc.keys())[0]
            if component_values is None:
                component_values = state._dict_of_valid_component_names[component_name]   
            else:
                new_values = state._dict_of_valid_component_names[component_name] 
                component_values = [n+m for n,m in zip(new_values, component_values)]
            # determin boson factor for this state
            boson_factor = 1
            for channel_value in state._dict_of_valid_component_names[component_name]:
                for photon_count in range(channel_value): 
                    boson_factor *= np.sqrt(photon_count+1)
            list_of_states_boson_factors.append(boson_factor)
        # determine the final state boson factor
        final_state_boson_factor = 1
        for channel_value in component_values:
            for photon_count in range(channel_value): 
                final_state_boson_factor  *= np.sqrt(photon_count+1)
        

        new_optical_component = state._dict_of_optical_values[tuple(component_values)]
        new_state = state.copy()
        new_state.optical_components = [(new_optical_component,1)]
        new_boson_factor = original_column_boson_factor*np.prod(list_of_states_boson_factors)/final_state_boson_factor
        if use_group_cumulative_probability_for_state:
            new_state.cumulative_probability = (self.group_cumulative_probability * np.abs(self.column_amplitude) **2)
        self.list_of_states = [new_state]
        self.column_boson_factor = new_boson_factor
        return
    
    def generate_optical_component_corresponding_to_column(self, 
                                                           return_values_instead_of_component_name_as_string: bool = False,
                                                           use_stored_values: bool = False) -> str:
        """ Return the optical component as string (like '122' or '212') from a column. For states in the column
            only the first components are used. The function is intended for columns where the states only have
            a single optical component.

            If the bool 'return_values_instead_of_component_name_as_string' is set to 'True' the function will return a tuple representing
            the photon numbers per channel instead of a string representing the component name.

            If the bool 'use_stored_values' is set to 'True' function will return a values from earlier calculation. This is faster and can be used
            if column is not changed since last time function was called.

            If states in the column contain more than one optical component only the first optical component is considered.

            The original column remains unchanged.

            If the photon numbers in the states add up to an invalid photon number in one channel the function will not return the
            component as a string, but a tuple of photon values in the channels. To avoid this increase 'circuit.length_of_fock_state' to 
            a higher number (if you want to model channels with 4 photons, the length_of_fock_state has to be 5. For n photons be channels set
            length_of_fock_state to n+1).

        Returns:
            str: Optical component in string format
        """
        if not use_stored_values:
            component_values = None
            for state in self:
                oc = state.optical_components
                component_name = list(oc.keys())[0]
                if component_values is None:
                    component_values = state._dict_of_valid_component_names[component_name]   
                else:
                    new_values = state._dict_of_valid_component_names[component_name] 
                    component_values = [n+m for n,m in zip(new_values, component_values)]

            try:
                self._column_as_component = state._dict_of_optical_values[tuple(component_values)]
            except:
                self._column_as_component = None

            self._column_as_tuple_of_values = tuple(component_values)
            
        if return_values_instead_of_component_name_as_string:
            return self._column_as_tuple_of_values 
        else:
            return self._column_as_component
        
    def generate_single_state_from_column(self, use_group_cumulative_probability_for_state: bool = True) -> State:
        """ This function first condenses the column to a single state and then returns a copy of that state. The photons from the 
            various states in the column are added without taking into account and photon properties. The returned state will not 
            contain any information under 'photon_resolution' in auxiliary_information. The cumulative probability for the returned state will be the 
            cumulative probability for the states in the original column.

            For the boson factor the operation is the reverse of splitting a state in single components.
            NOTE: if the 'photons' in the column are not fully identical care needs to be taken in setting the right boson factor. This function
            'assumes' the photons are identical. For correcting column boson factors in an interference group call InterferenceGroup.rescale_boson_factors()
            
            If states in the column contain more than one optical component only the first optical component is considered.

            If use_group_cumulative_probability_for_state is True (default) the cumulative_probability of the state will be 
            the group_cumulative_probability x |column_amplitude|**2

            The original column remains unchanged.

            If the photon numbers for the states add up to a number that is larger or equal to circuit.length_of_fock_state the function
            will throw an error.

            Example 1:
                - If the column consists of states with components '001', '100' and '100' the resulting state will have optical component
                    '201'. The amplitude of this component will be 1 and the column_boson_factor will be sqrt(1/2)
            Example 2: 
                - We start with state '22' with column_boson_factor 1
                - We expand into 4 single photon states '01','01','10','10' in a column with column_boson_factor 2
                - after condense_column_to_single_state the column_boson_factor is again 1
            Example 3: 
                - We start with a column with states '20' and '02' with column_boson_factor 1
                - After condense_column_to_single_state the column_boson_factor is 1 and the state is '22'
            Example 4: 
                - We start with a column with states '11' and '11' with column_boson_factor 1
                - After condense_column_to_single_state the column_boson_factor is 1/2 and the state is '22'

        Args:
            use_group_cumulative_probability_for_state (bool, optional): If this is True the formula
                                column.group_cumulative_probability x |column_amplitude|**2
                                is used to set state.cumulative_probability. Otherwise the state.cumulative_probability is not
                                changed.

        Returns:
            State: state (new instance of State) containing photons in the time column
        """
        temporary_column = self.copy()
        temporary_column.condense_column_to_single_state(use_group_cumulative_probability_for_state = use_group_cumulative_probability_for_state)
        new_state = temporary_column.list_of_states[0]
        new_state.auxiliary_information.pop('photon_resolution', None)
        return new_state

    


