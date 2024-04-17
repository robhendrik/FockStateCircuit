from __future__ import annotations
from fock_state_circuit.state import State
from fock_state_circuit.temporal_and_spectral_gate_functionality.column_of_states import ColumnOfStates
import numpy as np

class InterferenceGroup():
    """ An InterferenceGroup holds ColumnsOfStates which can during a detection/measurement event interfere. 
        Interference groups are part of CollectionsOfStateColumns. See either the class description for 
        ColumnOfStates and/or for CollectionOfStateColumns for more information (not added here to avoid duplication
        and inconsistencies).

        Attributes:
            interference_group_identifier: Identifier for the interference group
            group_cumulative_probability: The sum of cumulative probabilities of all columns in the group
        
        Methods:
            __getitem__(self, identifier):
                    Return columns by identifier
        
            __len__(self):
                    Number of columns in the group
        
            __str__(self) -> str:
                    Return the string for 'pretty printing' the interference group.
        
        by_state(self):
            Used to iterate through states in an interference group as in: for state in group.by_state(): 

        by_column(self):
            Used to iterate through columns in an interference group as in: for column in group.by_column(): 
        
        extract_column_by_identifier(self,identifier) -> ColumnOfStates:
            Return the column of states for the given identifier

        set_next_new_column_identifier(self) -> None:
            Find the next available number for a column_identifier and set this value to self._next_new_column_identifier

        set_interference_group_identifier(self) -> None:
            Ensure that the interference_group_identifier is correctly set for all columns in the group.

        add_column(self, column, column_identifier: int = None) -> None:
            Add a new column to the interference group. The paramater column has to be of type 'ColumnOfStates' in FockStateCircuit.
            ault the new column will be assigned a random identifier. Only when column_identifier is specifically passed as 
            parameter the use of this identifier will be enfored.

            Option 1) column_identifier is not specified: 
                Add the column to the interference group with the next available column identifier (so a new column in the group
                will be created). The column_identifier in the group that is added will be overwritten.
            Option 2) When column_identifier is specified:
                The column will be force to carry the specified column_identifier. If this column_identifier is already used the next available
                identifier will be used
        
        add_state(self, state, column_identifier: int = None) -> None:
            Add a new state to the interference group. 

            Option 1:
                If column_identifier is given in the arguments for this function then
                either create a new column with that identifier, or add 
                the state to the existing column with this column_identifier.

            Option 2:
                If column identifier is not given look in state.auxiliary_information['photon_resolution']['column_identifier'] 
                for the column identifier to use (and either add to existing column if this column exists, or create a new column).

            Option 3:
                If column_identifier is not given and the state does also not provide an identifier create a new column, add
                the state and add this column to the interference group (with random identifier) 

        split_columns(self) -> None:
            Split all columns into states with a single optical components.

            For the interference group the arguments 'interference_group_identifier' and 
            'group_cumulative_probability' do not change.

            The 'column_cumulative_probabilities' for teh columns in the group will also after executing this 
            function add up to the 'group_cumulative_probability'

            Example 1:
            original_state.optical_components = [('110', np.sqrt(1/2)),('011',1/2), ('101',1/2)]
            interference_group = fsc.InterferenceGroup(state = original_state)
                # interference_group.group_cumulative_probability is one after initialization
            interference_group.split_columns()
            
                ==> The interference group now contains three columns
                    - One column with a state [('110',1)] with column_amplitude equal to np.sqrt(1/2)
                    - One column with a state [('011',1)] with column_amplitude equal to 1/2
                    - One column with a state [('101',1)] with column_amplitude equal to 1/2

            #The value for 'column.group_cumulative_probability' will be 0.5, 0.25 and 0.25 after splittig.
            #'group_cumulative_probability' will still be one.
        
        single_photon_states(self) -> None:
            Expand all columns in the interference group into single photon states.

            Function will return an error if not all states have a single optical component. In that case first call
            interference_group.split_columns() to create states with just one single optical component.

            The attributes for the group will not change (same group identifier and cumulative probability). For the columns
            the number of states in the columns can change, as well as the column.column_boson_factor
        
        generate_single_state_from_interference_group(self) -> State:
            Create a new state (of type State). This state will have as optical components the sum of photon states
            in the individual columns. The amplitudes of the components will be the column_amplitudes.
                
            This function performs the reverse operation from creating an interference group from a state. The function 
            works if states are full expanded into single photon states as well as when the state is still a single
            state with multiple photons per optical channel. A mixed situation where some columns are expanded can easily
            lead to erroneous results.

            Example 1:
                original_state.optical_components = [('110',r),('011',1/2), ('101',1/2)]
                interference_group = fsc.InterferenceGroup(state = original_state)
                interference_group.split_columns()
                interference_group.single_photon_states()
                # we now have all single photon states in the interference group
                new_state = interference_group.generate_single_state_from_interference_group()

                ==> new_state will now be identical to original_state (a new instance, but identical content)

            Example 2:
                original_state.optical_components = [('110',r),('011',1/2), ('101',1/2)]
                interference_group = fsc.InterferenceGroup(state = original_state)
                # we have multi-photon states in the interference group
                new_state = interference_group.generate_single_state_from_interference_group()

                ==> new_state will now be identical to original_state (a new instance, but identical content)

        rescale_column_amplitudes(self) -> None:
            Adapt the column amplitudes such that the sum of their squares adds up to 1

        rescale_boson_factors(self):
            Adapt the boson factors such that for each column the 'self detection probability' is equal to the square of the
            column amplitude 

        _valid_combinations(self,measurement_projection: list,state_column: ColumnOfStates) -> list:
            Generate all valid combinations between photon state and projection

            - Projection on 2130 will have measurement_projection [0,0,1,2,2,2]
            - A state column for component '2130' will have (for example) photon_channel_list [0,1,2,2,2,0]
            - Valid combinations will be for this case:
                [ (0,5,1,2,3,4), (5,0,1,2,3,4), (0,5,1,4,2,3), ... ]
                So, if you read projection from left to right the valid combination indicates which index in photon_channel_list
                corresponds to measurement_projection
            - Note 1: If the column contains empty states (like '000') then this state does not add a photon to
                column.list_of_states(). The index of states in the column is then not the same as the index of 
                photons in the list. e.g., if we have a column with state '00' and '01' and projection is on '01' 
                the the returned list of valid combinations will be [(0,)]. The state with the photon has index 1, 
                but in the photon list this is index 0
            - Note 2: if there is just one photon the Tuple notation forces the addition of a comma, like in example
                in note 1: the list can be [(0,)], meaning the first photon in the photon list matches with the first 
                photon in the measurement projection.

            The expected format for 'measurement_projection' is a list of integers like [0,0,1,2,2,2]
            The expected format for 'state_column' is a ColumnOfStates, where there the states are single photon states.

        _find_valid_photon_pairs_for_projection(self,measurement_projection,state_column_bra_side, state_column_ket_side) -> list:
            Generate all possible pairs photons that can contribute to a measurement projection. The return will be a list of
            'sequences'. A sequence is a tuple of pairs indicating which photons can combine to contribute to the projection.

            In this example the first and second photon can combine, but the photon with index 2 can only combine with itself, 
            leading to thre sequences in the list that will be returned:
                    [((0,1),(1,0),(2,2)), ((0,0),(1,1),(2,2)), ((1,0),(0,1),(2,2))] 

            Note: The indices are indices of photons in the photon list. If (one of) the column(s) contains the vacuum state 
                (like '000') this state contains no photons and the 'photon list index' is different from the 'state in column' 
                index. As Example: We have a bra-column with states '00' and '10' and a ket-column with states '10' and '00'. The list
                of valid photon pairs will be  [((0,0),) ]. The photon list of both columns only contain one entry (as there is one photon)
                and these photons with index 0 in the photon list form a valid combination when projecting on '10'. If we consider states then it
                is state with index 1 at bra side, and state with index 0 and ket side that combine. The indices for valid photon
                pairs are indices in the COLUMN.LIST_OF_PHOTONS() and not indices of the states.   

        _calculate_probability_for_photon_sequence(self, sequence, state_column_bra_side ,state_column_ket_side) -> float:
            Calculates the probability for a given sequence of photon pairs. The probability is the product of the probability
            to detect the pairs in the sequence. For any pair the probability is determined by the overlap of their wave packets, as 
            determined in the function self.photon_probability_function

        _measurement_projection(self, optical_component: str = None, target_values_and_channels: dict = None) -> tuple[list, float]:       
            Calculates the photon list and boson factor for the optical component which is measured. The component
            to be measured can be given as a string (argument optical_component) or as a dict of containing target channels and
            target values.

            Examples: For a circuit with 4 optical channels
                CollectionOfStateColumns._measurement_projection('1111') -> ([0, 1, 2, 3], 1.0)
                CollectionOfStateColumns._measurement_projection('2222') -> ([0, 0, 1, 1, 2, 2, 3, 3], 4.0)
                CollectionOfStateColumns._measurement_projection('2002') -> ([0, 0, 3, 3], 2.0)
                CollectionOfStateColumns.._measurement_projection('3000') -> ([0, 0, 0], 2.4494897427831783)
                CollectionOfStateColumns._measurement_projection('3003') -> ([0, 0, 0, 3, 3, 3], 6.00)

                CollectionOfStateColumns._measurement_projection({'channels': [0,1,2], 'values': [2,2,2]}) 
                            -> ([0, 0, 1, 1, 2, 2], 2.0)
        
        _calculate_probability_for_this_projection_for_these_state_columns(self,
                                                                            measurement_projection,
                                                                            projection_boson_factor,
                                                                            state_column_bra_side, 
                                                                            state_column_ket_side) -> float:
            Calculates the detection probability for a set of 'ColumnsOfStates" for a a given projection state.
            The formula used is:

            Probability =  pre-factor * overlap
            with:
            pre-factor = amplitude-bra * amplitude-ket / (boson-bra * boson-ket * boson_projection * boson-projection)
            overlap = SUM(overlap pair-0 * overlap pair-1 * overlap pair-2 * ..) 
                Here SUM is over all possible pairs of one bra and one ket photon that matches with the projection state.

            If detection is not possible (for instance because photon state in bra, ket and measurement_projection do not match)
            the function returns None. 

        total_probability_for_group_all_outcomes(self, list_of_projections: list = None) -> dict:
            Calculate all possible outcomes for a total measurement (all optical channels) and the probability to reach that outcome.
            (so these probabilities should add up to 1).

            If the provide a list_of_projections (format ['1101', '2201']) then only for those outcome the probability will be determined
            (the probabilities in that case could add up to a number lower than 1, or even be zero)

        measure_interference_group(self, 
                                    optical_channels_to_measure, 
                                    classical_channels_to_write_to,
                                    list_of_projections: list = None) -> list:
            Perform measurement on the interference group and return a list of groups, where each new group leads to one
            measurement outcome (so the number of groups in the returned list is equal to the number of different measurement
            results)

        Last modified: April 16th, 2024
    """
    _VERSION = '1.0.0'
    def __init__(self, list_of_state_columns: list = [], state: any = None, interference_group_identifier: int = None):
        """ Constructor for a new instance of class InterferenceGroup.
            There are 3 options to create a new InterferenceGroup instance:

            1) Provide a state and do not provide a 'list_of_state_columns' (or provide empty list)
                - We create a new group containing just this 'state'
                - If the 'state' has an 'interference_group_identifier' that is used as identifier for the group,
                    otherwise we choose 0 as identifier
                - If interference_group_identifier is passed as argument to this constructor then that value is used
                    as identifier for the group, overriding the information contained in the 'state'
                - For 'group_cumulative_probability' we use the value in 'state.cumulative_probability' for the given state.

            2) Provide a non-empty 'list_of_state_columns' and no 'state'
                - We create a new group containing the columns in the list
                - If the first 'column' has an 'interference_group_identifier' that is used as identifier for the group,
                    otherwise we choose 0 as identifier.
                - If interference_group_identifier is passed as argument to this constructor then that value is used
                    as identifier for the group, overriding the information contained in the 'list_of_columns'
                - The 'group_cumulative_probability' is set to 'column.group_cumulative_probability' for the first column in 
                    the list.

            3) If an empty 'list_of_columns' is provided as well as no 'state' the constructor creates an empty group.
                - An empty group is created, containing no columns
                - The interference_group_identifier is zero, or equal to the value for interference_group_identifier which
                     is passed as argument to this constructor.
                - The 'group_cumulative_probability' is set 1

        Args:
            list_of_state_columns (list, optional): List of ColumnsOfStates to form the InterferenceGroup. Defaults to [].
            state (State, optional): State to form the interference group in a new column (of type ColumnsOfStates). Defaults to None.
            interference_group_identifier (int, optional): Identifier for this interference group. Overrules information in the state. Defaults to None.

        Raises:
            Exception: If both list_of_state_columns and state are provided
        """
        self.group = dict([])
        self._interference_group_dict = dict([])

        # set the identifier for this interference group
        self.interference_group_identifier = interference_group_identifier

        # if only state is given as parameter create an interference group with a single column, containing
        # this single state
        if len(list_of_state_columns) == 0 and state is not None:
            try:
                group_id_from_state = state.auxiliary_information['photon_resolution']['interference_group_identifier']
            except:
                group_id_from_state = 0
            self.add_state(state)
            self.interference_group_identifier =  group_id_from_state
            self.group_cumulative_probability = state.cumulative_probability

        # if a list of columns is given as paramater create the interference group from these columns
        elif len(list_of_state_columns) > 0 and state is None:
            try:
                group_id_from_column = list_of_state_columns[0].interference_group_identifier
            except:
                group_id_from_column = 0
            self.group_cumulative_probability = list_of_state_columns[0].group_cumulative_probability
            for column in list_of_state_columns:
                self.add_column(column, column_identifier=column.column_identifier)
            self.interference_group_identifier =  group_id_from_column

        # if no parameter is given create an empty interference group
        elif len(list_of_state_columns) == 0 and state is None:
            if interference_group_identifier is None:
                self.interference_group_identifier = 0
            else:
                self.interference_group_identifier = interference_group_identifier
            self.group_cumulative_probability = 1

        else:
            raise Exception('Cannot pass both list_of_states and state as argument')
        
        # if interference_group_identifier is explicitely given overwrite the default values, or values from states
        if interference_group_identifier is not None:
            self.interference_group_identifier = interference_group_identifier

        # this is the function to determine how the probability to dectect two photons depends on their properties
        # (i.e., how we calculate the 'overlap' of the wave functions)
        self.photon_probability_function = None
        return
    
    @property
    def interference_group_identifier(self):
        return self.group['interference_group_identifier']
       
    @interference_group_identifier.setter
    def interference_group_identifier(self,index: int):
        self.group['interference_group_identifier'] = index
        for column in self:
            column.interference_group_identifier = index

    @property
    def group_cumulative_probability(self):
        return self.group['group_cumulative_probability']
       
    @group_cumulative_probability.setter
    def group_cumulative_probability(self, probability: float):
        self.group['group_cumulative_probability'] = probability
        for column in self.by_column():
            column.group_cumulative_probability = probability

    def __getitem__(self, identifier) -> ColumnOfStates:
        """ Return columns by identifier"""
        return list(self._interference_group_dict.values())[identifier]
    
    def __len__(self) -> int:
        """ Number of columns in the group"""
        return len(self._interference_group_dict)
    
    def __str__(self) -> str:
        """ Return the string for 'pretty printing' the interference group.

        Returns:
            str: string describing the group
        """    
        text = ''
        group = self
        text += '\t-----------------' + '\n'
        text += '\tgroup id:' + str(group.interference_group_identifier) + '\n'
        text += '\tgroup probability: ' + "{val:.2f}".format(val = group.group_cumulative_probability) + '\n'
        for column in self:
            text += '\t\t-----' + '\n'
            text += "\t\tcolumn id:" +str(column.column_identifier) + '\n'
            text += "\t\tcolumn amplitude:" + '({c.real:.2f} + {c.imag:.2f}i)'.format(c= column.column_amplitude) + '\n'
            text += "\t\tcolumn boson factor:" +str(column.column_boson_factor) + '\n'
            for state in column:
                try:
                    photon_info_string = ''
                    for k,v in state.auxiliary_information['photon_resolution']['photon_information'].items():
                        photon_info_string += " , " + str(k) + ': '
                        photon_info_string += "{val:.2f}".format(val = v)
                except:
                    photon_info_string = ''
                text += "\t\t\t" +state.print_optical_components() +  photon_info_string + '\n'
        return text
    
    def by_state(self):
        """ Used to iterate through states in an interference group as in: for state in group.by_state(): """
        for column in self._interference_group_dict.values():
            for state in column:
                yield state

    def by_column(self):
        """ Used to iterate through columns in an interference group as in: for column in group.by_column(): """
        for column in self._interference_group_dict.values():
            yield column

    def extract_column_by_identifier(self,identifier) -> ColumnOfStates:
        """ Return the column of states for the given identifier
        
        Args:
            identifier (int): Identifier for the column to be extracted."""
        return self._interference_group_dict[identifier]
    
    def set_next_new_column_identifier(self) -> None:
        """ Find the next available number for a column_identifier and set this value to self._next_new_column_identifier"""
        self._next_new_column_identifier = 0
        while self._next_new_column_identifier in self._interference_group_dict.keys():
            self._next_new_column_identifier += 1
        return
    
    def set_interference_group_identifier(self) -> None:
        """ Ensure that the interference_group_identifier is correctly set for all columns in the group."""
        for column in self:
            column.interference_group_identifier = self.interference_group_identifier
        return

    def add_column(self, column, column_identifier: int = None) -> None:
        """ Add a new column to the interference group. The paramater column has to be of type 'ColumnOfStates' in FockStateCircuit.
            Default the new column will be assigned a random identifier. Only when column_identifier is specifically passed as 
            parameter the use of this identifier will be enfored.

            Option 1) column_identifier is not specified: 
                Add the column to the interference group with the next available column identifier (so a new column in the group
                will be created). The column_identifier in the group that is added will be overwritten.
            Option 2) When column_identifier is specified:
                The column will be force to carry the specified column_identifier. If this column_identifier is already used the next available
                identifier will be used

        Args:
            column (ColumnOfStates): Column to be added to the interference group
            column_identifier (int, optional): Identifier for the column to which state should be added. Defaults to None.
        """
        if column_identifier is None or column_identifier in self._interference_group_dict.keys():
            self.set_next_new_column_identifier()
            column.column_identifier = self._next_new_column_identifier
            column.interference_group_identifier = self.interference_group_identifier
            self._interference_group_dict[self._next_new_column_identifier] = column
            return
        else:
            self._interference_group_dict[column_identifier] = column
            return
            
    
    def add_state(self, state, column_identifier: int = None) -> None:
        """ Add a new state to the interference group. 

            Option 1:
                If column_identifier is given in the arguments for this function then
                either create a new column with that identifier, or add 
                the state to the existing column with this column_identifier.

            Option 2:
                If column identifier is not given look in state.auxiliary_information['photon_resolution']['column_identifier'] 
                for the column identifier to use (and either add to existing column if this column exists, or create a new column).

            Option 3:
                If column_identifier is not given and the state does also not provide an identifier create a new column, add
                the state and add this column to the interference group (with random identifier)

        Args:
            state (State): State to be added to the interference group
            column_identifier (int, optional): Identifier for the column to which state should be added. Defaults to None.
        """
        
        if 'photon_resolution' not in state.auxiliary_information.keys() and column_identifier is None:
            # create a new column_of_states and add that to the interference_group
            self.add_column(column = ColumnOfStates(state=state))
  
        elif 'photon_resolution' not in state.auxiliary_information.keys() and column_identifier is not None:
            # try to add state to the specifief column_identifier. If that does not exist create a new column and
            # with the given column_identifier
            for column in self:
                if column.column_identifier == column_identifier:
                    column.add_state(state = state)
                    break
            else:
                # create new column for this state since there was no column with the right column identifier
                column = ColumnOfStates(state=state)
                self.add_column(column = ColumnOfStates(state=state), column_identifier=column_identifier)
   
        elif 'photon_resolution' in state.auxiliary_information.keys() and column_identifier is None:
            for column in self:
                if column.column_identifier == state.auxiliary_information['photon_resolution']['column_identifier']:
                    column.add_state(state = state)
                    break
            else:
                # create new column for this state since there was no column with the right column identifier
                column = ColumnOfStates(state=state)
                self.add_column(column = ColumnOfStates(state=state), column_identifier=state.auxiliary_information['photon_resolution']['column_identifier'])

  
        elif 'photon_resolution' in state.auxiliary_information.keys() and column_identifier is not None:
            # try to add state to the specifief column_identifier. If that does not exist create a new column and
            # ignore the column_identifier
            for column in self:
                if column.column_identifier == column_identifier:
                    column.add_state(state = state)
                    break
            else:
                # create new column for this state since there was no column with the right column identifier
                column = ColumnOfStates(state=state)
                self.add_column(column = ColumnOfStates(state=state), column_identifier=column_identifier)
                
        else:
            return
        
        return

    def split_columns(self) -> None:
        """ Split all columns into states with a single optical components.

            For the interference group the arguments 'interference_group_identifier' and 
            'group_cumulative_probability' do not change.

            The 'column_cumulative_probabilities' for teh columns in the group will also after executing this 
            function add up to the 'group_cumulative_probability'

            Example 1:
            original_state.optical_components = [('110', np.sqrt(1/2)),('011',1/2), ('101',1/2)]
            interference_group = fsc.InterferenceGroup(state = original_state)
                # interference_group.group_cumulative_probability is one after initialization
            interference_group.split_columns()
            
                ==> The interference group now contains three columns
                    - One column with a state [('110',1)] with column_amplitude equal to np.sqrt(1/2)
                    - One column with a state [('011',1)] with column_amplitude equal to 1/2
                    - One column with a state [('101',1)] with column_amplitude equal to 1/2

            #The value for 'column.group_cumulative_probability' will be 0.5, 0.25 and 0.25 after splittig.
            #'group_cumulative_probability' will still be one.
        
        """
        found_column_to_split = False
        for column in self:
            result = column.split()
            if result is None: # if column.split() returns None the column already had only single component states
                continue
            else:
                self.add_column(result)
                found_column_to_split = True
                break
        # If a column has been split make a recursive call to this same function, if no column has been split
        # (i.e., if all columns already were columns with only single-component states) return nothing.
        if found_column_to_split:
            self.split_columns()
            return
        else:
            return
        
    def single_photon_states(self) -> None:
        """ Expand all columns in the interference group into single photon states.

            Function will return an error if not all states have a single optical component. In that case first call
            interference_group.split_columns() to create states with just one single optical component.

            The attributes for the group will not change (same group identifier and cumulative probability). For the columns
            the number of states in the columns can change, as well as the column.column_boson_factor
        """
        for column in self:
            column.single_photon_states()
        return
    
    def generate_single_state_from_interference_group(self) -> State:
        """ Create a new state (of type State). This state will have as optical components the sum of photon states
            in the individual columns. The amplitudes of the components will be the column_amplitudes.
             
            This function performs the reverse operation from creating an interference group from a state. The function 
            works if states are full expanded into single photon states as well as when the state is still a single
            state with multiple photons per optical channel. A mixed situation where some columns are expanded can easily
            lead to erroneous results.

            Example 1:
                original_state.optical_components = [('110',r),('011',1/2), ('101',1/2)]
                interference_group = fsc.InterferenceGroup(state = original_state)
                interference_group.split_columns()
                interference_group.single_photon_states()
                # we now have all single photon states in the interference group
                new_state = interference_group.generate_single_state_from_interference_group()

                ==> new_state will now be identical to original_state (a new instance, but identical content)

            Example 2:
                original_state.optical_components = [('110',r),('011',1/2), ('101',1/2)]
                interference_group = fsc.InterferenceGroup(state = original_state)
                # we have multi-photon states in the interference group
                new_state = interference_group.generate_single_state_from_interference_group()

                ==> new_state will now be identical to original_state (a new instance, but identical content)

        """
        self.split_columns()
        optical_components = dict([])
        for column in self:
            group_cumulative_probability = column.group_cumulative_probability
            if len(column) != 0:
                state = column.generate_single_state_from_column()
                for k,v in state.optical_components.items():
                    new_amplitude = column.column_amplitude * v['amplitude']
                    new_probability = np.abs(new_amplitude)**2
                    amp_prob = {'amplitude': new_amplitude, 'probability': new_probability}
                    optical_components.update({k:amp_prob})
            else:
                continue

        new_state= state.copy()
        new_state.cumulative_probability = group_cumulative_probability
        new_state.optical_components = optical_components
        new_state.auxiliary_information.pop('photon_resolution', None)
        return new_state
    
    def rescale_column_amplitudes(self) -> None:
        """ Adapt the column amplitudes such that the sum of their squares adds up to 1"""
        sumsquared = 0
        for column in self.by_column():
            sumsquared += np.abs(column.column_amplitude)**2
        for column in self.by_column():
            column.column_amplitude = column.column_amplitude / np.sqrt(sumsquared)
        return
    
    def rescale_boson_factors(self):
        """ Adapt the boson factors such that for each column the 'self detection probability' is equal to the square of the
            column amplitude """
        for column in self.by_column():
            outcome = column.generate_optical_component_corresponding_to_column()
            measurement_projection, projection_boson_factor = self._measurement_projection(optical_component = outcome)
            measured_probability = self._calculate_probability_for_this_projection_for_these_state_columns(
                                                                        measurement_projection,
                                                                        projection_boson_factor,
                                                                        column, 
                                                                        column)
            expected_probability = np.abs(column.column_amplitude)**2
            correction_needed = expected_probability/measured_probability
            column.column_boson_factor = column.column_boson_factor / np.sqrt(correction_needed)
        return
        

    def _valid_combinations(self,measurement_projection: list,state_column: ColumnOfStates) -> list:
        """ Generate all valid combinations between photon state and projection

            - Projection on 2130 will have measurement_projection [0,0,1,2,2,2]
            - A state column for component '2130' will have (for example) photon_channel_list [0,1,2,2,2,0]
            - Valid combinations will be for this case:
                [ (0,5,1,2,3,4), (5,0,1,2,3,4), (0,5,1,4,2,3), ... ]
                So, if you read projection from left to right the valid combination indicates which index in photon_channel_list
                corresponds to measurement_projection
            - Note 1: If the column contains empty states (like '000') then this state does not add a photon to
                column.list_of_states(). The index of states in the column is then not the same as the index of 
                photons in the list. e.g., if we have a column with state '00' and '01' and projection is on '01' 
                the the returned list of valid combinations will be [(0,)]. The state with the photon has index 1, 
                but in the photon list this is index 0
            - Note 2: if there is just one photon the Tuple notation forces the addition of a comma, like in example
                in note 1: the list can be [(0,)], meaning the first photon in the photon list matches with the first 
                photon in the measurement projection.

            The expected format for 'measurement_projection' is a list of integers like [0,0,1,2,2,2]
            The expected format for 'state_column' is a ColumnOfStates, where there the states are single photon states.

        Args:
            measurement projection (list of integers): List of channels for the photons in the projection
            state_column (ColumnOfStates): Column with photons to be measured. The column has to contain only
                single photon states.

        Return:
            (list) : List of valid combinations
            
        """
        list_of_valid_combinations = [[]]
        photon_list = state_column.list_of_photons()
        if len(photon_list) == 0:
            photon_channel_list = []
        elif isinstance(photon_list[0], tuple):
            photon_channel_list = [photon[0] for photon in photon_list]
        else:
            photon_channel_list = photon_list
        # for every position in measurement projection, check for which index in photon list there is a matching photon that so far
        # has not been used
        for projection in measurement_projection:
            new_list_of_valid_combinations = []
            this_projection_matches_with_photon_indices = [i for i,n in enumerate(photon_channel_list) if n == projection]
            # loop through all existing matched combinations, if the index has not be used add it
            for valid_combination in list_of_valid_combinations:
                for option in this_projection_matches_with_photon_indices:
                    if option not in valid_combination:
                        new_combination = valid_combination + [option]
                        new_list_of_valid_combinations.append(new_combination)
                list_of_valid_combinations = new_list_of_valid_combinations
        # return the combinations, but only if the length is equal to the measurement projection
        return [tuple(valid_combi) for valid_combi in list_of_valid_combinations if len(valid_combi) == len(measurement_projection)]
    
    def _find_valid_photon_pairs_for_projection(self,measurement_projection,state_column_bra_side, state_column_ket_side) -> list:
        """ Generate all possible pairs photons that can contribute to a measurement projection. The return will be a list of
            'sequences'. A sequence is a tuple of pairs indicating which photons can combine to contribute to the projection.

            In this example the first and second photon can combine, but the photon with index 2 can only combine with itself, 
            leading to thre sequences in the list that will be returned:
                    [((0,1),(1,0),(2,2)), ((0,0),(1,1),(2,2)), ((1,0),(0,1),(2,2))] 

            Note: The indices are indices of photons in the photon list. If (one of) the column(s) contains the vacuum state 
                (like '000') this state contains no photons and the 'photon list index' is different from the 'state in column' 
                index. As Example: We have a bra-column with states '00' and '10' and a ket-column with states '10' and '00'. The list
                of valid photon pairs will be  [((0,0),) ]. The photon list of both columns only contain one entry (as there is one photon)
                and these photons with index 0 in the photon list form a valid combination when projecting on '10'. If we consider states then it
                is state with index 1 at bra side, and state with index 0 and ket side that combine. The indices for valid photon
                pairs are indices in the COLUMN.LIST_OF_PHOTONS() and not indices of the states.

        Args:
            measurement projection (list of integers): List of channels for the photons in the projection
            state_column_bra_side (ColumnOfStates): Column with photons to be measured. The column has to contain only
                single photon states.
            state_column_ket_side (ColumnOfStates): Column with photons to be measured. The column has to contain only
                single photon states.

        Returns:
            list: list of all possible combinations as tuples
        """
        valid_combinations_bra_side = self._valid_combinations(measurement_projection,state_column_bra_side )
        valid_combinations_ket_side = self._valid_combinations(measurement_projection,state_column_ket_side)

        valid_bra_ket_pairs = []
        for valid_bra in valid_combinations_bra_side:
            for valid_ket in valid_combinations_ket_side:
                sequence = tuple([(bra,ket) for bra,ket in zip(valid_bra,valid_ket)])
                valid_bra_ket_pairs.append(sequence)  
        return valid_bra_ket_pairs



    def _calculate_probability_for_photon_sequence(self, sequence, state_column_bra_side ,state_column_ket_side) -> float:
        """ Calculates the probability for a given sequence of photon pairs. The probability is the product of the probability
            to detect the pairs in the sequence. For any pair the probability is determined by the overlap of their wave packets, as 
            determined in the function self.photon_probability_function

        Args:
            sequence (_type_): A tuple of photon pairs, e.g., ((0,1),(1,0),(2,2))
            state_column_bra_side (ColumnOfStates): Column with photons to be measured. The column has to contain only
                                                    single photon states.
            state_column_ket_side (ColumnOfStates): Column with photons to be measured. The column has to contain only
                                                    single photon states.

        Returns:
            float: Probability
        """
        # the pair should contain indices in photon list not in state column, otherwise it goes wrong for empty state with zero photons
        list_of_probabilities = []
        photon_list_bra_side = state_column_bra_side.list_of_photons()
        photon_list_ket_side = state_column_ket_side.list_of_photons()
        for pair in sequence:
            # column.list_of_photons() is of form [(channel photon0, photon_info photon0), (channel photon1, photon_info photon1), ...]
            # we want to retreive the photon_info to calculate overlap of the wave-packets/wave-function
            # sequence is of form ((0,1),(1,2),(2,3),(3,0)) where (0,1) is a pair of photons with indices 0 and 1 in photon lists at bra- and ket-
            # side respectively
            photon_info_dict_bra_side = photon_list_bra_side[pair[0]][1]
            photon_info_dict_ket_side = photon_list_ket_side[pair[1]][1]
            
            # the dictionary photon information has to be of  form that can be read by the function 'self.photon_probability_function'
            # For temporal overlap the dict is {'time_delay' (float): , 'pulse_width': (float)} but for other situations the photon_probability_function
            # can be redefined, taking different inputs.
            list_of_probabilities.append(self.photon_probability_function(photon_info_dict_bra_side, photon_info_dict_ket_side))
        return np.prod(list_of_probabilities)
    
    def _measurement_projection(self, optical_component: str = None, target_values_and_channels: dict = None) -> tuple[list, float]:       
        """ Calculates the photon list and boson factor for the optical component which is measured. The component
            to be measured can be given as a string (argument optical_component) or as a dict of containing target channels and
            target values.

            Examples: For a circuit with 4 optical channels
                CollectionOfStateColumns._measurement_projection('1111') -> ([0, 1, 2, 3], 1.0)
                CollectionOfStateColumns._measurement_projection('2222') -> ([0, 0, 1, 1, 2, 2, 3, 3], 4.0)
                CollectionOfStateColumns._measurement_projection('2002') -> ([0, 0, 3, 3], 2.0)
                CollectionOfStateColumns.._measurement_projection('3000') -> ([0, 0, 0], 2.4494897427831783)
                CollectionOfStateColumns._measurement_projection('3003') -> ([0, 0, 0, 3, 3, 3], 6.00)

                CollectionOfStateColumns._measurement_projection({'channels': [0,1,2], 'values': [2,2,2]}) 
                            -> ([0, 0, 1, 1, 2, 2], 2.0)
        Args:
            optical_component (str): Optical component in string form (i.e., '1001')
            target_values_and_channels (dict): values in dictionary form (i.e., {'channels': [0,1,2], 'values': [2,2,2]})
        Returns:
            tuple: First element is the photon list (as list of integers), second element the bospn factor as float
        """
        if optical_component is not None:
            # we need at least one state to be ableto look up the optical component as a 
            try:
                for state in self.by_state():
                    reference_state = state
                    break
                values = reference_state._dict_of_valid_component_names[optical_component]
                channels = [index for index in range(len(values))]
            except: 
                Exception('If we use optical_component as a string we need at least one state in the InterferenceGroup')
        elif target_values_and_channels is not None:
            values = target_values_and_channels['values']
            channels = target_values_and_channels['channels']
        destinations_list = []
        for channel,value in zip(channels, values):
            for count in range(value):
                destinations_list.append(channel)
        boson_factor = 1
        for channel_value in values:
            for photon_count in range(channel_value): 
                boson_factor *= np.sqrt(photon_count+1)

        return (destinations_list, boson_factor)
    
    def _calculate_probability_for_this_projection_for_these_state_columns(self,
                                                                           measurement_projection,
                                                                           projection_boson_factor,
                                                                           state_column_bra_side, 
                                                                           state_column_ket_side) -> float:
        """ Calculates the detection probability for a set of 'ColumnsOfStates" for a a given projection state.
            The formula used is:

            Probability =  pre-factor * overlap
            with:
            pre-factor = amplitude-bra * amplitude-ket / (boson-bra * boson-ket * boson_projection * boson-projection)
            overlap = SUM(overlap pair-0 * overlap pair-1 * overlap pair-2 * ..) 
                Here SUM is over all possible pairs of one bra and one ket photon that matches with the projection state.

            If detection is not possible (for instance because photon state in bra, ket and measurement_projection do not match)
            the function returns None.

        Args:
            measurement projection (list of integers): List of channels for the photons in the projection
            projection_boson_factor (float): boson factor for the projection 
            state_column_bra_side (ColumnOfStates): Column with photons to be measured. The column has to contain only
                                                    single photon states.
            state_column_ket_side (ColumnOfStates): Column with photons to be measured. The column has to contain only
                                                    single photon states.
        Returns:
            float: detection probability, or None if detection is not possible
        """
    
        # find all sequences of photons pairs which match with the optical component for the projection
        # based on this sequence the overlap of wave functions will be calculated
        valid_bra_ket_pairs = self._find_valid_photon_pairs_for_projection(measurement_projection,state_column_bra_side, state_column_ket_side)
        if len(valid_bra_ket_pairs) == 0:
            return None

        # The pre factor consists of amplitude and 'boson factor' for projection
        # Any boson factor for the state has to be incorporated in the amplitude
        # The overall state has to be normalized
        pre_factor = state_column_ket_side.column_amplitude * state_column_bra_side.column_amplitude
        pre_factor = pre_factor/ (projection_boson_factor * projection_boson_factor)
        pre_factor = pre_factor/(state_column_ket_side.column_boson_factor * state_column_bra_side.column_boson_factor)

        # calculate the overlap of the photon wave functions
        probability = 0
        for sequence in valid_bra_ket_pairs:
            overlap_probability= self._calculate_probability_for_photon_sequence(sequence, state_column_bra_side ,state_column_ket_side )
            probability += overlap_probability

        probability = pre_factor * probability

        return np.real(probability)
    
    def total_probability_for_group_all_outcomes(self, list_of_projections: list = None) -> dict:
        """ Calculate all possible outcomes for a total measurement (all optical channels) and the probability to reach that outcome.
            (so these probabilities should add up to 1).

            If the provide a list_of_projections (format ['1101', '2201']) then only for those outcome the probability will be determined
            (the probabilities in that case could add up to a number lower than 1, or even be zero)

        Args:
            list_of_projections (list, optional): List or measurement outcomes to consider. Defaults to None.

        Returns:
            dict: Dictionary of form {projection:probability}
        """
        outcomes_and_their_probabilities = dict([])
        if list_of_projections is None:
            # determine all possible outcomes
            list_of_outcomes = []
            for column in self.by_column():
                list_of_outcomes.append(column.generate_optical_component_corresponding_to_column(use_stored_values=True))
            list_of_outcomes = set(list_of_outcomes)
        else:
            list_of_outcomes = list_of_projections
        
        for outcome in list_of_outcomes:
            total_probability = []
            measurement_projection, projection_boson_factor = self._measurement_projection(optical_component = outcome)
            list_of_probabilities_for_this_outcome = []
            # explore all possible combinations of state_columns for the bra and ket side
            for state_column_bra_side in self.by_column():
                if state_column_bra_side.generate_optical_component_corresponding_to_column(use_stored_values = True) != outcome:
                    continue
                for state_column_ket_side in self.by_column():
                    if state_column_ket_side.generate_optical_component_corresponding_to_column(use_stored_values = True) != outcome:
                        continue
                    probability = self._calculate_probability_for_this_projection_for_these_state_columns(
                                                                        measurement_projection,
                                                                        projection_boson_factor,
                                                                        state_column_bra_side, 
                                                                        state_column_ket_side)
                    
                    
                    if probability is not None:
                        list_of_probabilities_for_this_outcome.append(probability)

            total_probability.append(sum(list_of_probabilities_for_this_outcome))
            outcomes_and_their_probabilities.update({outcome:sum(total_probability)})
        return outcomes_and_their_probabilities
    
    def _probability_column_list(self, column_list):
        """ Calculates the probability for a column list where ALL COLUMNS HAVE THE SAME COMPONENT. For speed the function does not NOT CHECK, but 
            assumes that column_list[0]._column_as_component is representative for all columns in the list."""
        measurement_projection, projection_boson_factor = self._measurement_projection(optical_component = column_list[0]._column_as_component)
        total_probability = 0
        for state_column_bra_side in column_list:
            for state_column_ket_side in column_list:
                probability = self._calculate_probability_for_this_projection_for_these_state_columns(
                                                                        measurement_projection,
                                                                        projection_boson_factor,
                                                                        state_column_bra_side, 
                                                                        state_column_ket_side)               
                if probability is not None:
                    total_probability += probability
        return total_probability
    

    def measure_interference_group_alternative(self, 
                                   optical_channels_to_measure, 
                                   classical_channels_to_write_to,
                                   list_of_projections: list = None) -> list:
        """ Perform measurement on the interference group and return a list of groups, where each new group leads to one
            measurement outcome (so the number of groups in the returned list is equal to the number of different measurement
            results)

            This is an alternative implementation of 'measure_interference_group' which seems a little bit faster.

        Args:
            optical_channels_to_measure (list): list of of optical channel numbers to be measured
            classical_channels_to_write_to (list): list of classical channel numbers to write the measurement result to
            list_of_projections (list, optional): List of projections to limit the 'search' and speed up measurement. Defaults to None.

        Returns:
            list: List of interference groups
        """
        # prepare the list that will be returned
        list_of_groups = []

        columns_with_this_partial_outcome = dict([])
        for column in self.by_column():
            optical_component = column.generate_optical_component_corresponding_to_column(
                                                                    return_values_instead_of_component_name_as_string=False,
                                                                    use_stored_values = False)
            if list_of_projections is not None and not optical_component in list_of_projections:
                continue
            values = column._column_as_tuple_of_values
            partial_outcome = "".join([str(values[channel]) for channel in optical_channels_to_measure])
            try:
                columns_with_this_partial_outcome[partial_outcome].append(column)
            except:
                columns_with_this_partial_outcome.update({partial_outcome:[column]})

        for partial_outcome, column_list in columns_with_this_partial_outcome.items():
            
            columns_with_this_full_outcome = dict([])
            for column in column_list:
                full_outcome = column._column_as_component
                try:
                    columns_with_this_full_outcome[full_outcome].append(column)
                except:
                    columns_with_this_full_outcome.update({full_outcome:[column]})
            probability_for_partial_outcome = sum([self._probability_column_list(list_full_outcome) for list_full_outcome in columns_with_this_full_outcome.values()])
            new_group = InterferenceGroup(list_of_state_columns=column_list)
            new_group.photon_probability_function = self.photon_probability_function
            
            # set the new cumulative probability for the group
            new_group.group_cumulative_probability *= probability_for_partial_outcome 

            # renormalize column_amplitudes and boson factors
            new_group.rescale_column_amplitudes()
            new_group.rescale_boson_factors()

            # write classical channels per outcome
            values_to_measure = [column_list[0]._column_as_tuple_of_values[channel] for channel in optical_channels_to_measure]
            for state in new_group.by_state():
                for index, value in enumerate(values_to_measure):
                    classical_channel = classical_channels_to_write_to[index]
                    state.classical_channel_values[classical_channel] = value
                    state.measurement_results.append({'measurement_results':state.classical_channel_values, 'probability': probability_for_partial_outcome})  
            # add the group for this partial outcome to the list which will be returned
            list_of_groups.append(new_group)

        return list_of_groups



    def measure_interference_group(self, 
                                   optical_channels_to_measure, 
                                   classical_channels_to_write_to,
                                   list_of_projections: list = None) -> list:
        """ Perform measurement on the interference group and return a list of groups, where each new group leads to one
            measurement outcome (so the number of groups in the returned list is equal to the number of different measurement
            results)

        Args:
            optical_channels_to_measure (list): list of of optical channel numbers to be measured
            classical_channels_to_write_to (list): list of classical channel numbers to write the measurement result to
            list_of_projections (list, optional): List of projections to limit the 'search' and speed up measurement. Defaults to None.

        Returns:
            list: List of interference groups
        """
        # call the faster alternative implementation, but leave original in code untill 100% sure that alternative is faster in all cases
        return self.measure_interference_group_alternative( optical_channels_to_measure, 
                                                            classical_channels_to_write_to,
                                                            list_of_projections)

        # prepare the list that will be returned
        list_of_groups = []

        columns_with_this_outcome = dict([])
        values_for_this_outcome = dict([])
        total_outcomes_for_this_partial_outcome = dict([])

        # check all column on the measurement outcome they would generate, and group the columns together that would have identical
        # measurement outcomes. These columns will form a new interference group. So the single original group is split in groups per
        # measurement outcome. If we do a 'partial' measurement (i.e., not measure all channels) then the outcome is the outcome for the 
        # the detected channels only, not for the full state.
        for column in self.by_column():
            optical_component = column.generate_optical_component_corresponding_to_column(
                                                                                return_values_instead_of_component_name_as_string=False,
                                                                                use_stored_values = False)
            if not optical_component in list_of_projections:
                continue
            # we first set 'use stored values' to false to force recalculation. Afterwards we recover the earlier calculated result.
            values = column.generate_optical_component_corresponding_to_column(return_values_instead_of_component_name_as_string=True,
                                                                               use_stored_values = True)
            outcome = "".join([str(values[channel]) for channel in optical_channels_to_measure])
            if outcome in columns_with_this_outcome.keys():
                columns_with_this_outcome[outcome].append(column)
                # in 'total_outcomes_for_this_partial_outcome' we list the outcomes for a full (all channel) measurement 
                # that lead to the same outcome for the specific partial measurement
                # so if 'total outcomes' would be 201, 211 and 111 and we measure channel 0 then 201 and 211 would be total outcomes
                # belonging  to partial outcome 2, and 111 would be the single total outcome belonging to partial outcome 1.
                total_outcomes_for_this_partial_outcome[outcome].append(column.generate_optical_component_corresponding_to_column(
                                                                                return_values_instead_of_component_name_as_string=False,
                                                                                use_stored_values = True))
            else:
                columns_with_this_outcome.update({outcome : [column]})
                values_for_this_outcome.update({outcome : tuple([values[channel] for channel in optical_channels_to_measure])})
                total_outcomes_for_this_partial_outcome.update({outcome: [column.generate_optical_component_corresponding_to_column(
                                                                                return_values_instead_of_component_name_as_string=False,
                                                                                use_stored_values = True)]})
        
        probs = self.total_probability_for_group_all_outcomes(list_of_projections=list_of_projections)
        for outcome, column_list in columns_with_this_outcome.items():
            new_group = InterferenceGroup(list_of_state_columns=column_list)
            new_group.photon_probability_function = self.photon_probability_function
            
            # determine total probability for this partial outcome by adding the total outcome probabilities
            probability = 0
            for total_outcome in set(total_outcomes_for_this_partial_outcome[outcome]):
                probability += probs[total_outcome]

            # set the new cumulative probability for the group
            new_group.group_cumulative_probability *= probability

            # renormalize column_amplitudes and boson factors
            new_group.rescale_column_amplitudes()
            new_group.rescale_boson_factors()

            # write classical channels per outcome
            for state in new_group.by_state():
                for index, value in enumerate(values_for_this_outcome[outcome]):
                    classical_channel = classical_channels_to_write_to[index]
                    state.classical_channel_values[classical_channel] = value
                    state.measurement_results.append({'measurement_results':state.classical_channel_values, 'probability': probability})  
            # add the group for this partial outcome to the list which will be returned
            list_of_groups.append(new_group)

        return list_of_groups
