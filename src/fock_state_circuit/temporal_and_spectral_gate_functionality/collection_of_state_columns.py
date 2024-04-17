from __future__ import annotations
from fock_state_circuit.state import State
from fock_state_circuit.temporal_and_spectral_gate_functionality.column_of_states import ColumnOfStates
from fock_state_circuit.temporal_and_spectral_gate_functionality.interference_group import InterferenceGroup
import numpy as np

class CollectionOfStateColumns():
    """ CollectionOfStateColumns
        __str__(self): Return the string for 'pretty printing' the column. Used as print(column)
    
        as_dictionary(self) -> dict
                returns the collection in dictionary form
        
        by_state(self):
                Generator for states (States) in the collection

        by_column(self):
                Generator for columns (ColumnOfStates) in the collection


        by_interference_group(self):
                Generator for interference groups (InterferenceGroups) in the collection"
    
        find_next_free_interference_group_identifier(self,initial_state) -> int:
                Generates the lowest identifier (integer) that is not yet used as an interference group identifier in the collection
    
        generate_collection_of_states(self) -> CollectionOfStates
                Returns a collection of states from the collection of time columns.
    
        new_instance_from_state_information(self) -> CollectionOfStateColumns
                Generates a 'copy' of this instance of CollectionOfStatesColumns with added feature that
                all paramaters a reset based on the values of state.auxiliary_information['photon_resolution'] for the 
                states in the state. If this information in state.auxiliary_information has been changed the 
                collection that is returned form this function will again be fully consistent.
            
        split(self) -> None
                Split all states into states with single optical components. The split of states will
                be in new columns (ColumnOfStates) in the same InteferenceGroup.    

        single_photon_states(self) -> None
            Split all states into single photon states. The states will be in the same columns (ColumnOfStates). 
        
        perform_measurement(    self,
                                optical_channels_to_measure: list = [],
                                classical_channels_to_write_to: list = [], 
                                list_of_projections: list = None
                                ) -> any:
            Perform a measurement on this CollectionOfStateColumns and return a collection_of_states. In the returned collection
            the classical channels will be populated, the quantum states will have 'collapsed' and the cumulative probabilities are
            adjusted to reflect detection probability for the different outcomes.

        Last modified: April 16th, 2024
                
    """
    _VERSION = '1.0.0'
    def __init__(self, collection_of_states):
        """ Constructor for an instance of the class CollectionOfStateColumns.The constructor will create new instances
            ColumnOfStates and InterferenceGroups which are part of this new instance of CollectionOfStateColumns.

        Args:
            collection_of_states (CollectionOfStates): collection of states from which the new instance of 
                ColumnOfStates is to be formed.
        """

        self.collection_by_column = dict([])
        
        # the number of channels defining the circuit.
        self._no_of_optical_channels = collection_of_states._no_of_optical_channels
        self._no_of_classical_channels = collection_of_states._no_of_classical_channels    

        # the original collection_of_states
        self._collection_of_states = collection_of_states

        
        # create a dict with as keys the valid state names as strings and the list of photon numbers as value
        self._dict_of_valid_component_names = collection_of_states._dict_of_valid_component_names

        # this is the function to determine how the probability to dectect two photons depends on their properties
        # (i.e., how we calculate the 'overlap' of the wave functions)
        self.collection = {'photon_probability_function': None}

        # create a dict with as keys the tuple of the list of photon numbers as value, 
        # and valid state names as strings as value
        self._dict_of_optical_values = collection_of_states._dict_of_optical_values  

        for state in collection_of_states:
            initial_state = state.initial_state
            if initial_state not in self.collection_by_column.keys():
                self.collection_by_column.update({initial_state:dict([])})
            
            # When initializing we can have two situations. Either the collection_of_states has states which already
            # carry information on interference groups and columns, or the states do not have this information. A mixed situation
            # (where some states do, and some do not have an interference group and column will lead to errors).
            # If states have and interference group identifier and column identifier we have to add them to the right place in 
            # in the dict 'collection by column'. If they do not have this information we create a new interference group, with a single 
            # column for every state in the collection.
 
            # check if state has 'photon_resolution' information
            if 'photon_resolution' in state.auxiliary_information.keys():
                interference_group_id = state.auxiliary_information['photon_resolution']['interference_group_identifier']
                if interference_group_id not in self.collection_by_column[initial_state].keys():
                    # create a new interference group with a new column for this state
                    new_column = ColumnOfStates(state=state)
                    new_group = InterferenceGroup(list_of_state_columns=[new_column])
                    self.collection_by_column[initial_state].update({interference_group_id:new_group})
                else:
                    self.collection_by_column[initial_state][interference_group_id].add_state(state)

            # otherwise this is a new state without interference group identifier and column identifier
            else:
                new_interference_group_identifier = self.find_next_free_interference_group_identifier(initial_state)
                new_group = InterferenceGroup(state=state,interference_group_identifier=new_interference_group_identifier)
                self.collection_by_column[initial_state].update({new_interference_group_identifier : new_group})
            
        return
    
    @property
    def photon_probability_function(self):
        return self.collection['photon_probability_function']
       
    @photon_probability_function.setter
    def photon_probability_function(self,function):
        self.collection['photon_probability_function'] = function
        for group in self.by_group():
            group.photon_probability_function = function

    def __str__(self) -> str:
        """ Return the string for 'pretty printing' the CollectionOfStateColumns.

        Returns:
            str: string describing the collection.
        """ 
        text = ''
        for initial_state in self.collection_by_column.keys():
            text += '-----------------------------------' + '\n'
            text += 'Initial_state: ' + str(initial_state) + '\n'
            for interference_group in self.collection_by_column[initial_state].keys():
                group = self.collection_by_column[initial_state][interference_group]
                text += '\t-----------------' + '\n'
                text += '\tgroup id:' + str(group.interference_group_identifier) + '\n'
                text += '\tgroup probability: ' + "{val:.2f}".format(val = group.group_cumulative_probability) + '\n'
                for column in self.collection_by_column[initial_state][interference_group]:
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
    
    def as_dictionary(self) -> dict:
        """ Returns the collection in dictionary form"""
        return self.collection_by_column
    
    def by_state(self):
        """ Generator for states (States) in the collection"""
        for initial_state in self.collection_by_column.keys():
            for interference_group_identifier in self.collection_by_column[initial_state].keys():
                for column in self.collection_by_column[initial_state][interference_group_identifier]:
                    for state in column:
                        yield state

    def by_column(self):
        """ Generator for columns (ColumnOfStates) in the collection"""
        for initial_state in self.collection_by_column.keys():
            for interference_group_identifier in self.collection_by_column[initial_state].keys():
                for column in self.collection_by_column[initial_state][interference_group_identifier]:
                    yield column

    def by_group(self):
        " Generator for interference groups (InterferenceGroups) in the collection"
        for initial_state in self.collection_by_column.keys():
            for interference_group in self.collection_by_column[initial_state].values():
                yield interference_group

    def find_next_free_interference_group_identifier(self,initial_state) -> int:
        """ Generates the lowest identifier (integer) that is not yet used as an interference group identifier in the collection"""
        identifier = 0
        while True:
            if identifier not in self.collection_by_column[initial_state].keys():
                return identifier
            identifier += 1

    def generate_collection_of_states(self) -> any:
        """ Returns a collection of states from the collection of time columns."""
        new_collection_of_states = None
        for state in self.by_state():
            if new_collection_of_states is None:
                new_collection_of_states = state._collection_of_states.copy()
                new_collection_of_states.clear()
            new_collection_of_states.add_state(state.copy())

        return new_collection_of_states
    
    def new_instance_from_state_information(self) -> CollectionOfStateColumns:
        """ Generates a 'copy' of this instance of CollectionOfStatesColumns with added feature that
            all paramaters a reset based on the values of state.auxiliary_information['photon_resolution'] for the 
            states in the state. If this information in state.auxiliary_information has been changed the 
            collection that is returned form this function will again be fully consistent.
        """
        temp_collection = self.generate_collection_of_states()
        new_collection_of_columns = CollectionOfStateColumns(collection_of_states=temp_collection)
        return new_collection_of_columns
    
    def split(self) -> None:
        """ Split all states into states with single optical components. The split of states will
            be in new columns (ColumnOfStates) in the same InteferenceGroup.    
        """
        for initial_state in self.collection_by_column.keys():
            for interference_group in self.collection_by_column[initial_state].values():
                interference_group.split_columns()

        return
    
    def single_photon_states(self) -> None:
        """ Split all states into single photon states. The states will be in the same columns (ColumnOfStates).
        """
        for initial_state in self.collection_by_column.keys():
            for interference_group in self.collection_by_column[initial_state].values():
                interference_group.single_photon_states()
        return
      
    def perform_measurement(    self,
                                optical_channels_to_measure: list = [],
                                classical_channels_to_write_to: list = [], 
                                list_of_projections: list = None
                                ) -> any:
        """ Perform a measurement on this CollectionOfStateColumns and return a collection_of_states. In the returned collection
            the classical channels will be populated, the quantum states will have 'collapsed' and the cumulative probabilities are
            adjusted to reflect detection probability for the different outcomes.

        Args:
            optical_channels_to_measure (list): list of of optical channel numbers to be measured
            classical_channels_to_write_to (list): list of classical channel numbers to write the measurement result to
            list_of_projections (list, optional): List of projections to limit the 'search' and speed up measurement. Defaults to None.

        Returns:
            CollectionOfStates: CollectionOfStates after measurement
        """

        resulting_collection = self._collection_of_states.copy()
        resulting_collection.clear()
        
        group_counter = 0   
        for group in self.by_group():
            list_of_groups = group.measure_interference_group(optical_channels_to_measure, classical_channels_to_write_to, list_of_projections)
            for new_group in list_of_groups:
                new_group.interference_group_identifier = group_counter
                group_counter+=1

                for state in new_group.by_state():
                    resulting_collection.add_state(state)

        return resulting_collection
    
    