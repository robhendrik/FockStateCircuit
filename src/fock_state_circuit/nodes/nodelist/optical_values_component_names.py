""" Module optical_values_components_names.py

    This module contains two classes: ComponentNames() and OpticalValues()

    These classes deal with managing the relation between component names as strings (e.g., '0001') and the values (photon numbers)
    in the optical channels (e.g., (0,0,0,1)). 
    
    1) Classes manage the option for 'reversed notation', meaning the situation where the user want the value for the 
    channel 0 to be at the right of the name string (with FockStateCircuit.channel_0_left_in_state_name set to False
    leading to '0001' = (1,0,0,0)) or the default situation with channel 0 at the left side of the name string (with 
    FockStateCircuit.channel_0_left_in_state_name set to True leading to '0001' = (0,0,0,1))
    
    2) Classes manage the option for 'double digit photon numbers', meaning the situation where the we can have 10 or more photons
    in a channel. In that case optical values (1,0,0,0) correspond to state name '01000000' (with .channel_0_left_in_state_name == True ) or
    '00000001' (with .channel_0_left_in_state_name == False ). NOTE: We can reverse the order in which the optical values
    are captured in the name string, but we do not reverse the notation of a specific value. So optical value 10 will always be string '10' and 
    optical value 1 will be string '01', irrespective of whether we reverse the state notation.

                            Examples:
                            length_of_fock_state = 5 (i.e., single digit optical value notation)
                            with channel_0_left_in_state_name == True: 
                            * (0,0,1) <--> '001'
                            * (2,3,2) <--> '232'
                            * (1,5,1) <--> not allowed, with length_of_fock_state = 5 we only allow photon numbers 0,1,2,3 or 4

                            length_of_fock_state = 15 (i.e., double digit optical value notation)
                            with channel_0_left_in_state_name == True: 
                            * (0,0,1) <--> '000001'
                            * (2,3,4) <--> '020304'
                            * (1,4,1) <--> '010401'
                            * (10,11,10) <--> '101110'

                            length_of_fock_state = 15 (i.e., double digit optical value notation)
                            with channel_0_left_in_state_name == False: 
                            * (0,0,1) <--> '010000'
                            * (2,3,4) <--> '040302'
                            * (1,4,1) <--> '010401'
                            * (10,11,10) <--> '101110'

                            length_of_fock_state = 115 (i.e., triple digit optical value notation)
                            with channel_0_left_in_state_name == True: 
                            * (0,0,1) <--> '000000001'
                            * (2,3,4) <--> '002003004'
                            * (1,4,1) <--> '001004001'
                            * (10,11,10) <--> '010011010'

                            length_of_fock_state = 115 (i.e., triple digit optical value notation)
                            with channel_0_left_in_state_name == False: 
                            * (0,0,1) <--> '001000000'
                            * (2,3,4) <--> '004003002'
                            * (1,4,1) <--> '001004001'
                            * (10,11,10) <--> '010011010'

    3) Classes can serve as iterator just like a python dictionary:
                            Examples:
                            * [optical_value for optical_value in circuit._dict_of_optical_values.keys()]
                            * [component_name for component_name in circuit._dict_of_valid_component_names.keys()]
                            * [name,value for name,values in circuit._dict_of_valid_component_names.items()]

    4) We can check existence/validity of an optical state with the - in - keyword
                            Examples:
                            For these settigs:
                                comp_names = ComponentNames(length_of_fock_state=115,
                                                            no_of_optical_channels=3,
                                                            channel_0_left_in_state_name=False)
                                optical_values = OpticalValues(comp_names)
                            All these statement are True
                                '001000000' in comp_names
                                [2,3,4] in optical_values
                                (2,2,2) in optical_values # accepts values as lists or as tuples
                                '201000000' not in comp_names 
                                (2,3,4,6) not in optical_values

    5) We can get the index of a state in the full basis
                            Examples:
                            For these settings:
                                comp_names = ComponentNames(length_of_fock_state=56,
                                                            no_of_optical_channels=3,
                                                            channel_0_left_in_state_name=False)
                                optical_values = OpticalValues(comp_names)
                            All these statement are True
                                optical_values.index_of([1,0,0]) == 1 # note that we use reversed state notation, not the default !
                                optical_values.index_of((0,0,1)) == 56**2 # note that we use reversed state notation, not the default !
                                comp_names.index_of('000010') == 10 # note that we use reversed state notation, not the default !

                            -- or --
                            For these settings:
                                length_of_fock_state = 56
                                channel_0_left_in_state_name = True
                                comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
                                optical_values = OpticalValues(comp_names)
                            All these statements are True
                                optical_values.index_of((0,0,1)) == 56**2
                                optical_values.index_of((1,0,0)) == 1
                                comp_names.index_of('000001') == 56**2
                                comp_names.index_of('010000') == 1


    Last modified: June 7th, 2024     
"""
from typing import Optional, Union

class ComponentNames():
    """ Class ComponentNames. This class is used to translate component names as strings to lists of values in optical channels. 

        These classes deal with managing the relation between component names as strings (e.g., '0001') and the values (photon numbers)
            in the optical channels (e.g., (0,0,0,1)). 
            
            1) Classes manage the option for 'reversed notation', meaning the situation where the user want the value for the 
            channel 0 to be at the right of the name string (with FockStateCircuit.channel_0_left_in_state_name set to False
            leading to '0001' = (1,0,0,0)) or the default situation with channel 0 at the left side of the name string (with 
            FockStateCircuit.channel_0_left_in_state_name set to True leading to '0001' = (0,0,0,1))
            
            2) Classes manage the option for 'double digit photon numbers', meaning the situation where the we can have 10 or more photons
            in a channel. In that case optical values (1,0,0,0) correspond to state name '01000000' (with .channel_0_left_in_state_name == True ) or
            '00000001' (with .channel_0_left_in_state_name == False ). NOTE: We can reverse the order in which the optical values
            are captured in the name string, but we do not reverse the notation of a specific value. So optical value 10 will always be string '10' and 
            optical value 1 will be string '01', irrespective of whether we reverse the state notation.

                                    Examples:
                                    length_of_fock_state = 5 (i.e., single digit optical value notation)
                                    with channel_0_left_in_state_name == True: 
                                    * (0,0,1) <--> '001'
                                    * (2,3,2) <--> '232'
                                    * (1,5,1) <--> not allowed, with length_of_fock_state = 5 we only allow photon numbers 0,1,2,3 or 4

                                    length_of_fock_state = 15 (i.e., double digit optical value notation)
                                    with channel_0_left_in_state_name == True: 
                                    * (0,0,1) <--> '000001'
                                    * (2,3,4) <--> '020304'
                                    * (1,4,1) <--> '010401'
                                    * (10,11,10) <--> '101110'

                                    length_of_fock_state = 15 (i.e., double digit optical value notation)
                                    with channel_0_left_in_state_name == False: 
                                    * (0,0,1) <--> '010000'
                                    * (2,3,4) <--> '040302'
                                    * (1,4,1) <--> '010401'
                                    * (10,11,10) <--> '101110'

                                    length_of_fock_state = 115 (i.e., triple digit optical value notation)
                                    with channel_0_left_in_state_name == True: 
                                    * (0,0,1) <--> '000000001'
                                    * (2,3,4) <--> '002003004'
                                    * (1,4,1) <--> '001004001'
                                    * (10,11,10) <--> '010011010'

                                    length_of_fock_state = 115 (i.e., triple digit optical value notation)
                                    with channel_0_left_in_state_name == False: 
                                    * (0,0,1) <--> '001000000'
                                    * (2,3,4) <--> '004003002'
                                    * (1,4,1) <--> '001004001'
                                    * (10,11,10) <--> '010011010'

            3) Classes can serve as iterator just like a python dictionary:
                                    Examples:
                                    * [optical_value for optical_value in circuit._dict_of_optical_values.keys()]
                                    * [component_name for component_name in circuit._dict_of_valid_component_names.keys()]
                                    * [name,value for name,values in circuit._dict_of_valid_component_names.items()]

            4) We can check existence/validity of an optical state with the - in - keyword
                                    Examples:
                                    For these settigs:
                                        comp_names = ComponentNames(length_of_fock_state=115,
                                                                    no_of_optical_channels=3,
                                                                    channel_0_left_in_state_name=False)
                                        optical_values = OpticalValues(comp_names)
                                    All these statement are True
                                        '001000000' in comp_names
                                        [2,3,4] in optical_values
                                        (2,2,2) in optical_values # accepts values as lists or as tuples
                                        '201000000' not in comp_names 
                                        (2,3,4,6) not in optical_values

            5) We can get the index of a state in the full basis
                                    Examples:
                                    For these settings:
                                        comp_names = ComponentNames(length_of_fock_state=56,
                                                                    no_of_optical_channels=3,
                                                                    channel_0_left_in_state_name=False)
                                        optical_values = OpticalValues(comp_names)
                                    All these statement are True
                                        optical_values.index_of([1,0,0]) == 1 # note that we use reversed state notation, not the default !
                                        optical_values.index_of((0,0,1)) == 56**2 # note that we use reversed state notation, not the default !
                                        comp_names.index_of('000010') == 10 # note that we use reversed state notation, not the default !

                                    -- or --
                                    For these settings:
                                        length_of_fock_state = 56
                                        channel_0_left_in_state_name = True
                                        comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
                                        optical_values = OpticalValues(comp_names)
                                    All these statements are True
                                        optical_values.index_of((0,0,1)) == 56**2
                                        optical_values.index_of((1,0,0)) == 1
                                        comp_names.index_of('000001') == 56**2
                                        comp_names.index_of('010000') == 1


            Last modified: June 7th, 2024                 
    """
  
    def __init__(self,length_of_fock_state: int, no_of_optical_channels: int, channel_0_left_in_state_name: bool):
        """ Constructor for instane of Class ComponentNames"""
        self.length_of_fock_state = length_of_fock_state
        self.no_of_optical_channels = no_of_optical_channels
        self.channel_0_left_in_state_name = channel_0_left_in_state_name
        self.dict_of_optical_values, self.dict_of_valid_component_names = dict([]), dict([])
        self.digits_per_channel = len(str(length_of_fock_state-1))
        self.string_format_in_state_as_word = "{:0"+str(self.digits_per_channel)+ "d}"
        self.tuple_of_strings = tuple(self.string_format_in_state_as_word.format(number) for number in range(self.length_of_fock_state))
        self.size_of_basis = self.length_of_fock_state**self.no_of_optical_channels
        return

    def __getitem__(self,component_name: str) -> list:
        """ Return list of optical values for the given optical state in string format."""
        if component_name in self.dict_of_valid_component_names:
            return list(self.dict_of_valid_component_names[component_name])
        else:
            component_name_as_integer_tuple = tuple( int(component_name[a:a+self.digits_per_channel]) for a in range(0,len(component_name),self.digits_per_channel))
            if not self.channel_0_left_in_state_name:
                component_name_as_integer_tuple = tuple(reversed(component_name_as_integer_tuple))
            self.dict_of_valid_component_names.update({component_name: component_name_as_integer_tuple})
            self.dict_of_optical_values.update({component_name_as_integer_tuple: component_name})
            return list(component_name_as_integer_tuple)
        
    def _values_from_index(self, index: int) -> list:
        return [(index//(self.length_of_fock_state**c))%self.length_of_fock_state for c in range(self.no_of_optical_channels)]
        
    def __iter__(self):
        """ Iterator for optical component names as a string"""
        if len(self.dict_of_valid_component_names) == self.size_of_basis:
            return iter(self.dict_of_optical_values)

        self.list_of_fock_states = [self._values_from_index(n)  for n in range(self.length_of_fock_state**self.no_of_optical_channels)]
        if self.channel_0_left_in_state_name == True:
            self.dict_of_valid_component_names = {''.join([self.tuple_of_strings[number] for number in optical_state]):optical_state for optical_state in self.list_of_fock_states}
        else: #self.state_least_significant_digit_left == False:
            self.dict_of_valid_component_names = {''.join([self.tuple_of_strings[number] for number in optical_state[::-1]]):optical_state for optical_state in self.list_of_fock_states}       

        return iter(self.dict_of_valid_component_names)
    
    def __contains__(self, key: str) -> bool:
        """ True if the componet name (as string) represent a valid optical state."""
        values = self.__getitem__(key)
        return len(values) == self.no_of_optical_channels and min(values) >= 0 and max(values) < self.length_of_fock_state
    
    def __len__(self) -> int:
        """ Total number of possible states as number_of_possible_optical_values ^ number_of_optical_channels."""
        return self.size_of_basis

    def keys(self):
        """ Keys are component names as string, Values are the optical values in list format."""
        self.__iter__()
        return self.dict_of_valid_component_names.keys()

    def items(self):
        """ Keys are component names as string, Values are the optical values in list format."""
        self.__iter__()
        return self.dict_of_valid_component_names.items()

    def values(self):
        """ Keys are component names as string, Values are the optical values in list format."""
        self.__iter__()
        return self.dict_of_valid_component_names.values()
    
    def index_of(self,key: str) -> int:
        """ Return the index of the key, which is interpreted as a component name in string form. The index is in
            the basis which contains all valid optical states/photon number states in standard order.
        """
        values = self.__getitem__(key)
        return sum([value*(self.length_of_fock_state**index) for index,value in enumerate(values)])
    
    def at_index(self,index: int) -> str:
        """ Return the component name (as string) for the index. The index is in
            the basis which contains all valid optical states/photon number states in standard order.
        """
        optical_values = self._values_from_index(index) 
        if self.channel_0_left_in_state_name:              
            component_name = ''.join([self.tuple_of_strings[number] for number in optical_values])                                                                 
        else:
            component_name = ''.join([self.tuple_of_strings[number] for number in reversed(optical_values)])   
        return component_name


class OpticalValues():
    """ Class ComponentNames. This class is used to translate component names as strings to lists of values in optical channels. 
    
        These classes deal with managing the relation between component names as strings (e.g., '0001') and the values (photon numbers)
        in the optical channels (e.g., (0,0,0,1)). 
        
        1) Classes manage the option for 'reversed notation', meaning the situation where the user want the value for the 
        channel 0 to be at the right of the name string (with FockStateCircuit.channel_0_left_in_state_name set to False
        leading to '0001' = (1,0,0,0)) or the default situation with channel 0 at the left side of the name string (with 
        FockStateCircuit.channel_0_left_in_state_name set to True leading to '0001' = (0,0,0,1))
        
        2) Classes manage the option for 'double digit photon numbers', meaning the situation where the we can have 10 or more photons
        in a channel. In that case optical values (1,0,0,0) correspond to state name '01000000' (with .channel_0_left_in_state_name == True ) or
        '00000001' (with .channel_0_left_in_state_name == False ). NOTE: We can reverse the order in which the optical values
        are captured in the name string, but we do not reverse the notation of a specific value. So optical value 10 will always be string '10' and 
        optical value 1 will be string '01', irrespective of whether we reverse the state notation.

                                Examples:
                                length_of_fock_state = 5 (i.e., single digit optical value notation)
                                with channel_0_left_in_state_name == True: 
                                * (0,0,1) <--> '001'
                                * (2,3,2) <--> '232'
                                * (1,5,1) <--> not allowed, with length_of_fock_state = 5 we only allow photon numbers 0,1,2,3 or 4

                                length_of_fock_state = 15 (i.e., double digit optical value notation)
                                with channel_0_left_in_state_name == True: 
                                * (0,0,1) <--> '000001'
                                * (2,3,4) <--> '020304'
                                * (1,4,1) <--> '010401'
                                * (10,11,10) <--> '101110'

                                length_of_fock_state = 15 (i.e., double digit optical value notation)
                                with channel_0_left_in_state_name == False: 
                                * (0,0,1) <--> '010000'
                                * (2,3,4) <--> '040302'
                                * (1,4,1) <--> '010401'
                                * (10,11,10) <--> '101110'

                                length_of_fock_state = 115 (i.e., triple digit optical value notation)
                                with channel_0_left_in_state_name == True: 
                                * (0,0,1) <--> '000000001'
                                * (2,3,4) <--> '002003004'
                                * (1,4,1) <--> '001004001'
                                * (10,11,10) <--> '010011010'

                                length_of_fock_state = 115 (i.e., triple digit optical value notation)
                                with channel_0_left_in_state_name == False: 
                                * (0,0,1) <--> '001000000'
                                * (2,3,4) <--> '004003002'
                                * (1,4,1) <--> '001004001'
                                * (10,11,10) <--> '010011010'

        3) Classes can serve as iterator just like a python dictionary:
                                Examples:
                                * [optical_value for optical_value in circuit._dict_of_optical_values.keys()]
                                * [component_name for component_name in circuit._dict_of_valid_component_names.keys()]
                                * [name,value for name,values in circuit._dict_of_valid_component_names.items()]

        4) We can check existence/validity of an optical state with the - in - keyword
                                Examples:
                                For these settigs:
                                    comp_names = ComponentNames(length_of_fock_state=115,
                                                                no_of_optical_channels=3,
                                                                channel_0_left_in_state_name=False)
                                    optical_values = OpticalValues(comp_names)
                                All these statement are True
                                    '001000000' in comp_names
                                    [2,3,4] in optical_values
                                    (2,2,2) in optical_values # accepts values as lists or as tuples
                                    '201000000' not in comp_names 
                                    (2,3,4,6) not in optical_values

        5) We can get the index of a state in the full basis
                                Examples:
                                For these settings:
                                    comp_names = ComponentNames(length_of_fock_state=56,
                                                                no_of_optical_channels=3,
                                                                channel_0_left_in_state_name=False)
                                    optical_values = OpticalValues(comp_names)
                                All these statement are True
                                    optical_values.index_of([1,0,0]) == 1 # note that we use reversed state notation, not the default !
                                    optical_values.index_of((0,0,1)) == 56**2 # note that we use reversed state notation, not the default !
                                    comp_names.index_of('000010') == 10 # note that we use reversed state notation, not the default !

                                -- or --
                                For these settings:
                                    length_of_fock_state = 56
                                    channel_0_left_in_state_name = True
                                    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
                                    optical_values = OpticalValues(comp_names)
                                All these statements are True
                                    optical_values.index_of((0,0,1)) == 56**2
                                    optical_values.index_of((1,0,0)) == 1
                                    comp_names.index_of('000001') == 56**2
                                    comp_names.index_of('010000') == 1


        Last modified: June 7th, 2024       
    """
    def __init__(self, component_names: str) -> None:
        """ Constructor for class OpticalValues"""
        self.component_names = component_names
        self.length_of_fock_state = component_names.length_of_fock_state
        self.no_of_optical_channels = component_names.no_of_optical_channels
        self.channel_0_left_in_state_name = component_names.channel_0_left_in_state_name
        self.dict_of_optical_values, self.dict_of_valid_component_names =  component_names.dict_of_optical_values, component_names.dict_of_valid_component_names
        self.digits_per_channel = component_names.digits_per_channel
        self.string_format_in_state_as_word = component_names.string_format_in_state_as_word
        self.tuple_of_strings = component_names.tuple_of_strings
        self.size_of_basis = component_names.size_of_basis
        return
    
    def __getitem__(self,optical_values: Union[list, tuple]) -> str:
        optical_values = tuple(optical_values)
        if optical_values in self.dict_of_optical_values:
            return self.dict_of_optical_values[optical_values]
        else:         
            if self.channel_0_left_in_state_name:              
                component_name = ''.join([self.tuple_of_strings[number] for number in optical_values])                                                                 
            else:
                component_name = ''.join([self.tuple_of_strings[number] for number in reversed(optical_values)])   
            self.dict_of_optical_values.update({optical_values: component_name})
            self.dict_of_valid_component_names.update({component_name: optical_values})
            return component_name
    
    def __iter__(self):
        if len(self.dict_of_optical_values) == self.size_of_basis:
            return iter(self.dict_of_optical_values)
        self.dict_of_optical_values = {tuple(v):k for k,v in self.component_names.items()}      
        return iter(self.dict_of_optical_values)
    
    def __len__(self) -> int:
        return self.size_of_basis
    
    def __contains__(self, values: Union[tuple, list]) -> bool:
        return (isinstance(values, tuple) or isinstance(values, list)) and len(values) == self.no_of_optical_channels and min(values) >= 0 and max(values) < self.length_of_fock_state

    def keys(self):
        self.__iter__()
        return self.dict_of_optical_values.keys()

    def items(self):
        self.__iter__()
        return self.dict_of_optical_values.items()

    def values(self):
        self.__iter__()
        return self.dict_of_optical_values.values()    
    
    def index_of(self,values: Union[list,tuple]) -> int:
        return sum([value*(self.length_of_fock_state**index) for index,value in enumerate(values)])
    
    def at_index(self,index: int) -> str:
        """ Return the component name (as list) for the index. The index is in
            the basis which contains all valid optical states/photon number states in standard order.
        """
        return self.component_names._values_from_index(index) 
   
    
    def reorder(self,values: Union[tuple, list], tensor_list: list) -> int:
        """ Return the index for a list of values that is 're-ordered' with the tensor_list
        
            Examples:
                Use tensor_list that does nothing (keeps same order):
                    OpticalValues.reorder(values,[2,1,0]) == OpticalValues.index_of(values)
                Use tensor_list that reverses the order:
                    OpticalValues.reorder(values,[2,1,0]) == OpticalValues.index_of(reversed(values))
       
        """
        return self.index_of([values[index] for index in tensor_list])