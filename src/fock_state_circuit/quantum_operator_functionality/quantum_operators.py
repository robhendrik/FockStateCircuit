''' quantum_operators.py

This module contains the classes Hamiltonian and Quantumoperator to be used in a FockStateCircuit. The classes execute an operator
on the optical states in a FockStateCircuit where the operation is specified by its Hamiltonian.

Example: 
    For a beamsplitter between two channels the Hamiltonian is "T a+ b - T a b+". a+ and b+ represent the creation operator on channel a and be respectively,
    and a,b represent the annihilation operator. T is the 'coupling strength' between the channels. The unitary transformation is then exp(H).
    See for instance this discussion on StackExchange for more background 
    https://physics.stackexchange.com/questions/438723/is-the-beam-splitter-transformation-related-to-hamiltonian

The notation of the operators is as a tuple of a string and a float (i.e., ('+0-0', 0.71)). This tuple is accompanied by the operator_channels (a list of 
integers like [0,0,1,1]). If there are 4 characters in the string then there shoudl be 4 channels in the operator_channels. The characters in the string define 
whether it is the creation operator ('+'), annihilation operation ('-') or should be ignored ('0'). The operator_channels specify on which channel to apply the 
annihilation/creation operators. If a channel number repeats in the operator_channels it means multiple annihilation/creation operators are applied to the same 
channel.

We can combine operators as Hamiltonian or a QuantumOperator. The QuantumOperator is the unitary transformation that follows from the Hamiltonian as U = exp(H).
If two operators A and B commute (AB = BA) then we can add them as Hamiltonian (H = A+B). If they do not commute we have to create UA = exp(A) and UB=exp(B) to 
create U = UA x UB.

Classes:
    Hamiltonian:        This is a simple dataclass containing an operator as a list of tuples to describe the Hamiltonian. 
                        For a beamsplitter this could the operator could be [('+-', 0.71),('-+',0.71)] and the operator_channels [0,1]
                        For each tuple in the list the first character defines whether it is the creation operator ('+'), annihilation operation ('-') or 
                        should be ignored ('0'). The operator_channels specify on which channel to apply the annihilation/creation operators. If the operators
                        commute they can be combined in a list and will be added to come to the overall Hamiltonian.

    
    QuantumOperator:    This is the class containing the operator in the optical quantum state. It is initialized with list of Hamiltonians that will be combined 
                        to generate the transformation. Note that the order of these operators matters. Index 0 will be applied first.


Last modified: June 1st, 2024     
'''
from __future__ import annotations
import numpy as np
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Hamiltonian:
    """ This is a simple dataclass containing an operator as a list of tuples to describe the Hamiltonian. 
        For a beamsplitter this could the operator could be [('+-', 0.71),('-+',0.71)] and the operator_channels [0,1]
        For each tuple in the list the first character defines whether it is the creation operator ('+'), annihilation operation ('-') or 
        should be ignored ('0'). The operator_channels specify on which channel to apply the annihilation/creation operators. If the operators
        commute they can be combined in a list and will be added to come to the overall Hamiltonian.

        NOTE: operator channels are uses to specify whether the the operators work on the same channel or not. This is not the same channel number as
            we use in the Fock state for a completer circuit. If the Hamiltonian acts on two channels the operator_channels are [0,1]. if we call
            apply_operator_to_collection/apply_operator_to_state we can specify on which channels in teh circuit the operator acts through the argument 
            optical_channels.

            This approach allows to use a lower channel number to define the operator. If the Hamiltonian describes mixing of two channels in a 14-channel 
            circuit we only need matrices covering two channels to describe the interaction. If we work on the full optical state we would need a 14 channel
            matric making the operations a lot slower.

    Arguments:
        operators (list[tuple]): list of operators as a list of tuples in teh form [('+0-0',1), ]. A '+' represent a creation operator and a '-' an annihilation
                                    operation. Any other character (like '0' will be ignored).
        operator_channels (list[int]): list of channels describing whether there creation/annihilation operators act on the same or different channels.  If the 
                                    string in the operator has 4 characters (like '+-00') we need 4 channels in the operator_channels (like [0,0,1,1]).

    """
    _VERSION = '1.0.0'
    operators : list[tuple]
    operator_channels: list[int]

@dataclass(frozen=True)
class QuantumOperator:
    """ QuantumOperator is the class containing the operator in the optical quantum state. It is initialized with list of Hamiltonians that will be combined 
        to generate the transformation. Note that the order of these operators matters. Index 0 will be applied first.
    
    Arguments:
        hamiltonians (list[Hamiltonian]): List of Hamiltonians applied in the order from index 0 upwards
        length_of_fock_state (int): Maximum number of photons in a optical channel, can be taken from FockStateCircuit._length_of_fock_state, 
                        CollectionOfStates._length_of_fock_state or State._length_of_fock_state
        power_for_exponent_taylor_expansion (int): Maximum power in the Taylor expansion of the exponent when calculating exp(H). Default is 15.

    Methods:
        apply_operator_to_state(    state: State,
                                    optical_channels: list[int]
                                ) -> "State": # type: ignore
            Applies the operator to a state and returns the result.

        apply_operator_to_collection(   input_collection: CollectionOfStates, 
                                        optical_channels: list[int]
                                    ) -> CollectionOfStates: # type: ignore
            Applies the operator to a collection and returns the result.
         
    """
    _VERSION = '1.0.0'
    hamiltonians: list[Hamiltonian]
    length_of_fock_state: int
    power_for_exponent_taylor_expansion: int = 25

    def __post_init__(self) -> None:
        """ This is the initialization routine which runs after the initialization of the class QuantumOperator (which is a dataclass with is has its own initialization).
            We create the class as frozen, so as immutable. We want to run calculations at initialization to ensure we do the calculation only once. Because this is a 
            'frozen' dataclass we need to use settattr to be able to modify the arguments (so it looks more complex than it actually is)
        """
        object.__setattr__(self, 'operator_channels', self.hamiltonians[0].operator_channels)
        lim_basis_to_values, lim_basis_to_index = self._generate_limited_basis(self.operator_channels, self.length_of_fock_state)
        object.__setattr__(self, '_lim_basis_to_values', lim_basis_to_values)
        object.__setattr__(self, '_lim_basis_to_index', lim_basis_to_index)
        matrices = [self._generate_limited_matrix(hamiltonian.operators, self.operator_channels, self.length_of_fock_state) for hamiltonian in self.hamiltonians]
        object.__setattr__(self, '_matrices', matrices)
        object.__setattr__(self, '_transition_matrices_list', self._exponentiaze())
        object.__setattr__(self, '_transition_matrix', self._combine_operators())
       
        return

    def _generate_limited_basis(self, operator_channels: list[int], length_of_fock_state: int) -> tuple[dict,dict]:
        """ Generate lookup-tables for the limited basis. The limited basis covers all possible photon numbers for the channels relevant for the operator.

            Example:
                If length_of_fock_state is 3 we allow photon numbers 0,1,2. If wehave operator channels [0,0,1,1]. So we need a basis for two channels with up to 2 photons.
                The basis states will be ('00','10','20','01','11','21','02','12','22')

        Args:
            operator_channels (list[int]): list of channels describing whether there creation/annihilation operators act on the same or different channels.  If the 
                                    string in the operator has 4 characters (like '+-00') we need 4 channels in the operator_channels (like [0,0,1,1]).
            length_of_fock_state (int): Maximum number of photons in a optical channel, can be taken from FockStateCircuit._length_of_fock_state, 
                                CollectionOfStates._length_of_fock_state or State._length_of_fock_state

        Returns:
            tuple[dicts]: Tuple of two lookup-tables from index to state, and form state to index.
        """
        number_of_channels = len(set(operator_channels))
        lim_basis_to_values, lim_basis_to_index = dict([]), dict([])
        for number in range(length_of_fock_state** number_of_channels):
            d = number
            channel_values = []
            while d > 0:
                d,n = divmod(d,length_of_fock_state)
                channel_values.append(n)
            channel_values += [0] * ( number_of_channels - len(channel_values))
            lim_basis_to_values.update({number:tuple(channel_values)})
            lim_basis_to_index.update({tuple(channel_values):number})
        return lim_basis_to_values, lim_basis_to_index
    
    def _generate_limited_matrix(self, operators: list[Hamiltonian], operator_channels: list[int], length_of_fock_state: int) -> np.ndarray:
        """ Generate a matrix representing the effect of the operators on all states in the 'limited basis'.

        Args:
            hamiltonians (list[Hamiltonian]): List of Hamiltonians.operators applied in the order from index 0 upwards
            operator_channels (list[int]): list of channels describing whether there creation/annihilation operators act on the same or different channels.  If the 
                                    string in the operator has 4 characters (like '+-00') we need 4 channels in the operator_channels (like [0,0,1,1]).
            length_of_fock_state (int): Maximum number of photons in a optical channel, can be taken from FockStateCircuit._length_of_fock_state, 
                        CollectionOfStates._length_of_fock_state or State._length_of_fock_state

        Returns:
            ndarray: Matrix representing the effect of the operator.
        """
        number_of_channels = len(set(operator_channels))
        matrix_size = length_of_fock_state**number_of_channels
        matrix = np.zeros((matrix_size,matrix_size), np.csingle)
        for index, values in self._lim_basis_to_values.items():
            for operator in operators:
                new_values = list(values)
                coefficient = operator[1]
                for channel,command in zip(operator_channels, operator[0]):
                    if command == '+':
                        new_values[channel] += 1
                        coefficient *= np.sqrt(new_values[channel])
                    elif command == '-':
                        if new_values[channel] == 0:
                            break
                        else:
                            coefficient *= np.sqrt(new_values[channel])
                            new_values[channel] -= 1
                else:
                    if max(new_values) >= length_of_fock_state:
                        coefficient = 1
                        new_index = index
                    else:
                        new_index = self._lim_basis_to_index[tuple(new_values)]
                    matrix[index,new_index] += coefficient
        return matrix
    
    def _exponentiaze(self) -> list[np.ndarray]:
        """ Create the Taylor expansion of the exponent up to power 'power_for_exponent_taylor_expansion'. 
        """
        transition_matrices_list = []
        for matrix in self._matrices:
            transition_matrix = np.identity(n=matrix.shape[0], dtype=np.csingle)
            for n in range(1,self.power_for_exponent_taylor_expansion):
                transition_matrix = transition_matrix + (1/math.factorial(n))* np.linalg.matrix_power(matrix,n)
            transition_matrices_list.append(transition_matrix)
        return transition_matrices_list 
    
    def _combine_operators(self) -> np.ndarray:
        """ Create a combined operator for the list of operators. The operators are applied starting with index 0.
            So if the list_of_quantum_operators is [A,B,C] the the new operator will reflect CBA
        """
        transition_matrix = np.identity(n=self._transition_matrices_list[0].shape[0], dtype=np.csingle)
        for matrix in self._transition_matrices_list:
            transition_matrix = np.matmul(matrix,transition_matrix)
        return transition_matrix

    def apply_operator_to_collection(self,input_collection: CollectionOfStates, optical_channels: list[int]) -> CollectionOfStates: # type: ignore
        """ Applies the operator to a collection and returns the result.

        Args:
            state (State): input state on which to apply the operator.
            optical_channels (list[int]): Optical channels in the FockStateCircuit on which to apply the operator.

        Returns:
            CollectionOfState: Collection that results from applying the operator to each state in the input collection.
        """

        output_collection = input_collection.copy(empty_template = True)
        for state in input_collection:
            new_state = self.apply_operator_to_state(state, optical_channels)
            output_collection.add_state(state=new_state)
        return output_collection
    
    def apply_operator_to_state(self,state: State, optical_channels: list[int]) -> "State": # type: ignore
        """ Applies the operator to a state and returns the result.

        Args:
            state (State): input state on which to apply the operator
            optical_channels (list[int]): Optical channels in the FockStateCircuit on which to apply the operator

        Returns:
            State: State that results from applying the operator to the input state
        """
        new_state = state.copy()
        new_components = []
        for component, amp_prob in new_state.optical_components.items():
            values = state._dict_of_valid_component_names[component]       
            amplitude = amp_prob['amplitude']
            affected_values = [values[channel] for channel in optical_channels]
            input_index = self._lim_basis_to_index[tuple(affected_values)]
            for output_index, coeff in enumerate(self._transition_matrix[input_index,:]):
                if np.round(np.abs(coeff),4)**2 > state._threshold_probability_for_setting_to_zero:
                    output_values_in_limited_base = self._lim_basis_to_values[output_index]
                    full_output_values = values.copy()
                    for channel, value in zip(optical_channels, output_values_in_limited_base):
                        full_output_values[channel] = value
                    new_component = state._dict_of_optical_values[tuple(full_output_values)]
                    new_components.append((new_component, coeff*amplitude))
        new_state.optical_components = new_components
        return new_state