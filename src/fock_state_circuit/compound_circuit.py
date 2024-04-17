from fock_state_circuit.circuit import FockStateCircuit
from fock_state_circuit.collection_of_states import CollectionOfStates
   
class CompoundFockStateCircuit:
    """ Class for compound FockStateCircuits. The class is used to work with a list of circuits that have to be executed sequentially. 
        When initializing the instance or calling the instance with 'refresh()' an internal list 
        is created where 'bridges' are added between the circuits. This enables evaluation of the compound circuit with a "collection of states", 
        or to create a schematics of the compound circuit by calling the method 'draw()'.

        When a change is made to the 'list_of_circuits' attribute it is needed to call 'refresh()' before calling 'draw()' or 'evaluate_circuit()'.
        
        Attributes:
            self.list_of_circuits
            self.compound_circuit_name

        Methods:
            refresh(self
                    ) -> None:

                    Update the internal list of circuits connected with bridges. This method has to be called after making a change to 'list_of_circuits' before
                    running the circuit with 'evaluate_circuit()' or draw a circuit schematics with 'draw()

            clear(self
                    ) -> None:

                    Clears the internal list of circuits to release memory

            draw(   self, 
                    print_defaults: bool = False, 
                    settings_for_drawing_circuit: dict = None
                    ) -> None:

                    Draw the compound circuit. self. settings_for_drawing_circuit is the dict with settings for circuit drawing. 
                    Method is a shell which calls FockStateCircuit.draw() to execute the schematics.

            evaluate_circuit(self, 
                    collection_of_states_input: CollectionOfStates = None
                    ) -> CollectionOfStates:
                    
                    Evaluate the compount circuit for a given collection of input states.
                    Method is a shell which calls FockStateCircuit.evaluate_circuit() to evaluate the compound circuit for the given
                    input collection of states.

        Last modified: April 16th, 2024

    """
    _VERSION = '1.0.0'

    def __init__(self, 
                 list_of_circuits: list = [],
                 compound_circuit_name: str = None
                ):
        ''' Constructor for an instance of the class CompoundFockStateCircuit. The instance will be created the list_of_circuits passed as argument. 
            As part of the initialization the method 'refresh()' is called to create an internal list where the circuits are connected via 'bridges'.
        
        Args:
            list_of_circuits (list[FockStateCircuit], optional): Defaults to []
            compound_circuit_name (str, optional): Defaults to '' 
        '''

        self.list_of_circuits = list_of_circuits
        if compound_circuit_name is None:
            self.compound_circuit_name = FockStateCircuit._CIRCUIT_DRAW_DEFAULT_SETTINGS.get('compound_circuit_title','Optical circuit')
        else:
            self.compound_circuit_name = compound_circuit_name
        self.refresh()

    def __str__(self) -> str:
        text = "Compound FockStateCircuit: "
        text += self.compound_circuit_name + '\n'
        for index,circuit in enumerate(self.list_of_circuits):
            text += str(index) + "." + '\n'
            text += str(circuit)
        return text
    
    def refresh(self) -> None:
        ''' Update the internal list of circuits connected with bridges. This method has to be called after making a change to 'list_of_circuits' before
            running the circuit with 'evaluate_circuit()' or draw a circuit schematics with 'draw()
        '''
        self._list_of_circuits_with_bridges = [0]*len(self.list_of_circuits)
        for index, circuit in enumerate(self.list_of_circuits):
            length_of_fock_state = circuit._length_of_fock_state
            no_of_optical_channels= circuit._no_of_optical_channels
            no_of_classical_channels = circuit._no_of_classical_channels
            channel_0_left_in_state_name= circuit._channel_0_left_in_state_name
            threshold_probability_for_setting_to_zeros = circuit._threshold_probability_for_setting_to_zero
            use_full_fock_matrix = circuit._use_full_fock_matrix
            circuit_name = circuit._circuit_name
            self._list_of_circuits_with_bridges[index] = FockStateCircuit(length_of_fock_state=length_of_fock_state,
                                                     no_of_optical_channels=no_of_optical_channels,
                                                     no_of_classical_channels=no_of_classical_channels,
                                                     channel_0_left_in_state_name=channel_0_left_in_state_name,
                                                     threshold_probability_for_setting_to_zero=threshold_probability_for_setting_to_zeros,
                                                     use_full_fock_matrix=use_full_fock_matrix,
                                                     circuit_name=circuit_name)
            self._list_of_circuits_with_bridges[index].node_list = list(circuit.node_list)
        for index, circuit in enumerate(self.list_of_circuits):
            if not index == len(self.list_of_circuits)-1:
                self._list_of_circuits_with_bridges[index].bridge(next_fock_state_circuit=self._list_of_circuits_with_bridges[index+1])
        return

    def clear(self) -> None:
        ''' Clears the internal list of circuits to release memory
        '''
        for circuit in self._list_of_circuits_with_bridges:
            del circuit
        del self._list_of_circuits_with_bridges
        return

    def draw(   self, 
                print_defaults: bool = False, 
                settings_for_drawing_circuit: dict = None) -> None:
        ''' Draw the compound circuit. self. settings_for_drawing_circuit is the dict with settings for circuit drawing. 
            Method is a shell which calls FockStateCircuit.draw() to execute the schematics.

            If changes to the compound circuit are made call CompoundFockStateCircuit.refresh() before calling this method.
        '''
        if settings_for_drawing_circuit is None:
            settings_for_drawing_circuit = dict({'compound_circuit_title' : self.compound_circuit_name})
        self._list_of_circuits_with_bridges[0].draw(print_defaults=print_defaults, settings_for_drawing_circuit=settings_for_drawing_circuit)
        return
    
    def draw_station(   self, 
                stations: dict = None,
                settings_for_drawing_circuit: dict = None) -> None:
        ''' Draw the compound circuit for a single station. 
            self. settings_for_drawing_circuit is the dict with settings for circuit drawing. 
            
            Method is a shell which calls FockStateCircuit.draw_station() to execute the schematics.

            If changes to the compound circuit are made call CompoundFockStateCircuit.refresh() before calling this method.
        '''
        if settings_for_drawing_circuit is None:
            settings_for_drawing_circuit = dict({'compound_circuit_title' : self.compound_circuit_name})
        self._list_of_circuits_with_bridges[0].draw_station(stations = stations, settings_for_drawing_circuit = settings_for_drawing_circuit)

        return

    def evaluate_circuit(self, 
                        collection_of_states_input: CollectionOfStates = None) -> CollectionOfStates:
        ''' Evaluate the compount circuit for a given collection of input states.
            Method is a shell which calls FockStateCircuit.evaluate_circuit() to evaluate the compound circuit for the given
            input collection of states.

            If changes to the compound circuit are made call CompoundFockStateCircuit.refresh() before calling this method.
        '''
        return self._list_of_circuits_with_bridges[0].evaluate_circuit(collection_of_states_input=collection_of_states_input)
        