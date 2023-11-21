import random
import string
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.colors
import fock_state_circuit as fsc

_VERSION = '0.0.9'

class State():
    """ Class for states in circuit from class FockStateCircuit which can be combined to 'collection of states'. 

        Attributes:
            initial_state (str) : 
                Typically the state from which the current state has evolved in the circuit, but user can customize for other purposes. Also 
                used to group together states in a statistical mixture.

            cumulative_probability (float) :
                Probability to evolve from initial state to current state (as consequence of measurement or decoherence). Alternatively used
                to give the weight of this state in a statistical mixture of states indicated with the same initial_state

            optical_components (dict[str,dict]) : 
                Optical components of the state described as number states per channel (i.e., '1001' can mean one photon in channel 0 and channel 3). 
                Each component has an amplitude (complex number type np.cdouble) and a probability (float). The probability is always the square
                of the absolute value of the amplitude (so effectively redundant information). The format for optical_components is for example:
                { '1011': {'amplitude': 0.71 + 0j, 'probability': 0.5}, '1110': {'amplitude': 0.71 + 0j, 'probability': 0.5}}

            classical_channel_values (list[float]) :
                A list holding the values for classical channels in the fock state circuit.

            measurement_results (list[dict]) :
                Measurement_results holds the outcomes of all measurements. New measurements should appended to the end of the list.
                At each measurement the classical channel values after the measurement are store together with the probability to get 
                that measurement result. The format for measurement results is for example:
                [{'measurement_results': [1, 3, 3.14], 'probability': 0.5}, {'measurement_results': [0, 0, 3.14], 'probability': 0.25}]
        
        Methods:

            __bool__(self): bool(state) returns True or False depending on whether state is valid

            __eq__(self, number: int) : state == number returns True is the photon number in the optical component with most photons is 
            equal to the argument number
            
            __getitem__(self, key) : attributes can be accessed trough state['initial_state'], state['optical_components'] etc
            
            __int__(self): int(state) returns the photon number in the optical component with most photons

            __str__(self): Return the string for 'pretty printing' the state. Used as print(state)

            copy(self): state.copy() returns a deep copy of a state with same parameters (i.e., suitable for same fock state circuit)
    
            create_initialized_state(self): state.create_initialized_state() returns a new state with valid default values for the same 
                circuit paramaters as the current state. The new state will have 'initial_state' equal to '000' (with the right amount of digits) 
                and the classical channels will be [0,0] (again with right amount of digits). Measurememt history is empty. 
        
            initialize_this_state(self): state.initialize_this_state() initializes the current state to valid default values. 
                The initial state will be '000' (with the right amount of digits) and the classical channels will be [0,0] 
                (again with right amount of digits). Measurememt history is empty.

            photon_number(self): state.photon_number() returns the the photon number in the optical component with most photons and a 
                boolean stating whether all components have the same photon number. If the bool is True all components have the same number of 
                photons in the optical channels, otherwise the number returned by the function is the highest over all optical components
        
            set_state_components_from_vector(self, state_vector: list) state.set_state_components_from_vector(vector) sets the optical components
                for the state from the vector which is given as argument. The function assumes the basis is the same as used by function 
                'translate_state_components_to_vector'. For that function the basis is generated as output.

            translate_state_components_to_vector(self) : state.translate_state_components_to_vector() returns a tuple with a first element 
                a vector and as second element the basis. The vector represents the state in the given basis

    """
    
    _KEYS_IN_STATE = ['initial_state', 'cumulative_probability', 'optical_components', 'measurement_results', 'classical_channel_values']

    def __init__(self, collection_of_states : 'CollectionOfStates', input_state_as_a_dict: dict = None):
        """ Constructor for in instance of class "State". The state instance will be created in initialized form.

        Args:
            collection_of_states ('CollectionOfStates'): collection of states for which this state is valid. The state instance takes information
                on the number of optical channels, number of classical channels, maximum photon number in the fock state from the collection_of_states.

            input_state_as_a_dict (dict, optional): Content to populate the state. Dicationary needs to be in right format. Defaults to None.

        Raises:
            Exception: If paramaters for the state do not result in a valid state.
      
        """     
        self._collection_of_states = collection_of_states

        # The length of the fock state is the number of possible photon numbers. So if the 'length' is 2 the maximum
        # number of photons is 1 (either 0 or 1 photons). If the length is 4 we can have 0,1,2 of 3 photons per channel.
        self._length_of_fock_state = collection_of_states._length_of_fock_state

        # the number of channels defining the circuit.
        self._no_of_optical_channels = collection_of_states._no_of_optical_channels
        self._no_of_classical_channels = collection_of_states._no_of_classical_channels        

        # for naming the states we need a convention. if 'channel_0_left_in_state_name' is set to 'True' we
        # write a state with 2 photons in channel 0 and 5 photons in channel 1 as '05'. With this value set
        # to 'False' we would write this same state as '50'. e:
        self._channel_0_left_in_state_name = collection_of_states._channel_0_left_in_state_name
        
        # '_digits_per_optical_channel' defines the number of digits used when 
        # writing a fock state as word. For more than 10 <= photons <100 per channel we 
        # need 2 digits per channel. For 100 or more need 3 digits.
        self._digits_per_optical_channel = len(str(self._length_of_fock_state-1))
        
        # create a dict with as keys the valid state names as strings and the photon number as value
        self._dict_of_valid_component_names = collection_of_states._dict_of_valid_component_names
        self._string_format_in_state_as_word = "{:0"+str(self._digits_per_optical_channel)+ "d}" #obsolete?

        # probabilities below this threshold are set to zero. 
        # this means that if the probability to find an outcome is lower than this threshold the outcome is discarded.
        self._threshold_probability_for_setting_to_zero = collection_of_states._threshold_probability_for_setting_to_zero

        # parameter use for __str__. With print_only_last_measurement set to True only last measurement is printed. With False the
        # complete history is printed.
        self._print_only_last_measurement = collection_of_states.print_only_last_measurement
        
        # initialize the state as an empty state with default values or from the data in the 'input_state_as_a_dict' parameter
        self.initialize_this_state()

        # if a dictionary with values is given as argument use those values to populate the state
        if input_state_as_a_dict is not None and isinstance(input_state_as_a_dict, dict):
            for k,v in input_state_as_a_dict.items():
                if k in self._KEYS_IN_STATE:
                    self.state[k] = v     
            # check if state is valid, otherwise raise an exception
            if not True in (self._check_valid_state(state_to_check = self)):
                print(self._check_valid_state(state_to_check = self))
                raise Exception('invalid state')
            
    @property
    def initial_state(self):
        return self.state['initial_state']
       
    @initial_state.setter
    def initial_state(self, name: str):
        self.state['initial_state'] = name   

    @property
    def cumulative_probability(self):
        return self.state['cumulative_probability']
       
    @cumulative_probability.setter
    def cumulative_probability(self, probability: float):
        if not 0 <= probability <= 1:
            raise Exception('Invalid value for probability')
        self.state['cumulative_probability'] = probability  

    @property
    def optical_components(self):
        return self.state['optical_components']
       
    @optical_components.setter
    def optical_components(self, components: dict):
        if isinstance(components, dict):
            for k,v in components.items():
                if not('amplitude' in v and 'probability' in v) or not k in self._dict_of_valid_component_names:
                    raise Exception('Invalid dictionary for optical_components')
        elif isinstance(components, list):
            optical_components = dict([])
            for name, amplitude in components:
                if not name in self._dict_of_valid_component_names:
                    raise Exception('Invalid dictionary for optical_components')
                else:
                    try:
                        probability = np.abs(amplitude)**2
                    except:
                        raise Exception('Invalid dictionary for optical_components')
                    optical_components.update({name:{'amplitude':amplitude,'probability': probability}}) 
            components = optical_components           
        else:
            raise Exception('Invalid type for optical_components. Requires a dictionary')
        self.state['optical_components'] = components  

    @property
    def classical_channel_values(self):
        return self.state['classical_channel_values']
       
    @classical_channel_values.setter
    def classical_channel_values(self, classical_values: list):
        if isinstance(classical_values, list):
            if not len(classical_values) == self._no_of_classical_channels:
                    raise Exception('Invalid list for classical values')
        else:
            raise Exception('Invalid type for classical values. Requires a list')
        self.state['classical_channel_values'] = classical_values   

    @property
    def measurement_results(self):
        return self.state['measurement_results']
       
    @measurement_results.setter
    def measurement_results(self, measurement_results : list):
        if not isinstance(measurement_results, list):
            raise Exception('Invalid type for measurement results. Requires a list of dicts with keys \'measurement_results\' and \'probability\'')
        self.state['measurement_results'] = measurement_results   

    def __getitem__(self,key):
        return getattr(self,key)

    def __setitem__(self,key,value):
        setattr(self,key,value)

    def __str__(self) -> str:
        """ Return the string for 'pretty printing' the state.

        Returns:
            str: string describing the state
        """        
        text = 'Initial state: {!r}\n'.format(self.state.get('initial_state', "Warning: initial state not available for this state"))
        text += 'Cumulative probability: {:.02f}\n'.format(self.state.get('cumulative_probability'))
        classical_channel_values = self.state.get('classical_channel_values')
        if len(classical_channel_values) > 0:
            text += 'Classical values: ' + str(["{n:.2f}".format(n=val) for val in classical_channel_values]) + '\n'
        measurement_results = self.state.get('measurement_results', None)
        if measurement_results is not None and len(measurement_results) > 0:
            if self._print_only_last_measurement:
                text += 'Last measurement result:  \n'
                result = measurement_results[-1]
                if 'measurement_results' in result.keys() and 'probability' in result.keys():
                    text += '\tMeasurement results: ' + str(["{n:.2f}".format(n=val) for val in result['measurement_results']])
                    text +=  ", Probability: " + "{val:.2f}".format(val = result['probability'])  + "\n"
                else:
                    text += str(result)+ "\n"
            else:
                text += 'Measurement results (last result first):  \n'
                measurement_results = measurement_results[::-1]
                for result in measurement_results:
                    if 'measurement_results' in result.keys() and 'probability' in result.keys():
                        text += '\tValue: ' +str(["{n:.2f}".format(n=val) for val in result['measurement_results']]) 
                        text +=  ", Probability: " + "{val:.2f}".format(val = result['probability'] ) + "\n"
                    else:
                        text += str(result)+ "\n"
        text += 'Optical components: \n'
        for k,v in self.state['optical_components'].items():
            amp = v['amplitude']
            if amp.imag > 0:
                amp_as_word = '({c.real:.2f} + {c.imag:.2f}i)'.format(c=amp)
            else:
                amp_as_word = '({c.real:.2f} - {c.imag:.2f}i)'.format(c=np.conj(amp))
            prob = v['probability']
            word = '{!r}'.format(k)
            text += "\tComponent: " + word + ' Amplitude: ' + amp_as_word + ', Probability: ' + "{val:.2f}".format(val = prob) + '\n'
        return text
    
    def __bool__(self) -> bool:
        """ Returns True or False depending on whether state is valid

        Returns:
            bool:bool indicating whether state is valid
        """        
        if True in self._check_valid_state(state_to_check = self):
            return True
        else:
            return False
    
    def __eq__(self, number: int) -> bool:
        """ Returns True is the photon number in the optical component with most photons is equal to the argument number"""
        return number == self.__int__()
    
    def __int__(self) -> int:
        """ Returns the photon number in the optical component with most photons"""
        return int(self.photon_number()[0])

    def photon_number(self) -> tuple[int,bool]:
        """ Returns the the photon number in the optical component with most photons and a boolean stating whether all components have
            the same photon number. If the bool is True all components have the same number of photons in the optical channels, otherwise
            the number returned by the function is the highest over all optical components
            
        Returns:
            tuple: First element in the tuple is the photon number, the second is True if all components have the same photon number.
        """
        list_photon_numbers = []
        for component in self.state['optical_components'].keys():
            list_photon_numbers.append(sum(self._dict_of_valid_component_names[component]))
        if max(list_photon_numbers) == min(list_photon_numbers):
            return (max(list_photon_numbers), True)
        else:
            return (max(list_photon_numbers), False)

    def _check_valid_state(self, state_to_check: "State") -> None:
        """ Function to check validity of the state given as input.

        Args:
            state ("State"): state to be checked for validity

        Returns:
            tuple: first element in tuple 'True' or 'False' (indicating validity of the state). Second element is a string with error message
        """        
        if not isinstance(state_to_check, State):
            return (False, 'Invalid state: input is not a \'state\'')
        for key in state_to_check.state.keys():
            if not key in self._KEYS_IN_STATE:
                return (False,'Invalid state: invalid key')
            try:
                initial_state = state_to_check.state['initial_state']
                cumulative_probability = state_to_check.state['cumulative_probability']
                optical_components = state_to_check.state['optical_components']
                classical_channel_values = state_to_check.state['classical_channel_values']
                measurement_results = state_to_check.state['measurement_results']
            except:
                return (False,'Invalid state: cannot read values')
        if type(initial_state) != type('name'):
            return (False,'Invalid state: invalid initial_state label')
        try:
            float(cumulative_probability)
        except:
            return (False,'Invalid state: invalid type for cumulative probability')
        if cumulative_probability < 0 or cumulative_probability > 1:
            return (False,'Invalid state: invalid value for cumulative probability')
        if type(optical_components) != type(dict([])):
            return (False,'Invalid state"optical components is not a dictionary')
        for o_component, o_values in optical_components.items():
            if type(o_component) != type('name') or type(o_values) != type(dict([])):
                return (False,'Invalid state: issue with optical component label')
            try:
                amplitude = np.cdouble(o_values['amplitude'])
                probability = float(o_values['probability'])
                if probability < -1*self._threshold_probability_for_setting_to_zero or probability > 1+self._threshold_probability_for_setting_to_zero:
                    return (False,'Invalid state: invalid probability of optical component')
            except:
                return (False,'Invalid state: issue with optical components')
        if not type(classical_channel_values) == type([0,0]) and type(measurement_results) == type([0,0]):
            return (False,'Invalid state: issue with classical channel values')
        if len(classical_channel_values) != self._no_of_classical_channels:
            return (False,'Invalid state: classical channel values does not match with number of classical channels')
        return (True, 'Valid state')

    def copy(self) -> "State":       
        """ Return a deep copy of a state with same parameters (i.e., suitable for same fock state circuit)

        Returns:
            State: Deep copy of the current state
        """        
        new_initial_state = "{:s}".format(self.state['initial_state'])

        new_cumulative_probability = self.state['cumulative_probability']
        
        new_optical_components = dict([])
        for oc_name, oc_comp in self.state['optical_components'].items():
            new_comp_name = "{:s}".format(oc_name)
            new_comp_values = dict([])
            for k,v in oc_comp.items():
                new_comp_values.update({k:v})
            new_optical_components.update({new_comp_name :  new_comp_values})
        
        new_classical_channel_values = self.state['classical_channel_values'].copy()
        
        new_measurement_results = []
        for m_result in self.state['measurement_results']:
            new_m_result = dict([])
            for k,v in m_result.items():
                new_m_result.update({k:v})
            new_measurement_results.append(new_m_result)
        
        new_state_as_a_dict = {   'initial_state' : new_initial_state,
                            'cumulative_probability' : new_cumulative_probability,
                            'optical_components' : new_optical_components, 
                            'classical_channel_values' : new_classical_channel_values,
                            'measurement_results' : new_measurement_results
                            }

        return State(self._collection_of_states, new_state_as_a_dict )

    def initialize_this_state(self) -> None:
        """ Initializes the current state to valid default values. The initial state will be '000' (with the right amount of digits) 
            and the classical channels will be [0,0] (again with right amount of digits). Measurememt history is empty.
        """        
        new_initial_state = list(self._dict_of_valid_component_names)[0]

        new_cumulative_probability = 1.0
        
        new_optical_components = {new_initial_state :  {'amplitude': np.cdouble(1), 'probability': 1.0}}
        
        new_classical_channel_values =[0]*self._no_of_classical_channels       
        new_measurement_results = []
        
        self.state = {  'initial_state' : new_initial_state,
                        'cumulative_probability' : new_cumulative_probability,
                        'optical_components' : new_optical_components, 
                        'classical_channel_values' : new_classical_channel_values,
                        'measurement_results' : new_measurement_results
                        }
        
        return 
    
    def create_initialized_state(self) -> "State":
        """ Returns a new state with valid default values for the same circuit paramaters as the current state.
            The new state will have 'initial_state' equal to '000' (with the right amount of digits) 
            and the classical channels will be [0,0] (again with right amount of digits). Measurememt history is empty. 
            
            The current state is unchanged, the return value is a new state

            Args: None

        Returns:
            State: new initialized state for same circuit as current state

        """        
        new_initial_state = list(self._dict_of_valid_component_names)[0]

        new_cumulative_probability = 1.0
        
        new_optical_components = {new_initial_state :  {'amplitude': np.cdouble(1), 'probability': 1.0}}
        
        new_classical_channel_values =[0]*self._no_of_classical_channels       
        new_measurement_results = []
        
        empty_state_as_a_dict = {  'initial_state' : new_initial_state,
                                    'cumulative_probability' : new_cumulative_probability,
                                    'optical_components' : new_optical_components, 
                                    'classical_channel_values' : new_classical_channel_values,
                                    'measurement_results' : new_measurement_results
                                    }

        return State(self._collection_of_states, empty_state_as_a_dict )
    
    def translate_state_components_to_vector(self) -> tuple:
        """ Return a tuple with a first element a vector and as second element the basis. The vector represents the state in the given basis

        Returns:
            tuple: Tuple contains two lists. First one is the vector and second one the basis
        """        

        input_optical_components = self.state['optical_components']
        list_of_states = list(self._dict_of_valid_component_names.keys())
        # create an input state vector (numpy array)
        state_vector = np.array([0+0j]*len(list_of_states), dtype=np.cdouble)
        for component in input_optical_components.keys():
            index_component = list_of_states.index(component)
            state_vector[index_component] = input_optical_components[component]['amplitude']

        return (state_vector, list_of_states)
    
    def set_state_components_from_vector(self, state_vector: list ) -> None: 
        """ Set the optical components for the state from the vector which is given as argument. The function assumes the basis is the same 
            as used by function 'translate_state_components_to_vector'. For that function the basis is generated as output.

        Args:
            state_vector (list): list of amplitudes for the given basis
        """        

        list_of_states = list(self._dict_of_valid_component_names.keys())
        optical_components = dict([])
        for index, value in enumerate(state_vector):
            # if value is below threshold set to zero
            if np.abs(value)**2 < self._threshold_probability_for_setting_to_zero:
                state_vector[index] = 0
            amplitude = state_vector[index]
            component = list_of_states[index]
            probability = np.abs(amplitude)**2
            if probability != 0:
                optical_components.update({component: {'amplitude': amplitude, 'probability': probability}})
        self.state['optical_components'] = optical_components

        return
    
    def _rescale_optical_components(self) -> None:
        """ Rescales optical components by removing the ones with (too) low probability and re-normalizing the remaining ones.
            The limit is set by 'threshold_probability_for_setting_to_zero'
        """  
        oc = self.state['optical_components']
        total_probability = 0
        for component, amplitude_probability in oc.items():
            if amplitude_probability['probability'] < self._threshold_probability_for_setting_to_zero:
                amplitude_probability['probability'] = 0
                amplitude_probability['amplitude'] = np.csingle(0)
            else:
                total_probability += np.abs(amplitude_probability['amplitude'])**2

        if np.abs(total_probability - 1.0) >= self._threshold_probability_for_setting_to_zero:
            scale_factor = 1/math.sqrt(total_probability)
            new_oc = dict([])
            for component, amplitude_probability in oc.items():
                new_amplitude = scale_factor * amplitude_probability['amplitude']
                new_probability = np.abs(new_amplitude)**2
                if new_probability >= self._threshold_probability_for_setting_to_zero:
                    new_oc.update({component : {'amplitude':new_amplitude ,'probability': new_probability}})
            self.state['optical_components'] = new_oc 

        return
    
    def _identical_optical_components(self, other_state) -> bool:
        """ Returns 'True' if optical components are the same, otherwise 'False'. 
            The limit is set by 'threshold_probability_for_setting_to_zero'

            Returns:
                bool: indicates whether optical_components of the states are equal
        """
        optical_components_equal = True
        oc = self.state['optical_components']
        other_oc = other_state['optical_components']
        for component, amplitude_probability in oc.items():
            if component not in other_oc.keys():
                optical_components_equal = False
                break
            elif np.abs(amplitude_probability['probability'] - other_oc[component]['probability']) >= self._threshold_probability_for_setting_to_zero:
                optical_components_equal = False
                break
            elif np.abs(amplitude_probability['amplitude'] - other_oc[component]['amplitude'])**2 >= self._threshold_probability_for_setting_to_zero:
                optical_components_equal = False
                break
        return optical_components_equal

class CollectionOfStates():
    """ CollectionOfStates is a class describing collections of states as input and output to a FockStateCircuit. An instance can be generate by
    by passing a FockStateCircuit from which the collection of states takes parameters like the number of optical channels and the number of classical
    channels in the circuit. 

    The collection_of_states is an iterable. We can run though all states with 'for state in collection_of_states'.

    States are stored in a dictionary. The key is an 'identifier'. Data structure (including the structure of the underlying class State) is given below.

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
    Arguments:
        No arguments

    Methods
        __bool__(self): Check whether the collection of states is a valid collection (i.e., if bool(collection_of_states): )

        __init__(self, fock_state_circuit, input_collection_as_a_dict): Constructor for an instance of the class collection_of_states

        __len__(self): Function to return the number of states in the collection. len(collection_of_states) returns the number of states.

        __str__(self): Function to print a collection of states in a digestible format.

        add_state(self, state: 'State', identifier=None): Add a new state to the circuit. If identifier is given that will be used, 
            otherwise random new identifier will be generated. Take care when using existing identifier, the function will overwrite 
            the previous state with that identifier.

        adjust_length_of_fock_state(self, new_length_of_fock_state: int = 0): Function adjusts the 'length of Fock state' for the collection.

        clean_up(initial_state: str = ''): Function cleans up the collection_of_states. All optical components with low probability are 
        removed and the total set of optical components is renormalized.
        
        clear(self): Function to completely clear the collection of states.

        collection_as_dict(self) : Function to return the content of the collection of states as a dictionary. The keys will be the identifiers.
            The values will be the states.
        
        collection_from_dict(self, collection_of_states_as_dictionary): Function to load the collection_of_states with values from the 
            dictionary given as argument. All content in the orginal collection of states will be removed and replaced.
        
        copy(self): Function to create a deep copy of the collection of states. Changing the copy will not affect the original state.
        
        delete_state(self, identifier: str = None, initial_state: str = None): Delete (remove) a state from the collection of states. 
            Either the identifier or the initial_state is used to identify to state to remove. If more states exist with the same initial_state 
            the first one will be removed. If identier or initial_states given as argument are not present in the collection the function will 
            do nothing and leave the collection as it was. 
        
        density_matrix(initial_state: str = '', decimal_places_for_trace: int = 2): Function to create density matrix as well as 
            trace of the density matrix and trace of density matrix squared.

        extend(extra_optical_channels: int = 0, extra_classical_channels: int = 0, statistical_distribution: list[int] = []): Function extends 
            the original collection of states with new optical and/or classical channels. The optical channels will be filled according to the parameter 
            'statistical_distribution'.
        
        filter_on_classical_channel(self, classical_channel_numbers: list, values_to_filter: list): Removes all states from the collection which do not 
            match the filter criterium. In this case the classical channels need to have the right values. If no state has the correct values in 
            the classical channels an empty collection will result.
        
        filter_on_identifier(self, identifier_to_filter: list): Removes all states from the collection which do not match the filter criterium. 
            In this case the state identifier needs to be present in the list list "identifiers to filter". If the identifier is not present in 
            the collection and empty collection will result.
        

        filter_on_initial_state(self, initial_state_to_filter: list): Removes all states from the collection which do not match the filter criterium. 
            In this case the state's initial_state needs to be present in the list "initial_state_to_filter". If no state in the collection 
            has any of the initial_states given as input an empty collection will result.
        
        generate_allowed_components_names_as_list_of_strings(self) : Function to return a list of all allowed names for optical components.

        get_state(self, identifier=None, initial_state=None) : Function returns a state from the collection. The collection will remain unchanged.

        initial_states_as_list(self): Return the initial states in the collection as a list. 
        
        initialize_to_default(self): Function initializes collection of states to default for the given fock state circuit. 

        plot(self, classical_channels=[], initial_states=[], info_for_bar_plot={}): Function to plot a bar graph for a circuit.
        
        reduce(optical_channels_to_keep: list[int] = [], classical_channels_to_keep: list[int] = []) : Reduces the number of optical and/or classical channels. 

        set_collection_as_statistical_mixture(self, list_of_tuples_state_weight: list): Sets the collection of states to a statistical mixture. All
            content in the collection is replaced

        state_identifiers_as_list(self): Returns a list of all state identifiers in the circuit. 

    """
   
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

        # generate a list of all possible values in the optical channels       
        index_list = [index for index in range(0,self._length_of_fock_state**self._no_of_optical_channels)]
        self._list_of_fock_states = [[] for index in index_list]
        for _ in range(0,self._no_of_optical_channels):
            for index in range(len(index_list)):
                n = int(index_list[index]%self._length_of_fock_state)
                self._list_of_fock_states[index].append(n)
                index_list[index] = int(index_list[index]/self._length_of_fock_state)
        
        # create a dict with as keys the valid state names as strings and the list of photon numbers as value
        self._dict_of_valid_component_names = dict([])
        for optical_state in  self._list_of_fock_states:
            name = self._get_state_name_from_list_of_photon_numbers(optical_state)
            self._dict_of_valid_component_names.update({name : optical_state})

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
                raise Exception('Invalid collection of states')
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

    def copy(self) -> 'CollectionOfStates':
        """ Function to create a deep copy of the collection of states. Changing the copy will not affect the original state.
            Note: copy function will create new collection for "self._fock_state_circuit". Ensure this is set to the right circuit.

        Returns:
            CollectionOfStates: new collection of states with identical values and parameters as the original
        """        
        new_selection_of_states = dict([])
        for identifier,state in self._collection_of_states.items():
            new_state_identifier = "{:s}".format(identifier)
            new_state = state.copy()
            new_selection_of_states.update({new_state_identifier : new_state})

        return CollectionOfStates(self._fock_state_circuit, new_selection_of_states)
              
    def plot(self, classical_channels = [], initial_states = [], info_for_bar_plot = dict([]), histo_output_instead_of_plot = False):
        """ Function to plot a bar graph for a circuit. The bars indicate the probability to go from an initial state
            to an outcome in the classical channels. The circuit has to include measurement to the classical channels. 
            Optionally the classical channels to use can be specified as well as a selection of initial states. The 

        Args:
            result : dictionary which results from running 'evaluate_circuit()'
            classical_channels : classical channels which are used in the bar plot. 
                                Typically these are channels to which the measurement 
                                results for the circuit have been written.
            initial_states: list of initial states to limit the plot in case 'result' 
                                contains more states than should be plotted. Default all
                                initial states in 'result' are used.
            info_for_bar_plot: optional information for the bar plot. info_for_bar_plot.get['title']
                                sets the title for the graph.
            histo_output_instead_of_plot: if this bool is set to True the function will return a dict 
                                as histogram instead of creating a plot
        """        
        # if no classical channels given use all classical channels
        if classical_channels == []:
            classical_channels = [channel_no for channel_no in range(self._no_of_classical_channels)]
        plt.rcParams['figure.figsize'] = [15,6]
        dict_for_plotting = dict([])
        dict_for_tracking_coincidences = dict([])
        # output_states = []
        # create a dictionary with the initial state as key and a list of outcomes as value. Outcomes are the values in classical channel
        # and the probability to get to that outcome from an initial state
        # dict_for_plotting is {initial_state1 : [{'output_state':'1101', 'probability': 0.5}, {'output_state':'3000', 'probability': 0.5}] }
        map_outcome_to_classical_value = dict([])
        for state in self._collection_of_states.values():
            if len(initial_states) > 0 and state.initial_state not in initial_states:
                continue
            if state.initial_state not in dict_for_plotting.keys():
                dict_for_plotting[state.initial_state] = []     

            probability = state.cumulative_probability
            # to generate a string for the outcome we use the same function that is used for optical channels. Note that every classical channel value will
            # be mapped on an integer. 
            outcome = self._get_state_name_from_list_of_photon_numbers([state.classical_channel_values[index] for index in classical_channels])
            map_outcome_to_classical_value.update({outcome: [state.classical_channel_values[index] for index in classical_channels]})

            # run through measurement_results in reverse order and look for the last entry of ''coincidence indicator'. If non found use_coincidences will remain 'false'
            use_coincidences = False
            for result in state.measurement_results[::-1]:
                if 'coincidence indicator' in result.keys():
                    use_coincidences = True
                    coincidence_indicator = str(result['coincidence indicator'])
                    # the label is  typically 'ahead' or 'behind' but can be extended for when more than 2 subsets have to be combined.
                    label = result['label']
                    # dict_for_tracking_coincidences will group together all coincides with the same coincidence indicator
                    # this means that the output of these stats have to be plotted together as if the detection would 
                    # integrate over these states without the quantum interference.
                    if coincidence_indicator  not in dict_for_tracking_coincidences.keys():
                        dict_for_tracking_coincidences[coincidence_indicator ] = [{'output_state': outcome, 'probability': probability, 'initial_state': state.initial_state, 'label' : label}]
                    else:
                        dict_for_tracking_coincidences[coincidence_indicator ].append({'output_state': outcome, 'probability': probability, 'initial_state': state.initial_state, 'label' : label})
                    break

            # if there is no 'coincidence indicator' no time delay has been applied, treat the state as one for plotting
            if not use_coincidences:
                for outcomes_for_this_initial_state in dict_for_plotting[state.initial_state]:
                    # if the outcome already exists add the probability
                    if outcome == outcomes_for_this_initial_state['output_state']:
                        outcomes_for_this_initial_state['probability'] += probability
                        break
                else:
                    # if the outcome does not exist create a new entry in the list
                    dict_for_plotting[state.initial_state].append({'output_state': outcome, 'probability': probability})

        for list_with_same_coincidences in dict_for_tracking_coincidences.values():
            for element in list_with_same_coincidences:   
                normalization = 0          
                # first run through all possible combinations of state with same initial state and different label to create 
                # a normalization parameter
                for element_other in list_with_same_coincidences:
                    if element['label'] == element_other['label'] or element['initial_state'] != element_other['initial_state'] :
                        continue
                    normalization += element_other['probability']

                for element_other in list_with_same_coincidences:
                    if element['label'] == element_other['label'] or element['initial_state'] != element_other['initial_state'] :
                        continue

                    values_this = map_outcome_to_classical_value[element['output_state']].copy()
                    values_other = map_outcome_to_classical_value[element_other['output_state']].copy()
                    new_values = [value_this + value_other for value_this, value_other in zip(values_this, values_other)]
                    new_outcome = self._get_state_name_from_list_of_photon_numbers(new_values)

                    new_probability = element['probability']*element_other['probability']/normalization

                    for outcomes_for_this_initial_state in dict_for_plotting[element['initial_state']]:
                        # if the outcome already exists add the probability
                        if new_outcome == outcomes_for_this_initial_state['output_state']:
                            outcomes_for_this_initial_state['probability'] += new_probability
                            break
                    else:
                        # if the outcome does not exist create a new entry in the list
                        dict_for_plotting[element['initial_state']].append({'output_state': new_outcome, 'probability': new_probability})

        # create a list of all possible outcomes across all initial states
        output_states = []
        for initial_state, list_of_outcomes in dict_for_plotting.items():
            for output_probability in list_of_outcomes:
                outcome = output_probability['output_state']
                if outcome not in output_states:
                    output_states.append(outcome)

        # for states who do not lead to a certain outcome add the outcome with probability zero
        # this enables easier plotting of graphs. If the outcome is absent it will need to be corrected
        # later, so better to add the outcome with probability zero
        for initial_state, list_of_outcomes in dict_for_plotting.items():
            outcomes_for_this_initial_state = []
            for output_probability in list_of_outcomes:
                outcomes_for_this_initial_state.append(output_probability['output_state'])
            for outcome in list(set(output_states).difference(outcomes_for_this_initial_state)):
                dict_for_plotting[initial_state].append({'output_state': outcome, 'probability': 0})

        if histo_output_instead_of_plot:
            return dict_for_plotting

        no_initial_states = len(dict_for_plotting)
        no_output_states = len(output_states)
        width = 0.8/no_output_states # spread the bars over 80% of the distance between ticks on x -axis
        mid = no_output_states//2
        # cycle through standard color list 
        cycle = info_for_bar_plot.get('colors', list(matplotlib.colors.TABLEAU_COLORS))
        greys = ['whitesmoke','whitesmoke']
      
        for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
            x = list(dict_for_plotting.keys()).index(initial_state)
            for i in range(no_output_states):
                plt.bar(x+(i-mid)*width, 
                        1.2,
                        color = greys[i%len(greys)],
                        width = width
                        )
        for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
            for outcome in outcomes_for_that_initial_state:
                x = list(dict_for_plotting.keys()).index(initial_state)
                i = output_states.index(outcome['output_state'])
                plt.bar(x+(i-mid)*width, 
                        outcome['probability'],
                        color = cycle[i%len(cycle)],
                        width = width
                        )
        custom_lines = [matplotlib.lines.Line2D([0], [0], color = cycle[i%len(cycle)], lw=4) for i in range(len(output_states))]
        plt.legend(custom_lines, [outcome for outcome in  output_states])
        plt.xticks(rotation=90)
        plt.xticks([x for x in range(no_initial_states)], list(dict_for_plotting.keys()))
        plt.ylabel(info_for_bar_plot.get('ylabel', 'Probability'))
        text = info_for_bar_plot.get('title', 'Probabilities for going from input state to measurement result')
        plt.title(text) 
        plt.show()
        return

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
    
    def _get_state_name_from_list_of_photon_numbers(self, state: list) -> str:
        """ For a list of photon numbers generate a string which can serve as name for the state or a component in the state.
            Example: state [0,1,3] would become '310' with 0 representing the photon number in channel 0. If we use reversed
            state notation state [0,1,3] would become '013' (reversed or regular state notation is set in initialization of the 
            FockStateCircuit). If we allow per channel photon numbers which require more digits (e.g., 10) the format of the string will be adjusted.
            Example [10,1,3] would become '100103'

        Args:
            state (List): state as a list of values with channel 'n' at index 'n'. e.g., [0,1,3]

        Returns:
            str: name of the state of component derived from photon number per channel. e.g., '013'
        """        
        string_format_in_state_as_word = "{:0"+str(len(str(self._length_of_fock_state-1)))+ "d}"

        if self._channel_0_left_in_state_name == True:
            name = ''.join([string_format_in_state_as_word.format(number) for number in state])              
        else: #self.state_least_significant_digit_left == False:
            name = ''.join([string_format_in_state_as_word.format(number) for number in state[::-1]]) 

        return name
    
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
            dm_mixed_state = np.zeros((len(self._list_of_fock_states), len(self._list_of_fock_states)), dtype=np.cdouble)
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
        if optical_channels_to_keep == []:
            optical_channels_to_keep  = [*range(self._no_of_optical_channels)]
        if classical_channels_to_keep ==  []:
            classical_channels_to_keep  = [*range(self._no_of_classical_channels)]

        if not all([channel in [*range(self._no_of_optical_channels)] for channel in optical_channels_to_keep]):
            raise Exception('error in channel selection in CollectionOfStates.reduce()')
        if not all([channel in [*range(self._no_of_classical_channels)] for channel in classical_channels_to_keep]):
            raise Exception('error in channel selection in CollectionOfStates.reduce()')
        
        # create empty collection_of_states with reduced number of channels
        updated_circuit = fsc.FockStateCircuit(length_of_fock_state = self._length_of_fock_state, 
                                no_of_optical_channels = len(optical_channels_to_keep), 
                                no_of_classical_channels= len(classical_channels_to_keep),
                                use_full_fock_matrix=self._fock_state_circuit._use_full_fock_matrix
                                )
        
        new_collection_of_states = CollectionOfStates(fock_state_circuit=updated_circuit)
        new_collection_of_states.clear()
        new_collection_of_states._channel_0_left_in_state_name = self._channel_0_left_in_state_name
        new_collection_of_states._threshold_probability_for_setting_to_zero = self._threshold_probability_for_setting_to_zero
        new_collection_of_states.print_only_last_measurement = self.print_only_last_measurement

        # create a lookup table from original component names to reduced component names. 
        # for each original name there will be multiple names in the resulting collection
        lookup_table_component_names = dict([])
        for name, list in self._dict_of_valid_component_names.items():
            new_list = [list[channel_number] for channel_number in optical_channels_to_keep]
            new_name = self._get_state_name_from_list_of_photon_numbers(new_list)
            traced_out_values = [list[number] for number in range(self._no_of_optical_channels) if number not in optical_channels_to_keep]
            traced_out_name = self._get_state_name_from_list_of_photon_numbers(traced_out_values)
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
                    'measurement_results' : old_state['measurement_results']
                    }
                new_state = State(collection_of_states=new_collection_of_states, input_state_as_a_dict=new_state_as_a_dict)
                new_collection_of_states.add_state(state=new_state,identifier=identifier+traced_out_value)

            self._collection_of_states = new_collection_of_states.collection_as_dict()
            self._no_of_classical_channels = new_collection_of_states._no_of_classical_channels
            self._no_of_optical_channels = new_collection_of_states._no_of_optical_channels
            self._list_of_fock_states = new_collection_of_states._list_of_fock_states
            self._dict_of_valid_component_names = new_collection_of_states._dict_of_valid_component_names

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
        updated_circuit = fsc.FockStateCircuit(length_of_fock_state = new_length_of_fock_state, 
                                no_of_optical_channels = self._no_of_optical_channels, 
                                no_of_classical_channels= self._no_of_classical_channels,
                                use_full_fock_matrix=self._fock_state_circuit._use_full_fock_matrix
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
                    new_name = new_collection_of_states._get_state_name_from_list_of_photon_numbers(new_values)
                new_optical_components.update({new_name:amp_prob})
            new_state_as_a_dict = { 'initial_state' : state.initial_state,
                                    'cumulative_probability' : state.cumulative_probability,
                                    'optical_components' : new_optical_components,
                                    'classical_channel_values' : state.classical_channel_values,
                                    'measurement_results' : state.measurement_results
                                    }
            new_state = State(collection_of_states=new_collection_of_states,input_state_as_a_dict=new_state_as_a_dict)
            new_collection_of_states.add_state(state = new_state, identifier=identifier)
 
        self._collection_of_states = new_collection_of_states.collection_as_dict()
        self._length_of_fock_state = new_collection_of_states._length_of_fock_state
        self._list_of_fock_states = new_collection_of_states._list_of_fock_states
        self._dict_of_valid_component_names = new_collection_of_states._dict_of_valid_component_names
    
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
        
        # create a dict with values for extra channels and their probabilities
        extra_channel_values = dict({'initial_value' : (1,[])})
        for _ in range(extra_optical_channels):
            new_extra_channel_values = dict([])
            for old_probability, values in extra_channel_values.values():
                for index, fock_state_value in enumerate(range(self._length_of_fock_state)):
                    new_probability = statistical_distribution[index]*old_probability
                    new_values = values + [fock_state_value]
                    new_label = self._get_state_name_from_list_of_photon_numbers(new_values)
                    new_extra_channel_values.update({new_label: (new_probability,new_values)})
            extra_channel_values = new_extra_channel_values

        # create a new (still empty) collection_of_states with the extra optical channels
        updated_circuit  = fsc.FockStateCircuit(length_of_fock_state = self._length_of_fock_state, 
                        no_of_optical_channels = self._no_of_optical_channels+extra_optical_channels,
                        no_of_classical_channels= self._no_of_classical_channels+extra_classical_channels,
                        use_full_fock_matrix=self._fock_state_circuit._use_full_fock_matrix
                        )
        new_collection_of_states = CollectionOfStates(fock_state_circuit= updated_circuit )
        new_collection_of_states.clear()
        new_collection_of_states._channel_0_left_in_state_name = self._channel_0_left_in_state_name
        new_collection_of_states._threshold_probability_for_setting_to_zero = self._threshold_probability_for_setting_to_zero
        new_collection_of_states.print_only_last_measurement = self.print_only_last_measurement

        # for each existing state create a statistical mixture for possible values in new channels
        for probability, values in extra_channel_values.values():
            for old_state in self._collection_of_states.values():
                optical_components = old_state['optical_components']
                new_optical_components = dict([])
                for component_name, amplitude_probability in optical_components.items():
                    list = self._dict_of_valid_component_names[component_name]
                    new_list = list + values
                    new_component_name = self._get_state_name_from_list_of_photon_numbers(new_list)
                    new_optical_components.update({new_component_name:amplitude_probability})
                new_state_as_a_dict = {   'initial_state' : old_state['initial_state'],
                    'cumulative_probability' :  old_state['cumulative_probability']*probability,
                    'optical_components' : new_optical_components, 
                    'classical_channel_values' : old_state['classical_channel_values'] + [0]*extra_classical_channels,
                    'measurement_results' : old_state['measurement_results']
                    }
                if new_state_as_a_dict['cumulative_probability'] >= self._threshold_probability_for_setting_to_zero:
                    new_state = State(collection_of_states=new_collection_of_states, input_state_as_a_dict=new_state_as_a_dict)
                    new_collection_of_states.add_state(state=new_state)
            

        self._collection_of_states = new_collection_of_states.collection_as_dict()
        self._no_of_classical_channels = new_collection_of_states._no_of_classical_channels
        self._no_of_optical_channels = new_collection_of_states._no_of_optical_channels
        self._list_of_fock_states = new_collection_of_states._list_of_fock_states
        self._dict_of_valid_component_names = new_collection_of_states._dict_of_valid_component_names
       
        return