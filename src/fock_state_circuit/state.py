import math
import numpy as np
import copy
from fock_state_circuit.nodes.nodelist.node_list import NodeList

class State():
    """ Class for 'states' that can be used in circuits from class FockStateCircuit. The instances of State can be combined to 'collection of states'. 

        Attributes:
            initial_state (str) : 
                Identifier to track a state during its evolution in the circuit. Typically this identifier indicates the state from which the 
                current state has evolved in the circuit, but user can customize for other purposes. This identified is also used to group together 
                states in a statistical mixture. If we for instance start with a single state and in the circuit the state evolves in to a mixture of
                states then all states in that mixture will have the same 'initial_state'

            cumulative_probability (float) :
                Probability (or weight) of the state in a mixture of states. All states in the mixture typically have the same 'initial_state' value. 
                If we start with a single state with 'cumulative_probability' of 1.0 then the cumulative_probability of the current state indicates the
                probability to evolve from the original state to current state (as consequence of measurement or decoherence).

            optical_components (dict[str,dict]) : 
                Optical components of the state. This describes the (quantum) state of the photons. The optical state is described as the number of 
                photons optical per channel (i.e., '1001' can mean one photon in channel 0 and channel 3, and no photons in channel 1 and 2). If the 
                optical state is a pure photon-number-state (or Fock state) the optical components will consist of a single optical component. If the 
                state is superposition of photon-number-states then 'optical_components' will contain multiple components. Each component has an 
                amplitude (complex number type np.cdouble) and a probability (float). The probability is always the square
                of the absolute value of the amplitude (so effectively redundant information). The format for optical_components is for example:
                { '1011': {'amplitude': 0.71 + 0j, 'probability': 0.5}, '1110': {'amplitude': 0.71 + 0j, 'probability': 0.5}}

            classical_channel_values (list[float]) :
                A list holding the values for classical channels in the fock state circuit. These values can be floats, integeres, complex numbers, ...
                We can use the classical channels to store measurement results, but we can also use classical channels to set the properties of optical
                gates in the circuit (i.e., we can create a classically controlled wave plate where the orientation and phase delay are determined
                from the values in the classical channels).

            measurement_results (list[dict]) :
                Measurement_results holds the outcomes of all measurements. New measurements should appended to the end of the list.
                At each measurement the classical channel values after the measurement are store together with the probability to get 
                that measurement result. The format for measurement results is for example:
                [{'measurement_results': [1, 3, 3.14], 'probability': 0.5}, {'measurement_results': [0, 0, 3.14], 'probability': 0.25}]

            auxiliary_information (dict) :
                Information tracked in the state during execution of a circuit. This can relate to spectral information (bandwidth and timing),
                superquantum correlations or any other purpose. auxiliary information is absent when state is generated but will be preserved by copying
                a state.
        
        Methods:

            __bool__(self): bool(state) returns True or False depending on whether state is valid

            __eq__(self, number: int) : state == number returns True is the photon number in the optical component with most photons is 
            equal to the argument number
            
            __getitem__(self, key) : attributes can be accessed trough state['initial_state'], state['optical_components'] etc
            
            __int__(self): int(state) returns the photon number in the optical component with most photons

            __str__(self): Return the string for 'pretty printing' the state. Used as print(state)

            copy(self) -> "State"
            
                Returns a deep copy of a state with same parameters (i.e., suitable for same fock state circuit)
    
            create_initialized_state(self) -> "State"
            
                Returns a new state with valid default values for the same circuit paramaters as the current state. 
                The new state will have 'initial_state' equal to '000' (with the right amount of digits) 
                and the classical channels will be [0,0] (again with right amount of digits). Measurememt history is empty. 
        
            initialize_this_state(self)-> None
            
                Initializes the current state to valid default values. The initial state will be '000' (with the right amount of digits) 
                and the classical channels will be [0,0] (again with right amount of digits). Measurement history is empty. The state will not 
                contain any 'auxiliary_information'

            photon_number(self) -> tuple[int,bool]
            
                Returns the photon number in the optical component with most photons and a boolean stating whether all components 
                have the same photon number. If the bool is True all components have the same number of 
                photons in the optical channels, otherwise the number returned by the function is the highest over all optical components
        
            set_state_components_from_vector(self, state_vector: list) -> None
            
                Sets the optical components for the state from the vector which is given as argument. The function assumes the basis 
                is the same as used by function 'translate_state_components_to_vector'. For that function the basis is generated as output.

            translate_state_components_to_vector(self) -> tuple 
                returns a tuple with a first element a vector and as second element the basis. The vector represents the state in the 
                given basis

            inner(self, state) -> float:
                
                Returns the inner product between this state and the state passed as parameter. If states are orthogonal function returns 0, 
                if states are identical function returns 1. 

            outer(self, state) -> np.array
                
                Return the outer product between this state and the state passed as parameter. Function returns a numpy array.

            tensor(self, state) -> dict

                Generates optical components for an extended state as a tensor product of the current state and the state
                passed as parameter.

                The argument 'state' can be used in three ways:
                    - 'state' can be of type: 
                                new_components = state1.tensor(state = state2) with state2 of type fsc.State
                    - 'state' can be the optical components: 
                                new_components = state1.tensor(state = {'100':{'amplitude': 1.0,'probability': 1.0}})
                    - 'state' can be a list of tuples: 
                                new_components =  state1.tensor([('100',np.sqrt(1/2)),('110',1/2),('111',1/2)])

        Last modified: May 1st, 2024
    """
    _VERSION = '1.0.1'
    
    _KEYS_IN_STATE = ['initial_state', 'cumulative_probability', 'optical_components', 'measurement_results', 'classical_channel_values', 'auxiliary_information']

    def __init__(self, collection_of_states : any, input_state_as_a_dict: dict = None):
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

        # create a dict with as keys the valid state names as strings and the list of photon numbers as value
        self._dict_of_valid_component_names = collection_of_states._dict_of_valid_component_names

        # create a dict with as keys the tuple of the list of photon numbers as value, 
        # and valid state names as strings as value
        self._dict_of_optical_values = collection_of_states._dict_of_optical_values 

        # probabilities below this threshold are set to zero. 
        # this means that if the probability to find an outcome is lower than this threshold the outcome is discarded.
        self._threshold_probability_for_setting_to_zero = collection_of_states._threshold_probability_for_setting_to_zero

        # parameter use for __str__. With print_only_last_measurement set to True only last measurement is printed. With False the
        # complete history is printed.
        self._print_only_last_measurement = collection_of_states.print_only_last_measurement
        
        # initialize the state as an empty state with default values or from the data in the 'input_state_as_a_dict' parameter
        self.initialize_this_state()

        # if a dictionary with values is given as argument use those values to populate the state
        if input_state_as_a_dict:
            for k,v in input_state_as_a_dict.items():
                if k in self._KEYS_IN_STATE:
                    self.state[k] = v     
        #check if state is valid, otherwise raise an exception
        if not True in (self._check_valid_state(state_to_check = self)):
            print(self._check_valid_state(state_to_check = self))
            raise Exception('invalid state',self._check_valid_state(state_to_check = self), str(self) )
            
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
            print(probability)
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
                    if not name in optical_components.keys():
                        optical_components.update({name:{'amplitude':amplitude,'probability': probability}}) 
                    else:
                        new_amplitude = optical_components[name]['amplitude'] + amplitude
                        new_probability = np.abs(new_amplitude)**2
                        optical_components.update({name:{'amplitude':new_amplitude,'probability': new_probability}}) 
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

    @property
    def auxiliary_information(self):
        if not 'auxiliary_information' in self.state.keys():
            self.state['auxiliary_information'] = dict([])
        return self.state['auxiliary_information']
       
    @auxiliary_information.setter
    def auxiliary_information(self, auxiliary_information : dict):
        if not isinstance(auxiliary_information, dict):
            raise Exception('Invalid type for auxiliary information. Requires a dictionary')
        self.state['auxiliary_information'] = auxiliary_information 

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
                    text += 'Measurement results: ' + str(list(result.keys()))  + "\n"
            else:
                text += 'Measurement results (last result first):  \n'
                measurement_results = measurement_results[::-1]
                for result in measurement_results:
                    if 'measurement_results' in result.keys() and 'probability' in result.keys():
                        text += '\tValue: ' +str(["{n:.2f}".format(n=val) for val in result['measurement_results']])
                        text +=  ", Probability: " + "{val:.2f}".format(val = result['probability'] ) + "\n"
                    else:
                        text += 'Measurement results: ' + str(list(result.keys()))  + "\n"
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
        auxiliary_information = self.state.get('auxiliary_information', None)
        if auxiliary_information is not None and len(auxiliary_information) > 0:
            text += "Auxiliary information: "+ '\n' + '\t'
            for index,label in enumerate(auxiliary_information.keys()):
                text += label 
                if not index == len(auxiliary_information.keys()) -1:
                    text += ', '
                else:
                    text += '\n'
        return text
    
    def print_optical_components(self) -> str:
        """ Interim solution for pretty printing optical components. Future is to turn optical components in separate class"""
        text = ""
        for k,v in self.state['optical_components'].items():
            amp = v['amplitude']
            if amp.imag > 0:
                amp_as_word = '({c.real:.2f} + {c.imag:.2f}i)'.format(c=amp)
            else:
                amp_as_word = '({c.real:.2f} - {c.imag:.2f}i)'.format(c=np.conj(amp))
            word = '{!r}'.format(k)
            text += "Component: " + word + ', Amplitude: ' + amp_as_word
            text += ". "
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

            
        new_state_as_a_dict = {   'initial_state' : self.state['initial_state'],
                            'cumulative_probability' : self.state['cumulative_probability'],
                            'optical_components' : { name:amp_prob.copy() for name, amp_prob in self.state['optical_components'].items() }, 
                            'classical_channel_values' : self.state['classical_channel_values'].copy(),
                            'measurement_results' : [{k:v for k,v in m_result.items()} for m_result in self.state['measurement_results']]
                            }
        auxiliary_information = self.state.get('auxiliary_information', None)

        auxiliary_information = self.state.get('auxiliary_information', {})
        if auxiliary_information:
            new_state_as_a_dict.update({'auxiliary_information' : copy.deepcopy(self.state['auxiliary_information'])})

        return State(self._collection_of_states, new_state_as_a_dict )
    
        # This is an earlier implementation which looks slower but we measure very marginal performance differences.
        # new_initial_state = "{:s}".format(self.state['initial_state'])

        # new_cumulative_probability = self.state['cumulative_probability']
        
        # new_optical_components = dict([])
        # for oc_name, oc_comp in self.state['optical_components'].items():
        #     new_comp_name = "{:s}".format(oc_name)
        #     new_comp_values = dict([])
        #     for k,v in oc_comp.items():
        #         new_comp_values.update({k:v})
        #     new_optical_components.update({new_comp_name :  new_comp_values})
        
        # new_classical_channel_values = self.state['classical_channel_values'].copy()
        
        # new_measurement_results = []
        # for m_result in self.state['measurement_results']:
        #     new_m_result = dict([])
        #     for k,v in m_result.items():
        #         new_m_result.update({k:v})
        #     new_measurement_results.append(new_m_result)
        
        # new_state_as_a_dict = {   'initial_state' : new_initial_state,
        #                     'cumulative_probability' : new_cumulative_probability,
        #                     'optical_components' : new_optical_components, 
        #                     'classical_channel_values' : new_classical_channel_values,
        #                     'measurement_results' : new_measurement_results
        #                     }
        
        # auxiliary_information = self.state.get('auxiliary_information', None)
        # if auxiliary_information is not None and len(auxiliary_information) > 0:
        #     new_auxiliary_information = copy.deepcopy(self.state['auxiliary_information'])
        #     new_state_as_a_dict.update({'auxiliary_information' : new_auxiliary_information})

        # return State(self._collection_of_states, new_state_as_a_dict )

    def initialize_this_state(self) -> None:
        """ Initializes the current state to valid default values. The initial state will be '000' (with the right amount of digits) 
            and the classical channels will be [0,0] (again with right amount of digits). Measurement history is empty. The state will not 
            contain any 'auxiliary_information'
        """        
        new_initial_state = self._dict_of_valid_component_names.at_index(0)

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
        new_initial_state = self._dict_of_valid_component_names.at_index(0)

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
        # create an input state vector (numpy array)
        state_vector = np.array([0+0j]*len(self._dict_of_valid_component_names), dtype=np.cdouble)
        for component in input_optical_components.keys():
            index_component = self._dict_of_valid_component_names.index_of(component)
            state_vector[index_component] = input_optical_components[component]['amplitude']

        return (state_vector, list(self._dict_of_valid_component_names.keys()))
    
    def set_state_components_from_vector(self, state_vector: list ) -> None: 
        """ Set the optical components for the state from the vector which is given as argument. The function assumes the basis is the same 
            as used by function 'translate_state_components_to_vector'. For that function the basis is generated as output.

        Args:
            state_vector (list): list of amplitudes for the given basis
        """        
        optical_components = dict([])
        for index, value in enumerate(state_vector):
            # if value is below threshold set to zero
            if np.abs(value)**2 < self._threshold_probability_for_setting_to_zero:
                state_vector[index] = 0
                continue
            amplitude = state_vector[index]
            component = self._dict_of_valid_component_names.at_index(index)
            probability = np.abs(amplitude)**2
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

    def inner(self, state) -> float:
        """ Return the inner product between this state and the state passed as parameter. If states are orthogonal function returns 0, 
            if states are identical function returns 1. 

        Args:
            state (State): state

        Returns:
            float: inner product
        """
        result = np.cdouble(0)
        for comp1, amp_prob1 in self.optical_components.items():
            for comp2, amp_prob2 in state.optical_components.items():
                if comp1 == comp2:
                    result += np.conjugate(amp_prob1['amplitude']) * amp_prob2['amplitude']
        return result

    def outer(self, state) -> np.array:
        """ Return the outer product between this state and the state passed as parameter. Function returns a numpy array.

        Args:
            state (State): state

        Returns:
            np.array: outer product 
        """
        vector1, basis1 = self.translate_state_components_to_vector()
        vector2, basis2 = state.translate_state_components_to_vector()
        return  np.outer(np.conjugate(vector1),vector2)

    def tensor(self, state) -> dict:
        """ Generates optical components for an extended state as a tensor product of the current state and the state
            passed as parameter.

            The argument state can be used in three ways:
            - 'state' can be of type: 
                        new_components = state1.tensor(state = state2) with state2 of type fsc.State
            - 'state' can be the optical components: 
                        new_components = state1.tensor(state = {'100':{'amplitude': 1.0,'probability': 1.0}})
            - 'state' can be a list of tuples: 
                        new_components =  state1.tensor([('100',np.sqrt(1/2)),('110',1/2),('111',1/2)])

        Args:
            state (State, list, dict): State

        Returns:
            dict: optical components in the form {component_name (str) : {'amplitude': np.double, 'probability': float}}
        """

        if isinstance(state,State):
            new_list = NodeList(length_of_fock_state=self._length_of_fock_state,
                    no_of_optical_channels=self._no_of_optical_channels+state._no_of_optical_channels,
                    channel_0_left_in_state_name=self._channel_0_left_in_state_name)
            new_optical_components = dict([])
            for comp1, amp_prob1 in self.optical_components.items():
                for comp2, amp_prob2 in state.optical_components.items():
                    new_amp = amp_prob1['amplitude'] * amp_prob2['amplitude']
                    val1 = self._dict_of_valid_component_names[comp1]
                    val2 = state._dict_of_valid_component_names[comp2]
                    new_comp = new_list._dict_of_optical_values[tuple(val1 + val2)]
                    new_optical_components.update({new_comp: {'amplitude': new_amp, 'probability': np.abs(new_amp)**2 }})
            return new_optical_components
        
        if isinstance(state,list):
            new_optical_components = dict([])
            for comp1, amp_prob1 in self.optical_components.items():
                amp1 = amp_prob1['amplitude']
                for comp2, amp2 in state:
                    new_amp = amp1 * amp2
                    if self._channel_0_left_in_state_name:
                        new_comp = comp1 + comp2
                    else:
                        new_comp = comp2 + comp1
                    new_optical_components.update({new_comp: {'amplitude': new_amp, 'probability': np.abs(new_amp)**2 }})
            return new_optical_components
        
        if isinstance(state,dict):
            new_optical_components = dict([])
            for comp1, amp_prob1 in self.optical_components.items():
                for comp2, amp_prob2 in state.items():
                    if self._channel_0_left_in_state_name:
                        new_comp = comp1 + comp2
                    else:
                        new_comp = comp2 + comp1
                    new_amp = amp_prob1['amplitude'] * amp_prob2['amplitude']
                    new_optical_components.update({new_comp: {'amplitude': new_amp, 'probability': np.abs(new_amp)**2 }})
            return new_optical_components