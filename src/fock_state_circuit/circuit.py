import numpy as np
import math
import string
import matplotlib as mtplt
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib.patches  import Rectangle
import matplotlib.colors
from fock_state_circuit.collection_of_states import CollectionOfStates
from fock_state_circuit.state import State
from fock_state_circuit.temporal_and_spectral_gate_functionality.column_of_states import ColumnOfStates
from fock_state_circuit.temporal_and_spectral_gate_functionality.interference_group import InterferenceGroup
from fock_state_circuit.temporal_and_spectral_gate_functionality.collection_of_state_columns import CollectionOfStateColumns
from fock_state_circuit.nodes.optical_nodes import OpticalNodes
from fock_state_circuit.nodes.bridge_nodes import BridgeNodes
from fock_state_circuit.nodes.custom_nodes import CustomNodes
from fock_state_circuit.nodes.controlled_nodes import ControlledNodes
from fock_state_circuit.nodes.measurement_nodes import MeasurementNodes
from fock_state_circuit.nodes.classical_nodes import ClassicalNodes
from fock_state_circuit.nodes.spectral_nodes import SpectralNodes
from fock_state_circuit.nodes.superquantum_nodes import SuperQuantumNodes
from fock_state_circuit.nodes.nonlinear_optical_nodes import NonlinearOpticalNodes
from fock_state_circuit.visualization.draw import Draw
from fock_state_circuit.nodes.spectral_nodes import perform_measurement_photon_resolved
from fock_state_circuit.no_signalling_boxes import perform_measurement_no_signalling_boxes
from fock_state_circuit.temporal_and_spectral_gate_functionality.temporal_functions import _VERSION as tf_VERSION


def about():
    """
    About box for FockStateCircuit. Gives version numbers for
    FockStateCircuit, CollectionOfStates, NumPy and MatPlotLib.
    """
    print("")
    print("FockStateCircuit: Quantum Optics with Fock States for Python")
    print("Copyright (c) 2023 and later.")
    print("Rob Hendriks")
    print("")
    print("FockStateCircuit:            %s" % FockStateCircuit._VERSION)
    print("CollectionOfStates:          %s" % CollectionOfStates._VERSION)
    print("State:                       %s" % State._VERSION)
    print("ColumnOfStates:              %s" % ColumnOfStates._VERSION)
    print("InterferenceGroup:           %s" % InterferenceGroup._VERSION)
    print("CollectionOfStateColumns:    %s" % CollectionOfStateColumns._VERSION)    
    print("OpticalNodes:                %s" % OpticalNodes._VERSION)   
    print("BridgeNodes:                 %s" % BridgeNodes._VERSION)
    print("CustomNodes:                 %s" % CustomNodes._VERSION)
    print("ControlledNodes:             %s" % ControlledNodes._VERSION)
    print("MeasurementNodes:            %s" % MeasurementNodes._VERSION)   
    print("ClassicalNodes:              %s" % ClassicalNodes._VERSION)
    print("SpectralNodes:               %s" % SpectralNodes._VERSION)
    print("temporal_functions:          %s" % tf_VERSION)
    print("Numpy Version:               %s" % np.__version__)          
    print("Matplotlib version:          %s" % mtplt.__version__)

class FockStateCircuit(OpticalNodes,BridgeNodes, MeasurementNodes, ClassicalNodes, ControlledNodes, CustomNodes, SpectralNodes, NonlinearOpticalNodes, SuperQuantumNodes, Draw):
    """ Class for FockStateCircuit. The class is used to model the behavior of systems consisting of optical channels which can be 
        populates with Fock states, or photon number states (https://en.wikipedia.org/wiki/Fock_state). The circuit consistes of
        multiple channels (minimally 2 channels) which interact through optical components like beamsplitters and wave plates. The
        number states can be detected by detectors. Results from measurement are written to classical channels in the circuit. We also
        have the option to make the exact function of the optical components depending on the value in a classical channel. We can 
        for instance let the orientation of a wave plate depend on the value of the classical channels. 

        A FockStateCircuit takes a CollectionOfStates (a separate classs) as input and produces a CollectionOfStates as output. The 
        circuit models the evolution of optical and classical channels from initial state to final state.

        A CollectionOfStates consists of States (a separate class). States hold the values for classical and optical channels as well as
        the history of evolution through the circuit.

        The FockStateCircuit consists of 'nodes'. Each node performs a function, like interacting two optical channels, measuring a set of
        of channels, modifying the values in the classical channels. 

        Methods:

        Method(s) to create an instance:

            __init__(self, 
                    length_of_fock_state: int = 2, 
                    no_of_optical_channels: int = 2, 
                    no_of_classical_channels: int = 0, 
                    channel_0_left_in_state_name: bool = True, 
                    threshold_probability_for_setting_to_zero: float = 0.0001, 
                    use_full_fock_matrix: bool = False)

                    Constructor for an instance of the FockStateCircuit class. The instance will be created 'empty' without any nodes.
                    The instance has a fixed number of channels (optical channels for photons and classical channels which can contain
                    any number). The maximum number of photos in each channel is limited and specified by the argument 'length_of_fock_state'.
                    If 'length_of_fock_state' is for instance 3 the possible values for any optical channel are 0, 1 or 2 photons. So the
                    maximum photon number in this case is 2. In general the maximum photon number is equal to 'length_of_fock_state'-1.
                
                    The instance also has parameters for how to 'write' an optical state ('channel_0_left_in_state_name'). Of channel 0 has
                    one photon and the other channels zero photon then default the state is written as '1000'. With 
                    'channel_0_left_in_state_name' set to False this state would be written as '0001'. The value only affects the string value 
                    to indicate the state. All underlying mathematics and ordering of matrices and vectors is unaffected.
                
                    The paramater 'threshold_probability_for_setting_to_zero' forces rounding to zero for probabilities below the given level. 
                
                    Default the class optimizes calculations by reducing the size of the Fock state matrix to match the photon population
                    (i.e., if states are empty they are discarded from the fock matrix). When the bool 'use_full_fock_matrix' is set to True 
                    the system will always use the full size Fock matrix and skip the optimization.

                    
        Method(s) to run the simulation:

            evaluate_circuit(self, 
                    collection_of_states_input: collection_of_states.CollectionOfStates = None, 
                    nodes_to_be_evaluated: list = ['all']
                    ) -> collection_of_states.CollectionOfStates

                    Evaluate the fock state circuit for a given collection of input states. If no collection of input states
                    is given the function will generate a collection consisting of all basis states. If a list is given for
                    'nodes_to_be_evaluated' only those nodes are included (first node being '0', second node '1'). The function
                    returns a collection of states as the output of the fock state circuit consisting of the selected nodes.
                
                    This function is called recursively. At first call the function evaluates the first node(s) in the circuit 
                    and calculates the 'collection of states' that results from applying these nodes/this node to  
                    'collection_of_states_input', The the function 'evaluate_circuit' is called again for the next nodes. When
                    'evaluate_circuit' is finally called with an emply list of nodes not be evaluated the collection_of_states is retured.
                
                    The function groups the first nodes in the list of nodes_to_be_evaluated together if they are all of type 'optical'.
                    In this case a single fock state matrix can be calculated for these nodes combined. All other node types are handled 
                    one-by-one.

                    
        Method(s) to generate information on the circuit:

            basis(self)
                    Function returns a dictonary with valid components names as keys and the corresponding photon numbers in the channels as values.

            get_fock_state_matrix(self, 
                    nodes_to_be_evaluated: list[int] = 'all'
                    ) -> np.array:
                    Function returns the fock state matrix for a given set of nodes in the circuit            

        Last modified: May 1st, 2024              
    """
    _VERSION = '1.0.2'

    def __init__(self, length_of_fock_state: int = 2, 
                 no_of_optical_channels: int = 2, 
                 no_of_classical_channels: int = 0, 
                 channel_0_left_in_state_name: bool = True,
                 threshold_probability_for_setting_to_zero: float = 0.0001,
                 use_full_fock_matrix:bool = False,
                 circuit_name : str = None
                 ):
        """ Constructor for an instance of the FockStateCircuit class. The instance will be created 'empty' without any nodes.
            The instance has a fixed number of channels (optical channels for photons and classical channels which can contain
            any number). The maximum number of photos in each channel is limited and specified by the argument 'length_of_fock_state'.
            If 'length_of_fock_state' is for instance 3 the possible values for any optical channel are 0, 1 or 2 photons. So the
            maximum photon number in this case is 2. In general the maximum photon number is equal to 'length_of_fock_state'-1.

            The instance also has parameters for how to 'write' an optical state ('channel_0_left_in_state_name'). Of channel 0 has
            one photon and the other channels zero photon then default the state is written as '1000'. With 
            'channel_0_left_in_state_name' set to False this state would be written as '0001'. Teh value only affects the string value 
            to indicate the state. All underlying mathematics and ordering of matrices and vectors is unaffected.

            The parameter 'threshold_probability_for_setting_to_zero' forces rounding to zero for probabilities below the given level. 

            Default the class optimizes calculations by reducing the size of the Fock state matrix to match the photon population
            (i.e., if states are empty they are discarded from the fock matrix). When the bool 'use_full_fock_matrix' is set to True 
            the system will always use the full size Fock matrix and skip the optimization.

        Args:
            length_of_fock_state (int, optional): Defaults to 2.
            no_of_optical_channels (int, optional): Defaults to 2.
            no_of_classical_channels (int, optional): Defaults to 0.
            channel_0_left_in_state_name (bool, optional): Defaults to True.
            threshold_probability_for_setting_to_zero (float, optional): Defaults to 0.0001.
            use_full_fock_matrix (bool, optional): Defaults to False.
            circuit_name (str, optional): Defaults to None

        Raises:
            Exception: _description_
        """        

        super().__init__(length_of_fock_state, 
                 no_of_optical_channels, 
                 no_of_classical_channels, 
                 channel_0_left_in_state_name,
                 threshold_probability_for_setting_to_zero,
                 use_full_fock_matrix,
                 circuit_name
                 )
        # the length of the fock state is the number of possible photon numbers. So if the 'length' is 2 the maximum
        # number of photons is 1 (either 0 or 1 photons). If the length is 4 we can have 0,1,2 of 3 photons per channel.
        # self._length_of_fock_state = length_of_fock_state (executed in NodeList.__init__())

        # the number of channels defining the circuit.
        # self._no_of_optical_channels = no_of_optical_channels  (executed in NodeList.__init__())
        # self._no_of_classical_channels = no_of_classical_channels (executed in NodeList.__init__())
        
        # we need at least a fock states with length 2 (0 or 1 photon) and two optical channels. Anything else is a 
        # trivial circuit with either zero photons or one channel without interaction.
        # if self._length_of_fock_state < 1 or self._no_of_optical_channels < 2:
        #    raise Exception('length_of_fock_state minimal value is 1, no_of_optical_channels minimum value is 2') (executed in NodeList.__init__())

        # for naming the states we need a convention. if 'channel_0_left_in_state_name' is set to 'True' we
        # write a state with 2 photons in channel 0 and 5 photons in channel 1 as '05'. With this value set
        # to 'False' we would write this same state as '50'. 
        # self._channel_0_left_in_state_name = channel_0_left_in_state_name (executed in NodeList.__init__())

        # '_digits_per_optical_channel' defines the number of digits used when 
        # writing a fock state as word. For more than 10 <= photons <100 per channel we 
        # need 2 digits per channel. For 100 or more need 3 digits.
        # self._digits_per_optical_channel = len(str(self._length_of_fock_state-1)) (executed in NodeList.__init__())

        # list of nodes
        # self.node_list = [] (executed in NodeList.__init__())

        # probabilities below this threshold are set to zero. 
        # this means that if the probability to find an outcome is lower than this threshold the outcome is discarded.
        self._threshold_probability_for_setting_to_zero = threshold_probability_for_setting_to_zero
       
        # self.use_full_fock_matrix = True means the system always uses the full fock matrix. This means matrix only has to 
        # be calculated once. For small circuits with low 'length_of_fock_state' this is optimal. 
        # When self.use_full_fock_matrix = False the system reduces fock matrix to relevant states (i.e.,
        # states with lower photon number are discarded). This reduces operations with large matrices. 
        # This is optimal for system with large 'length_of_fock_state' and many optical channels, but only a few photons running 
        # through the system.
        self._use_full_fock_matrix = use_full_fock_matrix
        
        # name for the circuit
        self._circuit_name = circuit_name

        # generate a list of all possible values in the optical channels 
        index_list = [index for index in range(0,length_of_fock_state**self._no_of_optical_channels)]
        self._list_of_fock_states = [[] for index in index_list]
        for _ in range(0,self._no_of_optical_channels):
            for index in range(len(index_list)):
                n = int(index_list[index]%length_of_fock_state)
                self._list_of_fock_states[index].append(n)
                index_list[index] = int(index_list[index]/length_of_fock_state)

        return
    def __str__(self) -> str:
        """ Return the string for 'pretty printing' the circuit.

        Returns:
            str: string describing the circuit
        """ 
        text = 'FockStateCircuit name: ' + self._circuit_name + '\n'
        text += 'Optical Channels: ' + str(self._no_of_optical_channels) + '\n'
        text += 'Classical Channels: ' + str(self._no_of_classical_channels) + '\n'
        text += 'Length of Fock state: ' + str(self._length_of_fock_state) + '\n'
        return text   

    def evaluate_circuit(self, 
                         collection_of_states_input: CollectionOfStates = None, 
                         nodes_to_be_evaluated: list = ['all']
                         ) -> CollectionOfStates:
        """ Evaluate the fock state circuit for a give collection of input states. If no collection of input states
            is given the function will generate a collection consisting of all basis states. If a list is given for
            'nodes_to_be_evaluated' only those nodes are included (first node being '0', second node '1'). The function
            returns a collection of states as the output of the fock state circuit consisting of the selected nodes.

            This function is called recursively. At first call the function evaluates the first node(s) in the circuit 
            and calculates the 'collection of states' that results from applying these nodes/this node to  
            'collection_of_states_input', The the function 'evaluate_circuit' is called again for the next nodes. When
            'evaluate_circuit' is finally called with an emply list of nodes not be evaluated the collection_of_states is retured.

            The function groups the first nodes in the list of nodes_to_be_evaluated together if they are all of type 'optical'.
            In this case a single fock state matrix can be calculated for these nodes combined. All other node types are handled 
            one-by-one.

        Args:
            collection_of_states_input (CollectionOfStates, optional): Defaults to None.
            nodes_to_be_evaluated (list[int], optional): Defaults to ['all'].

        Raises:
            Exception: Exception will be triggered does not match with the nodes in the circuit
            Exception: Exception will be triggered if the nodes in the circuit contain an invalid node type
            Exception: Exception will be triggered if if the optical node with classical control is of unknown type

        Returns:
            CollectionOfStates : collection of states as output of the circuit
        """        
        # if nodes_to_be_evaluated == ['all'] make a list with all indices in node_list
        if nodes_to_be_evaluated == ['all']:
            nodes_to_be_evaluated = list(range(len(self.node_list)))
            if len(self.node_list) == 0: # case for empty circuit
                return collection_of_states_input
        # if nodes_to_be_evaluated is empty return the input state. This was last recursion loop
        elif len(nodes_to_be_evaluated) == 0:
            return collection_of_states_input
        # raise exception if the values in nodes_to_be_evaluated do not match with the length of self.node_list
        else:
            for node_index in nodes_to_be_evaluated:
                if not (0 <= node_index < len(self.node_list)) :
                    raise Exception('Invalid list of node indices in function evaluate_circuit')
        
        # if no state is given as input use initial collection of states 
        if collection_of_states_input is None:
            collection_of_states_input = CollectionOfStates(self)

        # check how many of the upcoming nodes are of type 'optical'. They will be evaluated in one step
        # other node types are evaluates one-by-one
        current_node_index = nodes_to_be_evaluated[0]
        if self.node_list[current_node_index].get('node_type') == 'optical':
            optical_nodes_list = [] # this list will contain all upcoming nodes which are 'optical'
            for index, node_index in enumerate(nodes_to_be_evaluated):
                if self.node_list[node_index].get('node_type') != 'optical':
                    break
                else:
                    optical_nodes_list.append(node_index)
            if max(optical_nodes_list) >= max(nodes_to_be_evaluated):
                remaining_nodes = []
            else:
                remaining_nodes = nodes_to_be_evaluated[len(optical_nodes_list):]
        else:
            current_node_index = nodes_to_be_evaluated[0]
            if current_node_index >= max(nodes_to_be_evaluated):
                remaining_nodes = []
            else:
                remaining_nodes = nodes_to_be_evaluated[1:]

        #****************************************************
        #                                                   *
        # return case per node type for evaluate circuit    *
        #                                                   *
        #****************************************************

        #***********************************************
        # case where next node(s) are of type 'optical'*
        #***********************************************
        if self.node_list[current_node_index].get('node_type') == 'optical':
            collection_of_states_output = self.__apply_node_or_nodes_to_collection(
                collection_of_states = collection_of_states_input,
                nodes_to_be_evaluated = optical_nodes_list,
                single_custom_node = None
                )

            return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)

        #****************************************************************
        # case where next node is of type 'custom fock matrix'          *
        #****************************************************************
        elif self.node_list[current_node_index].get('node_type') == 'custom fock matrix':

            custom_fock_matrix = self.node_list[current_node_index].get('custom_fock_matrix')
            # prepare a new collection of states which will be filled with the results 
            collection_of_states_output = CollectionOfStates(self, input_collection_as_a_dict=dict([]))

            for identifier, old_state in collection_of_states_input.items():
                new_state = old_state.copy()
                # create an input state vector (numpy array)
                state_vector, basis = old_state.translate_state_components_to_vector()

                # multiply with the matrix to get output state
                output_state_vector = np.matmul(custom_fock_matrix,state_vector)     

                # now translate the output vector to a state   
                new_state.set_state_components_from_vector(state_vector = output_state_vector)

                collection_of_states_output.add_state(state= new_state)
            return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)
        
        #**********************************************
        # case where next node is of type 'classical' *
        #**********************************************
        elif self.node_list[current_node_index].get('node_type') == 'classical':          
            for state in collection_of_states_input:
                # read the function from the node_list
                classical_function = self.node_list[current_node_index].get('function')

                # apply the function to the classical channels for every state in the collection
                state.classical_channel_values = classical_function(
                    state.classical_channel_values,
                    self.node_list[current_node_index]['new_input_values'],
                    self.node_list[current_node_index]['affected_channels'])
                
            return self.evaluate_circuit(collection_of_states_input = collection_of_states_input, nodes_to_be_evaluated = remaining_nodes)
  
        #**********************************************************************
        # case where next node is of type 'measurement'  *
        #**********************************************************************
        elif self.node_list[current_node_index].get('node_type') == 'measurement':

            if collection_of_states_input.is_photon_resolved():
                # list of projection to speed up measurement. The detection will only consider the 'projections' in the list
                list_of_projections = self.node_list[current_node_index].get('list_of_projections', None)

                # get parameters from node on what channels to measure and where to write the results
                optical_channels_to_be_read = self.node_list[current_node_index].get('optical_channels_to_be_read')
                classical_channels_to_be_written = self.node_list[current_node_index].get('classical_channels_to_be_written')  

                # perform the measurement
                collection_of_states_output = perform_measurement_photon_resolved(  collection_of_states=collection_of_states_input, 
                                                                                    optical_channels_to_measure=optical_channels_to_be_read, 
                                                                                    classical_channels_to_write_to=classical_channels_to_be_written,
                                                                                    list_of_projections = list_of_projections)
                return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)
            
            elif collection_of_states_input.has_no_signalling_boxes():
                # get parameters from node on what channels to measure and where to write the results
                optical_channels_to_be_read = self.node_list[current_node_index].get('optical_channels_to_be_read')
                classical_channels_to_be_written = self.node_list[current_node_index].get('classical_channels_to_be_written')  
                collection_of_states_output = perform_measurement_no_signalling_boxes(  collection_of_states=collection_of_states_input, 
                                                                    optical_channels_to_measure=optical_channels_to_be_read, 
                                                                    classical_channels_to_write_to=classical_channels_to_be_written)
                return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)

            else:
                # generate a number as a string to identify whether this is first, seconde, third etc measurement
                measurement_identifier = self.__get_identifier_for_measurement(current_node_index)
                        
                # get parameters from node on what channels to measure and where to write the results
                optical_channels_to_be_read = self.node_list[current_node_index].get('optical_channels_to_be_read')
                classical_channels_to_be_written = self.node_list[current_node_index].get('classical_channels_to_be_written')  

                # prepare a new collection of states which will be filled with the results of the measurements
                # on the states in the input collection of states          
                collection_of_states_output = CollectionOfStates(self, input_collection_as_a_dict=dict([]))
            
                for identifier, state in collection_of_states_input.items():
                    # for every state in the input collection_of_states we perform the measurement.
                    # The result is a collection of states (statistical mixture) for the different measurement outcomes.
                    collection_of_states_after_measurement = self._perform_measurement(state = state,
                                                                                optical_channels_to_measure = optical_channels_to_be_read,
                                                                                classical_channels_to_write_to = classical_channels_to_be_written)
                    # add all states in the collection resulting from the measurement to the output collection
                    # make some nice labels for the identifier to see history (can also be seen in measurement results)
                    marker = self.__next_state_marker()
                    for new_state in collection_of_states_after_measurement:
                        new_identifier = identifier + '-M' + measurement_identifier + next(marker)
                        collection_of_states_output.add_state(state = new_state, identifier=new_identifier)

                return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)

        #*****************************************************************
        # case where next node is of type 'controlled'
        #*****************************************************************
        elif self.node_list[current_node_index].get('node_type') == 'controlled':

            function = self.node_list[current_node_index].get('function')                
            variables_to_be_controlled = self.node_list[current_node_index]['control_parameters'].get('variables_to_be_controlled')
            classical_control_channels = self.node_list[current_node_index]['control_parameters'].get('classical_control_channels')
            function_parameters = self.node_list[current_node_index]['control_parameters'].get('function_parameters')
            
            # check if inputs are correct and otherwise raise an error
            error1 = (function != '__generic_optical_matrix(theta, phi)')
            error2 = (len(variables_to_be_controlled) != len(classical_control_channels))
            if any([error1, error2]):
                raise Exception('error in evaluate_circuit when assessing node of type \'optical and classical combined\'')
            
            # set default values
            for parameter in function_parameters.keys():
                if parameter == 'theta':
                    theta = function_parameters[parameter]
                if parameter == 'phi':
                    phi = function_parameters[parameter]

            # every state can have a different value in the classical channels
            # run through states and read classical value per state to make the matrix. 
            # apply this to the optical components of the same state
            collection_of_states_output = CollectionOfStates(self, input_collection_as_a_dict=dict([]))

            for identifier, state in collection_of_states_input.items():
                classical_values = [i for i in state.classical_channel_values]
                # load values from classical channels
                for index,parameter in enumerate(variables_to_be_controlled):
                    channel = classical_control_channels[index]
                    if parameter == 'theta':
                        theta = classical_values[channel]
                    if parameter == 'phi':
                        phi = classical_values[channel]

                # create the optical matrix corresponding to the function and classical channel values
                optical_matrix = self._generate_generic_optical_matrix(theta, phi)
                
                # generate custom node from which to calculate the fock state matrix
                tensor_list = self.node_list[current_node_index].get('tensor_list')
                single_custom_node = {'matrix_optical':optical_matrix, 'tensor_list': tensor_list}

                # make a collection for just one state
                collection_of_single_state = CollectionOfStates(self, input_collection_as_a_dict={identifier:state})

                collection_of_single_state_output = self.__apply_node_or_nodes_to_collection(
                    collection_of_states = collection_of_single_state,
                    nodes_to_be_evaluated = None,
                    single_custom_node = single_custom_node)
                    
                for identifier2, state2 in collection_of_single_state_output.items():
                    collection_of_states_output.add_state(state = state2, identifier = identifier2)
            
            return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)
        
        #************************************************************************
        # case where next node is of type 'generic function on collection'      *
        #************************************************************************
        elif self.node_list[current_node_index].get('node_type') == 'generic function on collection':
            
            generic_function = self.node_list[current_node_index].get('generic_function')
            generic_function_parameters = self.node_list[current_node_index].get('generic_function_parameters')
            
            collection_of_states_output = generic_function(collection_of_states_input,generic_function_parameters)
         
            return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)


        #*****************************************************************
        # case where next node is of type 'spectrum'      *
        #*****************************************************************
        elif self.node_list[current_node_index].get('node_type') == 'spectrum':

            spectral_information = self.node_list[current_node_index].get('spectral information')
            channels = spectral_information['channels']
            bandwidth_omega = spectral_information['bandwidth']

            # every state can have a different value in the classical channels
            # run through states and read classical value per state to make the matrix. 
            # apply this to the optical components of the same state
            collection_of_states_output = CollectionOfStates(self, input_collection_as_a_dict=dict([]))

            for identifier, state in collection_of_states_input.items():       
                if 'time delay' in spectral_information.keys():
                    time_delay_tau = spectral_information['time delay']
                else:
                    classical_channel_for_delay = spectral_information['classical_channel_for_delay']
                    time_delay_tau = state.classical_channel_values[classical_channel_for_delay]
                states_returned = self._apply_time_delay(channels, time_delay_tau,bandwidth_omega,state.copy())
                for state in states_returned:
                    collection_of_states_output.add_state(state = state)

            return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)
        #*****************************************************************
        # case where next node is of type 'bridge'      *
        #*****************************************************************
        elif self.node_list[current_node_index].get('node_type') == 'bridge':
            next_fock_state_circuit = self.node_list[current_node_index].get('next_fock_state_circuit')
            
            new_length = next_fock_state_circuit._length_of_fock_state
            new_optical = next_fock_state_circuit._no_of_optical_channels
            new_classical = next_fock_state_circuit._no_of_classical_channels
            
            collection_of_states_reshaped = collection_of_states_input.copy()
            if new_optical > self._no_of_optical_channels:
                collection_of_states_reshaped.extend(extra_optical_channels=(new_optical - self._no_of_optical_channels))
            elif new_optical < self._no_of_optical_channels:
                optical_to_keep = [channel for channel in range(new_optical)]
                collection_of_states_reshaped.reduce(optical_channels_to_keep=optical_to_keep)
            if new_classical > self._no_of_classical_channels:
                collection_of_states_reshaped.extend(extra_classical_channels=(new_classical - self._no_of_classical_channels))
            elif new_classical < self._no_of_classical_channels:
                classical_to_keep = [channel for channel in range(new_classical)]
                collection_of_states_reshaped.reduce(classical_channels_to_keep=classical_to_keep)

            collection_of_states_reshaped.adjust_length_of_fock_state(new_length)
            collection_of_states_output = next_fock_state_circuit.evaluate_circuit(collection_of_states_input = collection_of_states_reshaped)
            #return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)
            return collection_of_states_output
        raise Exception('Error in evaluate_circuit. Check if node_types are correct')
        return
 
    def __next_state_marker(self):
        """ Generate consistent identifier for multipe states coming from a measurement """

        alphabet = list(string.ascii_lowercase)
        status  = [0]       
        while True:
            marker = ''
            for number in status:
                marker = marker + alphabet[number]
            copy_status = status
            carry = 1
            for index,number in enumerate(copy_status):
                if alphabet[number] != 'z':
                    status[index] += carry
                    carry = 0
                    break
                else:
                    carry = 1
                    status[index] = 0
            if carry == 1:
                status.append(0)
            status = status 
            yield marker              
    
    def __get_identifier_for_measurement(self,current_node_index: int) -> str:
        """ Generate identifier for measurement to separate first, second, third etc measurement in the circuit """

        total_number_of_measurement_nodes_in_the_circuit = sum([
                1 if self.node_list[node_index].get('node_type') == 'measurement' else 0 
                for node_index in range(len(self.node_list))
                ])
        string_formatter = "{:0" + str(len(str(total_number_of_measurement_nodes_in_the_circuit)))+"d}"

        measurement_number = sum([
                1 if self.node_list[node_index].get('node_type') == 'measurement' else 0 
                for node_index in range(current_node_index+1)
                ])

        return string_formatter.format(measurement_number)

    def _perform_measurement(self,
                            state: State,
                            optical_channels_to_measure: list = [],
                            classical_channels_to_write_to: list = []
                            ) -> CollectionOfStates:
        """ Perform a measurement on a single state, where the given optical channels are measured and the result is written
            to the given classical channels. The function will return a collection of states, one state for each possible
            measurement outcome. The optical states will be the 'collapsed states' for the measurement outcome. The resulting 
            collection represents a 'statistical mixture' with as weights for each state the likelihood of the corresponding measurement
            result.

            List for optical and classical channels should be equally long to avoid unpredictable result

        Args:
            state (State): input state to be measured.
            optical_channels_to_measure (list, optional): optical channels to measure. list should have equal length list
                for classical channels to write to
            classical_channels_to_write_to (list, optional): classical channels to write the optical measurements to

        Returns:
            CollectionOfStates: collection of states after measurement of single input state
        """        
        # create empty collection which will be populated with the collapsed states and returned at end of the function
        collection_of_states_after_measurement = CollectionOfStates(fock_state_circuit = self, input_collection_as_a_dict=dict([]))

        # if the number of classical channels is larger than the number of opticl channels limit the number
        # of classical channels. 
        if len(classical_channels_to_write_to) > len(optical_channels_to_measure):
            classical_channels_to_write_to = classical_channels_to_write_to[:len(optical_channels_to_measure)]

        # create a list of tuples with a first tuple element the optical values and second the amplitude (as numpy complex number)
        val_amp_list = [(self._dict_of_valid_component_names[name], amp_prob['amplitude']) for name, amp_prob in state.optical_components.items()]
 
        # create a dictionary where the list of tuples is split for different measurement outcomes. The measurement outcome is the key
        dict_of_outcomes = dict([])
        for item in val_amp_list:
            dict_of_outcomes.setdefault(tuple(item[0][n] for n in optical_channels_to_measure),[]).append(item)

        # create for each measurement outcome a new 'collapsed' state and add this state to the collection as return value
        # outcome is a tuple of measured values (like (0,1,2) if you measured three channels). List of outcomes is a list
        # of tuples with values of all optical channels and the amplitudes
        # [((0,2,3,1), 0.71), ((0,2,3,2), 0.71)]. For measured optical channels the values should be the same in all items in the list.
        for outcome, list_of_outcomes in dict_of_outcomes.items():
            # create a list or amplitudes and take sum and sum squared for normalization
            old_amplitudes = np.array([item[1] for item in list_of_outcomes])
            old_probabilities = old_amplitudes.real**2 + old_amplitudes.imag**2
            tot_amp = np.sum(old_amplitudes )
            tot_prob = np.sum(old_probabilities )
            # renormalized to create new amplitudes and probabilities
            new_amps = old_amplitudes/np.sqrt(tot_prob)
            new_probs = new_amps.real**2 + new_amps.imag**2
            # create a new state as collapsed state for a specific measurement result
            new_state= state.copy()
            # 1. the new state has a cumulative probability that is multiplied with the probability to detect this specific outcome
            try:
                new_state.cumulative_probability *= tot_prob
            except:
                new_state.cumulative_probability = np.round(new_state.cumulative_probability*tot_prob,3)
            # 2. the new state should have classical channel values modified representing the measurement result
            for i,n in enumerate(classical_channels_to_write_to):
                new_state.classical_channel_values[n] = outcome[i]
            # 3. the new state's measurement results need to be updated with 
            try:
                new_state.measurement_results.append({'measurement_results':new_state.classical_channel_values, 'probability': tot_prob})   
            except:
                new_state.measurement_results = [{'measurement_results':new_state.classical_channel_values, 'probability': tot_prob}]
            new_state.optical_components = {self._dict_of_optical_values[tuple(name[0])]:{'amplitude':new_amplitude,'probability': new_probability} for name, new_amplitude, new_probability in zip(list_of_outcomes,new_amps, new_probs)}
            # 4. Other aspects like initial_state and auxiliary_information remain unchanged
            # add the new state to the collection
            collection_of_states_after_measurement.add_state(new_state)
        return collection_of_states_after_measurement

    
    def __apply_node_or_nodes_to_collection(self,
                                          collection_of_states: CollectionOfStates = None,
                                          nodes_to_be_evaluated: list[int] = ["all"],
                                          single_custom_node = None) -> CollectionOfStates:
        
        """ Apply the node(s) to a collection of states. If no  collection_of_states_input is given the function will generate 
            a collection containing all fock states for the circuit.

        Args:
            collection_of_states_input: (_dict_, optional): collection of states as input. Defaults to None.
            nodes_to_be_evaluated (_list_, optional):: nodes to be includes in the matrix calculation (first node is node 0). 
                default all nodes are included (value for nodes_to_be_evaluated = 'all' )
            single_custom_node: (_dict_, optional): dictionary with 'matrix_optical' and 'tensor_list'


        Raises:
            nothing

        Returns:
            collection_of_states_output: collection of states as output
        """ 

        if collection_of_states is None:
            collection_of_states = CollectionOfStates(fock_state_circuit=self)
        else:
            collection_of_states._fock_state_circuit = self
            collection_of_states = collection_of_states.copy()
        if self._use_full_fock_matrix == True:
            # if 'use_full_fock_matrix' is set to 'True' we always use the complete matrix. There is no need to group
            # states by photon number.
            states_grouped_by_photon_number = {'all': collection_of_states.state_identifiers_as_list()}
        else:
            # otherwise we create a new dictionary where states are grouped together that have components with same photon 
            # number. we use this to efficienlty use the fock matrix which can be reused for states within a group.
            # reduction of the fock matrix helps to avoid we always need a fock matric for the full set of states 
            # (which can become a huge matrix).
            states_grouped_by_photon_number = collection_of_states._group_states_together_by_photon_number()

        for key,grouped_states_by_photon_number in states_grouped_by_photon_number.items():
            # the key indicates the photon numbers in the components in the state.
            # either all components have the same number of photons (e.g., key = '=5' for components '500', '230' and '113')
            # of there are different photon numbers and then we use the max (e.g., key = '<=5'for components '500', '111' and '220')
            fock_matrix, list_of_state_indices_to_keep = self.__generate_fock_state_matrix(
                        nodes_to_be_evaluated = nodes_to_be_evaluated,
                        photon_number_selection = key,
                        single_custom_node = single_custom_node)
            
            for state_identifier in grouped_states_by_photon_number:
                # create an input state vector (numpy array)
                state_vector, basis = collection_of_states[state_identifier].translate_state_components_to_vector()
                reduced_vector = [state_vector[i] for i in list_of_state_indices_to_keep]

                # multiply with the matrix to get output state
                output_state_vector = np.matmul(fock_matrix,reduced_vector)     

                # expand the vector back to full basis
                expanded_vector = [np.cdouble(0)] * len(basis)
                for index,value in enumerate(list_of_state_indices_to_keep):
                    expanded_vector[value] = output_state_vector[index]
                # now translate the output vector to a state   
                collection_of_states[state_identifier].set_state_components_from_vector(state_vector = expanded_vector)
        return collection_of_states
    
    def __generate_fock_state_matrix(   self,
                                        nodes_to_be_evaluated: list[int] = 'all',
                                        photon_number_selection: str = 'all',
                                        single_custom_node: np.array = None
                                        ) -> np.array:
        
        """ Generate the fock_state_matrix for the circuit or for a custom node. This matrix describes the transition 
            from fock state basis at input to fock state basis as output. In other words: The elements
            in this matrix are the transition amplitudes from fock state to fock state ('0001' to '0100', 
            or '123' to '006'). In the argument a list can be passed which indicates the nodes to be included.
            if a single custom node is given the matrix is calculated for that node. Otherwise the list of 'nodes to be evaluated'
            is used. If this list is not given the fock_state_matrix is calculated for the complete circuit. 

        Important:
        - This can be a matrix consuming significant memory. The size of the fock state basis for a circuit with 
        'length' = 5 (max 4 photons in a channel) and 8 channels is 5**8 =  390.625. This matrix has 1.5E+11 values. 
        with the input photon_number_selection we can force the function to create a smaller matrix affecting a selected
        number of states. 
        - The matrix can only be calculated for a circuit with nodes of type 'optical'. With a measurement or
        classical control this is not possible.    
        
        Args:
            nodes_to_be_evaluated (list[int]): nodes to be includes in the matrix calculation (first node is node 0). 
                default all nodes are included (value for nodes_to_be_evaluated = 'all' )
            photon_number_selection (str) : string indicating whether to select states with a given photon number, or a photon 
                number below the given value.
            single_custom_node (np.array): dictionary with 'matrix_optical' and 'tensor_list'
        Returns:
            np.array: fock_state_matrix
        
        Raises:
            Exception if nodes of other node_type than 'optical' are included
        """
        if single_custom_node is not None:
            nodes_to_be_evaluated = [0]
            optical_matrix = single_custom_node['matrix_optical']
            tensor_list = single_custom_node['tensor_list']
        else:
            # if nodes_to_be_evaluated == 'all' make a list with all indices in node_list
            if nodes_to_be_evaluated == 'all':
                nodes_to_be_evaluated = list(range(len(self.node_list)))
                
            # check if all nodes are of type 'optical'        
            for index in nodes_to_be_evaluated:
                node = self.node_list[index]
                if node['node_type'] != 'optical':
                    raise Exception("Cannot create fock state matrix. Only nodes with node_type 'optical' can be used")
            
            # if there is only one node return the fock matrix from function '__generate_single_node_fock_matrix_from_optical_matrix'
            # otherwise multiply matrices from all nodes in 'nodes_to_be_evaluated' and return the resulting matrix
            index = nodes_to_be_evaluated[0]            
            node = self.node_list[index]
            optical_matrix = node['matrix_optical']
            tensor_list = node['tensor_list']

        # we now either have a list nodes to be converted one by one,or we have a single optical matrix to be converted.
        try: 
            matrix_fock_state_single_node, list_of_state_indices = self.__generate_single_node_fock_matrix_from_optical_matrix(
                matrix = optical_matrix, 
                tensor_list = tensor_list,
                photon_number_selection = photon_number_selection
                )
        except:
            raise Exception('Cannot create fock state matrix. Most likely matrix is too big')
        
        if len(nodes_to_be_evaluated) == 1: #case for one node only
            return (matrix_fock_state_single_node, list_of_state_indices)
        else:  #case for multiple nodes       
            fock_state_matrix = matrix_fock_state_single_node
            for index in nodes_to_be_evaluated[1:]:
                node = self.node_list[index]
                optical_matrix = node['matrix_optical']
                tensor_list = node['tensor_list']

                matrix_fock_state_single_node, list_of_state_indices  = self.__generate_single_node_fock_matrix_from_optical_matrix(
                    matrix = optical_matrix, 
                    tensor_list = tensor_list,
                    photon_number_selection = photon_number_selection
                    )
                fock_state_matrix = np.matmul(matrix_fock_state_single_node, fock_state_matrix)

            return (fock_state_matrix, list_of_state_indices)
    
    def __generate_single_node_fock_matrix_from_optical_matrix(self, 
                                                               matrix: np.array = np.identity(2), 
                                                               make_unitary: bool = True, 
                                                               tensor_list: list[int] = None,
                                                               photon_number_selection: str = 'all'
                                                               ) -> tuple[np.array, list[int]]:    
        """ Generate the transition matrix for fock states from the optical matrix for mixing two channels. 
            Currently only works for 2x2 optical matrix. If tensor_list is given the returned fock matrix will 
            be scaled up and re-arranged to apply to full set of optical channels. If tensor_list is not given 
            the returned fock matrix applies to two optical channels.
    
        Args:
            matrix (np.array): matrix for the optical transition (the electric fields)
                make_unitary (bool): If true the transition for any states with more photons than allowed per channel is set to 0 or 1.
                tensor_list (list[int]): list indicating to what channels the matrix should apply. Default = None
            photon_number_selection (str): If this is 'all' all states are included. If this is formated at '=5' or '=10'only states 
                with that amount of photons are included in the matrix (i.e., is value is 2 states are '11', '20', '02' but not
                '00' or '01'). If the format is '<=5' or '<=10' all states with the number or less photons are included

        Returns:
            tuple: (fock_matrix_scaled, states_indices_to_keep)  
            fock_matrix_scaled is a matrix in numpy array format describing the transition between fock stated |nm> to |uv> for two channels,
            or from |xxnxxmxx> to |yyuyyvyy> for more chanels
            states_indices_to_keep: list of state indices in the right order, corresponding to the matrix

        
        Raises:
            exception if the input optical matrix is not of size 2x2
        """     
        if len(matrix) != 2 and len(matrix[0]) != 2:
            raise Exception("incorrect optical matrix in '__fock_matrix_from_optical_matrix'")
        rac = matrix[0][0]
        rbd = matrix[1][1]
        tbc = matrix[0][1]   
        tad = matrix[1][0]  
        length = self._length_of_fock_state
        fock_matrix = np.zeros((length**2, length**2), dtype = np.csingle) # the matrix is pre-filled as zero matrix
        for l1 in range(0,length**2): #l1 is the vertical index in fock_matrix
            # n and m are the input photon numbers in the two channels
            n = int(l1%length)
            m = int((l1-n)/length)
            #n and m are input photon numbers, l1 is the counter for output data. l1 = m * Length + n
            #the input is a state |n>|m> . The mixing has a transmission (t) and a reflection (r)
            #the matrix is [t,r],[r,-t]. Output is a sum of states |u>|v>
            #so transformation is  |n>|m>|0>|0>  to (a**n)(b**m)|0>|0>|0>|0> to  Sum((c**u)(d**v)|0>|0>|0>|0>) to Sum(|0>|0>|u>|v>)
            #(tu + rv)**n (ru - tv)**m = 
            for j in range(0,n+1):
                for k in range(0,m+1):
                    u = (j+k)
                    v = (n-j)+(m-k)
                    #see https://en.wikipedia.org/wiki/Beam_splitter for formula
                    #the factor from the binomial expansion
                    coeff = math.comb(n,j)*math.comb(m,k)*(rac**(j))*(tad**(n-j))*(tbc**(k))*((rbd)**(m-k))
                    #factor in sqrt n for boson raising operation c|n> = sqrt(n+1)|n+1> and l|n> = sqrt(n)|n-1>. <n|a*a|n> = n
                    coeff = coeff * math.sqrt(math.factorial(u)*math.factorial(v)/(math.factorial(n)*math.factorial(m)))
                    #total number of photons cannot exceed (length-1) for each of the output channels
                    if ((u < length) and (v < length)):
                        l2 = v * length + u #l2 is the horizontal index in fock_matrix
                        fock_matrix[l1][l2] = fock_matrix[l1][l2] + coeff
                        # if make_unitary is set to True we have to set all coefficients for output states with 
                        if make_unitary:
                            if n + m >= length or u + v >= length:
                                if (n == u) and (m == v):
                                    fock_matrix[l1][l2] = 1.0
                                else:
                                    fock_matrix[l1][l2] = 0.0
            
        if tensor_list is None:
            return (fock_matrix, [])
            
        else:
            # if tensor_list is provided scale the fock_matrix up to the full set of optical channels and 
            # rearrange to make the operation apply to the right channels.
            # user tensor product/kronecker product with identity matrix to scale up to cover full circuit
            for n in range(1, self._no_of_optical_channels+1):
                if self._length_of_fock_state**n < len(fock_matrix):
                    continue
                elif self._length_of_fock_state**n == len(fock_matrix):
                    fock_matrix_scaled = fock_matrix
                elif self._length_of_fock_state**n > len(fock_matrix):
                    fock_matrix_scaled = np.kron(np.identity(self._length_of_fock_state), fock_matrix_scaled)

            # generate the right order for the base states to re-arrange the matrix to ensure it acts on the right channels
            # tensor list contains the new order for the channels
            new_state_list = [] # this will be the list of the new states with order of channels adjusted
            
            for state in self._list_of_fock_states: # iterate to all states in standard order
                new_state = [state[index] for index in tensor_list] # re-order the channels as per 'tensor_list'
                new_state_list.append(new_state) 
            
            # generate the new order
            new_order = np.array([self._list_of_fock_states.index(new_state_list[i]) for i in range(len(new_state_list))])

            # filter out only states with right photon number:
            if photon_number_selection == 'all':
                states_indices_to_keep = np.array([i for i in range(len(self._list_of_fock_states))])
            else:
                if photon_number_selection[0] == '=':
                    number = int(photon_number_selection[1:])
                    states_indices_to_keep = np.array([i for i, state in enumerate(self._list_of_fock_states) if sum(state) == number])
                elif photon_number_selection[0] == '<':
                    number = int(photon_number_selection[2:])
                    states_indices_to_keep = np.array([i for i, state in enumerate(self._list_of_fock_states) if sum(state) <= number])
                else:
                    raise Exception("error with 'photon_number_selection' parameter in '__generate_single_node_fock_matrix_from_optical_matrix'")
            
            # combine filtering and ordering in one list for indexing the fock matrix
            new_order_and_selection = new_order[np.ix_(states_indices_to_keep)]

            fock_matrix_scaled = fock_matrix_scaled[np.ix_(new_order_and_selection,new_order_and_selection)]
            
            #list_of_states = [self.generate_collection_of_states(return_type='list_of_strings')[i] for i in states_indices_to_keep]

            return (fock_matrix_scaled, states_indices_to_keep)   
 
    def basis(self) -> dict:
        """ Function returns a dictonary with valid components names as keys and the corresponding photon numbers in the channels as values.
        """

        collection_of_states = CollectionOfStates(fock_state_circuit=self, input_collection_as_a_dict=dict([]))
        return collection_of_states._dict_of_valid_component_names

    def get_fock_state_matrix(self, nodes_to_be_evaluated: list[int] = 'all') -> np.array:
        """ Function returns the fock state matrix for a given set of nodes in the circuit

            Args:
                nodes_to_be_evaluated (list[int]): nodes to be includes in the matrix calculation (first node is node 0). 
                    default all nodes are included (value for nodes_to_be_evaluated = 'all' )

            Returns:
                np.array: fock_state_matrix
        """
        if len(nodes_to_be_evaluated) == 1 and self.node_list[nodes_to_be_evaluated[0]].get('node_type') == 'custom fock matrix':
            return  self.node_list[nodes_to_be_evaluated[0]].get('custom_fock_matrix')
      
        fock_matrix, list_of_state_indices_to_keep = self.__generate_fock_state_matrix(nodes_to_be_evaluated=nodes_to_be_evaluated)
        return fock_matrix

    
