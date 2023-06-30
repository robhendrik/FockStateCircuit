import numpy as np
import qutip as qt
import math
import string
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.colors
import collection_of_states as cos


class FockStateCircuit:
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
                    'channel_0_left_in_state_name' set to False this state would be written as '0001'. Teh value only affects the string value 
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

                    Evaluate the fock state circuit for a give collection of input states. If no collection of input states
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

            draw(self, 
                    print_defaults: bool = False, 
                    settings_for_drawing_circuit: dict = None
                    ) -> None

                    Draw the optical circuit. self. settings_for_drawing_circuit is the dict with settings for circuit drawing. 
                    If this dict is not defined default values will be taken. If the boolean print_defaults is set to True the function 
                    will print out the default values to console

            conventions(self) 
                    Function to print return the conventions used in this class as a string


        Method(s) to add wave plates to the circuit:

            half_wave_plate(self, 
                    channel_horizontal: int = 0, 
                    channel_vertical: int = 1, 
                    angle: float = 0, 
                    node_info: dict = None
                    ) -> None

                    Add a half wave plate to the fock state circuit with axis rotated over angle. The phase shift of 'pi' will be applied to the 
                    channel for vertical polarization.           

            half_wave_plate_225(self, 
                    channel_horizontal: int = 0, 
                    channel_vertical: int = 1, 
                    node_info: dict = None
                    ) -> None

                    Add a half wave plate at 22.5 degrees to the fock state circuit. The phase shift of 'pi' will be applied to the 
                    channel for vertical polarization in the frame of reference of the wave plate.
                    
            half_wave_plate_45(self, 
                    channel_horizontal: int = 0, 
                    channel_vertical: int = 1, 
                    node_info: dict = None
                    ) -> None

                    Add a half wave plate at 45 degrees to the fock state circuit. The phase shift of 'pi' will be applied to the 
                    channel for vertical polarization in the frame of reference of the wave plate.

            quarter_wave_plate(self, 
                    channel_horizontal: int = 0, 
                    channel_vertical: int = 1, 
                    angle: float = 0, 
                    node_info: dict = None
                    ) -> None

                    Add a quarter wave plate to the fock state circuit with axis rotated over a given angle (in radians).
                    The phase shift of 'pi/2' will be applied to the channel for vertical polarization in the frame of 
                    reference of the wave plate.           

            quarter_wave_plate_225(self, 
                    channel_horizontal: int = 0, 
                    channel_vertical: int = 1, 
                    node_info: dict = None
                    ) -> None

                    Add a quarter wave plate at 22.5 degrees to the fock state circuit. The phase shift of 'pi/2' will be applied 
                    to the channel for vertical polarization in the frame of reference of the wave plate                            

            quarter_wave_plate_45(self, 
                    channel_horizontal: int = 0, 
                    channel_vertical: int = 1, 
                    node_info: dict = None
                    ) -> None

                    Add a quarter wave plate at 45 degrees to the fock state circuit. The phase shift of 'pi/2' will be applied 
                    to the channel for vertical polarization in the frame of reference of the wave plate.


        Method(s) to add beamsplitters and 'mixers' to the circuit:

            polarizing_beamsplitter(self, 
                    input_channels_a: tuple[int, int] = (0, 1), 
                    input_channels_b: tuple[int, int] = (2, 3), 
                    node_info: dict = None
                    ) -> None

                    Apply polarizing beamsplitter to the fock state circuit.
                    Horizontal polarization will be transmitted without loss or phase shift.
                    Vertical polarization is reflected without loss or phase shift.

            non_polarizing_beamsplitter(self, 
                    input_channels_a: tuple[int, int] = (0, 1), 
                    input_channels_b: tuple[int, int] = (2, 3), 
                    reflection: float = 0, 
                    transmission: float = 1, 
                    node_info: dict = None
                    ) -> None

                    Apply non-polarizing generic beamsplitter to the fock state circuit. Beamsplitter transmission/reflection can be set.
                    All channels will be mixed input_channels_a[0] with input_channels_b[0] and input_channels_a[1] with input_channels_b[1]
                
                    Note 1: The sum (reflection + transmission) has to be equal to 1, e.g., as in reflection=sin(theta)**2, tranmission=cos(theta)**2.
                    The reflection and transmissions are coefficients for 'probability' not for the 'amplitude' of the electrical field.      
                
                    Note 2: Typical use case is that input_channels_a would contain the horizontal and vertical polarization channels at
                    input port 'a' of the beamsplitter, and input_channels_b the hor. and vert. channels at input port 'b'. Howver also different
                    'channels' can be mixed by this beamsplitter. 

            mix_50_50(self, 
                    first_channel: int = 0, 
                    second_channel: int = 1, 
                    node_info: dict = None
                    ) -> None

                    Mix two channels 50%/50%. Phase shift is applied to the second of the channels.
            
            non_polarizing_50_50_beamsplitter(self, 
                    input_channels_a: tuple[int, int] = (0, 1), 
                    input_channels_b: tuple[int, int] = (2, 3), 
                    node_info: dict = None
                    ) -> None

                    Apply non-polarizing 50%/50% beamsplitter to the fock state circuit.
                    input_channels_a will be mixed with input_channels_b:
                    input_channels_a[0] with input_channels_b[0] and input_channels_a[1] with input_channels_b[1].
                    Typical use case is that input_channels_a would contain the horizontal and vertical polarization channels at
                    input port 'a' of the beamsplitter, and input_channels_b the hor. and vert. channels at input port 'b'.
            
            mix_generic_refl_transm(self, 
                    first_channel: int = 0, 
                    second_channel: int = 1, 
                    reflection: float = 0, 
                    transmission: float = 1, 
                    node_info: dict = None
                    ) -> None

                    Mix two channels with reflection and transmission percentages. Phase shift is applied to one of the channels.
                    The sum (reflection + transmission) has to be equal to 1, e.g., as in reflection=sin(theta)**2, tranmission=cos(theta)**2.
                    Note: reflection and transmissions are coefficients for 'probability' not for the 'amplitude' of the electrical field.

        Method(s) to swap channels:
        
           swap(self, 
                    first_channel: int = 0, 
                    second_channel: int = 1,
                    node_info: dict = None
                    ) -> None: 
            
                    Swap two channels without loss or phase shift.
                       
        Method(s) to apply a phase shift to single channel:
     
            phase_shift_single_channel(self, 
                    channel_for_shift=0, 
                    phase_shift=0, 
                    node_info: dict = None
                    ) -> None

                    Apply phase shift of 'phi' (in radians) to specified channel of the fock state circuit.                       

                    
        Method(s) to control the values in the classical channels:                   

           classical_channel_function(self, 
                    function, 
                    affected_channels: list[int] = None, 
                    new_input_values: list[int] = None, 
                    node_info: dict = None) -> None

                    Perform a calculation/operation on the classical channels only. Function should use the full list of channels 
                    as input and as output.

            set_classical_channels(self, 
                    list_of_values_for_classical_channels: list[int] = None, 
                    list_of_classical_channel_numbers: list[int] = None, 
                    node_info: dict = None
                    ) -> None

                    Sets values for classical channels. Input is a list of channels and a list of values to be written to the classical channel.
                    The two lists should have same length. First value in the list of values is written to the first channel in list of channels.

                    
        Method(s) to perform a measurement:   
            
            measure_optical_to_classical(self, 
                    optical_channels_to_be_measured: list[int] = [0], 
                    classical_channels_to_be_written: list[int] = [0], 
                    node_info: dict = None
                    ) -> None

                    Read photon number in a optical channel and write to classical channel. 

                    
        Method(s) to create an optical node which reads settings from the classical channels

            phase_shift_single_channel_classical_control(self, 
                    optical_channel_to_shift: int = 0, 
                    classical_channel_for_phase_shift: int = 0, 
                    node_info: dict = None
                    ) -> None

                    Apply phase shift to a specified channel of the fock state circuit. The phase shift (in radians) is read from 
                    the given classical channel.
                    Example: circuit.phase_shift_single_channel_classical_control(optical_channel_to_shift = 2, classical_channel_for_phase_shift = 1) 
                    would read classical channel '1'. If that channel has value 'pi' a phase shift of 'pi' is applied to optical channel 2

                    
        Method(s) to create custom nodes, beyond what is provided in the standard set

            custom_optical_node(self, 
                    matrix_optical, 
                    optical_channels: list[int], 
                    node_type: str = 'optical', 
                    node_info: dict = None
                    ) -> None

                    Apply a custom optical matrix to the circuit. The matrix has to be a 2x2 numpy array with numpy cdouble entries. The function does
                    NOT check whether the matrix is physically possible (i.e.,invertible, unitary).

            custom_fock_state_node(self,
                    custom_fock_state,
                    node_type: str = 'custom optical', 
                    node_info: dict = None
                    ) -> None
                    TO BE IMPLEMENTED

                    
        Method(s) to export to Qiskit and Qutip
            TO BE IMPLEMENTED
 
    """    
    _TYPES_OF_NODES = ['optical','custom optical','optical and classical combined','classical', 'measurement optical to classical']

    def __init__(self, length_of_fock_state: int = 2, 
                 no_of_optical_channels: int = 2, 
                 no_of_classical_channels: int = 0, 
                 channel_0_left_in_state_name: bool = True,
                 threshold_probability_for_setting_to_zero: float = 0.0001,
                 use_full_fock_matrix:bool = False
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

            The paramater 'threshold_probability_for_setting_to_zero' forces rounding to zero for probabilities below the given level. 

            Default the class optimizes calculations by reducing the size of the Fock state matrix to match the photon population
            (i.e., if states are empty they are discarded from the fock matrix). When the bool 'use_full_fock_matrix' is set to True 
            the system will always use the full size Fock matrix and skip the optimization.

        Args:
            length_of_fock_state (int, optional): _description_. Defaults to 2.
            no_of_optical_channels (int, optional): _description_. Defaults to 2.
            no_of_classical_channels (int, optional): _description_. Defaults to 0.
            channel_0_left_in_state_name (bool, optional): _description_. Defaults to True.
            threshold_probability_for_setting_to_zero (float, optional): _description_. Defaults to 0.0001.
            use_full_fock_matrix (bool, optional): _description_. Defaults to False.

        Raises:
            Exception: _description_
        """        
        
        # the length of the fock state is the number of possible photon numbers. So if the 'length' is 2 the maximum
        # number of photons is 1 (either 0 or 1 photons). If the length is 4 we can have 0,1,2 of 3 photons per channel.
        self._length_of_fock_state = length_of_fock_state

        # the number of channels defining the circuit.
        self._no_of_optical_channels = no_of_optical_channels
        self._no_of_classical_channels = no_of_classical_channels
        
        # we need at least a fock states with length 2 (0 or 1 photon) and two optical channels. Anything else is a 
        # trivial circuit with either zero photons or one channel without interaction.
        if self._length_of_fock_state < 1 or self._no_of_optical_channels < 2:
            raise Exception('length_of_fock_state minimal value is 1, no_of_optical_channels minumum value is 2')

        # for naming the states we need a convention. if 'channel_0_left_in_state_name' is set to 'True' we
        # write a state with 2 photons in channel 0 and 5 photons in channel 1 as '05'. With this value set
        # to 'False' we would write this same state as '50'. 
        self._channel_0_left_in_state_name = channel_0_left_in_state_name

        # '_digits_per_optical_channel' defines the number of digits used when 
        # writing a fock state as word. For more than 10 <= photons <100 per channel we 
        # need 2 digits per channel. For 100 or more need 3 digits.
        self._digits_per_optical_channel = len(str(self._length_of_fock_state-1))

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
        
        # list of nodes for the optical states. each item is a dict
        self.node_list = [] 
        
        # generate a list of all possible values in the optical channels 
        index_list = [index for index in range(0,length_of_fock_state**self._no_of_optical_channels)]
        self._list_of_fock_states = [[] for index in index_list]
        for _ in range(0,self._no_of_optical_channels):
            for index in range(len(index_list)):
                n = int(index_list[index]%length_of_fock_state)
                self._list_of_fock_states[index].append(n)
                index_list[index] = int(index_list[index]/length_of_fock_state)

    def conventions(self) -> str:
        """ Function to print return the conventions used in this class as a string

        Returns:
            str: String describing the conventions used in this class
        """        
        text = "Conventions used in FockStateCircuit\n"
        text += "1. Beamsplitters:\n"
        text += "For beamsplitters we use the convention for a lossless dielectric beamsplitter. \n"
        text += "See for instance https://en.wikipedia.org/wiki/Beam_splitter#Phase_shift \n"
        text += "This means the generic matrix is 1/sqrt(2) [[1, 1],[1, -1]].\n" 
        text += "The minus sign is applied to light entering port b at input side and being transmitted to port b\n"
        text += "at the output side. The user can translate to other conventions (i.e., symmetric convention)\n"
        text += "by adding single channel phase shifts.\n "
        text += "\n"
        text += "2. Phase plates:\n"
        text += "For phase plates the \'horizontal\' polarization (default channel 0) is not affected and the \n"
        text += "\'vertical\' polarization (default channel 1) receives the phase shift\n"
        text += "So a quarter wave plate at angle of 0 degrees would apply a phase shift of 90 degrees or \'1j\'\n"
        text += "to the vertical polarization. A half wave plate would apply a phase shift of 180 degrees or \'-1\'\n"
        text += "to the vertical polarization. See section below on the impact of the photon number in the channel."
        text += "\n"
        text += "If we orient a phase plate at an angle we can rotate polarization but also apply a phase shift\n"
        text += "For a half wave plate oriented at 45 degrees or pi/4 radians the input horizontal polarization\n"
        text += "state will be \'swapped\' with the input vertical polarization without any phase shifts.\n"
        text += "So |H> will become |V> and |V> will become |H>. If H is channel 0 and V is channel 1 the\n"
        text += "mapping of Fock states will be |nm> to |mn> where m and n are photon numbers in the channels.\n"
        text += "\n"
        text += "For a half wave plate oriented at 22.5 degrees or pi/8 radians we create a superposition between\n"
        text += "horizontal and vertical polarization. Input horizontal |H> will become 1/sqrt(2) (|H> + |V>) and\n"
        text += "input vertical |V> will become 1/sqrt(2) (|H> - |V>). Note the minus sign!\n"
        text += "Expressed in polarization rotation this means:\n"
        text += "A half wave plate oriented at +22.5 degree will rotate horizontal polarization over +45 degree\n"
        text += "and will rotate vertical polarization over 45 degree plus a 180 degree phase shift\n"
        text += "or (same in other words) will rotation vertical polarization over +225 degree or -135 degree\n"
        text += "\n"
        text += "If the half wave plate is oriented at -22.5 degrees or -pi/8 radians we get:\n"
        text += "Input horizontal |H> will become 1/sqrt(2) (|H> - |V>)\n"
        text += "Input vertical |V> will become 1/sqrt(2) (-|H> - |V>)\n"
        text += "Expressed in polarization rotation this means:\n"
        text += "A half wave plate oriented at -22.5 degree will rotate horizontal polarization over -45 degree\n"
        text += "and will rotate vertical polarization over -45 degree plus a 180 dgree phase shift\n"
        text += "or (same in other words) will rotation vertical polarization over -225 degree or +135 degree\n"
        text += "\n"
        text += "3. Phase shift for higher photon states (2 or more photons in a channel):\n"
        text += "Swap or polarization rotation over 90 degrees is not impacted by photon number. A half wave plate\n"
        text += "at 45 degree will map state |20> to |02> and state |02> to |20> if the swap is between channel 0\n"
        text += "and channel 1, or if channel 0 and 1 represent horizontal and vertical polarization.\n"
        text += "\n"
        text += "For higher photon states the phase shift will be affected by the photon number. As example:\n"
        text += "if we have a quarter wave plate at 0 degree angle we apply a phase shift of 90 degree or \'1j\'\n"
        text += "to the vertically polarized channel. This means state |01> will become 1j |01>\n"
        text += "However state |02> will become -1 |02> (remember 1j squared is -1), \n"
        text += "state |03> will become -1j |03> and |04> will become |04>  \n"
        text += "\n"
        text += "4. Detection orientation and polarizers\n"
        text += "We can rotate polarization by phase plates. The detectors are always oriented along the channels,\n"
        text += "this means we can only detect in the polarization orientation of channel 0, channel 1 etc at\n"
        text += "the location of the detectors in the circuit. If we have channel 0 horizontal and channel 1 vertical\n"
        text += "polarization and we want to detect behind a polarizer at 45 degree the method is to place a half-\n"
        text += "wave plate at 22.5 degree before the detector. The detected signal in channel 0 after the half-wave\n"
        text += "plate is then the signal we look for, and the detected signal in channel 1 is what is absorbed in the\n"
        text += "polarizer.\n"
        return text
    
    # region Code region defining the different elements that can be added to a circuit
    def __update_list_of_nodes(self, node_to_be_added: dict = None) -> None:

        """ Update the list of nodes for node_list. This is the single function to update the list of nodes which serves as
            as a gatekeeper. node to be added is a dictionary describing the node.

        Args:
            node_to_be_added (_type_, optional): dictionary containing node to be added to the node list. Defaults to None.

        Raises:
            Exception: If invalid node in 'node_to_be_added
        """        

        if (node_to_be_added is None or
            'node_type' not in node_to_be_added.keys() or
            node_to_be_added['node_type'] not in FockStateCircuit._TYPES_OF_NODES):
            raise Exception('Error when updating the list of nodes for the fock state circuit')

        if node_to_be_added['node_type'] == 'optical':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'optical non-linear':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'optical and classical combined':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'classical':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'measurement optical to classical':
            self.node_list.append(node_to_be_added)

        return
     
    def __generate_generic_optical_matrix(self, theta: float = 0, phi: float = 0): # -> np.array[np.cdouble]:

        """ Generate the generic 2x2 matrix for mixing two optical channels. 

        Args:
            theta (float): rotation of the axis
            phi (float): phase difference between the two axis
        
        Returns:
            2x2 matrix in numpy array format for cdouble entries
        
        Raises:
            nothing
        """
        e = math.cos(phi) + 1j * math.sin(phi)
        matrix = np.array([
            [math.cos(theta)**2 + e*math.sin(theta)**2, (1-e)*math.cos(theta)*math.sin(theta)],
            [(1-e)*math.cos(theta)*math.sin(theta), math.sin(theta)**2 + e*math.cos(theta)**2]
        ])
        return matrix
    
    def __translate_channels_numbers_to_tensor_list(self, channel_numbers: list[int] = []) -> list[int]:

        """ Generate a tensor list. The tensor list indicates to which optical channel a gate should be applied.
            Default the gates are applied to the optical channels with the lowest index (e.g., channel 0 for single channel
            gate, channels 0 and 1 for two channel gates). When we generate transition matrices we have to swap the columns
            and rows to make them apply to right channel. This information is contained in the 'tensor_list'.
            Example: If for a 4 channel circuit a gate should apply to channel 3 the tensor list is [3,1,2,0,4]
            Example: If for a 3 channel circuit a gate should apply to channel 2 and channel 1 the list is [2,1,0]
            (so if the gate operates on channel 0 by default it will operate on channel_numbers[0] by action of the tensor_list,
            if the gate operates on channel 1 by default it will operate on channel_numbers[1] by action of the tensor_list)

        Args:
            channel_numbers (list, optional): List of channel numbers to apply the gate to Defaults to [].

        Raises:
            Exception: if the gate requires two channels and the give channels in channel_numbers are the same
            Exception: if too many channels are given in channel_numbers

        Returns:
            list: tensor_list describing how to re-order the rows and columns in the fock matrix describing the node
        """        
        # create a tensor list as regularly ordered list of channel numbers [0,1,2,3..]
        tensor_list = [n for n in range(self._no_of_optical_channels)]
        if not channel_numbers or len(channel_numbers) == 0:
            # if no channel numbers are given return the default list [0,1,2,..]
            return tensor_list
        elif len(channel_numbers) == 1:
            # if there is only one channel number we swap channel number '0' and the given channel number
            channel_for_shift = channel_numbers[0]
            if channel_for_shift != 0:
                tensor_list[0], tensor_list[channel_for_shift] = tensor_list[channel_for_shift], tensor_list[0] 
        
        elif len(channel_numbers) == 2:
            channel_horizontal, channel_vertical = channel_numbers[0], channel_numbers[1]
            if channel_horizontal == channel_vertical:
                raise Exception('channel numbers identical where two different channel numbers are needed')
            
            if (channel_horizontal, channel_vertical) == (0,1):
                pass # tensor_list is already ordered correctly
            elif (channel_horizontal, channel_vertical) == (1,0):
                tensor_list[0], tensor_list[1] = tensor_list[1], tensor_list[0] #swap
            else:
                tensor_list[0], tensor_list[channel_horizontal] = tensor_list[channel_horizontal],  tensor_list[0]
                tensor_list[1], tensor_list[channel_vertical] = tensor_list[channel_vertical],  tensor_list[1]  
        
        else:
            raise Exception('error creating tensor list, more than two channels provided')
        
        return tensor_list
        

    def __apply_generic_rotation_2_channels(self, 
                                            channel_horizontal: int = 0,
                                            channel_vertical: int = 1,
                                            theta: float = 0, 
                                            phi: float = 0,
                                            node_type: str = 'optical',
                                            node_info: dict = None
                                            ) -> None:  
        
        """ Apply a generic rotation to channels over angles theta and phi.

        Args:
            channel_horizontal: channel number for horizontal polarization
            channel_vertical: channel number for vertical polarization
            theta: rotation of the axis in radians
            phi: phase difference between fast and slow axis in radians
        
        Returns:
            nothing
        
        Raises:
            nothing
        """
        channel_numbers = [channel_horizontal, channel_vertical]
        tensor_list = self.__translate_channels_numbers_to_tensor_list(channel_numbers = channel_numbers)
        matrix_optical = self.__generate_generic_optical_matrix(theta, phi)
        self.__update_list_of_nodes({'matrix_optical': matrix_optical, 'tensor_list':tensor_list, 'node_type': node_type, 'node_info': node_info})

        return
    
    def custom_optical_node(self, 
                            matrix_optical, #: np.array[np.cdouble], 
                            optical_channels: list[int],  
                            node_type: str = 'optical', 
                            node_info: dict = None
                            ) -> None:  
        
        """ Apply a custom optical matrix to the circuit. The matris has to be a 2x2 numpy array with numpy cdouble entries. The function does
            NOT check whether the matrix is physically possible (i.e.,invertible, unitary). 

        Args:
            matrix_optical (np.array[np.cdouble]): optical matrix to be applied
            optical_channels (list[int]): channels to apply the optical matrix to
            node_type (str): type of optical node ('optical' or 'optical non-linear' )
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.

        Returns:
            nothing
        
        Raises:
            Exception: If the matrix is not 2x2
        """
        if len(matrix_optical) != 2 and len(matrix_optical[0]) != 2:
            raise Exception('Only optical nodes with 2-channel interaction are implemented')
        channel_numbers = optical_channels
        tensor_list = self.__translate_channels_numbers_to_tensor_list(channel_numbers = channel_numbers)
        matrix_optical = matrix_optical
        self.__update_list_of_nodes({'matrix_optical': matrix_optical, 'tensor_list':tensor_list, 'node_type': node_type, 'node_info': node_info})

        return

    def half_wave_plate(self, 
                        channel_horizontal: int = 0, 
                        channel_vertical: int = 1, 
                        angle: float = 0, 
                        node_info: dict = None) -> None:  
        """ Add a half wave plate to the fock state circuit with axis rotated over angle. The phase shift of 'pi' will be applied to the 
            channel for vertical polarization.

        Args:
            channel_horizontal (int): channel number for horizontal polarization
            channel_vertical (int): channel number for vertical polarization
            angle (float): rotation of the axis in radians
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.

        Returns:
            nothing
        
        Raises:
            nothing
        """
        if node_info == None:
            node_info = {
                    'label' : "half-wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$\frac{\lambda}{2}$",''],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }

        self.__apply_generic_rotation_2_channels(channel_horizontal, channel_vertical, theta = angle, phi = math.pi, node_info = node_info)
        return
    
    def half_wave_plate_45(self, 
                           channel_horizontal: int = 0, 
                           channel_vertical: int = 1, 
                           node_info: dict = None) -> None:  
        """ Add a half wave plate at 45 degrees to the fock state circuit. The phase shift of 'pi' will be applied to the 
            channel for vertical polarization in the frame of reference of the wave plate.
        
        Args:
            channel_horizontal: channel number for horizontal polarization
            channel_vertical: channel number for vertical polarization
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.

        Returns:
            nothing
        
        Raises:
            nothing
        """
        if node_info == None:
            node_info = {
                    'label' : "half-wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$\frac{\lambda}{2}$",''],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }

        self.half_wave_plate(channel_horizontal, channel_vertical, angle = math.pi/4, node_info = node_info)
        return
    
    def half_wave_plate_225(self, 
                            channel_horizontal: int = 0, 
                            channel_vertical: int = 1, 
                            node_info: dict = None) -> None: 
        """ Add a half wave plate at 22.5 degrees to the fock state circuit. The phase shift of 'pi' will be applied to the 
            channel for vertical polarization in the frame of reference of the wave plate.
                
        Args:
            channel_horizontal: channel number for horizontal polarization
            channel_vertical: channel number for vertical polarization
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.

        Returns:
            nothing
        
        Raises:
            nothing
        
        """
        if node_info == None:
            node_info = {
                    'label' : "half-wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$\frac{\lambda}{2}$",''],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        self.half_wave_plate(channel_horizontal, channel_vertical, angle = math.pi/8, node_info = node_info)
        return
    
    def quarter_wave_plate(self, 
                           channel_horizontal: int = 0, 
                           channel_vertical: int = 1, 
                           angle: float = 0, 
                           node_info: dict = None) -> None:    
        """ Add a quarter wave plate to the fock state circuit with axis rotated over a given angle (in radians).
            The phase shift of 'pi/2' will be applied to the channel for vertical polarization in the frame of 
            reference of the wave plate.

        Args:
            channel_horizontal: channel number for horizontal polarization
            channel_vertical: channel number for vertical polarization
            angle: rotation of the axis in radians
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            nothing
        """
        if node_info == None:
            node_info = {
                    'label' : "Qtr-wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$\frac{\lambda}{4}$",''],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        self.__apply_generic_rotation_2_channels(channel_horizontal, channel_vertical, theta = angle, phi = math.pi/2, node_info = node_info)
        return
    
    def quarter_wave_plate_45(self, 
                              channel_horizontal: int = 0, 
                              channel_vertical: int = 1, 
                              node_info: dict = None) -> None:   
        """ Add a quarter wave plate at 45 degrees to the fock state circuit. The phase shift of 'pi/2' will be applied 
            to the channel for vertical polarization in the frame of reference of the wave plate.
                        
        Args:
            channel_horizontal: channel number for horizontal polarization
            channel_vertical: channel number for vertical polarization
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            nothing
        """  
        if node_info == None:
            node_info = {
                    'label' : "Qtr-wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$\frac{\lambda}{4}$",''],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        self.quarter_wave_plate(channel_horizontal, channel_vertical, angle = math.pi/4, node_info = node_info)
        return
    
    def quarter_wave_plate_225(self, 
                               channel_horizontal: int = 0, 
                               channel_vertical: int = 1, 
                               node_info: dict = None) -> None:     
        """ Add a quarter wave plate at 22.5 degrees to the fock state circuit. The phase shift of 'pi/2' will be applied 
            to the channel for vertical polarization in the frame of reference of the wave plate
                        
        Args:
            channel_horizontal: channel number for horizontal polarization
            channel_vertical: channel number for vertical polarization
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            nothing
        """
        if node_info == None:
            node_info = {
                    'label' : "Qtr-wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$\frac{\lambda}{4}$",''],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        self.quarter_wave_plate(channel_horizontal, channel_vertical, angle = math.pi/8, node_info = node_info)
        return
    
    def phase_shift_single_channel(self, channel_for_shift = 0, phase_shift = 0, node_info: dict = None) -> None:    
        """ Apply phase shift of 'phi' (in radians) to specified channel of the fock state circuit.
                        
        Args:
            channel_for_shift: channel to apply phase shift to
            phase_shift: phase shift in radians to be applied
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            nothing
        """ 
        if node_info == None:
            node_info = {
                    'label' : "phase-shift",
                    'channels' : [channel_for_shift],
                    'markers' : ['o'],
                    'color' : ['pink'],
                    'markerfacecolor' : ['pink'],
                    'market_text' : [r"$\phi$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        if channel_for_shift == 0:
            (channel_horizontal, channel_vertical) = 1, 0
        else:
            (channel_horizontal, channel_vertical) = 0, channel_for_shift
            
        # we use a generic rotation with theta is zero, so phase shift is applied to the second ('the vertical') channel. 
        # so channel_to_be_shifted has to be the vertical channel. Another option would be to rotate the phase plate over 
        # 90 degrees with theta and apply phase shift to horizontal channel
        self.__apply_generic_rotation_2_channels(channel_horizontal = channel_horizontal,
                                                channel_vertical = channel_vertical,
                                                theta = 0,
                                                phi = phase_shift,
                                                node_info = node_info)
        return
    
    def phase_shift_single_channel_classical_control(self, 
                                                     optical_channel_to_shift: int = 0, 
                                                     classical_channel_for_phase_shift: int = 0, 
                                                     node_info: dict = None) -> None:   
        """ Apply phase shift to a specified channel of the fock state circuit. The phase shift (in radians) is read from 
            the given classical channel.
            Example: circuit.phase_shift_single_channel_classical_control(optical_channel_to_shift = 2, classical_channel_for_phase_shift = 1) 
            would read classical channel '1'. If that channel has value 'pi' a phase shift of 'pi' is applied to optical channel 2
                        
        Args:
            optical_channel_to_shift: channel to apply phase shift to
            classical_channel_for_phase_shift: classical control channel to read phase shift from
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            exception if the requested channels are not available
        """ 
        if classical_channel_for_phase_shift > self._no_of_classical_channels:
            raise Exception('the classical channel is a not defined channel in the circuit')
        if optical_channel_to_shift > self._no_of_optical_channels:
            raise Exception('the optical channel is a not defined channel in the circuit')

        if node_info == None:
            node_info = {
                    'label' : "phase-shift(c)",
                    'channels' : [optical_channel_to_shift],
                    'channels_classical' : [classical_channel_for_phase_shift],
                    'classical_use_case': 'read', 
                    'markers' : ['o'],
                    'color' : ['pink'],
                    'markerfacecolor' : ['pink'],
                    'market_text' : [r"$\phi$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        # we use a generic rotation with theta is zero, so phase shift is applied to the second ('the vertical') channel. 
        # so channel_to_be_shifted has to be the vertical channel. Another option would be to rotate the phase plate over 
        # 90 degrees with theta and apply phase shift to horizontal channel
        # channel_for_shift has to become the 'channel_vertical'. This is the channel to which the phase shift is applied.
        if optical_channel_to_shift == 0:
            channel_horizontal, channel_vertical = 1,0
        else:
            channel_horizontal, channel_vertical = 0,optical_channel_to_shift

        channel_numbers = [channel_horizontal, channel_vertical]
        tensor_list = self.__translate_channels_numbers_to_tensor_list(channel_numbers = channel_numbers)

        control_parameters =    {'variables_to_be_controlled' : ['phi'],
                                'function_parameters'  : {'theta': 0, 'phi': 0},
                                'classical_control_channels' : [classical_channel_for_phase_shift]
                                } 

        node_to_be_added = {'function' : '__generic_optical_matrix(theta, phi)',
                                'control_parameters': control_parameters,
                                'tensor_list' : tensor_list,
                                'node_type' : 'optical and classical combined',
                                'node_info' : node_info
                                }
        #self.__apply_generic_rotation_2_channels(channel_horizontal, channel_for_shift, theta = 0, phi = phase_shift, node_info = node_info)

        self.__update_list_of_nodes(node_to_be_added)

        return
    
    def wave_plate_classical_control(self, optical_channel_horizontal: int = 0, 
                                     optical_channel_vertical: int = 1, 
                                     classical_channel_for_orientation: int = 0, 
                                     classical_channel_for_phase_shift: int = 0,
                                     node_info: dict = None) -> None:   
        """ Apply a wave plate to specified channels of the fock state circuit getting the phase shift and plate orientation 
            from a the classical channels. 
            Example: wave_plate_classical_control(self, channel_horizontal = 3, 
                        channel_vertical = 2, 
                        classical_channel_for_orientation = 0, 
                        classical_channel_for_phase_shift = 2)
            would read a orientation angel 'theta' from classical channel 0 and a phase shift 'phi' from classical channel 2. This
            would be applied as a wave plate to optical channels 3 and 2).
                        
        Args:
            optical_channel_horizontal: channel number for horizontal polarization
            optical_channel_vertical: channel number for vertical polarization
            classical_channel_for_orientation: classical control channel to read orientation angle 'theta' (radians) from
            classical_channel_for_phase_shift: classical control channel to read phase shift 'phi' (radians) from
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            exception if the requested channels are not available
        """ 
        if max(classical_channel_for_phase_shift, classical_channel_for_orientation) > self._no_of_classical_channels:
            raise Exception('the classical channel is a not defined channel in the circuit')
        if max(optical_channel_horizontal, optical_channel_vertical) > self._no_of_optical_channels:
            raise Exception('the optical channel is a not defined channel in the circuit')
        
        if node_info == None:
            node_info = {
                    'label' : "wave_plate(c)",
                    'channels' : [optical_channel_horizontal, optical_channel_vertical],
                    'channels_classical' : [classical_channel_for_phase_shift, classical_channel_for_orientation],
                    'classical_use_case': 'read', 
                    'markers' : ['o'],
                    'color' : ['pink'],
                    'markerfacecolor' : ['pink'],
                    'market_text' : [r"$\theta$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        
        # add to list of nodes
        channel_numbers = [optical_channel_horizontal, optical_channel_vertical]
        tensor_list = self.__translate_channels_numbers_to_tensor_list(channel_numbers = channel_numbers)

        control_parameters =    {'variables_to_be_controlled' : ['theta', 'phi'],
                                'function_parameters'  : {'theta': math.pi/2, 'phi': 0},
                                'classical_control_channels' : [classical_channel_for_orientation, classical_channel_for_phase_shift]
                                } 

        node_to_be_added = {'function' : '__generic_optical_matrix(theta, phi)',
                                'control_parameters': control_parameters,
                                'tensor_list' : tensor_list,
                                'node_type' : 'optical and classical combined',
                                'node_info' : node_info
                                }

        self.__update_list_of_nodes(node_to_be_added = node_to_be_added)

        return
    
    def measure_optical_to_classical(self, 
                                     optical_channels_to_be_measured: list[int] = [0], 
                                     classical_channels_to_be_written: list[int] = [0], 
                                     node_info: dict = None) -> None: 
        """ Read photon number in a optical channel and write to classical channel.
                        
        Args:
            optical_channels_to_be_measured (list[int]): list of of optical channel numbers to be measured
            classical_channels_to_be_written (list[int]): list of classical channel numbers to write the measurement result to
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Raises:
            Exception: when list of optical channels and list of classical channels are not of equal length
        """ 
        optical_channels_allowed = all([( 0 <= c < self._no_of_optical_channels) for c in optical_channels_to_be_measured])
        classical_channels_allowed = all([( 0 <= c < self._no_of_classical_channels) for c in classical_channels_to_be_written])
        if len(classical_channels_to_be_written) != len(optical_channels_to_be_measured) or not optical_channels_allowed or not classical_channels_allowed:
            raise Exception('error in defining measurement from optical to classical')
    
        if node_info == None:
            node_info = {
                    'label' : "Measurement",
                    'channels' : optical_channels_to_be_measured,
                    'channels_classical'  : classical_channels_to_be_written,
                    'classical_use_case': 'measure', 
                    'color' : ['black'],
                    'markerfacecolor' : ['black'],
                    'market_text' : [r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        
        node_to_be_added = {
            'optical_channels_to_be_read' : optical_channels_to_be_measured,
            'classical_channels_to_be_written' : classical_channels_to_be_written,
            'node_type' : 'measurement optical to classical', 
            'node_info' : node_info
        }

        self.__update_list_of_nodes(node_to_be_added = node_to_be_added)
        return
    
    def classical_channel_function(self, 
                                   function, 
                                   affected_channels: list[int] = None, 
                                   new_input_values: list[int] = None, 
                                   node_info: dict = None) -> None: 
        """ Perform a calculation/operation on the classical channels only. Function should use the full list of channels 
            as input and as output.
            
            Example function to inverse the order of classical channels:
                
                def test_function(current_values, new_input_values, affected_channels):
                    output_list = current_values[::-1]
                    return output_list
            
            Example function to set values in the classical channels to fixed values
                
                def function(current_values, new_input_values, affected_channels):
                    for index, channel in enumerate(affected_channels):
                        current_values[channel] = new_input_values[index]
                    return current_values
                        
        Args:
            function: function using list of classical channels as input and as output
            affected_channels (list[int]): = None, 
            new_input_values (list[int]): = None, 
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            nothing
        """ 
        if node_info == None:
            node_info = {
                    'label' : "class-fnct",
                    'channels' : [],
                    'channels_classical' : [channel for channel in range(self._no_of_classical_channels)],
                    'classical_use_case': 'only', 
                    'color' : ['black'],
                    'markerfacecolor' : ['black'],
                    'market_text' : [],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        if affected_channels == None:
            affected_channels = []
        
        if new_input_values == None:
            new_input_values = []
        
        node_to_be_added = {
            'function' : function,
            'affected_channels' : affected_channels,
            'new_input_values' : new_input_values,
            'node_type' : 'classical', 
            'node_info' : node_info
        }
        self.__update_list_of_nodes(node_to_be_added = node_to_be_added)
  
        return
    
    def set_classical_channels(self, 
                               list_of_values_for_classical_channels: list[int] = None, 
                               list_of_classical_channel_numbers: list[int] = None, 
                               node_info: dict = None) -> None:
        """ Sets values for classical channels. Input is a list of channels and a list of values to be written to the classical channel.
            The two lists should have same length. First value in the list of values is written to the first channel in list of channels.
    
        Args:
            list_of_values_for_classical_channels (list[int]): list of values
            list_of_classical_channel_numbers (list[int]): list of channel numbers
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used
        
        Returns:
            nothing

        Raises:
            exception is channel number is/are invalid or list of values does not match channels
        """
        conditions = [
            len(list_of_values_for_classical_channels) != len(list_of_classical_channel_numbers),
            max(list_of_classical_channel_numbers) >= self._no_of_classical_channels
            ]
        if any(conditions):
            raise Exception('Incorrect input in function set_classical_channels')
        
        def function(classical_channel_values, list_of_values_for_classical_channels, list_of_classical_channel_numbers ):
            for index, channel in enumerate(list_of_classical_channel_numbers):
                classical_channel_values[channel] = list_of_values_for_classical_channels[index]
            return classical_channel_values
        self.classical_channel_function(function, 
                                        affected_channels=list_of_classical_channel_numbers, 
                                        new_input_values=list_of_values_for_classical_channels,
                                        node_info = node_info)
        
        return
    
    def swap(self, 
             first_channel: int = 0, 
             second_channel: int = 1,
             node_info: dict = None) -> None: 
        """ Swap two channels without loss or phase shift.
        
        Args:
            first_channel (int): channel number for swap
            second_channel (int): channel number for swap
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            nothing
        """
        if node_info == None:
            node_info = {
                    'label' : "Swap gate",
                    'channels' : [first_channel, second_channel],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$\triangledown$",r"$\triangle$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        self.__apply_generic_rotation_2_channels(first_channel, second_channel, theta = math.pi/4, phi = math.pi, node_info = node_info)
        return
    
    def mix_50_50(self, 
                  first_channel: int = 0, 
                  second_channel: int  = 1, 
                  node_info: dict = None
                  ) -> None: 
        """ Mix two channels 50%/50%. Phase shift is applied to the second of the channels.
        
        Args:
            first_channel (int): channel number 
            second_channel (int): channel number 
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
        
        Returns:
            nothing
        
        Raises:
            nothing
        """
        if node_info == None:
            node_info = {
                    'label' : "Mix gate",
                    'channels' : [first_channel, second_channel],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        self.__apply_generic_rotation_2_channels(first_channel, second_channel, theta = math.pi/8, phi = math.pi, node_info = node_info)
        return

    def mix_generic_refl_transm(self, 
                                first_channel: int = 0, 
                                second_channel: int = 1, 
                                reflection: float = 0, 
                                transmission: float = 1, 
                                node_info: dict = None) -> None:    
        """ Mix two channels with reflection and transmission percentages. Phase shift is applied to one of the channels.
            The sum (reflection + transmission) has to be equal to 1, e.g., as in reflection=sin(theta)**2, tranmission=cos(theta)**2.
            Note: reflection and transmissions are coefficients for 'probability' not for the 'amplitude' of the electrical field.
        
        Args:
            first_channel (int): channel number 
            second_channel (int): channel number 
            reflection (float): reflection fraction
            transmission (float): transmission fraction
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.

        Raises:
            Exception: If (reflection + transmission) is not 1
        """
        if node_info == None:
            node_info = {
                    'label' : "Mix",
                    'channels' : [first_channel, second_channel],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        if reflection + transmission != 1:
            raise Exception(
                'For generic mixing the sum or reflection and transmission has to be 1 and both have to between 0 and 1 (including)'
                )
        theta = np.arctan2(math.sqrt(reflection),math.sqrt(transmission))
        self.__apply_generic_rotation_2_channels(first_channel, second_channel, theta = theta, phi = math.pi, node_info = node_info)
        return
    
    def polarizing_beamsplitter(self, 
                                input_channels_a: tuple[int,int] = (0,1), 
                                input_channels_b: tuple[int,int] = (2,3), 
                                node_info: dict = None) -> None: 
        """ Apply polarizing beamsplitter to the fock state circuit.
            Horizontal polarization will be transmitted without loss or phase shift.
            Vertical polarization is reflected without loss or phase shift.
                                
        Args:
            input_channels_a (tuple[int,int]): tuple with channel numbers for input port a. First index is hor., 2nd index vert. polarization
            input_channels_b (tuple[int,int]): tuple with channel numbers for input port b. First index is hor., 2nd index vert. polarization
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used
        
        Raises:
            Exception: If number of channels in the circuit does not match requirement for the operation
        """
        if node_info == None:
            node_info = {
                    'label' : "PBS",
                    'channels' : [input_channels_a[1], input_channels_b[1]],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        if self._no_of_optical_channels < 4:
            raise Exception('For beamsplitter 4 channels are needed in the circuit')
        self.swap(first_channel = input_channels_a[1], second_channel = input_channels_b[1], node_info = node_info)
        return
    
    def non_polarizing_50_50_beamsplitter(self, 
                                          input_channels_a: tuple[int,int] = (0,1), 
                                          input_channels_b: tuple[int,int] = (2,3), 
                                          node_info: dict = None) -> None:    
        """ Apply non-polarizing 50%/50% beamsplitter to the fock state circuit.
            input_channels_a will be mixed with input_channels_b:
            input_channels_a[0] with input_channels_b[0] and input_channels_a[1] with input_channels_b[1].
            Typical use case is that input_channels_a would contain the horizontal and vertical polarization channels at
            input port 'a' of the beamsplitter, and input_channels_b the hor. and vert. channels at input port 'b'.
                                
        Args:
            input_channels_a (tuple[int,int]): tuple with channel numbers for input port a
            input_channels_b (tuple[int,int]): tuple with channel numbers for input port b
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used
        
        Raises:
            Exception: If number of channels in the circuit does not match requirement for the operation
        """
        if node_info == None:
            node_info_0 = {
                    'label' : "NPBS",
                    'channels' : [input_channels_a[0], input_channels_b[0]],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
            node_info_1 = {
                    'label' : "NPBS",
                    'channels' : [input_channels_a[1], input_channels_b[1]],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        if self._no_of_optical_channels < 4:
            raise Exception('For beamsplitter 4 channels are needed in the circuit')
        self.mix_50_50(first_channel = input_channels_a[0], second_channel = input_channels_b[0], node_info = node_info_0)
        self.mix_50_50(first_channel = input_channels_a[1], second_channel = input_channels_b[1], node_info = node_info_1)
        return
    
    def non_polarizing_beamsplitter(self, 
                                    input_channels_a: tuple[int,int] = (0,1), 
                                    input_channels_b: tuple[int,int] = (2,3), 
                                    reflection: float = 0, 
                                    transmission: float = 1, 
                                    node_info: dict = None
                                    ) -> None:     
        """ Apply non-polarizing generic beamsplitter to the fock state circuit. Beamsplitter transmission/reflection can be set.
            All channels will be mixed input_channels_a[0] with input_channels_b[0] and input_channels_a[1] with input_channels_b[1]

            Note 1: The sum (reflection + transmission) has to be equal to 1, e.g., as in reflection=sin(theta)**2, tranmission=cos(theta)**2.
            The reflection and transmissions are coefficients for 'probability' not for the 'amplitude' of the electrical field.      

            Note 2: Typical use case is that input_channels_a would contain the horizontal and vertical polarization channels at
            input port 'a' of the beamsplitter, and input_channels_b the hor. and vert. channels at input port 'b'. Howver also different
            'channels' can be mixed by this beamsplitter.      
        
        Args:
            input_channels_a (tuple[int,int]): tuple with channel numbers for input port a
            input_channels_b (tuple[int,int]): tuple with channel numbers for input port b
            reflection (float): reflection fraction
            transmission (float): transmission fraction
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used
        
        Raises:
            Exception if number of channels in the circuit does not match requirement for the operation
        """
        if node_info == None:
            node_info_0 = {
                    'label' : "NPBS",
                    'channels' : [input_channels_a[0], input_channels_b[0]],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
            node_info_1 = {
                    'label' : "NPBS",
                    'channels' : [input_channels_a[1], input_channels_b[1]],
                    'markers' : ['s'],
                    'color' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'market_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        if self._no_of_optical_channels < 4:
            raise Exception('For beamsplitter 4 channels are needed in the circuit')
        self.mix_generic_refl_transm(
            first_channel = input_channels_a[0], 
            second_channel = input_channels_b[0], 
            reflection = reflection, 
            transmission = transmission,
            node_info = node_info_0
            )
        self.mix_generic_refl_transm(
            first_channel = input_channels_a[1], 
            second_channel = input_channels_b[1], 
            reflection = reflection, 
            transmission = transmission,
            node_info = node_info_1
            )
        return
    # endregion
 
    # region Code region defining the functions for evaluating the circuit
    def evaluate_circuit(self, 
                         collection_of_states_input: cos.CollectionOfStates = None, 
                         nodes_to_be_evaluated: list = ['all']
                         ) -> cos.CollectionOfStates:
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
            collection_of_states_input (cos.CollectionOfStates, optional): Defaults to None.
            nodes_to_be_evaluated (list[int], optional): Defaults to ['all'].

        Raises:
            Exception: Exception will be triggered does not match with the nodes in the circuit
            Exception: Exception will be triggered if the nodes in the circuit contain an invalid node type
            Exception: Exception will be triggered if if the optical node with classical control is of unknown type

        Returns:
            cos.CollectionOfStates : collection of states as output of the circuit
        """        
        
        # if nodes_to_be_evaluated == ['all'] make a list with all indices in node_list
        if nodes_to_be_evaluated == ['all']:
            nodes_to_be_evaluated = list(range(len(self.node_list)))
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
            collection_of_states_input = cos.CollectionOfStates(self)

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
        # case where next node is of type 'measurement optical to classical'  *
        #**********************************************************************
        elif self.node_list[current_node_index].get('node_type') == 'measurement optical to classical':

            # generate a number as a string to identify whether this is first, seconde, third etc measurement
            measurement_identifier = self.__get_identifier_for_measurement(current_node_index)
                      
            # get parameters from node on what channels to measure and where to write the results
            optical_channels_to_be_read = self.node_list[current_node_index].get('optical_channels_to_be_read')
            classical_channels_to_be_written = self.node_list[current_node_index].get('classical_channels_to_be_written')  

            # prepare a new collection of states which will be filled with the results of the measurements
            # on the states in the input collection of states          
            collection_of_states_output = cos.CollectionOfStates(self, input_collection_as_a_dict=dict([]))
          
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
        # case where next node is of type 'optical and classical combined'
        #*****************************************************************
        elif self.node_list[current_node_index].get('node_type') == 'optical and classical combined':

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
            collection_of_states_output = cos.CollectionOfStates(self, input_collection_as_a_dict=dict([]))

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
                optical_matrix = self.__generate_generic_optical_matrix(theta, phi)
                
                # generate custom node from which to calculate the fock state matrix
                tensor_list = self.node_list[current_node_index].get('tensor_list')
                single_custom_node = {'matrix_optical':optical_matrix, 'tensor_list': tensor_list}

                # make a collection for just one state
                collection_of_single_state = cos.CollectionOfStates(self, input_collection_as_a_dict={identifier:state})

                collection_of_single_state_output = self.__apply_node_or_nodes_to_collection(
                    collection_of_states = collection_of_single_state,
                    nodes_to_be_evaluated = None,
                    single_custom_node = single_custom_node)
                    
                for identifier2, state2 in collection_of_single_state_output.items():
                    collection_of_states_output.add_state(state = state2, identifier = identifier2)
            
            return self.evaluate_circuit(collection_of_states_input = collection_of_states_output, nodes_to_be_evaluated = remaining_nodes)

        raise Exception('Error in evaluate_circuit. Check if node_types are correct')
        return
 
    def __next_state_marker(self) -> str:
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
                1 if self.node_list[node_index].get('node_type') == 'measurement optical to classical' else 0 
                for node_index in range(len(self.node_list))
                ])
        string_formatter = "{:0" + str(len(str(total_number_of_measurement_nodes_in_the_circuit)))+"d}"

        measurement_number = sum([
                1 if self.node_list[node_index].get('node_type') == 'measurement optical to classical' else 0 
                for node_index in range(current_node_index+1)
                ])

        return string_formatter.format(measurement_number)

    def _perform_measurement(self,
                            state: cos.State,
                            optical_channels_to_measure: list = [],
                            classical_channels_to_write_to: list = []
                            ) -> cos.CollectionOfStates:
        
        """ Perform a measurement on a single state, where the given optical channels are measured and the result is written
            to the given classical channels. The function will return a collection of states, one state for each possible
            measurement outcome. The optical states will be the 'collapsed states' for the measurement outcome. The resulting 
            collection represents a 'statistical mixture' with as weights for each state the likelihood of the corresponding measurement
            result.

            List for optical and classical channels should be equally long to avoid unpredictable result

        Args:
            state (cos.State): input state to be measured.
            optical_channels_to_measure (list, optional): optical channels to measure. list should have equal length list
                for classical channels to write to
            classical_channels_to_write_to (list, optional): classical channels to write the optical measurements to

        Returns:
            cos.CollectionOfStates: collection of states after measurement of single input state
        """        

        # outcomes will be a dictionary with as key the possible measurement outcomes and as value a list of 
        # indices for the components that lead to that outcome. The index indicates the place of the components
        # in the self._self._list_of_fock_states.
        outcomes = dict([])
        string_format_in_state_as_word = "{:0"+str(self._digits_per_optical_channel)+ "d}"
        # deep copy existing values of the classical values
        classical_channel_values_before_measurement = [val for val in state.classical_channel_values]

        vector_in, basis = state.translate_state_components_to_vector()

        for component_index, amplitude in enumerate(vector_in):

            probability = np.abs(amplitude)**2
            
            # if amplitude of the component is zero move to next
            if probability < self._threshold_probability_for_setting_to_zero:
                continue
            
            # get the photon count in the optical component
            values = self._list_of_fock_states[component_index]

            # determine the new values in classical channels.
            # first load with the old values, later on overwrite with new values
            classical_channel_after_measurement = [val for val in classical_channel_values_before_measurement]

            # if the number of classical channels is larger than the number of opticl channels limit the number
            # of classical channels. 
            if len(classical_channels_to_write_to) > len(optical_channels_to_measure):
                classical_channels_to_write_to = classical_channels_to_write_to[:len(optical_channels_to_measure)]
            measurement_result = ''
            for channel_index, classical_channel in enumerate(classical_channels_to_write_to):
                optical_channel = optical_channels_to_measure[channel_index]
                classical_channel_after_measurement[classical_channel] = values[optical_channel]
                measurement_result += string_format_in_state_as_word.format(values[optical_channel])

            # make a list of optical components that contribute to a given measurement result
            if measurement_result not in outcomes.keys():
                outcomes[measurement_result] = {'probability' : probability,
                                                'results' :classical_channel_after_measurement,
                                                'set_of_component_indices': set([component_index])
                                                }
            else:
                outcomes[measurement_result]['probability'] += probability
                outcomes[measurement_result]['set_of_component_indices'].add(component_index) 
        
        # for each measurement result from this state, for this specific measurement there is a key-value pair in the dictionary 'outcomes'
        # the values are the probability to get that results (so these should ideally add up to 1 when adding up all possible measurement outcomes)
        # and the list of indices indicated the components, or basis states leading to that outcome.

        # Next we need to make the 'collapsed state' for each measurement outcome. All components contributing to that outcome should be in 
        # that collapsed state, the components giving a different outcome should get amplitude zero. The result is a collection of states where 
        # there is one state in the collection for each measurement outcome. 
        # The cumulative_weight is the likelihood to get to a measurement result, the optical components all contribute to that result and the 
        # probabilities for the components in each state again add up to one.

        # create an empty collection of states with for same circuit
        collection_of_states_after_measurement = cos.CollectionOfStates(fock_state_circuit = self, input_collection_as_a_dict=dict([]))

        # loop over all measurement results and create a new, collapsed state for each result.
        for measurement_result, components_with_that_result in outcomes.items():
            
            # make a collapsed state for each outcome
            collapsed_state = state.create_initialized_state()

            # prepare the output 'vector' to generate the optical state. First set all to zero.
            vector_out = [np.cdouble(0)] * len(vector_in)
            
            # loop over all components contributing to a particular outcomes
            for component_index in components_with_that_result['set_of_component_indices']:
                               
                # the cumulative probability for this state in the collection is equal to the likelihood of getting to a 
                # measurement result
                cumulative_probability = components_with_that_result['probability']
                if cumulative_probability != 0:
                    # load amplitudes for remaining components in the output vector for optical state
                    scale_factor = math.sqrt((1/cumulative_probability))
                    vector_out[component_index] = vector_in[component_index] * scale_factor
                
            # the classical channels contain the measurement result
            new_classical_channel_values = [i for i in components_with_that_result['results']]

            # prepare the attribute 'measurement_results' for the collapsed state. If there are already
            # measurement results in the input state copy these to the new collapsed state
            if state.measurement_results and state.measurement_results is not None:
                measurement_results = [previous_values for previous_values in state.measurement_results]   
                measurement_results.append({'measurement_results':new_classical_channel_values, 'probability': cumulative_probability})   
            else:
                measurement_results = [{'measurement_results':new_classical_channel_values, 'probability': cumulative_probability}]

            if state.cumulative_probability is not None:
                cumulative_probability = state.cumulative_probability * cumulative_probability
            else:
                cumulative_probability = cumulative_probability
            

            collapsed_state.initial_state = state.initial_state
            collapsed_state.cumulative_probability = cumulative_probability
            # use the vector to populate the attribute 'optical_components' from the collapsed state
            collapsed_state.set_state_components_from_vector(vector_out)
            collapsed_state.classical_channel_values = new_classical_channel_values
            collapsed_state.measurement_results = measurement_results

            collection_of_states_after_measurement.add_state(state=collapsed_state, identifier=measurement_result)

        return collection_of_states_after_measurement
    
    def __apply_node_or_nodes_to_collection(self,
                                          collection_of_states: cos.CollectionOfStates = None,
                                          nodes_to_be_evaluated: list[int] = ["all"],
                                          single_custom_node = None) -> cos.CollectionOfStates:
        
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
            collection_of_states = cos.CollectionOfStates(fock_state_circuit=self)
        else:
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
            states_grouped_by_photon_number = collection_of_states.group_states_together_by_photon_number()

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
        fock_matrix = np.zeros((length**2, length**2), dtype = np.csingle) # the matrix is pre-filled as diagonal matrix
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

    # endregion

    # region Code region for drawing the circuit

    def draw(self, print_defaults: bool = False, settings_for_drawing_circuit: dict = None) -> None:
        """ Draw the optical circuit. self. settings_for_drawing_circuit is the dict with settings for circuit drawing. 
            If this dict is not defined default values will be taken. If the boolean print_defaults is set to True the function will print
            out the default values to console
        
        Args:
            print_defaults (bool): If 'True' function will print default settings to console
            ettings_for_drawing_circuit (dict): dict with settings for circuit drawing. If none given default will be taken.

        Returns:
            nothing
        
        Raises:
            nothing
        """
        # default settings for circuit drawing (can be overwritten when calling function)
        circuit_draw_settings_dict_default = {
                'channel_line_spacing': 5,
                'channel_line_length': 80,
                'nodes_per_page': 8,
                'spacing_between_nodes' : 10, 
                'canvas_width': 100,
                'plt.figure.figsize': [15, 6], 
                'circuit_title': 'Optical circuit',
                'channel_labels': ['optical '+ str(index) for index in range(self._no_of_optical_channels)],
                'channel_labels_classical': ['classical '+ str(index) for index in range(self._no_of_classical_channels)],
                'channel_label_string_max_length': 15,
                'node_label_string_max_length': 15
            }
        # default settings for drawing nodes in the circuit (can be overwritten when adding node to circuit)
        default_node_info = {
            'label' : '',
            'connection_linestyle' : 'solid',
            'connection_linewidth': 2,
            'connection_linecolor': 'blue',
            'channels' : [],
            'channels_classical': [],
            'markers' : ['o'],
            'markercolor' : ['blue'],
            'markerfacecolor' : ['white'],
            'market_text' : [''],
            'markersize' : 20,
            'fillstyle' : 'full'
        }
        if print_defaults == True:

            print("These are the default setting for drawing a circuit stored as: \n'circuit_draw_settings_dict_default'")
            print("The defaults can be overwritten by setting the variable circuit.settings_for_drawing_circuit")
            print("Here 'circuit' is the name of your FockStateCircuit ")
            print("----------------------------------------------------- ")
            for item in circuit_draw_settings_dict_default.items():
                print(item)
            print("\n")
            print("These are the default setting for drawing a node stored as: \n'default_node_info'")
            print("The defaults can be overwritten by setting the variable circuit.settings_for_drawing_node")
            print("Here 'circuit' is the name of your FockStateCircuit ")
            print("----------------------------------------------------- ")
            for item in default_node_info.items():
                print(item)
        # if no settings for circuit drawing are passed use default
        if not settings_for_drawing_circuit:
            circuit_draw_settings_dict = circuit_draw_settings_dict_default       
        else:
            # check whether all settings are passed, if not add the default value
            circuit_draw_settings_dict = settings_for_drawing_circuit
            for key in circuit_draw_settings_dict_default.keys():
                if key not in circuit_draw_settings_dict.keys():
                    circuit_draw_settings_dict.update({key:circuit_draw_settings_dict_default[key] })

        # determine how many pages are needed to draw the complete circuit
        number_of_nodes = len(self.node_list)
        nodes_on_last_page = number_of_nodes%circuit_draw_settings_dict['nodes_per_page']       
        if nodes_on_last_page == 0:
            number_of_pages = int(number_of_nodes/circuit_draw_settings_dict['nodes_per_page'])
        else:
            number_of_pages = 1 + int(number_of_nodes/circuit_draw_settings_dict['nodes_per_page'])

        # determine the height needed per page, depending on the number of channels and the spacing
        canvas_height = circuit_draw_settings_dict['channel_line_spacing']*(self._no_of_optical_channels+self._no_of_classical_channels+ 1)

        fig_width = circuit_draw_settings_dict['plt.figure.figsize'][0]
        fig_height = circuit_draw_settings_dict['plt.figure.figsize'][1]*canvas_height/40
        plt.rcParams['figure.figsize'] = [fig_width,fig_height]
        fig, ax = plt.subplots(nrows= number_of_pages, ncols=1)
        fig.suptitle(circuit_draw_settings_dict['circuit_title'])
        
        # draw page by page
        for page_number in range(number_of_pages):

            if number_of_pages == 1:
                axis = ax
            else:
                axis = ax[page_number]

            # make an 'invisible' curve to size the canvas from (0,0) to (..dict['canvas_width'], canvas_height )
            axis.axis('off') #axis invisible, border invisible
            xpoints = [0,circuit_draw_settings_dict['canvas_width']] 
            ypoints = [0,canvas_height]            
            axis.plot(xpoints,ypoints,'o:r', alpha=0) # alpha = 0 means invisible

            # draw an horizontal line for each channel
            line_start_x = (circuit_draw_settings_dict['canvas_width']-circuit_draw_settings_dict['channel_line_length'])/2
            line_end_x = line_start_x + circuit_draw_settings_dict['channel_line_length']
            line_y_values = [canvas_height - (line_no+1)*circuit_draw_settings_dict['channel_line_spacing'] for line_no in range(self._no_of_optical_channels)]
            for line_no in range(self._no_of_optical_channels):
                line_y = line_y_values[line_no]
                axis.plot([line_start_x,line_end_x],[line_y,line_y], linestyle = 'solid', marker='o', color = 'blue', alpha=1)
            # add an horizontal line for each classical channel
            line_y_values_classical = [min(line_y_values) - (line_no+1)*circuit_draw_settings_dict['channel_line_spacing'] for line_no in range(self._no_of_classical_channels)]
            for line_no in range(self._no_of_classical_channels):
                line_y = line_y_values_classical[line_no-1]
                axis.plot([line_start_x,line_end_x],[line_y,line_y], linestyle = 'solid', marker='o', color = 'black', alpha=1)

            # add the labels for the channels
            max_characters = circuit_draw_settings_dict['channel_label_string_max_length']
            for line_no in range(self._no_of_optical_channels):
                axis.annotate(                  
                    circuit_draw_settings_dict['channel_labels'][line_no][:max_characters].replace('\n', ' '), 
                    (line_start_x-0.2*circuit_draw_settings_dict['spacing_between_nodes'], line_y_values[line_no]),
                    fontsize=8,
                    horizontalalignment = 'right',
                    verticalalignment =  'center'
                    )
            for line_no in range(self._no_of_classical_channels):
                axis.annotate(                  
                    circuit_draw_settings_dict['channel_labels_classical'][line_no][:max_characters].replace('\n', ' '), 
                    (line_start_x-0.2*circuit_draw_settings_dict['spacing_between_nodes'], line_y_values_classical[line_no]),
                    fontsize=8,
                    horizontalalignment = 'right',
                    verticalalignment =  'center'
                    )

            # run through the nodes that have to be plotted on this page
            first_node_on_page = circuit_draw_settings_dict['nodes_per_page']*page_number
            last_node_on_page = min(number_of_nodes, first_node_on_page + circuit_draw_settings_dict['nodes_per_page'])
            for node_number, node in enumerate(self.node_list[first_node_on_page:last_node_on_page]):
                # if specific node information is given use that, otherwise use default for everything or per item that is not specified
                if node['node_info'] == None:
                    current_node = default_node_info
                else:
                    current_node = dict([])
                    for key in default_node_info.keys():
                        if key in node['node_info'].keys():
                            current_node.update({key : node['node_info'][key]})
                        else:
                            current_node.update({key : default_node_info[key]})

                # for some items we need a list if you want to mark the node different per channel (i.e., a target and a control channel)
                # if there is no list given we artificially make the list by adding the 0th element to the end
                for item_needing_list in [key for key in current_node.keys() if ( type(current_node[key]) == type([]) )]:
                    if current_node[item_needing_list]:
                        current_node[item_needing_list] += [ current_node[item_needing_list][0] ]*len(current_node['channels'])

                # bool to identify node with connection to classical channel
                node_has_classical_channel = (len(current_node['channels_classical']) != 0)
                node_has_optical_channel = ((len(current_node['channels']) != 0))
                # if node affects no channels skip
                if (not node_has_classical_channel) and (not node_has_optical_channel):
                    continue
                
                # determine x value and y values for each node
                node_x = (node_number+ 0.5) * circuit_draw_settings_dict['spacing_between_nodes'] + line_start_x
                node_y_values = [line_y_values[channel] for channel in current_node['channels']]
                node_y_values_classical = [line_y_values_classical[channel] for channel in current_node['channels_classical']]
                lowest_y_value = min(node_y_values + node_y_values_classical)
                highest_y_value = max(node_y_values + node_y_values_classical)

                if node_has_optical_channel:
                    # plot a wide white vertical line
                    axis.plot(
                        [node_x]*len(node_y_values),
                        node_y_values,
                        linestyle = 'solid',
                        linewidth = 5,
                        color = 'white',
                        alpha=1
                        )
                    # plot a vertical line connecting the channels affected by the node (target, control, ...)
                    axis.plot(
                        [node_x]*len(node_y_values),
                        node_y_values,
                        linestyle = current_node['connection_linestyle'],
                        linewidth = current_node['connection_linewidth'],
                        marker = 'none',
                        color = current_node['connection_linecolor'],
                        alpha=1
                        )
                #plot a black line connecting the lowest no classical channel and the highest no optical channel for the node
                if node_has_classical_channel:
                    if node_has_optical_channel:
                        highest_y_value_classical_line = min(node_y_values) 
                    else:
                        highest_y_value_classical_line = max(node_y_values_classical)
                    # plot a wide white vertical line
                    axis.plot(
                        [node_x, node_x],
                        [lowest_y_value, highest_y_value_classical_line],
                        linestyle = 'solid',
                        linewidth = 5,
                        color = 'white',
                        alpha=1
                        )
                    axis.plot(
                        [node_x, node_x],                
                        [lowest_y_value, highest_y_value_classical_line],
                        linestyle = 'solid',
                        linewidth = 2,
                        marker = 'none',
                        color = 'black',
                        alpha=1
                        )
                # plot dashed rectangle with the label
                max_characters = circuit_draw_settings_dict['node_label_string_max_length']
                axis.annotate(                  
                    current_node['label'][:max_characters].replace('\n', ' '), 
                    (node_x - circuit_draw_settings_dict['spacing_between_nodes']*0.4, 
                     highest_y_value+circuit_draw_settings_dict['channel_line_spacing']*0.55),
                     fontsize=8
                    )
                axis.plot(
                    [node_x - circuit_draw_settings_dict['spacing_between_nodes']*0.4,
                     node_x + circuit_draw_settings_dict['spacing_between_nodes']*0.4,
                     node_x + circuit_draw_settings_dict['spacing_between_nodes']*0.4,
                     node_x - circuit_draw_settings_dict['spacing_between_nodes']*0.4,
                     node_x - circuit_draw_settings_dict['spacing_between_nodes']*0.4
                    ],
                    [lowest_y_value-circuit_draw_settings_dict['channel_line_spacing']*0.5,
                     lowest_y_value-circuit_draw_settings_dict['channel_line_spacing']*0.5,
                     highest_y_value+circuit_draw_settings_dict['channel_line_spacing']*0.5,
                     highest_y_value+circuit_draw_settings_dict['channel_line_spacing']*0.5,
                     lowest_y_value-circuit_draw_settings_dict['channel_line_spacing']*0.5
                     ],
                     linestyle = ':',
                     marker = 'none',
                     linewidth = 0.5,
                     color = 'grey'
                    )
                
                # plot a marker per channel
                for index in range(len(current_node['channels'])):
                    axis.plot(
                        node_x,
                        node_y_values[index],
                        markeredgewidth = 1,
                        marker = current_node['markers'][index],
                        markersize = 20,
                        color = current_node['markercolor'][index],
                        markerfacecolor = current_node['markerfacecolor'][index],
                        fillstyle='full',
                        alpha=1
                        )
                    axis.plot(
                        [node_x]*len(node_y_values),
                        node_y_values,
                        linestyle = 'none',
                        markeredgewidth = 0.3,
                        marker = current_node['market_text'][index],
                        markersize = 10,
                        color = 'white',
                        markerfacecolor = 'white',
                        fillstyle='full',
                        alpha=1
                        )
                if node_has_classical_channel:
                    for index in range(len(current_node['channels_classical'])):
                        axis.plot(
                            node_x,
                            node_y_values_classical[index],
                            markeredgewidth = 1,
                            marker = 'o',
                            markersize = 10,
                            color = 'black',
                            markerfacecolor = 'black',
                            fillstyle='full',
                            alpha=1
                            )
        plt.show()
        return
    
    # endregion