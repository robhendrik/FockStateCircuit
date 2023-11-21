import numpy as np
import qutip as qt
import math
import string
import matplotlib as mtplt
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib.patches  import Rectangle
import matplotlib.colors
import collection_of_states as cos

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
    print("FockStateCircuit:   %s" % FockStateCircuit._VERSION)
    print("CollectionOfStates: %s" % cos._VERSION)
    print("Numpy Version:      %s" % np.__version__)
    print("Matplotlib version: %s" % mtplt.__version__)



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

            draw(self, 
                    print_defaults: bool = False, 
                    settings_for_drawing_circuit: dict = None
                    ) -> None

                    Draw the optical circuit. self. settings_for_drawing_circuit is the dict with settings for circuit drawing. 
                    If this dict is not defined default values will be taken. If the boolean print_defaults is set to True the function 
                    will print out the default values to console

            conventions(self) 
                    Function to print return the conventions used in this class as a string

            basis(self)
                    Function returns a dictonary with valid components names as keys and the corresponding photon numbers in the channels as values.

            get_fock_state_matrix(self, 
                    nodes_to_be_evaluated: list[int] = 'all'
                    ) -> np.array:
                    Function returns the fock state matrix for a given set of nodes in the circuit


        Method(s) to add wave plates to the circuit:

            wave_plate(self, 
                    channel_horizontal: int = 0, 
                    channel_vertical: int = 1, 
                    theta: float = 0, 
                    phi: float = 0,
                    node_info: dict = None
                    ) -> None:  

                    Add a wave plate to the fock state circuit with axis rotated over angle 
                    theta and phase delay angle phi. The phase shift will be applied to the 
                    channel for vertical polarization.

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
                    custom_fock_matrix,
                    node_type: str = 'custom fock matrix', 
                    node_info: dict = None
                    ) -> None

                    Apply a custom Fock state matrix to the circuit. The matrix has to be an LxL numpy array with numpy cdouble entries. L is the total size
                    of the Fock state basis (which can be retrieved via FockStateCircuit.basis() )The function does NOT check whether the matrix is physically 
                    possible (i.e.,invertible, unitary). 

        Method(s) to create special nodes

            bridge(self,
                    next_fock_state_circuit,
                    node_type: str = 'bridge', 
                    node_info: dict = None
                    ) -> None:
                    Apply a bridge node to the circuit to transfer the collection of states from one circuit to another. Used when the characteristics
                    of the circuit change (i.e., change number of optical/classical channels). 
                                
            channel_coupling(self, 
                    control_channels: list[int] = [0],
                    target_channels: list[int] = [1],
                    coupling_strength: float = 0,
                    node_info: dict = None
                    ) -> None:  
                    Apply a node to the circuit to couple channels with the given 'coupling_strength'. The node will effectively 
                    apply a controlled shift from control channels to target channels.

            shift(  self, 
                    target_channels: list[int] = [1],
                    shift_per_channel: list[int] = [0],
                    node_info: dict = None
                    ) -> None:  
                    Apply a shift node to the circuit. The photon value in the target channel(s) is increased by the value in 
                    the list 'shift_per_channel' modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, 
                    shift value is 2 and target channel value is 1 the result is that target becomes (2+1)%3 = 0. This is not a 
                    linear optical operation (photons are created in this process). 

            c_shift(self, 
                    control_channels: list[int] = [0],
                    target_channels: list[int] = [1],
                    node_info: dict = None
                    ) -> None:  
                    Apply a controlled-shift (c-shift) node to the circuit. The photon value in the target channel(s) is increased by the value in 
                    the control channels(s) modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, control channel value is 2 and 
                    target channel valie is 1 the result is that control value remains 2 and target becomes (2+1)%3 = 0

            time_delay(self,
                    affected_channels: list[int] = None,
                    delay: float = 0,
                    bandwidth: float = 0,
                    node_type: str = 'spectrum', 
                    node_info: dict = None
                    ) -> None:

                    Apply a time delay for a given bandwidth. The node will turn a pure state into 
                    a statistical mixture where states obtain a phase shift that depends on the bandwidth and the time delay

                    Note: Do not use more than one time delay gates in one circuit.

            time_delay_classical_control(self,
                    affected_channels: list[int] = None,
                    classical_channel_for_delay: int = 0,
                    bandwidth: float = 0,
                    node_type: str = 'spectrum', 
                    node_info: dict = None
                    ) -> None:

                    Apply a time delay for a given bandwidth. Read the delay value from classical channel. 
                    The node will turn a pure state into a statistical mixture where states
                    obtain a phase shift that depends on the bandwidth and the time delay. 

                    Note: Do not use more than one time delay gates in one circuit.
    """
    _VERSION = '0.0.9'

    _TYPES_OF_NODES = ['optical','custom fock matrix','optical and classical combined','classical', 'measurement optical to classical', 'spectrum', 'bridge']

    # default settings for circuit drawing (can be overwritten when calling function)
    _CIRCUIT_DRAW_DEFAULT_SETTINGS = {
            'figure_width_in_inches' : 16,
            'channel_line_length_as_fraction_of_figure_width' : 0.8,
            'number_of_nodes_on_a_line': 10,
            'spacing_between_lines_in_relation_to_spacing_between_nodes' : 1,
            'compound_circuit_title' : 'Optical circuit',
            'channel_label_string_max_length': 15,
            'node_label_string_max_length': 15,
            'compound_plot_title_font_size' : 25,
            'circuit_name_font_size': 15,
            'channel_label_font_size': 10,
            'node_label_font_size' : 6.5,
            'classical_channel_line_color' : 'black',
            'classical_channel_line_marker' : 'o',
            'classical_channel_line_marker_size' : 5,
            'optical_channel_line_color' :'blue',
            'optical_channel_line_marker': 'o',
            'optical_channel_line_marker_size' : 5,
            'bridge_marker' : 'o',
            'bridge_marker_size' : 12,
            'bridge_markeredgewidth' : 2,
            'box_around_node_linestyle' : ':',
            'box_around_node_linewidth': 0.5,
            'box_around_node_color' : 'grey'
        }
    
    # default settings for drawing nodes in the circuit (can be overwritten when adding node to circuit)
    _NODE_DRAW_DEFAULT_SETTINGS = {
            'label' : '',
            'connection_linestyle' : 'solid',
            'connection_linewidth': 2,
            'connection_linecolor_optical': 'blue',
            'connection_linecolor_classical': 'black',
            'channels' : [],
            'channels_classical': [],
            'markers' : ['o'],
            'markercolor' : ['blue'],
            'markerfacecolor' : ['white'],
            'marker_text' : [''],
            'marker_text_fontsize': [10],
            'marker_text_color': ['white'],
            'markersize' : [20],
            'markeredgewidth' : 1,
            'fillstyle' : ['full'],
            'classical_marker_color' : ['black'],
            'classical_marker' : ['o'],
            'classical_marker_size' : ['10'],
            'classical_marker_text' : [''],
            'classical_marker_text_color' : ['white'],
            'classical_marker_text_fontsize': [10],
            'combined_gate': 'single'
        }

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
        
        # the length of the fock state is the number of possible photon numbers. So if the 'length' is 2 the maximum
        # number of photons is 1 (either 0 or 1 photons). If the length is 4 we can have 0,1,2 of 3 photons per channel.
        self._length_of_fock_state = length_of_fock_state

        # the number of channels defining the circuit.
        self._no_of_optical_channels = no_of_optical_channels
        self._no_of_classical_channels = no_of_classical_channels
        
        # we need at least a fock states with length 2 (0 or 1 photon) and two optical channels. Anything else is a 
        # trivial circuit with either zero photons or one channel without interaction.
        if self._length_of_fock_state < 1 or self._no_of_optical_channels < 2:
            raise Exception('length_of_fock_state minimal value is 1, no_of_optical_channels minimum value is 2')

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
        if node_to_be_added['node_type'] == 'custom fock matrix':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'optical non-linear':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'optical and classical combined':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'classical':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'measurement optical to classical':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'spectrum':
            self.node_list.append(node_to_be_added)
        if node_to_be_added['node_type'] == 'bridge':
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
        node_info = {
                'label' : "custom",
                'channels' : optical_channels,
                'channels_classical' : [],
                'markers' : ['*'],
                'markercolor' : ['blue'],
                'markerfacecolor' : ['lightblue'],
                'marker_text' : [r"$c$"],
                'marker_text_fontsize' : [7],
                'markersize' : 25,
                'fillstyle' : 'full'
            }
 
        if len(matrix_optical) != 2 and len(matrix_optical[0]) != 2:
            raise Exception('Only optical nodes with 2-channel interaction are implemented')
        channel_numbers = optical_channels
        tensor_list = self.__translate_channels_numbers_to_tensor_list(channel_numbers = channel_numbers)
        matrix_optical = matrix_optical
        self.__update_list_of_nodes({'matrix_optical': matrix_optical, 'tensor_list':tensor_list, 'node_type': node_type, 'node_info': node_info})

        return

    def custom_fock_state_node(self,
            custom_fock_matrix,
            node_type: str = 'custom fock matrix', 
            node_info: dict = None
            ) -> None:
        """ Apply a custom Fock state matrix to the circuit. The matrix has to be an LxL numpy array with numpy cdouble entries. L is the total size
            of the Fock state basis (which can be retrieved via FockStateCircuit.basis() )The function does NOT check whether the matrix is physically 
            possible (i.e.,invertible, unitary). 
        """
        if node_info == None:
            node_info = {
                    'label' : "custom",
                    'channels' : [channel for channel in range(self._no_of_optical_channels)],
                    'channels_classical' : [],
                    'markers' : ['*'],
                    'markercolor' : ['blue'],
                    'markerfacecolor' : ['lightblue'],
                    'marker_text' : [r"$c$"],
                    'marker_text_fontsize' : [7],
                    'markersize' : 25,
                    'fillstyle' : 'full'
                }
        self.__update_list_of_nodes({'custom_fock_matrix':custom_fock_matrix,'node_type': node_type, 'node_info': node_info})

        return
    
    def wave_plate(self, 
                        channel_horizontal: int = 0, 
                        channel_vertical: int = 1, 
                        theta: float = 0, 
                        phi: float = 0,
                        node_info: dict = None) -> None:  
        """ Add a wave plate to the fock state circuit with axis rotated over angle 
            theta and phase delay angle phi. The phase shift will be applied to the 
            channel for vertical polarization.

        Args:
            channel_horizontal (int): channel number for horizontal polarization
            channel_vertical (int): channel number for vertical polarization
            theta (float): rotation of the axis in radians
            phi (float): phase delay applied to channel for vertical polarization
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.

        Returns:
            nothing
        
        Raises:
            nothing
        """
        if node_info == None:
            node_info = {
                    'label' : "wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\lambda$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }

        self.__apply_generic_rotation_2_channels(channel_horizontal, channel_vertical, theta = theta, phi = phi, node_info = node_info)
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
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{2}$"],
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
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{2}$"],
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
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{2}$"],
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
                    'label' : "qtr-wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{4}$"],
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

        default_node_info_for_this_node = {
                    'label' : "qtr-wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{4}$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        if node_info == None:
            node_info = default_node_info_for_this_node
        else:
            node_info =  default_node_info_for_this_node | node_info
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
                    'label' : "qtr-wave plate",
                    'channels' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{4}$"],
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
                    'markercolor' : ['pink'],
                    'markerfacecolor' : ['pink'],
                    'marker_text' : [r"$\phi$"],
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
                    'markers' : ['o'],
                    'markercolor' : ['pink'],
                    'markerfacecolor' : ['pink'],
                    'marker_text' : [r"$\phi$"],
                    'markersize' : 20,
                    'fillstyle' : 'full',
                    'classical_marker_text' : [r"$c$"],
                    'classical_marker_text_fontsize' : [5]
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
                    'markers' : ['o'],
                    'markercolor' : ['pink'],
                    'markerfacecolor' : ['pink'],
                    'marker_text' : [r"$\theta$"],
                    'markersize' : 20,
                    'fillstyle' : 'full',
                    'classical_marker_text' : [r"$c$"],
                    'classical_marker_text_fontsize' : [5]
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
                    'label' : "measurement",
                    'channels' : optical_channels_to_be_measured,
                    'channels_classical'  : classical_channels_to_be_written,
                    'markercolor' : ['black'],
                    'markerfacecolor' : ['black'],
                    'marker_text' : [r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full',
                    'classical_marker_text' : [r"$t$"],
                    'classical_marker_text_fontsize' : [5]
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
        if affected_channels == None:
            affected_channels = []
        
        if new_input_values == None:
            new_input_values = []

        if node_info == None:
            node_info = {
                    'label' : "class. function",
                    'channels' : [],
                    'channels_classical' : affected_channels,
                    'markercolor' : ['black'],
                    'markerfacecolor' : ['black'],
                    'marker_text' : [],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }

        
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

        if node_info == None:
            node_info = {
                    'label' : "set classic",
                    'channels' : [],
                    'channels_classical' : list_of_classical_channel_numbers,
                    'markercolor' : ['black'],
                    'markerfacecolor' : ['black'],
                    'marker_text' : [],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
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
        if first_channel > second_channel:
            first_channel, second_channel = second_channel, first_channel
        if node_info == None:
            node_info = {
                    'label' : "swap gate",
                    'channels' : [first_channel, second_channel],
                    'markers' : ['v','^'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$s$",r"$s$"],
                    'marker_text_fontsize' : [5,5],
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
                    'label' : "mix 50/50",
                    'channels' : [first_channel, second_channel],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
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
                    'label' : "mix generic",
                    'channels' : [first_channel, second_channel],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
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
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
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
                    'label' : "NPBS 50/50",
                    'channels' : [input_channels_a[0], input_channels_b[0]],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full',
                    'combined_gate': 'NPBS5050'
                }
            node_info_1 = {
                    'label' : "NPBS",
                    'channels' : [input_channels_a[1], input_channels_b[1]],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full',
                    'combined_gate': 'NPBS5050'
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
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full',
                    'combined_gate': 'NPBS'
                }
            node_info_1 = {
                    'label' : "NPBS",
                    'channels' : [input_channels_a[1], input_channels_b[1]],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full',
                    'combined_gate': 'NPBS'
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
    
    def shift(  self, 
                target_channels: list[int] = [1],
                shift_per_channel: list[int] = [0],
                node_info: dict = None
                ) -> None:  
        """ Apply a shift node to the circuit. The photon value in the target channel(s) is increased by the value in 
            the list 'shift_per_channel' modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, 
            shift value is 2 and target channel value is 1 the result is that target becomes (2+1)%3 = 0. This is not a 
            linear optical operation (photons are created in this process). 

            Args: 
                target_channels (list[int]): Channels that change value based on the values in 'shift_per_channel'
                shift_per_channel (list[int]): Shift value per channel
                node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
            
            Raises: 
                Exception when the input channel numbers do not match with the circuit.
                
        """
        # check if lists with channel numbers are ok, otherwise throw exception
        error = False
        if len(shift_per_channel) == len(target_channels):
            for shift, target in zip(shift_per_channel, target_channels):
                if min(shift, target) < 0 or target > self._no_of_optical_channels:
                    error = True
        else:
            error = True
        if error:
            Exception('Error in channel input for function FockStateCircuit.shift')

        this_is_first_round = True  
        # for each channel create a transition matrix
        # multiply the matrices to get to the final overall matrix
        for shift, channel_to_shift in zip(shift_per_channel,target_channels):
            # create an identity matrix for the given combination of one target and one control channel
            coupling_matrix_one_target_control_combination = np.zeros((len(self._list_of_fock_states),len(self._list_of_fock_states)), dtype = np.ubyte)
            for input_index, input in enumerate(self._list_of_fock_states):
                for output_index, output in enumerate(self._list_of_fock_states):
                    # except for target channel values should be equal
                    if all([input[channel] == output[channel] for channel in range(len(output)) if channel != channel_to_shift]):
                    # for target channel the value has to be (shift + original target value) % length. So shift with control value modulus length
                        if (output[channel_to_shift] == (shift + input[channel_to_shift])%self._length_of_fock_state):
                                coupling_matrix_one_target_control_combination[output_index][input_index] = np.ubyte(1)
            if this_is_first_round:
                coupling_matrix = coupling_matrix_one_target_control_combination
                this_is_first_round = False
            else:
                coupling_matrix  = np.matmul(coupling_matrix, coupling_matrix_one_target_control_combination)

        node_info = {
            'label' : "shift",
            'channels' : target_channels ,
            'channels_classical' : [],
            'markers' : ['h']*len(target_channels),
            'markercolor' : ['blue']*len(target_channels),
            'markerfacecolor' : ['lightblue']*len(target_channels),
            'marker_text' : [r'$s$']*len(target_channels),
            'marker_text_fontsize' : [8],
            'marker_text_color' : ['black']*len(target_channels),
            'markersize' : 15,
            'fillstyle' : 'full'
        }
    
        self.custom_fock_state_node(custom_fock_matrix = coupling_matrix, 
                                    node_type='custom fock matrix', 
                                    node_info = node_info)
        return

    
    def c_shift(self, 
                control_channels: list[int] = [0],
                target_channels: list[int] = [1],
                node_info: dict = None
                ) -> None:  
        """ Apply a controlled-shift (c-shift) node to the circuit. The photon value in the target channel(s) is increased by the value in 
            the control channels(s) modulo the maximum allowed photon number. So if 'length_of_fock_state' is 3, control channel value is 2 and 
            target channel valie is 1 the result is that control value remains 2 and target becomes (2+1)%3 = 0The control channel(s) remain unaffected. 
            This is not a linear optical operation (photons are created in this process). The node can create entanglement and allows 'coupling' 
            between optical channels.

            Args: 
                control_channels (list[int]): Channels that 'control' the target channel and remain unchanged themselves
                target_channels (list[int]): Channels that change value based on the values in the control channels
                node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used.
            
            Raises: 
                Exception when the input channel numbers do not match with the circuit.
                
        """
               # if channel input is a channel number (integer) and not a list make it a list
        if type(control_channels) == type(1):
            control_channels = [control_channels]
        if type(target_channels) == type(1):
            target_channels = [target_channels]

        if node_info == None:
            node_info = {
                    'label' : "c-shift",
                    'channels' : control_channels + target_channels ,
                    'channels_classical' : [],
                    'markers' : ['o']*len(control_channels) + ['h']*len(target_channels),
                    'markercolor' : ['black']*len(control_channels) + ['blue']*len(target_channels),
                    'markerfacecolor' : ['black']*len(control_channels) + ['lightblue']*len(target_channels),
                    'marker_text' : [r'$c$']*len(control_channels) + [r'$t$']*len(target_channels),
                    'marker_text_fontsize' : [8],
                    'marker_text_color' : ['white']*len(control_channels) + ['black']*len(target_channels),
                    'markersize' : 15,
                    'fillstyle' : 'full'
                }


        # check if lists with channel numbers are ok, otherwise throw exception
        error = False
        if len(control_channels) == len(target_channels):
            for control, target in zip(control_channels,target_channels):
                if min(control, target) < 0 or max(control, target) > self._no_of_optical_channels:
                    error = True
                if control == target:
                    error = True
        else:
            error = True
        if error:
            Exception('Error in channel input for function FockStateCircuit.c_shift')
            
        
        this_is_first_round = True  
        # run through the combinations of control and target 
        # for each combination create a transition matrix
        # multiply the matrices to get to the final overall matrix
        for control,target in zip(control_channels,target_channels):
            # create an identity matrix for the given combination of one target and one control channel
            coupling_matrix_one_target_control_combination = np.zeros((len(self._list_of_fock_states),len(self._list_of_fock_states)), dtype = np.ubyte)
            for input_index, input in enumerate(self._list_of_fock_states):
                for output_index, output in enumerate(self._list_of_fock_states):
                    # except for target channel values should be equal
                    if all([input[channel] == output[channel] for channel in range(self._no_of_optical_channels) if channel != target]):
                    # for target channel the value has to be (control + original target value) % length. So shift with control value modulus length
                        if (output[target] == (input[control] + input[target])%self._length_of_fock_state):
                                coupling_matrix_one_target_control_combination[output_index][input_index] = np.ubyte(1)
            if this_is_first_round:
                coupling_matrix = coupling_matrix_one_target_control_combination
                this_is_first_round = False
            else:
                coupling_matrix  = np.matmul(coupling_matrix, coupling_matrix_one_target_control_combination)

        self.custom_fock_state_node(custom_fock_matrix = coupling_matrix, node_type='custom fock matrix', node_info = node_info)

        return

    def channel_coupling(self, 
                        control_channels: list[int] = [0],
                        target_channels: list[int] = [1],
                        coupling_strength: float = 0,
                        node_info: dict = None
                        ) -> None:  
        """ Apply a node to the circuit to couple channels with the given 'coupling_strength'. The node will effectively 
            apply a controlled shift from control channels to target channels.

            Args: 
                control_channels (list[int]): Channels that 'control' the target channel and remain unchanged themselves
                target_channels (list[int]): Channels that change value based on the values in the control channels
                coupling_strength (float): Set coupling strength between 0 (no coupling, node does nothing) and 1 (full coupling)
                node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case 
                        default image is used.
            
            Raises: 
                Exception when the input channel numbers do not match with the circuit.
                
        """
        # if channel input is a channel number (integer) and not a list make it a list
        if type(control_channels) == type(1):
            control_channels = [control_channels]
        if type(target_channels) == type(1):
            target_channels = [target_channels]

        if node_info == None:
            node_info = {
                    'label' : "coupling",
                    'channels' : control_channels + target_channels ,
                    'channels_classical' : [],
                    'markers' : ['o']*len(control_channels) + ['h']*len(target_channels),
                    'markercolor' : ['black']*len(control_channels) + ['blue']*len(target_channels),
                    'markerfacecolor' : ['black']*len(control_channels) + ['lightblue']*len(target_channels),
                    'marker_text' : [r'$c$']*len(control_channels) + [r'$t$']*len(target_channels),
                    'marker_text_fontsize' : [8],
                    'marker_text_color' : ['white']*len(control_channels) + ['black']*len(target_channels),
                    'markersize' : 15,
                    'fillstyle' : 'full'
                }


        # check if lists with channel numbers are ok, otherwise throw exception
        error = False
        if len(control_channels) == len(target_channels):
            for control, target in zip(control_channels,target_channels):
                if min(control, target) < 0 or max(control, target) > self._no_of_optical_channels:
                    error = True
                if control == target:
                    error = True
        else:
            error = True
        if error:
            Exception('Error in channel input for function FockStateCircuit.channel_coupling')
            
        
        # the coupling matrix is pre-filled as identity matrix
        coupling_matrix = np.identity(len(self._list_of_fock_states), dtype = np.csingle)
        
        # run through the combinations of control and target 
        # for each combination create a transition matrix
        # multiply the matrices to get to the final overall matrix
        for control,target in zip(control_channels,target_channels):
            # create an identity matrix for the given combination of one target and one control channel
            coupling_matrix_one_target_control_combintation = np.identity(len(self._list_of_fock_states), dtype = np.csingle)
            for input_index, input in enumerate(self._list_of_fock_states):
                for output_index, output in enumerate(self._list_of_fock_states):
                    # except for target channel values should be equal
                    valid_transition = all([input[channel] == output[channel] for channel in range(self._no_of_optical_channels) if channel != target])
                    # for target channel the value has to be (control + original target value) % length. So shift with control value modulus length
                    valid_transition = valid_transition and (output[target] == (input[control] + input[target])%self._length_of_fock_state)
                    # check if on diagonal
                    on_diagonal = (input_index == output_index)
                    if valid_transition and not on_diagonal:
                        coupling_matrix_one_target_control_combintation[output_index][input_index] = np.sqrt(np.cdouble(coupling_strength)*(1 + 0*1j))
                    elif valid_transition and on_diagonal:
                        coupling_matrix_one_target_control_combintation[output_index][input_index] = np.cdouble(1)
                    elif (not valid_transition) and on_diagonal:
                        coupling_matrix_one_target_control_combintation[output_index][input_index] = np.sqrt(np.cdouble((1-coupling_strength))*(1 + 0*1j))
                    else:
                        coupling_matrix_one_target_control_combintation[output_index][input_index] = np.cdouble(0)
            coupling_matrix  = np.matmul(coupling_matrix, coupling_matrix_one_target_control_combintation)

        self.custom_fock_state_node(custom_fock_matrix = coupling_matrix, node_type='custom fock matrix', node_info = node_info)

        return
    
    def time_delay(self,
            affected_channels: list[int] = None,
            delay: float = 0,
            bandwidth: float = 0,
            node_type: str = 'spectrum', 
            node_info: dict = None
            ) -> None:
        """ IN BETA: Apply a time delay for a given bandwidth. The node will turn a pure state into 
            a statistical mixture where states obtain a phase shift that depends on the bandwidth and the time delay

            Note: Do not use more than one time delay gates in one circuit.
        """
        if node_info == None:
            node_info = {
                    'label' : "delay",
                    'channels' : affected_channels,
                    'channels_classical' : [],
                    'markers' : ['s'],
                    'markercolor' : ['darkblue'],
                    'markerfacecolor' : ['blue'],
                    'marker_text' : [r"$\tau$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
            
        self.__update_list_of_nodes({'spectral information': {'bandwidth': bandwidth, 'time delay': delay, 'channels': affected_channels},'node_type': node_type, 'node_info': node_info})
    
    def time_delay_classical_control(self,
            affected_channels: list[int] = None,
            classical_channel_for_delay: int = 0,
            bandwidth: float = 0,
            node_type: str = 'spectrum', 
            node_info: dict = None
            ) -> None:
        """ IN BETA: Apply a time delay for a given bandwidth. Read the delay value from classical channel. 
            The node will turn a pure state into a statistical mixture where states
            obtain a phase shift that depends on the bandwidth and the time delay. 

            Note: Do not use more than one time delay gates in one circuit.
        """
        if node_info == None:
            node_info = {
                    'label' : "delay(c)",
                    'channels' : affected_channels,
                    'channels_classical' : [classical_channel_for_delay],
                    'markers' : ['s'],
                    'markercolor' : ['darkblue'],
                    'markerfacecolor' : ['blue'],
                    'marker_text' : [r"$\tau$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }
        
        self.__update_list_of_nodes({'spectral information': {'bandwidth': bandwidth, 'classical_channel_for_delay': classical_channel_for_delay, 'channels': affected_channels},'node_type': node_type, 'node_info': node_info})
    
    def bridge(self,
            next_fock_state_circuit,
            node_type: str = 'bridge', 
            node_info: dict = None
            ) -> None:
        """ Apply a bridge node to the circuit to transfer the collection of states from one circuit to another. Used when the characteristics
            of the circuit change (i.e., change number of optical/classical channels). 
        """
        if node_info == None:
            node_info = {
                    'label' : "bridge",
                    'channels' : [],
                    'channels_classical' : [],
                    'markers' : ['o'],
                    'markercolor' : ['blue'],
                    'markerfacecolor' : ['white'],
                    'marker_text' : [''],
                    'markersize' : 40,
                    'fillstyle' : 'none'
                }
        self.__update_list_of_nodes({'next_fock_state_circuit':next_fock_state_circuit,'node_type': node_type, 'node_info': node_info})

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

        #****************************************************************
        # case where next node is of type 'custom fock matrix'              *
        #****************************************************************
        elif self.node_list[current_node_index].get('node_type') == 'custom fock matrix':

            custom_fock_matrix = self.node_list[current_node_index].get('custom_fock_matrix')
            # prepare a new collection of states which will be filled with the results 
            collection_of_states_output = cos.CollectionOfStates(self, input_collection_as_a_dict=dict([]))

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
            collection_of_states_output = cos.CollectionOfStates(self, input_collection_as_a_dict=dict([]))

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

    # endregion

    # region Code region for drawing the circuit
    def draw(self, print_defaults: bool = False, settings_for_drawing_circuit: dict = None, compound_circuit_settings: dict = None) -> None:
        """ Draw the optical circuit. self.settings_for_drawing_circuit is the dict with settings for circuit drawing. 
            If this dict is not defined default values will be taken. If the boolean print_defaults is set to True the function will print
            out the default values to console. The parameter compound_circuit_settings is used when drawing a circuit with bridges to other circuits 
            (a.k.a. compound circuit).
        
        Args:
            print_defaults (bool): If 'True' function will print default settings to console.
            settings_for_drawing_circuit (dict): dict with settings for circuit drawing. If none given default will be taken.
            compound_circuit_settings (dict): only used for drawing compound circuits.

        Returns:
            nothing
        
        Raises:
            nothing
        """
        # default settings for circuit drawing (can be overwritten when calling function)
        if settings_for_drawing_circuit is not None:
            circuit_draw_settings_dict = dict(FockStateCircuit._CIRCUIT_DRAW_DEFAULT_SETTINGS | settings_for_drawing_circuit)
        else:
            circuit_draw_settings_dict = dict(FockStateCircuit._CIRCUIT_DRAW_DEFAULT_SETTINGS)

        if print_defaults == True:
            print(FockStateCircuit._CIRCUIT_DRAW_DEFAULT_SETTINGS)
            print(FockStateCircuit._NODE_DRAW_DEFAULT_SETTINGS)

        if compound_circuit_settings is None:
            # this is either the first of a series of connection circuits, or a single circuit.
            number_of_nodes_compound_circuit = 0
            number_of_optical_channels_compound_circuit = 0
            number_of_classical_channels_compound_circuit = 0

            # loop through all circuits in the compound circuit until you find a circuit that does not end with a 'bridge'
            circuit = self
            compound_circuit_names = []
            while True:
                # count total number of nodes in compound circuit
                number_of_nodes_compound_circuit += len(circuit.node_list)

                # make a list of names for the various circuits in the compound circuit
                if circuit._circuit_name is None:
                    this_circuit_name = 'circuit ' + str(len(compound_circuit_names))
                    circuit._circuit_name = this_circuit_name
                else:
                    this_circuit_name = circuit._circuit_name
                compound_circuit_names.append(this_circuit_name)

                # determine maximum number of optical or classical channels in the compound circuit
                number_of_optical_channels_compound_circuit = max(number_of_optical_channels_compound_circuit, circuit._no_of_optical_channels)                
                number_of_classical_channels_compound_circuit = max(number_of_classical_channels_compound_circuit, circuit._no_of_classical_channels)

                # if last node is not a bridge we have reached the end, otherwise another circuit to be added
                if len(circuit.node_list) > 0 and circuit.node_list[-1]['node_type'] == 'bridge':
                    circuit = circuit.node_list[-1]['next_fock_state_circuit']
                    number_of_nodes_compound_circuit -= 1
                else:
                    break
            
            # define a coordinate system to position the elements in the circuit drawing. 
            canvas_width = 100

            # determine the horizontal positions of the nodes and the lines
            target_line_length = canvas_width * circuit_draw_settings_dict['channel_line_length_as_fraction_of_figure_width']
            nodes_per_page = circuit_draw_settings_dict['number_of_nodes_on_a_line']
            line_start_x = (canvas_width-target_line_length)/2 # centre the lines on the canvas, determine left side starting point
            spacing_between_nodes = math.floor(target_line_length/nodes_per_page)
            line_end_x = line_start_x + nodes_per_page * spacing_between_nodes
            line_x_values = (line_start_x,line_end_x)
            node_x_values = [line_x_values[0] + (node_on_page+0.5)*spacing_between_nodes for node_on_page in range(nodes_per_page)]

            # determine the canvas height based on spacing between lines a number of channels
            channel_line_spacing = circuit_draw_settings_dict['spacing_between_lines_in_relation_to_spacing_between_nodes'] * spacing_between_nodes
            canvas_height = channel_line_spacing * (number_of_optical_channels_compound_circuit+number_of_classical_channels_compound_circuit+ 2)

            # determine the vertical positions for the lines and circuit names
            circuit_name_y_value = canvas_height
            line_y_values_optical = [canvas_height - (line_no+2)*channel_line_spacing for line_no in range(number_of_optical_channels_compound_circuit)]
            line_y_values_classical = [min(line_y_values_optical) - (line_no+1)*channel_line_spacing for line_no in range(number_of_classical_channels_compound_circuit)]
            
            # make default labels for the channels
            if 'channel_labels_optical' not in circuit_draw_settings_dict.keys():
                optical_channel_labels = ['optical '+ str(index) for index in range(number_of_optical_channels_compound_circuit)]
                circuit_draw_settings_dict.update({'channel_labels_optical': optical_channel_labels})
            if 'channel_labels_classical' not in circuit_draw_settings_dict.keys():
                classical_channel_labels = ['classical '+ str(index) for index in range(number_of_classical_channels_compound_circuit)]
                circuit_draw_settings_dict.update({'channel_labels_classical': classical_channel_labels})

            # determine how many pages are needed to draw the compound circuit
            nodes_on_last_page = number_of_nodes_compound_circuit%nodes_per_page       
            if nodes_on_last_page == 0:
                number_of_pages = int(number_of_nodes_compound_circuit/nodes_per_page)
            else:
                number_of_pages = 1 + int(number_of_nodes_compound_circuit/nodes_per_page)
       
            # create the plot, use two times expected number of pages as size. left over pages
            # will be deleted at the end
            # the better solution would be to dynamically add subplots when starting a new page.
            figure_size_in_inches = (circuit_draw_settings_dict['figure_width_in_inches'], 
                                     circuit_draw_settings_dict['figure_width_in_inches']*number_of_pages*canvas_height/canvas_width
                                     )
            fig, axs = plt.subplots(nrows= 2*number_of_pages, ncols=1, squeeze = False, figsize = figure_size_in_inches)

            # give each page of the plot a title
            for page in range(number_of_pages):
                axs[page][0].set_title(circuit_draw_settings_dict['compound_circuit_title'] + ' page ' + str(page+1),
                                    fontsize=circuit_draw_settings_dict['compound_plot_title_font_size']
                                    )

            # gather all parameters in a dictionary which can be passed on in the recursive call to next circuits
            compound_circuit_settings = {
                'canvas_height' : canvas_height,
                'canvas_width' : canvas_width,
                'plot_axs' : axs,
                'figure' : fig,
                'circuit_name_y_value' : circuit_name_y_value,
                'line_y_values_optical' : line_y_values_optical,
                'line_y_values_classical' : line_y_values_classical ,
                'node_x_values' : node_x_values ,
                'line_x_values': line_x_values,
                'channel_labels_classical' : circuit_draw_settings_dict['channel_labels_classical'],
                'channel_labels_optical' : circuit_draw_settings_dict['channel_labels_optical'],
                'spacing_between_nodes' : spacing_between_nodes,
                'channel_line_spacing' : channel_line_spacing,
                'node_positions_occupied' : [],
                'active_optical_channels_per_node' : [],
                'active_classical_channels_per_node' : []
            }
        if len(self.node_list) == 0:
            raise Exception('Error drawing empty circuit. Circuits need to contain at least one node')
        
        for node_index, node in enumerate(self.node_list):

            # if specific node information is given use that, otherwise use default for everything or per item that is not specified
            if node['node_info'] is not None:
                current_node_info = FockStateCircuit._NODE_DRAW_DEFAULT_SETTINGS | node['node_info']
            else:
                current_node_info = FockStateCircuit._NODE_DRAW_DEFAULT_SETTINGS

            for key in current_node_info.keys():
                if key not in FockStateCircuit._NODE_DRAW_DEFAULT_SETTINGS.keys():
                    print("Unrecognized key in parameter node_info: ", key, " Run circuit.draw(print_defaults = True) to get a list of recognized keys.")
            
            # for some items we need a list if you want to mark the node different per channel (i.e., a target and a control channel)
            # if there is no list given we artificially make the list by adding the 0th element to the end
            for item_needing_list in current_node_info.keys():
                if type(FockStateCircuit._NODE_DRAW_DEFAULT_SETTINGS[item_needing_list]) == type([]):
                    maximum_needed_list_length = len(current_node_info['channels']) + len(current_node_info['channels_classical'])
                    if type(current_node_info[item_needing_list]) == type([]) and len(current_node_info[item_needing_list]) == 0 :
                        current_node_info[item_needing_list] = []
                    elif type(current_node_info[item_needing_list]) == type([]) and len(current_node_info[item_needing_list]) < maximum_needed_list_length:
                        current_node_info[item_needing_list] = current_node_info[item_needing_list] + [current_node_info[item_needing_list][0]] * (maximum_needed_list_length - len(current_node_info[item_needing_list]) )
                    elif type(current_node_info[item_needing_list]) != type([]):
                        current_node_info[item_needing_list] = [ current_node_info[item_needing_list]]*(maximum_needed_list_length)

            # determine if node is a bridge
            this_node_is_a_bridge = (node['node_type'] == 'bridge')
            
            # determine if this is the very first node of a circuit
            this_node_is_first_in_circuit = (node_index == 0)

            # determine if this is the last node in the circuit
            this_node_is_last_in_circuit = (node_index == len(self.node_list)-1)

            # determine what type of node this
            node_has_classical_channel = (len(current_node_info['channels_classical']) != 0)
            node_has_optical_channel = ((len(current_node_info['channels']) != 0))

            # determine if node is part of a combined node
            this_is_a_combined_node = (not this_node_is_a_bridge) and ('combined_gate' in current_node_info.keys()) and  (current_node_info['combined_gate'] != 'single')

            # determine if it is the first node in a combined node
            if this_is_a_combined_node:
                this_is_first_of_a_combined_node = (node_index == 0 or (self.node_list[node_index - 1]['node_info'].get('combined_gate', 'single') != current_node_info['combined_gate']))
            else:
                this_is_first_of_a_combined_node = False

            # determine the length of the combined node
            if this_is_a_combined_node and this_is_first_of_a_combined_node:
                number_of_combined_nodes = 0 # count combined nodes
                while True:
                    if node_index + number_of_combined_nodes >= len(self.node_list):
                        break
                    elif self.node_list[node_index + number_of_combined_nodes]['node_type'] == 'bridge':
                        break
                    elif self.node_list[node_index + number_of_combined_nodes]['node_info'].get('combined_gate','single') != current_node_info['combined_gate']:
                        break
                    else:
                        number_of_combined_nodes += 1

                nodes_occupied_by_combined_node = number_of_combined_nodes//2
                # if combined node does not fit on a page treat it as individual nodes (combined node is too large)
                if (number_of_combined_nodes//2) > len(compound_circuit_settings['node_x_values']):
                    this_is_a_combined_node = False
                # if combined node is a single node treat it as a single node
                if number_of_combined_nodes <= 1:
                    this_is_a_combined_node = False

            # create a number which in binary format has a 1 for occupied channels and 0 otherwise
            if this_node_is_a_bridge:
                # for a bridge fill string with all 1's. All channels are occupied in a bridge. 
                # we do not want the bridge to be move on the drawing, and also other nodes should not move 'through' a bridge
                bitlist_all_occupied = ['1'] * (len(compound_circuit_settings['line_y_values_optical'])+len(compound_circuit_settings['line_y_values_classical']))
                bitnumber_occupied_channels = int(''.join(bitlist_all_occupied), 2)
            
            elif this_is_a_combined_node:
                if this_is_first_of_a_combined_node:
                    # for a combined node create a bitnumber indicating all channels occupied by all nodes in the combination. Avoid that another node is place somewhere inbetween
                    bitnumber_used = 0
                    for node in self.node_list[node_index:node_index+number_of_combined_nodes]:
                        # we have to look forward in node list  to determine complete size of the combined node
                        optical_channels = node['node_info'].get('channels',[])
                        classical_channels = node['node_info'].get('channels_classical',[])
                        # make a bitstring to indicate occupied channels
                        bitlist_optical = ['1' if (channel in optical_channels) else '0' for channel in range(len(compound_circuit_settings['line_y_values_optical']))]
                        bitlist_classical = ['1' if (channel in classical_channels) else '0' for channel in range(len(compound_circuit_settings['line_y_values_classical']))]
                        bitnumber_used_individual_node = int(''.join(bitlist_optical + bitlist_classical), 2)
                        bitnumber_used = bitnumber_used | bitnumber_used_individual_node

                    # make another bitstring filled up between channels 
                    memory, bitnumber_occupied_channels = 0, 0
                    for bit_index in range(len(compound_circuit_settings['line_y_values_optical']) + len(compound_circuit_settings['line_y_values_classical'])):
                        bit_value_occupied = int((bitnumber_used & (1 << bit_index )) != 0)
                        memory |= bit_value_occupied
                        memory &= int((bitnumber_used >> bit_index) != 0)
                        bitnumber_occupied_channels = bitnumber_occupied_channels | (memory << bit_index )

            else: # not a bridge and not a combined node
                # make a bitstring to indicate occupied channels
                bitlist_optical = ['1' if (channel in current_node_info['channels']) else '0' for channel in range(len(compound_circuit_settings['line_y_values_optical']))]
                bitlist_classical = ['1' if (channel in current_node_info['channels_classical']) else '0' for channel in range(len(compound_circuit_settings['line_y_values_classical']))]
                bitnumber_used = int(''.join(bitlist_optical + bitlist_classical), 2)
                # make another bitstring filled up between channels 
                memory, bitnumber_occupied_channels = 0, 0
                for bit_index in range(len(compound_circuit_settings['line_y_values_optical']) + len(compound_circuit_settings['line_y_values_classical'])):
                    bit_value_occupied = int((bitnumber_used & (1 << bit_index )) != 0)
                    memory |= bit_value_occupied
                    memory &= int((bitnumber_used >> bit_index) != 0)
                    bitnumber_occupied_channels = bitnumber_occupied_channels | (memory << bit_index )
                # and finally also make the 'surrounding' bits 1 to avoid that channel labels overlap
                # we always want an empty channel inbetween two occupied channels
                # we only need this if we want to find a position for the node that is already used, i.e., do not force node to be added to end
                bitnumber_incl_boundaries = ((bitnumber_occupied_channels << 1) | bitnumber_occupied_channels | (bitnumber_occupied_channels >> 1))

            # determine whether we force node to be added to end, or whether we try to fit it in next to an existing node
            if this_node_is_first_in_circuit or this_node_is_a_bridge or (not node_has_classical_channel and not node_has_optical_channel) or this_is_a_combined_node:
                add_new_node_to_end = True
            else:
                add_new_node_to_end = False

            # determine the node x position and page number for the various cases we can encounter
            if not add_new_node_to_end:              
                # try to shift to left in drawing
                # find the earliest node position that has no overlap with the current node
                node_position = len(compound_circuit_settings['node_positions_occupied'])-1
                while True:
                    if node_position < 0 or compound_circuit_settings['node_positions_occupied'][node_position] & bitnumber_occupied_channels != 0:
                        node_position += 1
                        break
                    else:
                        node_position -= 1

                while True:
                    if node_position >= len(compound_circuit_settings['node_positions_occupied']) or (compound_circuit_settings['node_positions_occupied'][node_position] & bitnumber_incl_boundaries) == 0:
                        break
                    else:
                        node_position += 1
                
                if node_position >= len(compound_circuit_settings['node_positions_occupied']):
                    # it could be that we try to fit the node in an existing position but it does not fit, we have to add it to the end after all
                    page_number, node_number_on_page= divmod(len(compound_circuit_settings['node_positions_occupied']), len(compound_circuit_settings['node_x_values']))
                    compound_circuit_settings['node_positions_occupied'].append(bitnumber_occupied_channels)
                    start_empty_new_page = (node_number_on_page == 0)
                else:
                    # we found a place to fit the node, update the list of occupied node positions
                    compound_circuit_settings['node_positions_occupied'][node_position] = compound_circuit_settings['node_positions_occupied'][node_position] | bitnumber_occupied_channels
                    page_number, node_number_on_page= divmod(node_position, len(compound_circuit_settings['node_x_values']))
                    start_empty_new_page = False

                # determine where to write this node
                node_x = compound_circuit_settings['node_x_values'][node_number_on_page]
                axis = compound_circuit_settings['plot_axs'][page_number][0]

            elif add_new_node_to_end and not this_is_a_combined_node:
                # add node in next position after the already occupied node positions
                page_number, node_number_on_page= divmod(len(compound_circuit_settings['node_positions_occupied']), len(compound_circuit_settings['node_x_values']))
                # if node is most left on page start a new page
                start_empty_new_page = (node_number_on_page == 0)
                # if this is a bridge it will be drawn 'inbetween' nodes so we have to shorten the list of nodes occupied to avoid a gap
                compound_circuit_settings['node_positions_occupied'].append(bitnumber_occupied_channels)
                if this_node_is_a_bridge:
                    compound_circuit_settings['node_positions_occupied']  = compound_circuit_settings['node_positions_occupied'][1:]

                # determine where to write this node
                node_x = compound_circuit_settings['node_x_values'][node_number_on_page]
                axis = compound_circuit_settings['plot_axs'][page_number][0]

            else: # this_is_a_combined_node:
                if this_is_first_of_a_combined_node:
                    # add node in next position after the already occupied node positions
                    page_number_first_combined_node, node_number_on_page_first_combined_node= divmod(len(compound_circuit_settings['node_positions_occupied']), len(compound_circuit_settings['node_x_values']))
                    nodes_occupied_by_combined_node = (1+ number_of_combined_nodes)//2 # 0 -> 0, 1->1,  2-> 1, 3 -> 2, 4->2, ....
                    page_number_last_combined_node, node_number_on_page_last_combined_node= divmod(len(compound_circuit_settings['node_positions_occupied']) + nodes_occupied_by_combined_node, len(compound_circuit_settings['node_x_values']))
                    page_number = page_number_last_combined_node

                    # check if full combined nodes still fits on the page, otherwise move everything to next page
                    if page_number_last_combined_node != page_number_first_combined_node:
                        # if we move combined node to next page fill up the empty positions in the current page
                        for _ in range(node_number_on_page_first_combined_node, len(compound_circuit_settings['node_x_values'])):
                            compound_circuit_settings['node_positions_occupied'].append(0)
                        node_number_on_page_first_combined_node = 0
                    # occupy all channels needed for the combined node
                    for _ in range(nodes_occupied_by_combined_node):
                        compound_circuit_settings['node_positions_occupied'].append(bitnumber_occupied_channels)
                    
                    # set bool for starting new page if the combined node requires a new page
                    start_empty_new_page = (node_number_on_page_first_combined_node == 0 or (page_number_last_combined_node != page_number_first_combined_node))
                  
                    # generate the x-values for all nodes in the combined node
                    combined_node_x_values = []
                    node_spacing = compound_circuit_settings['node_x_values'][1]-compound_circuit_settings['node_x_values'][0]
                    node_x_original_first_node_in_compound = compound_circuit_settings['node_x_values'][node_number_on_page_first_combined_node]
                    if number_of_combined_nodes%2 == 0: #case for even number of nodes
                        shift_combined_node = (-1/4)*node_spacing
                    else: #case for odd number of nodes
                        shift_combined_node = 0
                    for combined_node_number in range(number_of_combined_nodes):                            
                        combined_node_x_values.append(node_x_original_first_node_in_compound + shift_combined_node + combined_node_number*node_spacing/2.0)

                    # generate x coordinates for box around combined node
                    box_xs = (  min(combined_node_x_values)- compound_circuit_settings['spacing_between_nodes']*0.2,
                                max(combined_node_x_values)+ compound_circuit_settings['spacing_between_nodes']*0.2
                                )
                    
                    # generate y coordinates for box around combined node
                    index_ys = []
                    bit_index, memory = 0, 0
                    while True:
                        bit_value_occupied = int(((bitnumber_occupied_channels >> bit_index ) & 1) != 0)
                        if bit_value_occupied != memory:
                            index_ys.append(bit_index-memory)
                        memory = bit_value_occupied
                        if (bitnumber_occupied_channels >> bit_index ) == 0:
                            break
                        bit_index += 1
                    box_ys = (  (compound_circuit_settings['line_y_values_optical']+ compound_circuit_settings['line_y_values_classical'])[::-1][min(index_ys)]- compound_circuit_settings['channel_line_spacing']*0.5,
                                (compound_circuit_settings['line_y_values_optical']+ compound_circuit_settings['line_y_values_classical'])[::-1][max(index_ys)]+ compound_circuit_settings['channel_line_spacing']*0.5
                                )                                                                       
                else:
                    start_empty_new_page = False

                # determine where to write this node
                node_x = combined_node_x_values[0]
                del combined_node_x_values[0]
                axis = compound_circuit_settings['plot_axs'][page_number][0]

            # if this is last node in circuit update the 'active_channels_per_node_position'
            for node_position in range(len(compound_circuit_settings['active_optical_channels_per_node']), len(compound_circuit_settings['node_positions_occupied'])):
                compound_circuit_settings['active_optical_channels_per_node'].append(self._no_of_optical_channels)
                compound_circuit_settings['active_classical_channels_per_node'].append(self._no_of_classical_channels)


            # add functionality t dynamically add subplots by    fig.axes[i].change_geometry(n+1, 1, i+1)
            # axis = fig.add_subplot(n+1, 1, n+1)

            # determine how to write this node in one of four categories
            same_page_existing_circuit = (not start_empty_new_page) and (not this_node_is_first_in_circuit)
            new_page_existing_circuit = (start_empty_new_page) and (not this_node_is_first_in_circuit)
            same_page_new_circuit = (not start_empty_new_page) and (this_node_is_first_in_circuit)
            new_page_new_circuit = (start_empty_new_page) and (this_node_is_first_in_circuit)

            # before drawing the node draw the page where needed
            if same_page_existing_circuit:
                # nothing needed, page is already prepared
                prepare_new_page = False
                draw_channel_lines = False
                add_channel_labels = False
                add_circuit_label = False
            elif new_page_existing_circuit:
                prepare_new_page = True
                draw_channel_lines = True
                add_channel_labels = True
                add_circuit_label = True
            elif same_page_new_circuit:
                prepare_new_page = False
                draw_channel_lines = True
                add_channel_labels = False
                add_circuit_label = True
            elif new_page_new_circuit:
                prepare_new_page = True
                draw_channel_lines = True
                add_channel_labels = True
                add_circuit_label = True
            
            # determine what to do with the channels lines after the node
            if this_node_is_last_in_circuit and this_node_is_a_bridge and new_page_existing_circuit:
                # this is the end of a circuit on a new page
                draw_bridge_symboles = True
                modify_end_of_line_symbols_previous_page = True
                add_circuit_label = False # cancel writing of circuit label on new page
            elif this_node_is_last_in_circuit and this_node_is_a_bridge and not new_page_existing_circuit:
                # this is the end of a circuit on existing page (so space for minimally one node left)
                draw_bridge_symboles = True
                modify_end_of_line_symbols_previous_page = False
            elif this_node_is_last_in_circuit and not this_node_is_a_bridge:
                draw_bridge_symboles = False
                modify_end_of_line_symbols_previous_page = False
            else:
                draw_bridge_symboles = False
                modify_end_of_line_symbols_previous_page = False

            if prepare_new_page:
                # make an 'invisible' curve to size the canvas from (0,0) to (..dict['canvas_width'], canvas_height )
                axis.axis('off') #axis invisible, border invisible
                xpoints = [0,compound_circuit_settings['canvas_width']] 
                ypoints = [0,compound_circuit_settings['canvas_height']]            
                axis.plot(xpoints,ypoints,'o:r', alpha=0) # alpha = 0 means invisible

                # determine horizontal start and stop for lines. 
                line_start_x = compound_circuit_settings['line_x_values'][0]
                line_end_x = compound_circuit_settings['line_x_values'][1]

            if draw_channel_lines:
                if not this_is_first_of_a_combined_node:
                    line_start_x =  node_x - 0.5 * compound_circuit_settings['spacing_between_nodes']
                else:
                    line_start_x =  node_x - 0.5 * compound_circuit_settings['spacing_between_nodes'] - shift_combined_node
                # add an horizontal line for each optical channel
                line_end_x = compound_circuit_settings['line_x_values'][1]
                for line_no in range(self._no_of_optical_channels):
                    line_y = compound_circuit_settings['line_y_values_optical'][line_no]
                    axis.plot([ line_start_x,line_end_x],[line_y,line_y],
                                linestyle = 'solid',
                                marker=circuit_draw_settings_dict['optical_channel_line_marker'],
                                markersize= circuit_draw_settings_dict['optical_channel_line_marker_size'],
                                color = circuit_draw_settings_dict['optical_channel_line_color'],
                                alpha=1
                                )

                # add an horizontal line for each classical channel
                for line_no in range(self._no_of_classical_channels):
                    line_y = compound_circuit_settings['line_y_values_classical'][line_no]
                    axis.plot([ line_start_x,line_end_x],[line_y,line_y],
                                linestyle = 'solid',
                                marker=circuit_draw_settings_dict['classical_channel_line_marker'],
                                markersize= circuit_draw_settings_dict['classical_channel_line_marker_size'],
                                color = circuit_draw_settings_dict['classical_channel_line_color'], 
                                alpha=1
                                )
                    
            if add_channel_labels:
                # add the labels for the channels if this is first node on the page
                max_characters = circuit_draw_settings_dict['channel_label_string_max_length']
                for line_no in range(min(self._no_of_optical_channels, len(compound_circuit_settings['channel_labels_optical']) )):  
                    axis.annotate(               
                        compound_circuit_settings['channel_labels_optical'][line_no][:max_characters], 
                        (line_start_x-0.2*compound_circuit_settings['spacing_between_nodes'], compound_circuit_settings['line_y_values_optical'][line_no]),
                        fontsize=circuit_draw_settings_dict['channel_label_font_size'],
                        horizontalalignment = 'right',
                        verticalalignment =  'center'
                        )
                for line_no in range(min(self._no_of_classical_channels, len(compound_circuit_settings['channel_labels_classical']) )):
                    axis.annotate(                  
                        compound_circuit_settings['channel_labels_classical'][line_no][:max_characters], 
                        (line_start_x-0.2*compound_circuit_settings['spacing_between_nodes'], compound_circuit_settings['line_y_values_classical'][line_no]),
                        fontsize=circuit_draw_settings_dict['channel_label_font_size'],
                        horizontalalignment = 'right',
                        verticalalignment =  'center'
                        )
                    
            if add_circuit_label:
                if not this_is_first_of_a_combined_node:
                    circuit_label_x =  node_x - 0.5 * compound_circuit_settings['spacing_between_nodes']
                else:
                    circuit_label_x =  node_x - 0.5 * compound_circuit_settings['spacing_between_nodes'] - shift_combined_node
                # add a label for the circuit at the top
                axis.annotate(               
                    self._circuit_name, 
                    (circuit_label_x, compound_circuit_settings['circuit_name_y_value']),
                    fontsize=circuit_draw_settings_dict['circuit_name_font_size'],
                    horizontalalignment = 'left',
                    verticalalignment =  'center'
                    )
            
            if modify_end_of_line_symbols_previous_page:
                if page_number > 0:
                    axis_prev = compound_circuit_settings['plot_axs'][page_number-1][0]
                    line_end = compound_circuit_settings['line_x_values'][1]
                    for index in range(self._no_of_optical_channels):                      
                        axis_prev.plot(
                            line_end,
                            compound_circuit_settings['line_y_values_optical'][index],
                            markeredgewidth = circuit_draw_settings_dict['bridge_markeredgewidth'],
                            marker = circuit_draw_settings_dict['bridge_marker'],
                            markersize = circuit_draw_settings_dict['bridge_marker_size'],
                            color = circuit_draw_settings_dict['optical_channel_line_color'],
                            markerfacecolor = 'white',
                            fillstyle='full',
                            alpha=1
                            )
                    for index in range(self._no_of_classical_channels):
                        axis_prev.plot(
                            line_end,
                            compound_circuit_settings['line_y_values_classical'][index],
                            markeredgewidth = circuit_draw_settings_dict['bridge_markeredgewidth'],
                            marker = circuit_draw_settings_dict['bridge_marker'],
                            markersize = circuit_draw_settings_dict['bridge_marker_size'],
                            color = circuit_draw_settings_dict['classical_channel_line_color'],
                            markerfacecolor = 'white',
                            fillstyle='full',
                            alpha=1
                            )
                    axis_prev.add_patch(Rectangle((line_end, 0), compound_circuit_settings['canvas_width'] - line_end, compound_circuit_settings['canvas_height'],
                                edgecolor = 'white',
                                facecolor = 'white',
                                fill=True,
                                lw=1,
                                zorder = 2))

            if draw_bridge_symboles:
                if not this_is_first_of_a_combined_node:
                    bridge_x = node_x - 0.5 * compound_circuit_settings['spacing_between_nodes']
                else:
                    bridge_x = node_x - 0.5 * compound_circuit_settings['spacing_between_nodes']+shift_combined_node
                for index in range(self._no_of_optical_channels):                      
                    axis.plot(
                        bridge_x,
                        compound_circuit_settings['line_y_values_optical'][index],
                        markeredgewidth = circuit_draw_settings_dict['bridge_markeredgewidth'],
                        marker = circuit_draw_settings_dict['bridge_marker'],
                        markersize = circuit_draw_settings_dict['bridge_marker_size'],
                        color = circuit_draw_settings_dict['optical_channel_line_color'],
                        markerfacecolor = 'white',
                        fillstyle='full',
                        alpha=1
                        )
                for index in range(self._no_of_classical_channels):
                    axis.plot(
                        bridge_x,
                        compound_circuit_settings['line_y_values_classical'][index],
                        markeredgewidth = circuit_draw_settings_dict['bridge_markeredgewidth'],
                        marker = circuit_draw_settings_dict['bridge_marker'],
                        markersize = circuit_draw_settings_dict['bridge_marker_size'],
                        color = circuit_draw_settings_dict['classical_channel_line_color'],
                        markerfacecolor = 'white',
                        fillstyle='full',
                        alpha=1
                        )
                axis.add_patch(Rectangle((bridge_x, 0), compound_circuit_settings['canvas_width'] - bridge_x, compound_circuit_settings['canvas_height'],
                                edgecolor = 'white',
                                facecolor = 'white',
                                fill=True,
                                lw=1,
                                zorder = 2))
              
            # if node affects no channels skip
            if node_has_classical_channel or node_has_optical_channel:
            
                # determine y values for each node
                node_y_values_optical = [compound_circuit_settings['line_y_values_optical'][channel] for channel in current_node_info['channels']]
                node_y_values_classical = [compound_circuit_settings['line_y_values_classical'][channel] for channel in current_node_info['channels_classical']]
                lowest_y_value = min(node_y_values_optical + node_y_values_classical)
                highest_y_value = max(node_y_values_optical + node_y_values_classical)

                if node_has_optical_channel:
                    # plot a wide white vertical line
                    axis.plot(
                        [node_x]*len(node_y_values_optical),
                        node_y_values_optical,
                        linestyle = 'solid',
                        linewidth = current_node_info['connection_linewidth'] * 2,
                        color = 'white',
                        alpha=1
                        )
                    # plot the connection line
                    axis.plot(
                        [node_x]*len(node_y_values_optical),
                        node_y_values_optical,
                        linestyle = current_node_info['connection_linestyle'],
                        linewidth = current_node_info['connection_linewidth'],
                        marker = 'none',
                        color = current_node_info['connection_linecolor_optical'],
                        alpha=1
                        )

                if node_has_classical_channel:
                    # plot a wide white vertical line
                    axis.plot(
                        [node_x]*len(node_y_values_classical),
                        node_y_values_classical,
                        linestyle = 'solid',
                        linewidth = current_node_info['connection_linewidth'] * 2,
                        color = 'white',
                        alpha=1
                        )
                    # plot the connection line
                    axis.plot(
                        [node_x]*len(node_y_values_classical),
                        node_y_values_classical,
                        linestyle = 'solid',
                        linewidth = current_node_info['connection_linewidth'],
                        marker = 'none',
                        color = current_node_info['connection_linecolor_classical'],
                        alpha=1
                        )
                    
                if node_has_classical_channel and node_has_optical_channel:
                    # plot line connecting optical and classical channels
                    axis.plot(
                        [node_x]*2,
                        [max(node_y_values_classical), min(node_y_values_optical)],
                        linestyle = 'solid',
                        linewidth = current_node_info['connection_linewidth'] * 2,
                        color = 'white',
                        alpha=1
                        )
                    axis.plot(
                        [node_x]*2,
                        [max(node_y_values_classical), min(node_y_values_optical)],
                        linestyle = 'solid',
                        linewidth = current_node_info['connection_linewidth'],
                        marker = 'none',
                        color = current_node_info['connection_linecolor_classical'],
                        alpha=1
                        )
                
                # draw box around node and plot label
                if not this_is_a_combined_node:
                    box_xs = (node_x - compound_circuit_settings['spacing_between_nodes']*0.4 ,
                              node_x + compound_circuit_settings['spacing_between_nodes']*0.4
                              )
                    box_ys = (lowest_y_value-compound_circuit_settings['channel_line_spacing']*0.5,
                              highest_y_value+compound_circuit_settings['channel_line_spacing']*0.5
                              )
                    do_print_box = True
                elif this_is_first_of_a_combined_node:
                    do_print_box = True
                else:
                    do_print_box = False

                if do_print_box:
                    x_position_node_label = min(box_xs)
                    y_position_node_label = max(node_y_values_classical + node_y_values_optical) + 0.55*compound_circuit_settings['channel_line_spacing']
                    axis.annotate(                  
                        current_node_info['label'], 
                        (x_position_node_label, y_position_node_label),
                        fontsize=circuit_draw_settings_dict['node_label_font_size']
                        )
                    axis.plot(
                        [box_xs[0],
                            box_xs[1],
                            box_xs[1],
                            box_xs[0],
                            box_xs[0]
                        ],
                        [box_ys[0],
                            box_ys[0],
                            box_ys[1],
                            box_ys[1],
                            box_ys[0]
                            ],
                            linestyle = circuit_draw_settings_dict['box_around_node_linestyle'],
                            marker = 'none',
                            linewidth = circuit_draw_settings_dict['box_around_node_linewidth'],
                            color = circuit_draw_settings_dict['box_around_node_color']
                        )
            
                if node_has_optical_channel:
                    # plot a marker per channel
                    for index in range(len(current_node_info['channels'])):
                        # plot the node marker for optical channels
                        axis.plot(
                            node_x,
                            node_y_values_optical[index],
                            markeredgewidth = current_node_info['markeredgewidth'],
                            marker = current_node_info['markers'][index],
                            markersize = current_node_info['markersize'][index],
                            color = current_node_info['markercolor'][index],
                            markerfacecolor = current_node_info['markerfacecolor'][index],
                            fillstyle= current_node_info['fillstyle'][index],
                            alpha=1
                            )
                        # write the text in the node marker
                        axis.plot(
                            node_x,
                            node_y_values_optical[index],
                            linestyle = 'none',
                            markeredgewidth = 0.3,
                            marker = current_node_info['marker_text'][index],
                            markersize = current_node_info['marker_text_fontsize'][index],
                            color = current_node_info['marker_text_color'][index],
                            markerfacecolor = current_node_info['marker_text_color'][index],
                            fillstyle='full',
                            alpha=1
                            )
                # plot a classical markers for the relevant channels
                if node_has_classical_channel:
                    for index in range(len(current_node_info['channels_classical'])):
                        axis.plot(
                            node_x,
                            node_y_values_classical[index],
                            markeredgewidth = 1,
                            marker = current_node_info['classical_marker'][index],
                            markersize = current_node_info['classical_marker_size'][index],
                            color = current_node_info['classical_marker_color'][index],
                            markerfacecolor = current_node_info['classical_marker_color'][index],
                            fillstyle='full',
                            alpha=1
                            )
                        # write the text in the node marker
                        axis.plot(
                            node_x,
                            node_y_values_classical[index],
                            linestyle = 'none',
                            markeredgewidth = 0.3,
                            marker = current_node_info['classical_marker_text'][index],
                            markersize = current_node_info['classical_marker_text_fontsize'][index],
                            color = current_node_info['classical_marker_text_color'][index],
                            markerfacecolor = current_node_info['classical_marker_text_color'][index],
                            fillstyle='full',
                            alpha=1
                            )

        # if last node of circuit was 'bridge' plot next circuit. Otherwise show the plot.
        if this_node_is_a_bridge:
            next_fock_state_circuit = self.node_list[-1].get('next_fock_state_circuit')
            next_fock_state_circuit.draw(print_defaults = False, 
                                        settings_for_drawing_circuit = settings_for_drawing_circuit,
                                        compound_circuit_settings =compound_circuit_settings)
        else:
            # if this is the end of the compound circuit go through all pages and ensure 
            # that there are no empty channel lines extending. Cut the lines right after
            # last node on the page
            last_node_on_page = []
            for node_position, channels_occupied in enumerate(compound_circuit_settings['node_positions_occupied']):
                            
                page_number, node_number_on_page= divmod(node_position, len(compound_circuit_settings['node_x_values']))
                if channels_occupied != 0:
                    if page_number < len(last_node_on_page):
                        if node_number_on_page > last_node_on_page[page_number][0]:
                            last_node_on_page[page_number] = (node_number_on_page,node_position)
                    else:
                        last_node_on_page.append( (node_number_on_page,node_position) )
            for page_number, node_indices in enumerate(last_node_on_page):
                axis = compound_circuit_settings['plot_axs'][page_number][0]
                last_node_x = compound_circuit_settings['node_x_values'][node_indices[0]]     

                active_optical_channels = compound_circuit_settings['active_optical_channels_per_node'][node_indices[1]]      
                active_classical_channels = compound_circuit_settings['active_classical_channels_per_node'][node_indices[1]]      

                eol_x = last_node_x + 0.5 * compound_circuit_settings['spacing_between_nodes']
                axis.add_patch(Rectangle((eol_x, 0), compound_circuit_settings['canvas_width'] - eol_x, compound_circuit_settings['canvas_height'],
                edgecolor = 'white',
                    facecolor = 'white',
                    fill=True,
                    lw=1,
                    zorder = 2))
                for index, y_value in enumerate(compound_circuit_settings['line_y_values_optical']):
                    if index < active_optical_channels:
                        axis.plot(
                            eol_x,
                            y_value,
                            marker = circuit_draw_settings_dict['optical_channel_line_marker'],
                            color = circuit_draw_settings_dict['optical_channel_line_color'],
                            alpha=1
                            )
                for index, y_value in enumerate(compound_circuit_settings['line_y_values_classical']):
                    if index < active_classical_channels:
                        axis.plot(
                            eol_x,
                            y_value,
                            marker = circuit_draw_settings_dict['classical_channel_line_marker'],
                            color = circuit_draw_settings_dict['classical_channel_line_color'],
                            )
            for page_number in range(len( last_node_on_page), len(compound_circuit_settings['plot_axs'])):
                compound_circuit_settings['figure'].delaxes(compound_circuit_settings['plot_axs'][page_number,0])                   
            plt.show()
        return
    
    
    # endregion
 
    def basis(self) -> dict:
        """ Function returns a dictonary with valid components names as keys and the corresponding photon numbers in 
            the channels as values.
        """

        collection_of_states = cos.CollectionOfStates(fock_state_circuit=self, input_collection_as_a_dict=dict([]))
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
    
    def _apply_time_delay(self, channel_numbers, time_delay_tau,bandwidth_omega, input_state):
        """ BETA: This gate will model the impact of a time delay for channels that have light pulses with 
            a finite duration. The gate mimics the 'decoherence' but does not actually add spectral or temporal 
            information to the state.

            The gate splits the state in three parts, a part representing the temporal overlap and two 
            states represent a 'ahead' and a 'behind' part. The states will receive a label indicating to 
            the 'plot' function that they should be grouped together.

            The impact of delay is determined by the overlap percentage which follows from the formula 
            overlap = e^(-1*(time_delay_tau*bandwidth_omega)**2). So if time_delay_tau*bandwidth_omega = 1 the 
            overlap is 1/e which is roughly 50%.

            Important note: This function creates a statistical mixture based on bandwidth and time delay. 
            this cannot be reversed. So adding more than one of these gates in the circuit will cause unexpected 
            behavior (e.g., adding two time delay gates one with +t and one with -t will NOT cancel out).

            Args:
                channel_numbers (list[int]): nodes to be includes in the matrix calculation (first node is node 0). 
                time_delay_tau (float): time delay used to calculate the impact of the delay
                bandwidth_omega (float): bandwidth used to calculate the impact of delay. 
                input_state (cos.State): State on which to apply the time delay


            Returns:
                tuple: tuple of states, or tuple with single state if delay is zero
        """
        if time_delay_tau == 0:
            return (input_state,)

        if not isinstance(channel_numbers, list):
            channel_numbers = [channel_numbers]
        
        overlap = np.exp(-1.0*np.log(2)*np.power(time_delay_tau*bandwidth_omega, 2.0))
        separation = 1-overlap

        identification_code = np.random.randint(1000000)
        # first create state for the overlap situation
        state_overlap = input_state.copy()
        state_overlap .cumulative_probability = input_state.cumulative_probability * overlap
        
        # then add states for the non-overlap situation 'behind'
        state_behind = input_state.copy()
        state_behind.cumulative_probability = input_state.cumulative_probability * separation/2.0
        state_behind.measurement_results.append({'coincidence indicator': identification_code, 'label' : 'ahead'})
        new_optical_components = []
        for name, amp_prob in state_behind.optical_components.items():
            values = input_state._dict_of_valid_component_names[name].copy()
            amplitude = amp_prob['amplitude']
            for channel_index, value in enumerate(values):
                if channel_index in channel_numbers:
                    values[channel_index] = 0
            for trial_name, trial_values in input_state._dict_of_valid_component_names.items():
                if values == trial_values:
                    new_name = trial_name
                    break
            new_optical_components.append((new_name,amplitude)) 
        state_behind.optical_components = new_optical_components

        # then add states for the non-overlap situation 'ahead'
        state_ahead = input_state.copy()
        state_ahead.cumulative_probability = input_state.cumulative_probability * separation/2.0
        state_ahead.measurement_results.append({'coincidence indicator': identification_code, 'label' : 'behind'})
        new_optical_components = []
        for name, amp_prob in state_ahead.optical_components.items():
            values = input_state._dict_of_valid_component_names[name].copy()
            amplitude = amp_prob['amplitude']
            for channel_index, value in enumerate(values):
                if channel_index not in channel_numbers:
                    values[channel_index] = 0
            for trial_name, trial_values in input_state._dict_of_valid_component_names.items():
                if values == trial_values:
                    new_name = trial_name
                    break    
            new_optical_components.append((new_name,amplitude)) 
        state_ahead.optical_components = new_optical_components
        
        return (state_overlap, state_behind, state_ahead)
    
class CompoundFockStateCircuit:
    ''' Class for compound FockStateCircuits. The class is used to work with a list of circuits that have to be executed sequentially. 
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
                    collection_of_states_input: cos.CollectionOfStates = None
                    ) -> cos.CollectionOfStates:
                    
                    Evaluate the compount circuit for a given collection of input states.
                    Method is a shell which calls FockStateCircuit.evaluate_circuit() to evaluate the compound circuit for the given
                    input collection of states.

    '''
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
        ''' Update the internal list of circuits connected with bridges. This method has to be called 
            after making a change to 'list_of_circuits' before  running the circuit with 
            'evaluate_circuit()' or draw a circuit schematics with 'draw()'. 
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
        ''' Draw the compound circuit. 'settings_for_drawing_circuit' is the dict with settings 
            for circuit drawing. Method is a shell which calls FockStateCircuit.draw() to execute the schematics.

            If changes to the compound circuit are made call CompoundFockStateCircuit.refresh() before 
            calling this method.
        '''
        if settings_for_drawing_circuit is None:
            settings_for_drawing_circuit = dict({'compound_circuit_title' : self.compound_circuit_name})
        self._list_of_circuits_with_bridges[0].draw(print_defaults=print_defaults, settings_for_drawing_circuit=settings_for_drawing_circuit)
        return

    def evaluate_circuit(self, 
                        collection_of_states_input: cos.CollectionOfStates = None) -> cos.CollectionOfStates:
        ''' Evaluate the compount circuit for a given collection of input states. Method is a shell which 
            calls FockStateCircuit.evaluate_circuit() to evaluate the compound circuit for the given
            input collection of states.

            If changes to the compound circuit are made call CompoundFockStateCircuit.refresh() before 
            calling this method.
        '''
        return self._list_of_circuits_with_bridges[0].evaluate_circuit(collection_of_states_input=collection_of_states_input)
        