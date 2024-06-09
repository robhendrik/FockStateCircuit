from __future__ import annotations
from fock_state_circuit.nodes.nodelist.node_list import NodeList
import math
import numpy as np

class OpticalNodes(NodeList):    
    """Method(s) to add wave plates to the circuit:

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
                    'channels_optical' can be mixed by this beamsplitter. 

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

        Last modified: April 16th, 2024
                
    """
    _VERSION = '1.0.0'
    def _generate_generic_optical_matrix(self, theta: float = 0, phi: float = 0): # -> np.array[np.cdouble]:

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
        tensor_list = self._translate_channels_numbers_to_tensor_list(channel_numbers = channel_numbers)
        matrix_optical = self._generate_generic_optical_matrix(theta, phi)
        self._update_list_of_nodes({'matrix_optical': matrix_optical, 'tensor_list':tensor_list, 'node_type': node_type, 'node_info': node_info})

        return
    
    def wave_plate(self,
                    channel_horizontal: int = 0,
                    channel_vertical: int = 1,
                    theta: float = 0, 
                    phi: float = 0,
                    node_type: str = 'optical',
                    node_info: dict = None
                    ) -> None:  
        """ Apply a wave plate with phase delay between fast and slow axis of Phi, and a rotation of the axis over Theta to the two channels. 
            If Phi is equal to pi this will be a half wave plate, if Phi is equal to pi/2 this is a quarter wave plate.

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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "half-wave plate",
                    'channels_optical' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\theta\phi$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info

        self.__apply_generic_rotation_2_channels(channel_horizontal, channel_vertical, theta = theta, phi = phi, node_info = node_info)
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
            'channels_optical' can be mixed by this beamsplitter.      
        
        Args:
            input_channels_a (tuple[int,int]): tuple with channel numbers for input port a
            input_channels_b (tuple[int,int]): tuple with channel numbers for input port b
            reflection (float): reflection fraction
            transmission (float): transmission fraction
            node_info (dict): Information for displaying the node when drawing circuit. Defaults to None, in that case default image is used
        
        Raises:
            Exception if number of channels in the circuit does not match requirement for the operation
        """
        if node_info == None: node_info = {}

        node_info_0 = {
                    'label' : "NPBS",
                    'channels_optical' : [input_channels_a[0], input_channels_b[0]],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full',
                    'combined_gate': 'NPBS'
                }|node_info
        node_info_1 = {
                    'label' : "NPBS",
                    'channels_optical' : [input_channels_a[1], input_channels_b[1]],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full',
                    'combined_gate': 'NPBS'
                }|node_info
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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "half-wave plate",
                    'channels_optical' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{2}$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info

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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "half-wave plate",
                    'channels_optical' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{2}$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info

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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "half-wave plate",
                    'channels_optical' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{2}$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "qtr-wave plate",
                    'channels_optical' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{4}$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "qtr-wave plate",
                    'channels_optical' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{4}$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info

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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "qtr-wave plate",
                    'channels_optical' : [channel_horizontal, channel_vertical],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$\frac{\lambda}{4}$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "phase-shift",
                    'channels_optical' : [channel_for_shift],
                    'markers' : ['o'],
                    'markercolor' : ['pink'],
                    'markerfacecolor' : ['pink'],
                    'marker_text' : [r"$\phi$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "swap gate",
                    'channels_optical' : [first_channel, second_channel],
                    'markers' : ['v','^'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$s$",r"$s$"],
                    'marker_text_fontsize' : [5,5],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "mix 50/50",
                    'channels_optical' : [first_channel, second_channel],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "mix generic",
                    'channels_optical' : [first_channel, second_channel],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
        if reflection + transmission != 1:
            raise Exception(
                'For generic mixing the sum or reflection and transmission has to be 1 and both have to between 0 and 1 (including)'
                )
        theta = math.atan2(math.sqrt(reflection),math.sqrt(transmission))
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
        if node_info == None: node_info = {}

        node_info = {
                    'label' : "PBS",
                    'channels_optical' : [input_channels_a[1], input_channels_b[1]],
                    'markers' : ['s'],
                    'markercolor' : ['purple'],
                    'markerfacecolor' : ['grey'],
                    'marker_text_fontsize' : [10],
                    'marker_text' : [r"$M$",r"$M$"],
                    'markersize' : 20,
                    'fillstyle' : 'full'
                }|node_info
        
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
        if node_info == None: node_info = {}

        node_info_0 = {
                'label' : "NPBS 50/50",
                'channels_optical' : [input_channels_a[0], input_channels_b[0]],
                'markers' : ['s'],
                'markercolor' : ['purple'],
                'markerfacecolor' : ['grey'],
                'marker_text' : [r"$M$",r"$M$"],
                'markersize' : 20,
                'fillstyle' : 'full',
                'combined_gate': 'NPBS5050'
            }|node_info
        node_info_1 = {
                'label' : "NPBS",
                'channels_optical' : [input_channels_a[1], input_channels_b[1]],
                'markers' : ['s'],
                'markercolor' : ['purple'],
                'markerfacecolor' : ['grey'],
                'marker_text' : [r"$M$",r"$M$"],
                'markersize' : 20,
                'fillstyle' : 'full',
                'combined_gate': 'NPBS5050'
                }|node_info
        if self._no_of_optical_channels < 4:
            raise Exception('For beamsplitter 4 channels are needed in the circuit')
        self.mix_50_50(first_channel = input_channels_a[0], second_channel = input_channels_b[0], node_info = node_info_0)
        self.mix_50_50(first_channel = input_channels_a[1], second_channel = input_channels_b[1], node_info = node_info_1)
        return
    
    