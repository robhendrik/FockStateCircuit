""" This modules contains the class PopescuRohrlichPhotonPairAmplitudes(). The purpose is to take
    care of the calculations to 'unravel' an optical state containing a number of photons into 
    the states of these contributing photons. If the overall state is not entangled this 'unravelling'
    leads to a unique outcome which can be used, if the state is 'entangled' than there is no
    meaning to the state of the contributing photons (only the overall state is defined in this case.). 
    In case of entanglement the values for individual photons returned by this class have no meaning. 
    For this reason the class contains methods to check whether the original state is entangled.

    NOTE: This class will return values for the individual photons in any case and will not throw]
    and error in case of entanglement. The higher level code needs to take care of checking this 
    before processing the result (use PopescuRohrlichPhotonPairAmplitudes.is_dict_factorable() and 
    PopescuRohrlichPhotonPairAmplitudes.are_all_pairs_factorable() ). These functions contain a 
    threshold setting to suppress rounding noise in the determination whether the state can be factored. 
    Default this threshold is set at PPM level (i.e., errors below order of magnitude 1e-6 will accepted)

    This class takes as input a dictionary with as keys strings consisting of the characters '0', '1', 
    '2' or '3' and length d. The values are complex floats. So the input dictionary looks like
    {'000': 1, '001': 0.2j, '321': 0.1 + 0.3j}

    1. The constructor first generates a d-dimensional matrix representing the data. The value in the 
    dict at key '131' will be in the matrix at index matrix[1][3][1]. This data is stored as 
    self.resulting_matrix

    2. As next step the contstructor decomposes the matrix in d vectors of length 4 such that
    the values in the matrix are the product of the values in the vectors. So the matrix element matrix[1][3][1]
    is then vector[0][1] x vector[1][3] x vector[2][1]. This data is stored as self.reproduced vectors

    3. As third step the vectors of (length 4) are decomposed in two vectors of length 2 such that if we have
    the 2-vectors as [a,b] and [c,d] the original vector was [ab,ad,bc,bd]. The vectors of length 2 are stored
    as self.photon_amplitudes

    This means that IF the original dictionary represents the situation where and optical state in composed of
    non-entangled photons (i.e., if the wave function can be written as the product of the wave functions of
    the individual photons) this constructor will generate the wave functions of these photons.  
    
    If the dictionary represents and entangled state the constructor will also generate photon values, but these
    have no physical meaning. When using this class to always have to check with self.is_dict_factorable() and 
    self.are_all_pairs_factorable() whether the output is sensible.
    
    The convention used (for interfacing with the package FockStateCircuit) is that if the photon states represent
    polarization the states are written as [h,v] (i.e., [amplitude of horizontal component, amplitude of vertical 
    component]). The vectors for the photon pair are then [hh,hv,vh,vv]. The dictionary the amplitude 'hh' is then
    labelled as '0', the state 'hv' is labelled as '1', the state 'vh' is labelled as '2' and the state 'vv' is 
    labelled as '3' 

    Last modified: June 12th, 2024
"""
import numpy as np

_VERSION = '1.0.0'

class PopescuRohrlichPhotonPairAmplitudes():
    def __init__(self,input_dictionary):
        """ Constructor takes as input a dictionary with as keys strings consisting of the characters '0', '1', 
            '2' or '3' and length d. The values are complex floats. So the input dictionary looks like
            {'000': 1, '001': 0.2j, '321': 0.1 + 0.3j}

            1. The constructor first generates a d-dimensional matrix representing the data. The value in the 
            dict at key '131' will be in the matrix at index matrix[1][3][1]. This data is stored as 
            self.resulting_matrix

            2. As next step the contstructor decomposes the matrix in d vectors of length 4 such that
            the values in the matrix are the product of the values in the vectors. So the matrix element matrix[1][3][1]
            is then vector[0][1] x vector[1][3] x vector[2][1]. This data is stored as self.reproduced vectors

            3. As third step the vectors of (length 4) are decomposed in two vectors of length 2 such that if we have
            the 2-vectors as [a,b] and [c,d] the original vector was [ab,ad,bc,bd]. The vectors of length 2 are stored
            as self.photon_amplitudes
        
            This means that IF the original dictionary represents the situation where and optical state in composed of
            non-entangled photons (i.e., if the wave function can be written as the product of the wave functions of
            the individual photons) this constructor will generate the wave functions of these photons.  
            
            If the dictionary represents and entangled state the constructor will also generate photon values, but these
            have no physical meaning. When using this class to always have to check with self.is_dict_factorable() and 
            self.are_all_pairs_factorable() whether the output is sensible.
            
            The convention used (for interfacing with the package FockStateCircuit) is that if the photon states represent
            polarization the states are written as [h,v] (i.e., [amplitude of horizontal component, amplitude of vertical 
            component]). The vectors for the photon pair are then [hh,hv,vh,vv]. The dictionary the amplitude 'hh' is then
            labelled as '0', the state 'hv' is labelled as '1', the state 'vh' is labelled as '2' and the state 'vv' is 
            labelled as '3' 
        """
        # reproduced_vectors will be a list of vectors (numpy array). The list has length d and the vectors have length 4
        self.reproduced_vectors = None 

        # resulting_matrix  will be a d-dimensional matrix (numpy array of complex numbers) with each axis of length 4  
        self.resulting_matrix = None 

        # photon_amplitudes will be a list of length 2d where each element is a list (numpy array) of length 2
        self.photon_amplitudes = None 

        # this will be a list of stokes vectors in form ([x_coordinate,y_coordinate,z_coordinate] , np.sqrt(normalization_factor))
        self.stokes_vectors = None 

        self.overall_probability = None

        # this is a mapping to determine which photons in the list self.stokes-vector and self.photon_amplitude
        # belong to what pair. Format is [(0,1),(2,3),(4,5)] to indicate that photons 0 and 1 form pair 0, photons
        # 2 and 3 form pair 1 etc. Right now this mapping is trivial, included for future use if needed.
        self.photon_indices_per_pair = None 

        if len(input_dictionary) > 0:
            self.resulting_matrix = self.generate_matrix_from_dict(input_dictionary)

            number_of_boxes = len(self.resulting_matrix.shape)
            if number_of_boxes > 1:
                self.reproduced_vectors = []
                for box_number in range(number_of_boxes):
                    swapped_matrix = np.swapaxes(self.resulting_matrix,box_number,0)
                    new_vector = self.vector_from_matrix(swapped_matrix)
                    self.reproduced_vectors.append(new_vector)
            else:
                self.reproduced_vectors = [self.resulting_matrix]
            
            self.photon_amplitudes = []
            self.photon_indices_per_pair = []
            photon_index = 0
            for reproduced_vector in self.reproduced_vectors:
                self.photon_amplitudes.append(self.photon_amplitudes_from_reproduced_vector(reproduced_vector)[0])
                self.photon_amplitudes.append(self.photon_amplitudes_from_reproduced_vector(reproduced_vector)[1])
                self.photon_indices_per_pair.append((photon_index, photon_index + 1))
                photon_index += 2

            self.stokes_vectors = []
            for photon_amplitude in self.photon_amplitudes:
                self.stokes_vectors.append(self.stokes_vector_from_amplitudes(photon_amplitude))

            self.overall_probability = sum([np.abs(amp)**2 for amp in input_dictionary.values()])

        return

    def vector_from_matrix(self,matrix: np.array) -> np.array:
        """ Generate a vector of length 4, reprsenting a photon pair, for the first index of the matrix, 
            assuming that the matrix can be factored (i.e., represents a state that is the product of states
            of the photon pairs.)
        """
        # first rescale the matrix to make the sum of the absolute values squared equal to one
        absolute_square = np.square(np.abs(matrix))
        rescaled_matrix = matrix/ np.sqrt(np.sum(absolute_square))
        # then make the phase of the highest element 0 (so the highest value is a positive, real number)
        indices_of_max= np.unravel_index(np.argmax(absolute_square, axis=None), absolute_square.shape)
        phase = rescaled_matrix[indices_of_max] / np.abs(rescaled_matrix[indices_of_max])
        rescaled_matrix = matrix / phase

        # determine the ratios of the absolute values
        squares = np.array([np.sum(absolute_square[d,:]) for d in range(absolute_square.shape[0])])
        squares = squares / np.sum(squares)
        ratios = np.sqrt(squares)
        
        # determine the phase between the elements
        angles = np.array([np.angle(rescaled_matrix[tuple([d]+list(indices_of_max[1:]))]) for d in range(rescaled_matrix.shape[0])])

        new_vector = np.array([radius*np.exp(1j*angle) for radius,angle in zip(ratios,angles)])
        return new_vector
    
    def dict_from_reproduced_vectors(self) -> dict:
        """ Generate a dictionary from a set of vectors of length 4 (representing photon pairs). This is the
            reverse operation of what is done in the constructor. If you would create a new instance with the dictionary
            that comes out of this function the values in self.reproduced_vectors are the same as the reproduced
            vectors in this instance. We use this to check if we can indeed have a situation where the wave function
            can be factored.
        """
        dictionary_representing_matrix = dict([])
        for overall_index in range(4**len(self.reproduced_vectors)):
            coordinates = [(overall_index//(4**n))%4 for n in range(len(self.reproduced_vectors))]
            label = "".join([str(index) for index in coordinates])
            value = np.prod([vector[coordinates[n]] for n, vector in enumerate(self.reproduced_vectors)])
            dictionary_representing_matrix.update({label:value})
        return dictionary_representing_matrix
    
    def generate_matrix_from_dict(self,input_dictionary: dict) -> np.array:
        """ Generates a d-dimensional matrix representing the data. The value in the 
            dict at key '131' will be in the matrix at index matrix[1][3][1]. This data is stored as 
            self.resulting_matrix. Function takes as input a dictionary with as keys strings consisting 
            of the characters '0', '1', '2' or '3' and length d. The values are complex floats. 
            So the input dictionary looks like {'000': 1, '001': 0.2j, '321': 0.1 + 0.3j}
        """
        
        dimensions = len(list(input_dictionary.keys())[0])
        resulting_matrix = np.zeros(shape=tuple([4]*dimensions),dtype=np.csingle)
        for label,value in input_dictionary.items():
            coordinates = tuple([int(c) for c in label])
            resulting_matrix[coordinates] = value
        return resulting_matrix

    def compare_matrices_ppm_level(self,original_matrix: np.array, new_matrix: np.array) -> float:
        """ Return a value representing the difference between the two matrices. The matrices will be normalized first, 
            meaning the sum of the absolute value squared for all values is 1, and the element with the largest absolute 
            value will have phase zero (i.e., will be a positive real number). For these rescaled matrices we then sum per 
            element the square of the absolute value of the difference. This value we return x 1 million.

            If we have  a 4x4 matrix with 4 elements of value around 1 and the rest zero, a deviation of one element 
            of ~2e-6 would bring the return value of this function above 1.

            Matrices should be numpy array of same shape consisting of complex numbers.
        """
        # first we rescale the matrices such the sum of the absolute values squared is equal to one
        original_matrix = original_matrix/ np.sqrt(np.sum(np.square(np.abs(original_matrix))))
        new_matrix = new_matrix/ np.sqrt(np.sum(np.square(np.abs(new_matrix))))
        # then we establish the indices of the element with the largest absolute value in the original matrix
        indices_of_max= np.unravel_index(np.argmax(np.abs(original_matrix), axis=None), original_matrix.shape)
        # then we get the phase of the elements at these indices
        phase_original =  original_matrix[indices_of_max] / np.abs(original_matrix[indices_of_max])
        phase_new = new_matrix[indices_of_max] / np.abs(new_matrix[indices_of_max])
        # finally we scale the overall phase to make sure that the element at 'indices_of_max' is a positive real number
        original_matrix = original_matrix/ phase_original
        new_matrix = new_matrix/ phase_new
        # now we can compare the matrices by summing up the square of teh difference
        difference= np.sum(np.abs(original_matrix - new_matrix)**2,axis=None)
        return np.sqrt(difference) * 1000000

    def is_dict_factorable(self,threshold: float = 1.0) -> bool:
        """ Return True if the values in the dictionary can be written as the product of photon-pair
            values. With threshold at default value 1.0 the function will return True if the deviation is smaller
            than 1e-6 (order of magnitude, not exact value).
        """
        original_matrix = self.resulting_matrix
        reproduced_matrix = self.generate_matrix_from_dict(self.dict_from_reproduced_vectors())
        difference = self.compare_matrices_ppm_level(original_matrix, reproduced_matrix)
        return difference < threshold
    
    def photon_amplitudes_from_reproduced_vector(self,reproduced_vector: np.array) -> list[np.array]:
        """ Calculate the amplitudes for horizontal and vertical polarization for each photon in a photon pair 
            based on the 'reproduced_vector' representing the components for the HH, HV, VH and VV polarization 
            amplitudes of the photon pair (Here HH is the product of amplitudes for the horizontal polarization 
            components etc.).

            The function will return a list with the two vectors in format [amplitude for horizontal component, 
            amplitude for vertical component].

        """
        matrix = np.array([[reproduced_vector[0], reproduced_vector[1]],[reproduced_vector[2], reproduced_vector[3]]])
        number_of_boxes = len(matrix.shape)
        photon_amplitudes = []
        for box_number in range(number_of_boxes):
            swapped_matrix = np.swapaxes(matrix,box_number,0)
            new_vector = self.vector_from_matrix(swapped_matrix)
            photon_amplitudes.append(new_vector)
        return photon_amplitudes

    def reproduced_vector_from_photon_amplitudes(self, photon_amplitudes: list[np.array]) -> np.array:
        """ Generate a vector of length 4 representing the amplitudes for the photon pair ( as [hh,hv,vh,vv])
            from the amplitudes of the two photons in the pair (as [[h,v], [h,v]]).

        Args:
            photon_amplitudes (list[np.array]): Amplitudes of the two photons in the pair (as [[h,v], [h,v]]).

        Returns:
            np.array: A vector of length 4 representing the amplitudes for the photon pair ( as [hh,hv,vh,vv]).
        """
        order = ((0,0),(0,1),(1,0),(1,1))
        reproduced_vector = []
        for index in order:
            reproduced_vector.append(photon_amplitudes[0][index[0]] * photon_amplitudes[1][index[1]])
        return reproduced_vector

    def is_photon_pair_factorable(self, amplitudes, threshold: float = 1.0) -> bool:
        """ Return True if the values in the dictionary can be written as the product of photon-pair
            values. With threshold at default value 1.0 the function will return True if teh deviation is smaller
            than 1e-6 (order of magnitude, not exact value).
        """
        return self.compare_matrices_ppm_level(amplitudes, 
                                               self.reproduced_vector_from_photon_amplitudes(self.photon_amplitudes_from_reproduced_vector(amplitudes))
                                               ) < threshold
    
    def are_all_photon_pairs_factorable(self, threshold: float = 1.0) -> bool:
        """ Return True if the values in the dictionary can be written as the product of photon-pair
            values. With threshold at default value 1.0 the function will return True if teh deviation is smaller
            than 1e-6 (order of magnitude, not exact value).
        """
        all_factorable = True
        for amplitudes in self.reproduced_vectors:
            all_factorable &= self.is_photon_pair_factorable(amplitudes)
        return all_factorable
    

    def stokes_vector_from_amplitudes(self, photon_amplitude: list[np.csingle]) -> tuple:
        """ Calculates the Stokes vector from amplitudes for a single photon. The function returns a tuple containing
            a list with 3 floats representing x,y and z coordinates of the vector, and a float
            representing the normalization factor (the length of the vector before normalization). The vector
            represented by the list has length 1. The north-south pole is the z-axis (3rd coordinate). 
            If the phase difference between the amplitudes is zero the vector in the xy-plane is 
            pointing to the y-coordinate.

            As example: if northpole represents horizontal polarization and south pole vertical polarization 
            then the 'y-pole' represents diagonal polarization and the 'x-pole' circular polarization.

        Args:
            photon_amplitude: List with amplitudes for single photon like 
                [amplitude_north_pole (numpy complex), amplitude_south_pole (numpy complex)]
o
        Returns:
            tuple: tuple (vector, normalization). vector is a list of 3 floats representing x,y and x coordinates 
            of the Stokes vector.

        Raises:
            Exception when both amplitudes have length zero
        """
        amplitude_north_pole =  photon_amplitude[0]
        amplitude_south_pole =  photon_amplitude[1]
        normalization_factor = np.abs(amplitude_north_pole)**2 + np.abs(amplitude_south_pole)**2
        if normalization_factor == 0:
            raise Exception("Stokes vector cannot have length zero")
        z_coordinate = (np.abs(amplitude_north_pole)**2 -np.abs(amplitude_south_pole)**2)/normalization_factor

        xy_length = np.sqrt(1-z_coordinate**2)
        if np.abs(amplitude_north_pole) == 0 or np.abs(amplitude_south_pole) == 0:
            phase_between_amplitudes = 0
        else:
            phase_between_amplitudes = np.angle(amplitude_north_pole/amplitude_south_pole)
        y_coordinate = np.cos(phase_between_amplitudes) * xy_length
        x_coordinate = np.sin(phase_between_amplitudes) * xy_length

        return ([x_coordinate,y_coordinate,z_coordinate] , np.sqrt(normalization_factor))

