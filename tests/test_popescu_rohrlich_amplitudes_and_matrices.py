import sys  
sys.path.append("./src")
import fock_state_circuit as fsc

from fock_state_circuit.popescu_rohrlich_correlation_functionality.popescu_rohrlich_photon_pair_amplitude_functionality import PopescuRohrlichPhotonPairAmplitudes
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.colors
import random

from unittest.mock import Mock
from unittest.mock import patch
import matplotlib.testing.compare as plt_test
import pytest

def test_complete_class_PR_photon_pair_amplitude():
    # test complete class
    def inner(vector1,vector2):
        return np.round(np.inner(np.conj(vector1),vector2),2)

    no_error = True
    # 2 photons
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))

    photons = [[1,0],[1,0]]
    index = 0
    list_of_amplitudes = []
    while index < len(photons):
        pair_amplitudes_hh_hv_vh_vv =  test_case.reproduced_vector_from_photon_amplitudes([photons[index],photons[index+1]])
        index = index + 2
        list_of_amplitudes.append(pair_amplitudes_hh_hv_vh_vv)
    test_case.reproduced_vectors = list_of_amplitudes
    dict_from_list_of_amplitudes = test_case.dict_from_reproduced_vectors()

    new_case = PopescuRohrlichPhotonPairAmplitudes(input_dictionary=dict_from_list_of_amplitudes)
    no_error &= new_case.is_dict_factorable()
    no_error &= new_case.are_all_photon_pairs_factorable()


    reproduced_photons = new_case.photon_amplitudes
    for new_photon, old_photon in zip(reproduced_photons,photons):
        new_photon = np.array(new_photon) * np.exp(-1j*np.angle(new_photon[0]))
        old_photon = np.array(old_photon) * np.exp(-1j*np.angle(old_photon[0]))
        no_error &= inner(new_photon,old_photon) == 1

    # 4 photons
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))

    a = np.sqrt(1/2)
    photons = [[1j,0],[1,0],[a,a],[-1*a,a]]
    index = 0
    list_of_amplitudes = []
    while index < len(photons):
        pair_amplitudes_hh_hv_vh_vv =  test_case.reproduced_vector_from_photon_amplitudes([photons[index],photons[index+1]])
        index = index + 2
        list_of_amplitudes.append(pair_amplitudes_hh_hv_vh_vv)
    test_case.reproduced_vectors = list_of_amplitudes
    dict_from_list_of_amplitudes = test_case.dict_from_reproduced_vectors()

    new_case = PopescuRohrlichPhotonPairAmplitudes(input_dictionary=dict_from_list_of_amplitudes)
    no_error &= new_case.is_dict_factorable()
    no_error &= new_case.are_all_photon_pairs_factorable()


    reproduced_photons = new_case.photon_amplitudes
    for new_photon, old_photon in zip(reproduced_photons,photons):
        new_photon = np.array(new_photon) * np.exp(-1j*np.angle(new_photon[0]))
        old_photon = np.array(old_photon) * np.exp(-1j*np.angle(old_photon[0]))
        no_error &= inner(new_photon,old_photon) == 1

    # 4 photons
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))

    thetas = [0,np.pi,np.pi/2,np.pi]
    phis = [0,-np.pi/4,-np.pi/8,np.pi/2]
    photons = [ [np.cos(theta)*np.exp(1j*phi), np.sin(theta)*np.exp(-1j*phi)] for theta, phi in zip(thetas,phis)]
    index = 0
    list_of_amplitudes = []
    while index < len(photons):
        pair_amplitudes_hh_hv_vh_vv =  test_case.reproduced_vector_from_photon_amplitudes([photons[index],photons[index+1]])
        index = index + 2
        list_of_amplitudes.append(pair_amplitudes_hh_hv_vh_vv)
    test_case.reproduced_vectors = list_of_amplitudes
    dict_from_list_of_amplitudes = test_case.dict_from_reproduced_vectors()

    new_case = PopescuRohrlichPhotonPairAmplitudes(input_dictionary=dict_from_list_of_amplitudes)
    no_error &= new_case.is_dict_factorable()
    no_error &= new_case.are_all_photon_pairs_factorable()


    reproduced_photons = new_case.photon_amplitudes
    for new_photon, old_photon in zip(reproduced_photons,photons):
        new_photon = np.array(new_photon) * np.exp(-1j*np.angle(new_photon[0]))
        old_photon = np.array(old_photon) * np.exp(-1j*np.angle(old_photon[0]))
        no_error &= inner(new_photon,old_photon) == 1

    # 8 photons
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))
    a = np.sqrt(1/2)
    photons = [[1j,0],[1,0],[a,a],[-1*a,a],[1j,0],[1,0],[a,a],[-1*a,a]]
    index = 0
    list_of_amplitudes = []
    while index < len(photons):
        pair_amplitudes_hh_hv_vh_vv =  test_case.reproduced_vector_from_photon_amplitudes([photons[index],photons[index+1]])
        index = index + 2
        list_of_amplitudes.append(pair_amplitudes_hh_hv_vh_vv)
    test_case.reproduced_vectors = list_of_amplitudes
    dict_from_list_of_amplitudes = test_case.dict_from_reproduced_vectors()

    new_case = PopescuRohrlichPhotonPairAmplitudes(input_dictionary=dict_from_list_of_amplitudes)
    no_error &= new_case.is_dict_factorable()
    no_error &= new_case.are_all_photon_pairs_factorable()

    reproduced_photons = new_case.photon_amplitudes
    for new_photon, old_photon in zip(reproduced_photons,photons):
        new_photon = np.array(new_photon) * np.exp(-1j*np.angle(new_photon[0]))
        old_photon = np.array(old_photon) * np.exp(-1j*np.angle(old_photon[0]))
        no_error &= inner(new_photon,old_photon) == 1

    # 8 photons
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))
    thetas = [0,np.pi,np.pi/2,np.pi,-np.pi,-np.pi/2,-np.pi/8,np.pi/16]
    phis = [0,-np.pi/4,-np.pi/8,np.pi/2,np.pi,-np.pi/2,np.pi/16,0]

    phis = [np.pi/16,0]
    photons = [ [np.cos(theta)*np.exp(1j*phi), np.sin(theta)*np.exp(-1j*phi)] for theta, phi in zip(thetas,phis)]
    index = 0
    list_of_amplitudes = []
    while index < len(photons):
        pair_amplitudes_hh_hv_vh_vv =  test_case.reproduced_vector_from_photon_amplitudes([photons[index],photons[index+1]])
        index = index + 2
        list_of_amplitudes.append(pair_amplitudes_hh_hv_vh_vv)
    test_case.reproduced_vectors = list_of_amplitudes
    dict_from_list_of_amplitudes = test_case.dict_from_reproduced_vectors()


    new_case = PopescuRohrlichPhotonPairAmplitudes(input_dictionary=dict_from_list_of_amplitudes)
    no_error &= new_case.is_dict_factorable()
    no_error &= new_case.are_all_photon_pairs_factorable()

    reproduced_photons = new_case.photon_amplitudes

    for new_photon, old_photon in zip(reproduced_photons,photons):
        new_photon = np.array(new_photon) * np.exp(-1j*np.angle(new_photon[0]))
        old_photon = np.array(old_photon) * np.exp(-1j*np.angle(old_photon[0]))
        no_error &= inner(new_photon,old_photon) == 1
    
    assert no_error

def test_complete_class_non_factorizable():
    no_error = True
    # 4 photons
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))

    a = np.sqrt(1/2)
    photons = [[a,0],[a,0],[1j,0],[1,0]]
    index = 0
    list_of_amplitudes = []
    while index < len(photons):
        pair_amplitudes_hh_hv_vh_vv =  test_case.reproduced_vector_from_photon_amplitudes([photons[index],photons[index+1]])
        index = index + 2
        list_of_amplitudes.append(pair_amplitudes_hh_hv_vh_vv)
    test_case.reproduced_vectors = list_of_amplitudes
    dict_from_list_of_amplitudes_1 = test_case.dict_from_reproduced_vectors()

    a = np.sqrt(1/2)
    photons = [[0,a],[0,a],[1j,0],[1,0]]
    index = 0
    list_of_amplitudes = []
    while index < len(photons):
        pair_amplitudes_hh_hv_vh_vv =  test_case.reproduced_vector_from_photon_amplitudes([photons[index],photons[index+1]])
        index = index + 2
        list_of_amplitudes.append(pair_amplitudes_hh_hv_vh_vv)
    test_case.reproduced_vectors = list_of_amplitudes
    dict_from_list_of_amplitudes_2 = test_case.dict_from_reproduced_vectors()

    dict_from_list_of_amplitudes = dict_from_list_of_amplitudes_1.copy()
    for k,v in dict_from_list_of_amplitudes_2.items():
        if k in dict_from_list_of_amplitudes:
            dict_from_list_of_amplitudes[k] += v
        else:
            dict_from_list_of_amplitudes[k] = v

    new_case = PopescuRohrlichPhotonPairAmplitudes(input_dictionary=dict_from_list_of_amplitudes)
    no_error &= new_case.is_dict_factorable() == True
    no_error &= new_case.are_all_photon_pairs_factorable() == False

    no_error &= new_case.is_photon_pair_factorable(new_case.reproduced_vectors[0]) == False
    no_error &= new_case.is_photon_pair_factorable(new_case.reproduced_vectors[1]) == True

    assert no_error


def test_complete_class_non_factorizable_2():
    no_error = True
    # 6 photons
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))

    a = np.sqrt(1/2)
    photons = [[1,0],[1,0],[1,0],[-1,0],[1,0],[a,1j*a]]
    index = 0
    list_of_amplitudes = []
    while index < len(photons):
        pair_amplitudes_hh_hv_vh_vv =  test_case.reproduced_vector_from_photon_amplitudes([photons[index],photons[index+1]])
        index = index + 2
        list_of_amplitudes.append(pair_amplitudes_hh_hv_vh_vv)
    test_case.reproduced_vectors = list_of_amplitudes
    dict_from_list_of_amplitudes_1 = test_case.dict_from_reproduced_vectors()

    a = np.sqrt(1/2)
    photons = [[0,1],[1,0],[1,0],[-1,0],[0,1],[a,1j*a]]
    index = 0
    list_of_amplitudes = []
    while index < len(photons):
        pair_amplitudes_hh_hv_vh_vv =  test_case.reproduced_vector_from_photon_amplitudes([photons[index],photons[index+1]])
        index = index + 2
        list_of_amplitudes.append(pair_amplitudes_hh_hv_vh_vv)
    test_case.reproduced_vectors = list_of_amplitudes
    dict_from_list_of_amplitudes_2 = test_case.dict_from_reproduced_vectors()

    dict_from_list_of_amplitudes = dict_from_list_of_amplitudes_1.copy()
    for k,v in dict_from_list_of_amplitudes_2.items():
        if k in dict_from_list_of_amplitudes:
            dict_from_list_of_amplitudes[k] += v
        else:
            dict_from_list_of_amplitudes[k] = v

    new_case = PopescuRohrlichPhotonPairAmplitudes(input_dictionary=dict_from_list_of_amplitudes)
    no_error &= new_case.is_dict_factorable() == False
    assert no_error

def test_complete_matrices():
    # test compare matrices
    no_error = True
    matrix1 = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
    matrix2 = np.array([[1,0,1e-6,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
    matrix3 = np.array([[1,0,0,0],[1j,0,0,0],[1,0,0,0],[1,0,0,0]])
    matrix4 = np.array([[1,0,0,0],[1j,0,1e-6*1j,0],[1,0,0,0],[1,0,0,0]])
    matrix5 = np.array([[0.999999,0,0,0],[1j,0,1e-6*1j,0],[0.999999,0,0,0],[1.000001,0,1e-6,0]])
    matrix6 = np.array([[1,0,0,0],[1j,0,3e-6*1j,0],[1,0,0,0],[1,0,0,0]])

    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))
    # for identical matrix there should be no difference
    no_error &=(test_case.compare_matrices_ppm_level(matrix1,matrix1) < 0.0001)
    no_error &=(test_case.compare_matrices_ppm_level(matrix3,matrix3) < 0.0001)
    # small change for order 1e-6 should leave difference below 1
    no_error &=(test_case.compare_matrices_ppm_level(matrix1,matrix2) < 1)
    no_error &=(test_case.compare_matrices_ppm_level(matrix3,matrix4) < 1)
    # bigger changes lift the difference above 1
    no_error &=(test_case.compare_matrices_ppm_level(matrix1,matrix4) > 1)
    no_error &=(test_case.compare_matrices_ppm_level(matrix3,matrix5) > 1)
    # 3e-6 lifts difference just above 1
    no_error &=(test_case.compare_matrices_ppm_level(matrix3,matrix6) >1)
    # also works for higher dimensional matrices
    matrix10 = np.array([[[1j,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],[[1,0,0,0],[1,0,0,0],[1j,0,0,0],[1,0,0,0]],[[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],[[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]])
    matrix11 = np.array([[[1j,0,0,0],[1,0,0,0],[1,0,5e-6,0],[1,0,0,0]],[[1,0,0,0],[1,0,0,0],[1j,0,0,0],[1,0,0,0]],[[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],[[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]])
    no_error &=(test_case.compare_matrices_ppm_level(matrix10,matrix10) < 0.0001)
    no_error &=(test_case.compare_matrices_ppm_level(matrix11,matrix10) > 1)
    assert no_error

def test_matrix_creation_and_decomposition():

    def check_list_of_vectors(list_of_vectors):
        test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))
        test_case.reproduced_vectors = list_of_vectors
        matrix = test_case.generate_matrix_from_dict(test_case.dict_from_reproduced_vectors())
        no_error = True
        for box_number in range(len(list_of_vectors)):
            swapped_matrix = np.swapaxes(matrix,box_number,0)
            normalization = np.sqrt(np.inner(np.conj(list_of_vectors[box_number]),list_of_vectors[box_number]))
            no_error &= (np.round(np.abs(np.inner(np.conj(list_of_vectors[box_number]), test_case.vector_from_matrix(swapped_matrix))/normalization),2) == 1)
            return no_error
    no_error = True

    vector_1 = [1,1,1,1]
    vector_2 = [1,1,0,1]
    list_of_vectors = [vector_1, vector_2]
    no_error &= (check_list_of_vectors(list_of_vectors))


    vector_1 = [2j,2j,2j,2j]
    vector_2 = [0,1j,0,0]
    list_of_vectors = [vector_1, vector_2]
    no_error &= (check_list_of_vectors(list_of_vectors))

    vector_1 = [1,1,1,1]
    vector_2 = [1,1,0,1]
    vector_3 = [3,0,0,0]
    list_of_vectors = [vector_1, vector_2,vector_3]
    no_error &= (check_list_of_vectors(list_of_vectors))

    vector_1 = [1,1,1,1]
    vector_2 = [1,1,0,1]
    vector_3 = [10,10,-19,1]
    list_of_vectors = [vector_1, vector_2,vector_3]
    no_error &= (check_list_of_vectors(list_of_vectors))

    vector_1 = [1,1,1,1]
    vector_2 = [0,1,0,0]
    list_of_vectors = [vector_1, vector_2]
    no_error &= (check_list_of_vectors(list_of_vectors))

    vector_1 = [1j,1,1,1]
    vector_2 = [1,-1j,0,1]
    list_of_vectors = [vector_1, vector_2]
    no_error &= (check_list_of_vectors(list_of_vectors))
    assert no_error


def test_detection_of_matrices_that_cannot_be_factored():
    def dict_from_list_of_vectors(list_of_list_of_vectors):
        """ Function to create a non factorable dictionary"""
        test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))
    
        total_dictionary = dict([])
        for list_of_vectors in list_of_list_of_vectors:
            test_case.reproduced_vectors = list_of_vectors
            single_dictionary = test_case.dict_from_reproduced_vectors()
            for k,v in single_dictionary.items():
                if k in total_dictionary:
                    total_dictionary[k] += v 
                else:
                    total_dictionary[k] = v 
        return total_dictionary
    no_error = True

    vector_1 = [1j,1,1,1]
    vector_2 = [1,-1j,0,1]
    list_of_vectors_1 = [vector_1, vector_2]
    vector_1 = [1,1,1,1]
    vector_2 = [0,1,0,0]
    list_of_vectors_2 = [vector_1, vector_2]
    list_of_list_of_vectors = [list_of_vectors_1,list_of_vectors_2]

    non_factorable_dict = dict_from_list_of_vectors(list_of_list_of_vectors)
    test_case = PopescuRohrlichPhotonPairAmplitudes(non_factorable_dict)
    no_error &= (test_case.is_dict_factorable() == False)


    vector_1 = [1j,1,1,1]
    vector_2 = [1,-1j,0,1]
    list_of_vectors_1 = [vector_1, vector_2]
    vector_1 = [1,1,1,3]
    vector_2 = [1+5j,1,0,0]
    list_of_vectors_2 = [vector_1, vector_2]
    list_of_list_of_vectors = [list_of_vectors_1,list_of_vectors_2]

    non_factorable_dict = dict_from_list_of_vectors(list_of_list_of_vectors)
    test_case = PopescuRohrlichPhotonPairAmplitudes(non_factorable_dict)
    no_error &= (test_case.is_dict_factorable() == False)

    vector_1 = [2,2,2,2]
    vector_2 = [0,1,0,0]
    list_of_vectors_1 = [vector_1, vector_2]
    vector_1 = [2,2,2,2]
    vector_2 = [0,1,0,0]
    list_of_vectors_2 = [vector_1, vector_2]
    list_of_list_of_vectors = [list_of_vectors_1,list_of_vectors_2]

    non_factorable_dict = dict_from_list_of_vectors(list_of_list_of_vectors)
    test_case = PopescuRohrlichPhotonPairAmplitudes(non_factorable_dict)
    no_error &= (test_case.is_dict_factorable() == True)

    vector_1 = [2,2,2,2]
    vector_2 = [0,1,0,0]
    list_of_vectors_1 = [vector_1, vector_2]
    vector_1 = [2+2j,2+2j,2+2j,2+2j]
    vector_2 = [0, 1+1j,0,0]
    list_of_vectors_2 = [vector_1, vector_2]
    list_of_list_of_vectors = [list_of_vectors_1,list_of_vectors_2]

    non_factorable_dict = dict_from_list_of_vectors(list_of_list_of_vectors)
    test_case = PopescuRohrlichPhotonPairAmplitudes(non_factorable_dict)
    no_error &= (test_case.is_dict_factorable() == True)

    vector_1 = [2,2,2,2]
    vector_2 = [0,1,0,0]
    list_of_vectors_1 = [vector_1, vector_2]
    list_of_list_of_vectors = [list_of_vectors_1]
    non_factorable_dict = dict_from_list_of_vectors(list_of_list_of_vectors)
    test_case = PopescuRohrlichPhotonPairAmplitudes(non_factorable_dict)
    no_error &= (test_case.is_dict_factorable() == True)
    assert no_error

def test_pair_amplitude_from_photon_amplitudes():
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))
    # test pair_amplitudes_from_photon_amplitudes
    no_error = True
    no_error &= test_case.reproduced_vector_from_photon_amplitudes([[1,0],[1,0]]) == [1,0,0,0]
    no_error &= test_case.reproduced_vector_from_photon_amplitudes([[0,1],[0,-1]]) == [0,0,0,-1]
    no_error &= test_case.reproduced_vector_from_photon_amplitudes([[1j,0],[0,1]]) == [0,1j,0,0]
    no_error &= test_case.reproduced_vector_from_photon_amplitudes([[0,-1j],[1,0]]) == [0,0,-1j,0]
    no_error &= test_case.reproduced_vector_from_photon_amplitudes([[np.sqrt(1/2),np.sqrt(1/2)],[1,0]]) == [np.sqrt(1/2),0,np.sqrt(1/2),0]

    # test photon_amplitudes_from_pair_amplitudes for factorable state
    no_error = True
    amplitudes = [1,0,0,0]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) < 0.0001

    amplitudes = [0,1,0,0]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) < 0.0001

    amplitudes = [0,0,1,0] 
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) < 0.0001

    amplitudes = [0,0,0,1]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) < 0.0001

    amplitudes = [1,1,1,1]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) < 0.0001

    amplitudes = [1e-7,0,0,1j]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) < 1

    # test photon_amplitudes_from_pair_amplitudes for factorable state
    no_error = True
    amplitudes = [1,0,0,1]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) > 1

    amplitudes = [0,1,1,0]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) > 1

    amplitudes = [1e-6,0,0,1]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) > 1

    amplitudes = [1,1,1,1e-6]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) > 1

    amplitudes = [1,1,1,0.999995]
    no_error &= test_case.compare_matrices_ppm_level(amplitudes, test_case.reproduced_vector_from_photon_amplitudes(test_case.photon_amplitudes_from_reproduced_vector(amplitudes))) > 1

    # test photon_amplitudes_from_pair_amplitudes for factorable state
    no_error = True
    amplitudes = [1,0,0,0]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1)

    amplitudes = [1,0,0,0]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=0.001)

    amplitudes = [0,1,0,0]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1)

    amplitudes = [0,0,1,0] 
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1)

    amplitudes = [0,0,0,1]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1)

    amplitudes = [1,1,1,1]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1)

    amplitudes = [1e-7,0,0,1j]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1)

    amplitudes = [1e-7,0,0,1j]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=0.01) == False

    no_error = True
    amplitudes = [1,0,0,1]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1) == False

    amplitudes = [0,1,1,0]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1) == False

    amplitudes = [1e-6,0,0,1]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1) == False

    amplitudes = [1e-6,0,0,1]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=10) == True

    amplitudes = [1,1,1,1e-6]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1) == False

    amplitudes = [1,1,1,0.999995]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=1) == False

    amplitudes = [1,1,1,0.999995]
    no_error &= test_case.is_photon_pair_factorable(amplitudes,threshold=10) == True
    assert no_error

def test_amplitudes_for_photon_from_photon_pairs_hh_hv_vh_vv():
    no_error_found = True
    angles_s = [
        tuple([np.pi/4,0,0,0]),
        tuple([np.pi/2,0,0,0]),
        tuple([0,0,0,0]),
        tuple([np.pi/2,0,np.pi/2,0]),
        tuple([0,np.pi/2,0,np.pi/2]),
        tuple([np.pi/2,np.pi/2,np.pi/2,np.pi/2]),
        tuple([np.pi/4,0,0,1.1*np.pi]),
        tuple([np.pi/4,np.pi/2,0,0]),
        tuple([np.pi/5,np.pi/3,np.pi/6,np.pi/8]),
        tuple([0,np.pi/3,np.pi/6,np.pi/8]),
        tuple([np.pi/5,np.pi/3,0,np.pi/8]),
        tuple([np.pi/5,np.pi/3,np.pi/6,0]),
        tuple([np.pi/5,np.pi/16,1.9*np.pi,0.1]),
        tuple([10*np.pi/25,8*np.pi/17,0,0]),
        tuple([np.pi/5,10*np.pi/33,0,1.5]),
        tuple([np.pi/18,2*np.pi/3,0,0])
        ]    
    lengths = [0.1,0.001, 3.14, np.pi, np.sqrt(2)]
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))
    for index, angles in enumerate(angles_s):
        if angles[0] > np.pi/2 or angles[1] > np.pi:
            continue
        if angles[2] > np.pi/2 or angles[3] > np.pi:
            continue
        for indices in [(0,1),(2,3)]:

            length_1 = lengths[index%len(lengths)]
            length_2 = lengths[index%len(lengths)]

            
            if indices == (0,1):
                phase_1 = np.cos(angles[indices[1]]) + 1j*np.sin(angles[indices[1]])
                amplitude_north_1 = length_1*np.cos(angles[indices[0]])*phase_1
                amplitude_south_1 = length_1*np.sin(angles[indices[0]])
                vector_1, norm_1 = test_case.stokes_vector_from_amplitudes([amplitude_north_1, amplitude_south_1])
            else:
                phase_2 = np.cos(angles[indices[1]]) + 1j*np.sin(angles[indices[1]])
                amplitude_north_2 = length_2*np.cos(angles[indices[0]])*phase_2
                amplitude_south_2 = length_2*np.sin(angles[indices[0]])
                vector_2, norm_2 = test_case.stokes_vector_from_amplitudes([amplitude_north_2, amplitude_south_2])
        hh = amplitude_north_1 * amplitude_north_2
        hv = amplitude_north_1 * amplitude_south_2
        vh = amplitude_south_1 * amplitude_north_2
        vv = amplitude_south_1 * amplitude_south_2
        amplitudes = [hh, hv, vh, vv]
        length_of_amplitudes = norm_1*norm_2
        test_case.reproduced_vectors = [(hh, hv, vh, vv)]
        created_dictionary = test_case.dict_from_reproduced_vectors()
        test_case_2= PopescuRohrlichPhotonPairAmplitudes(input_dictionary=created_dictionary)
    
        vector_1_from_pair, length_1 = test_case_2.stokes_vectors[test_case_2.photon_indices_per_pair[0][0]]
        vector_2_from_pair, length_2 = test_case_2.stokes_vectors[test_case_2.photon_indices_per_pair[0][1]]


        no_error_found = no_error_found and np.round(length_1*length_2,4) == 1
    
        no_error_found = no_error_found and all([np.round(a,4) == np.round(b,4) for a,b in zip(vector_1,vector_1_from_pair)])
    
        no_error_found = no_error_found and all([np.round(a,4) == np.round(b,4) for a,b in zip(vector_2,vector_2_from_pair)])
    assert no_error_found

def test_stokes_vector_from_amplitudes():
    no_error_found = True
    angles_s = [
        tuple([np.pi/2,0,0,0]),
        tuple([0,0,0,0]),
        tuple([np.pi/2,0,np.pi/2,0]),
        tuple([0,np.pi/2,0,np.pi/2]),
        tuple([np.pi/2,np.pi/2,np.pi/2,np.pi/2]),
        tuple([np.pi/4,0,0,1.1*np.pi]),
        tuple([np.pi/4,np.pi/2,0,0]),
        tuple([np.pi/5,np.pi/3,np.pi/6,np.pi/8]),
        tuple([0,np.pi/3,np.pi/6,np.pi/8]),
        tuple([np.pi/5,np.pi/3,0,np.pi/8]),
        tuple([np.pi/5,np.pi/3,np.pi/6,0]),
        tuple([np.pi/5,np.pi/16,1.9*np.pi,0.1]),
        tuple([10*np.pi/25,8*np.pi/17,0,0]),
        tuple([np.pi/5,10*np.pi/33,0,1.5]),
        tuple([np.pi/18,2*np.pi/3,0,0])
        ]
    lengths = [0.1,0.001, 3.14, np.pi, np.sqrt(2)]
    test_case = PopescuRohrlichPhotonPairAmplitudes(dict([]))
    for index, angles in enumerate(angles_s):
        for indices in [(0,1),(2,3),(1,2),(0,3)]:
            if angles[indices[0]] > np.pi/2 or angles[indices[1]] > np.pi:
                      continue
            length = lengths[index%len(lengths)]
            phase = np.cos(angles[indices[1]]) + 1j*np.sin(angles[indices[1]])
            amplitude_north = length*np.cos(angles[indices[0]])*phase
            amplitude_south = length*np.sin(angles[indices[0]])
 
            vector, norm = test_case.stokes_vector_from_amplitudes([amplitude_north, amplitude_south])
            no_error_found = no_error_found and np.round(sum([c**2 for c in vector]),4) == 1
            no_error_found = no_error_found and np.round(length,4) == np.round(norm,4)
            angle_psi = np.arccos(vector[2])
            no_error_found = no_error_found and np.round(angle_psi,4) == np.round(2*angles[indices[0]],4)
            if np.round(np.cos(angles[indices[0]])) != 0 and np.round(np.sin(angles[indices[0]])) != 0:
                angle_phi = np.arctan2(vector[0],vector[1])
                no_error_found = no_error_found and np.round(angle_phi,4) == np.round(angles[indices[1]],4)
            else:
                angle_phi = 0
            if not no_error_found:
                print(angles,[angle/np.pi for angle in angles], indices)
                print(np.round(length,2),np.round(norm,2),'-',np.round(angle_psi,4),np.round(2*angles[indices[0]],4),'-',np.round(angle_phi,2),np.round(angles[indices[1]],2))
                print(vector)
    assert no_error_found