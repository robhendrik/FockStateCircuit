import sys  
sys.path.append("./src")
import fock_state_circuit as fsc

import numpy as np
import math
import pytest
from unittest.mock import Mock
import matplotlib.pyplot as plt
from unittest.mock import patch
import matplotlib.testing.compare as plt_test

def test_to_do_1():
    length_of_fock_state = 3
    _no_of_optical_channels = 5
    circuit1 = fsc.FockStateCircuit(length_of_fock_state=length_of_fock_state,
                                    no_of_optical_channels=_no_of_optical_channels,
                                    no_of_classical_channels=_no_of_optical_channels)
    circuit1.wave_plate_from_hamiltonian(0,1,np.pi/8,np.pi) 
    circuit1.wave_plate_from_hamiltonian(1,2,np.pi/8,np.pi) 
    circuit1.wave_plate_from_hamiltonian(2,3,np.pi/8,np.pi) 
    circuit1.wave_plate_from_hamiltonian(3,4,np.pi/8,np.pi) 
    circuit1.measure_optical_to_classical(optical_channels_to_be_measured=[0,1,2],
                                            classical_channels_to_be_written=[1,2,3]
                                            )  
    circuit1.wave_plate_from_hamiltonian(0,1,np.pi/8,np.pi) 
    circuit1.wave_plate_from_hamiltonian(1,2,np.pi/8,np.pi) 
    circuit1.wave_plate_from_hamiltonian(2,3,np.pi/8,np.pi) 
    circuit1.wave_plate_from_hamiltonian(3,4,np.pi/8,np.pi) 
    circuit1.measure_optical_to_classical(optical_channels_to_be_measured=[n for n in range(_no_of_optical_channels)],
                                            classical_channels_to_be_written=[n for n in range(_no_of_optical_channels)]
                                            )                                                        
    overall_collection = fsc.CollectionOfStates(fock_state_circuit=circuit1)
    result = circuit1.evaluate_circuit(collection_of_states_input=overall_collection)
    assert True