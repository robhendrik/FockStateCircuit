import sys  
sys.path.append("./src")
from fock_state_circuit import ComponentNames, OpticalValues,FockStateCircuit
import pytest
import sys  
import numpy as np

def test_get_values():
    no_error = True
    comp_names = ComponentNames(length_of_fock_state=3,no_of_optical_channels=3,channel_0_left_in_state_name=True)
    no_error &= comp_names['011'] == [0,1,1]
    no_error &= comp_names['211'] == [2,1,1]
    no_error &= comp_names['011'] == [0,1,1]
    no_error &= comp_names['012'] == [0,1,2]
    comp_names = ComponentNames(length_of_fock_state=3,no_of_optical_channels=3,channel_0_left_in_state_name=False)
    no_error &= comp_names['011'] == [1,1,0]
    no_error &= comp_names['211'] == [1,1,2]
    no_error &= comp_names['011'] == [1,1,0]
    no_error &= comp_names['012'] == [2,1,0]

    assert no_error

def test_get_component_names_edge_case():
    no_error = True
    comp_names = ComponentNames(length_of_fock_state=12,no_of_optical_channels=3,channel_0_left_in_state_name=False)
    optical_values = OpticalValues(comp_names)
    no_error &= optical_values[(0,0,0)] == '000000'
    no_error &= optical_values[(0,0,10)] == '10'+'00'+'00'
    no_error &= optical_values[(0,0,1)] == '01'+'00'+'00'
   
    comp_names = ComponentNames(length_of_fock_state=12,no_of_optical_channels=3,channel_0_left_in_state_name=True)
    optical_values = OpticalValues(comp_names)
    no_error &= optical_values[(0,0,0)] == '000000'
    no_error &= optical_values[(0,0,10)] == '00'+'00' + '10'
    no_error &= optical_values[(0,0,1)] == '00'+'00'+'01'

    assert no_error

def test_get_optical_values_various_situations():
    no_error = True
    length_of_fock_state = 5 
    channel_0_left_in_state_name =True

    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)

    no_error &= optical_values[(0,0,1)] == '001'
    no_error &=  optical_values[(2,3,2)] == '232'
    try:
        optical_values[(1,5,1)]
        no_error = False
    except:
        pass

    length_of_fock_state = 15 
    channel_0_left_in_state_name = True
    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)
    no_error &=  optical_values[(0,0,1)] == '000001'
    no_error &=  optical_values[(2,3,4)] == '020304'
    no_error &=  optical_values[(1,4,1)] == '010401'
    no_error &=  optical_values[(10,11,10)] == '101110'
    no_error &=  [0,0,1] == comp_names['000001']
    no_error &=  [2,3,4] == comp_names['020304']
    no_error &=  [1,4,1] == comp_names['010401']
    no_error &=  [10,11,10] == comp_names['101110']

    length_of_fock_state = 15
    channel_0_left_in_state_name = False
    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)
    no_error &=  optical_values[(0,0,1)] == '010000'
    no_error &=  optical_values[(2,3,4)] == '040302'
    no_error &=  optical_values[(1,4,1)] == '010401'
    no_error &=  optical_values[(10,11,10)] == '101110'
    no_error &=  [0,0,1] == comp_names['010000']
    no_error &=  [2,3,4] == comp_names['040302']
    no_error &=  [1,4,1] == comp_names['010401']
    no_error &=  [10,11,10] == comp_names['101110']

    length_of_fock_state = 115
    channel_0_left_in_state_name = True
    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)
    no_error &=  optical_values[(0,0,1)] == '000000001'
    no_error &=  optical_values[(2,3,4)] == '002003004'
    no_error &=  optical_values[(1,4,1)] == '001004001'
    no_error &=  optical_values[(10,11,10)] == '010011010'

    length_of_fock_state = 115 
    channel_0_left_in_state_name = False
    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)
    no_error &=  optical_values[(0,0,1)] == '001000000'
    no_error &=  optical_values[(2,3,4)] == '004003002'
    no_error &=  optical_values[(1,4,1)] == '001004001'
    no_error &=  optical_values[(10,11,10)] == '010011010'
    no_error &=  [0,0,1] == comp_names['001000000']
    no_error &=  [2,3,4] == comp_names['004003002']
    no_error &=  [1,4,1] == comp_names['001004001']
    no_error &=  [10,11,10] == comp_names['010011010']

    assert no_error

def test_iteration_names():
    no_error = True
    comp_names = ComponentNames(length_of_fock_state=3,no_of_optical_channels=3,channel_0_left_in_state_name=True)
    no_error &= '011' in comp_names.keys()
    no_error &= '211' in comp_names.keys()
    comp_names = ComponentNames(length_of_fock_state=3,no_of_optical_channels=3,channel_0_left_in_state_name=False)
    no_error &= '211' in comp_names.keys()
    no_error &= '011' in comp_names.keys()
    assert no_error



def test_iteration_values():
    no_error = True
    comp_names = ComponentNames(length_of_fock_state=3,no_of_optical_channels=3,channel_0_left_in_state_name=True)
    optical_values = OpticalValues(comp_names)
    no_error &= (0,1,1) in optical_values.keys()

    no_error &= (2,1,1) in optical_values.keys()
    comp_names = ComponentNames(length_of_fock_state=3,no_of_optical_channels=3,channel_0_left_in_state_name=False)
    optical_values = OpticalValues(comp_names)
    no_error &= (2,1,1) in optical_values.keys()
    no_error &= (0,1,1) in optical_values.keys()

    assert no_error

def test_changing_order_with_tensor_list():
    circuit = FockStateCircuit(length_of_fock_state = 3, no_of_optical_channels = 5)
    new_state_list = [] # this will be the list of the new states with order of channels adjusted
    tensor_list = [4,2,0,3,1]
    _list_of_fock_states = [list(optical_value) for optical_value in circuit._dict_of_optical_values.keys()]
    for state in _list_of_fock_states: # iterate to all states in standard order
        new_state = [state[index] for index in tensor_list] # re-order the channels as per 'tensor_list'
        new_state_list.append(new_state) 

    # generate the new order
    new_order_original = np.array([_list_of_fock_states.index(new_state_list[i]) for i in range(len(new_state_list))])

    new_order_2 = np.array([circuit._dict_of_optical_values.reorder(values,tensor_list) for values in circuit._dict_of_optical_values.keys()])

    assert all(new_order_original == new_order_2)


def test_iterate_through_values_and_components():
    circuit = FockStateCircuit(length_of_fock_state = 3, no_of_optical_channels = 5)
    list_of_names = [component_name for component_name in circuit._dict_of_valid_component_names.keys()]
    list_of_values = [values for values in circuit._dict_of_optical_values.keys()]
    no_error = True
    no_error &= len(list_of_names) == len(list_of_values)
    no_error &= '22222' in list_of_names
    no_error &= '222222' not in list_of_names
    no_error &= (0,1,2,1,0) in list_of_values
    no_error &= (0,1,3,1,0) not in list_of_values
    assert no_error

def test_iterate_through_values_and_components_items():
    no_error = True
    circuit = FockStateCircuit(length_of_fock_state = 3, no_of_optical_channels = 5)
    no_error &= all( [name == circuit._dict_of_optical_values[tuple(values)] for name,values in circuit._dict_of_valid_component_names.items()])
    no_error &= all( [list(values) == circuit._dict_of_valid_component_names[name] for values,name in circuit._dict_of_optical_values.items()])
    assert no_error

def test_contain():
    no_error = True
    length_of_fock_state = 115 
    channel_0_left_in_state_name = False
    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)
    no_error &=  '001000000' in comp_names
    no_error &=  [2,3,4] in optical_values
    no_error &=  (2,2,2) in optical_values
    no_error &=  '201000000' not in comp_names
    no_error &=  (2,3,4,6) not in optical_values
    assert no_error

def test_index():
    no_error = True
    length_of_fock_state = 56
    channel_0_left_in_state_name = False
    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)

    comp_name = optical_values[(1,2,2)]
    no_error &= comp_name == '020201'
    comp_name = optical_values[[2,0,0]]
    no_error &= comp_name == '000002'
    no_error &= len(optical_values.dict_of_optical_values) == 2
    index = comp_names.index_of('000010')
    no_error &= index ==10
    no_error &= len(optical_values.dict_of_optical_values) == 3
    index = optical_values.index_of([1,0,0])
    no_error &= index ==1

    values = comp_names['010101']
    no_error &= values == [1,1,1]
    values = comp_names['210201']
    no_error &= values == [1,2,21]
    no_error &= len(comp_names.dict_of_valid_component_names) == 5
    index = optical_values.index_of((0,0,1))
    no_error &= index == 56**2
    no_error &= len(comp_names.dict_of_valid_component_names) == 5
    index = optical_values.index_of((1,0,0))
    no_error &= index ==1

    length_of_fock_state = 56
    channel_0_left_in_state_name = True
    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)
    index = optical_values.index_of((0,0,1))
    no_error &= index == 56**2
    index = optical_values.index_of((1,0,0))
    no_error &= index ==1
    index = comp_names.index_of('000001')
    no_error &= index ==56**2
    index = comp_names.index_of('010000')
    no_error &= index ==1
    assert no_error

def test_re_order():
    no_error = True
    comp_names = ComponentNames(length_of_fock_state=12,no_of_optical_channels=3,channel_0_left_in_state_name=False)
    optical_values = OpticalValues(comp_names)

    for values in ([1,1,1], [0,1,2], [15,3,20]):
        no_error &= optical_values.reorder(values,[0,1,2]) == optical_values.index_of(values)

    for values in ([1,1,1], [0,1,2], [15,3,20],(1,2,3)):
        no_error &= optical_values.reorder(values,[2,1,0]) == optical_values.index_of(reversed(values))

    assert no_error

def test_index_2():
    no_error = True
    length_of_fock_state = 56
    channel_0_left_in_state_name = False
    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)

    index = comp_names.index_of('000010')
    no_error &= index ==10
    no_error &= comp_names.at_index(10) == '000010'
    index = optical_values.index_of([1,0,0])
    no_error &= index ==1
    no_error &= optical_values.at_index(1) == [1,0,0]

    index = optical_values.index_of((0,0,1))
    no_error &= index == 56**2
    no_error &= optical_values.at_index(56**2) == [0,0,1]
    
    length_of_fock_state = 56
    channel_0_left_in_state_name = True
    comp_names = ComponentNames(length_of_fock_state=length_of_fock_state ,no_of_optical_channels=3,channel_0_left_in_state_name=channel_0_left_in_state_name)
    optical_values = OpticalValues(comp_names)
    index = optical_values.index_of((0,0,1))
    no_error &= index == 56**2
    no_error &= optical_values.at_index(56**2) == [0,0,1]
    index = optical_values.index_of((1,0,0))
    no_error &= index ==1
    no_error &= optical_values.at_index(1) == [1,0,0]
    index = comp_names.index_of('000001')
    no_error &= index ==56**2
    no_error &= comp_names.at_index(56**2) == '000001'
    index = comp_names.index_of('010000')
    no_error &= index ==1
    no_error &= comp_names.at_index(1) == '010000'
    assert no_error
 