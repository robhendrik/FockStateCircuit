# FockStateCircuit
'FockStateCircuit' is intended to model (quantum) optical quantum circuits where the different channels carry well defined number of photons (typically low numbers like 0,1,2,3, ...). These states where the number of photons is well defined are called 'Fock states'. The class supports optical channels where interaction is established through optical components, like beamsplitters or waveplates. Circuits can also contain classical channels. These can be used to store the result of a measurement or to set the behavior of optical componens.

FockStateCircuits run instances from the class 'CollectionOfStates'. These are collections of states belonging to the FockStateCircuit. The states describe photon numbers in the optical channels and the values of the classical channels. When we run a collection of states on a circuit the collection will evolve through the circuit from an 'input' collection of states to an 'output' collection of states.

The states in the collections are instances of the class 'State'. The states describe photon numbers in the optical channels and the values of the classical channels. The states also carry the history of the different measurement results during evolution through the circuit, as well as a reference to the initial state they originate from. 

## Installing
* Install via pip install FockStateCircuit
* Copy the modules from the src folder on Github.

Dependencies are Numpy and Matplotlib. We tested with 

* Python 3.12
* numpy version 1.26.4 
* matplotlib 3.8.4

## Quick Start
This is an example modelling the HOM effect on a beamsplitter. The circuit has 4 optical channels (two polarizations for the two input ports of the beamsplitter). After the beamsplitter the photon number in two channels is measured and written to the two classical channels. The `initial_collection_of_states` is set up such that the input states are only those where we have one horizontally polarized photon in each input port, or where we have two horizontally polarized photon simultaneously at the two input ports. After ' evaluating' the circuit with the given collection of states the variable results contains the states which result if the input states evolve through the circuit. The resulting states are plotted and the schematics of the circuit are drawn.
```
circuit = fsc.FockStateCircuit(length_of_fock_state = 3, 
                                no_of_optical_channels = 4,
                                no_of_classical_channels=2
                                )
circuit.non_polarizing_50_50_beamsplitter(input_channels_a=(0,1), input_channels_b=(2,3))
circuit.measure_optical_to_classical(optical_channels_to_be_measured=[0,2], classical_channels_to_be_written=[0,1])

initial_collection_of_states = cos.CollectionOfStates(fock_state_circuit=circuit)
initial_collection_of_states.filter_on_initial_state(initial_state_to_filter=['1000', '0010', '1010'])

result = circuit.evaluate_circuit(collection_of_states_input=initial_collection_of_states)

result.plot()

circuit.draw()
```
The code should show this when running `result.plot()`: When using one photon the likelihood of finding it in any of the output ports is 50%. When using two photons (one for each input port) simultaneously we only see outcomes where both photons are in the same output ports, and not outcomes where they are each in a different output port. 
![title](./images/output_HOM_effect.png)



The schematics of the circuit (shown when calling `circuit.draw()` are these:
![title](./images/output_fock_state_circuit_example.png)

## Basic explanation
### Circuits
A FockStateCircuit consists of a series of 'nodes'. The variable `circuit.node_list` contains the nodes with all relevant information for evaluating the states and displaying the nodes. There are 4 types of nodes:
* Optical nodes: These only contain interactions between optical channels (beamsplitters, wave plates)
* Classical nodes: These only contain interactions between classical channels (e.g., to perform a calculation based on values in these channels)
* Measurement nodes: These contain measurements where classical channels are written with photon numbers measured in the optical channels. The optical channels will 'collapse' into the value corresponding to the measurement result.
* Combined nodes: These nodes have optical components which characteristics are set by values in the classical channels. This could be a wave plate whose angular orientation and/or phase delay is determined by a value the classical channels.
* Custom nodes: These nodes can perform and action on the optical circuit which is not limited to linear optical components. We can use this to model non-linear optical effects, or even non-physical effects.
* Bridge nodes: These nodes create a connection between two circuits with different characteristics. You use this for instance when you want to extend or shrink the number of channels in your circuit.

### States
Circuits run collections for states. The collections contain instances of the class `State`. Each state describes pure state which can be run on the circuit (i.e., it has the right number of classical and optical channels). The format for a state is:
```
Identifier: 'identifier_1-M1a'
Initial state: '10'
Cumulative probability: 1.00
Classical values: ['0.00', '1.00']
Last measurement result:  
	Value: ['0.00', '1.00'], Probability: 1.00
Optical components: 
	Component: '01' Amplitude: (1.00 - 0.00i), Probability: 1.00
```

* __initial_state__ (str) : 
                Typically the state from which the current state has evolved in the circuit, but user can customize for other purposes. Also 
                used to group together states in a statistical mixture.

* __cumulative_probability__ (float) :
                Probability to evolve from initial state to current state (as consequence of measurement or decoherence). Alternatively used
                to give the weight of this state in a statistical mixture of states indicated with the same initial_state

* __optical_components__ (dictionary) : 
                Optical components of the state described as number states per channel (i.e., '1001' can mean one photon in channel 0 and channel 3). 
                Each component has an amplitude (complex number type np.cdouble) and a probability (float). The probability is always the square
                of the absolute value of the amplitude (so effectively redundant information). The format for optical_components is for example:
                ```{ '1011': {'amplitude': 0.71 + 0j, 'probability': 0.5}, '1110': {'amplitude': 0.71 + 0j, 'probability': 0.5}}```

* __classical_channel_values__ (list of floats) :
                A list holding the values for classical channels in the fock state circuit.

* __measurement_results__ (list of dictionaries) :
                Measurement_results holds the outcomes of all measurements. New measurements should appended to the end of the list.
                At each measurement the classical channel values after the measurement are store together with the probability to get 
                that measurement result. The format for measurement results is for example:
                ```[{'measurement_results': [1, 3, 3.14], 'probability': 0.5}, {'measurement_results': [0, 0, 3.14], 'probability': 0.25}]```

## Watch outs
* When defining a circuit the parameter `length_of_fock_state` has to be set. If `length_of_fock_state` is for instance 3 the possible values for any optical channel are 0, 1 or 2 photons. So the maximum photon number in this case is 2. In general the maximum photon number is equal to `length_of_fock_state`-1. When the system encounters a transition between Fock States where the photon number is larger than `length_of_fock_state`-1 it will artificially set the transition amplitude to 1, leading to non-physical outcomes.
* After measurement the optical states 'collapse' to the value corresponding to the measurement. If more measurement outcomes are possible one state will turn into a statistical mixture. The photons will not disappear once they are measured (so this is a kind of 'non-destructive measurement')
* Under the hood the system uses quite large data structures. The basis is typically of size `length_of_fock_state` to the power `no_of_optical_channels`. There is some optimization to reduce the size of these matrices but it is still wise to minimize values to lowest number possible.
                
        
## Features
* Evaluate quantummechanical interaction between optical channels
* Easy to model famous experiments like Alain Aspects nobel prize winning experiment, the HOM effect, GHZ state generation as first demonstrator by Anton Zeilinger, quantum teleportation, ...
* Optical and quantum channels combined to easily process a variety of input states
* Features for easy plotting and visualizing the circuits and the states

## Documentation
On github you will find:
* This README.md in the main directory
* In the directory docs:

        tutorials
                fock_state_circuit_getting_started_tutorial.ipynb
                fock_state_circuit_tutorial.ipynb

        background on specific features
                combining_fock_state_circuits_tutorial.ipynb
                drawing_fock_state_circuit_tutorial.ipynb
        
        application examples:
                GHZ simulation with FockStateCircuit.ipynb
                Quantum Teleportation with FockStateCircuit.ipynb


You can also check https://armchairquantumphysicist.com/ where a number of applications are covered in blogposts

## Version history
### Version 1.05
* Minor bug fixes and performance improvement
### Version 1.04
* Minor bug fixes and performance improvements
### Version 1.0.2
* Minor bug fixes and performance improvements
### Version 1.0.0
* Added feature or drawing stations of a circuit. If we model communication between 'alice' and 'bob' we can draw their specific 'stations' to make 
        clear what is happening at either side. This only affects the visualization.
* Updated functionality for time delay. We can now model successive time delays. The way the code implements time delay is closer to actual mathemathical
        modelling of a system with (partially) distinguishable photons.
* The code has been 'refactored' from two .py files into a structure with more classes and an individual .py file per class. This makes the code 
        easier to maintain. 
#### Changes
* Backwards compatibility has been maintained EXCEPT for the fact that we can now use `import fock_state_circuit as fsc` for all code and do not have to
        separately import `CollectionOfStates as cos`. It does mean in that in the code `cos.CollectionOfStates` and `cos.State` have to be replaced by `fsc.CollectionOfStates` and `fsc.State`
### Version 0.0.9
* Added a new class CompoundFockStateCircuit
        The class is used to work with a list of circuits that have to be executed sequentially. 
        
* Added function: CollectionOfStates.extend
        Adds optical channels to every state and loads this with desired values. Creates statistical mixture if new channel is initiated with more than one value.
* Added function: CollectionOfStates.reduce
        Reduces number of optical channels by 'tracing out'. Most likely creates a statistical mixure in the remaining collection of states.
* Added function: CollectionOfStates.clean_up:
        Removes states and components with probability lower than threshold (self._threshold_probability_for_setting_to_zero)
        Groups states with same optical components together in one state while adding the cumulative probabilities.
* Added function: CollectionOfStates.density_matrix
        Returns a dictionary containing the density matric, trance of the density matrix and trace of the square of the density matric
* Added function: CollectionOfStates.adjust_length_of_fock_state
        Adjusts the length of the Fock state used in the calculation. This allows scaling up or down the maximum number of photons per channel
        and can enable more efficient calculation.

* Added function: State._rescale_optical_components
        Rescales optical components by removing the ones with (too) low probability and re-normalizing the remaining ones.

* Added function: State._identical_optical_components
        Returns True if optical components are the same, otherwise False.  
        
* Added function: FockStateCircuit.basis
        Returns a dictonary with valid components names as keys and the corresponding photon numbers in the values
* Added function: FockStateCircuit.custom_fock_state_node
        Apply a custom Fock state matrix to the circuit.
* Added function: FockStateCircuit.channel_coupling
        Apply a node to the circuit to couple channels with the given 'coupling_strength'.
* Added function: FockStateCircuit.bridge
        Apply a bridge node to the circuit to transfer the collection of states from one circuit to another.
* Added function FockStateCircuit.c_shift
        Apply a controlled shift node to the circuit
* Added function FockStateCircuit.get_fock_state_matrix
        Returns the fock state matrix for a given set of nodes in the circuit
* Added function fock_state_circuit.about() to module fock_state_circuit.py
        
#### Changes
* Added 'import fock_state_circuit as fsc'
* Added node-types 'bridge to other circuit' and 'custom optical'
* Added option to set optical components via a list of tuples, rather than a full dictionary
* Updated function FockStateCircuit.draw() to include bridge nodes
* Added a constant indicating version (_VERSION = '0.0.9')
    
#### Bug fixes
* Removed return value from CollectionOfStates.add_state

### Version 0.0.8

updated version: Published November 21, 2023 on https://github.com/robhendrik/FockStateCircuit/tree/main
### Version 0.0.8

initial version: Published June 20, 2023 on https://github.com/robhendrik/FockStateCircuit/tree/main
## Authors
Rob Hendriks




