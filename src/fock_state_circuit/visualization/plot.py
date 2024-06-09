import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.colors
from fock_state_circuit.temporal_and_spectral_gate_functionality.collection_of_state_columns import CollectionOfStateColumns

class Plot():
    """
        plot(self, classical_channels = [], initial_states = [], info_for_bar_plot = dict([]), histo_output_instead_of_plot = False):
            Function to plot a bar graph for a collection of states. The bars indicate the probability to go from an initial state
            to an outcome in the classical channels. The circuit has to include measurement to the classical channels. 
            Optionally the classical channels to use can be specified as well as a selection of initial states. The 

            The histogram returned will be of the form {'initial_state1': [  {'output_state': '1010', 'probability': 1.0},
                                                                            {'output_state': '0101', 'probability': 0.5}
                                                                            ]}

            If classical_channels is the empty list all channels will be used (this is the default value)
            If initial_states is the empty list all initial states will be used (this is the default value)

        plot_correlations(self, 
                          channel_combis_for_correlation : list, 
                          info_for_bar_plot = dict([]), 
                          correlation_output_instead_of_plot = False,
                          initial_states_to_assess = []):    
            Determine correlations between channels. If correlation is 1, the outcomes in the channels are always the same,
            if correlation is -1 the outcome is always different.

            Format for channel combinations is like: [(2,3),(4,5),(2,4),(3,5)]

            If correlation_output_instead_of_plot is set to True function will return correlation values instead of 
            creating a plot. Format for returned correlations is:
            Results in format: {'initial_state1': [0.7,1.0,0.7,-0.7], 
                                'initial_state2': [0.5,0.5,0.5,-0.5]}

        Last modified: April 16th, 2024
                                                                                                                                                                                                                                        ]}
    """
    _DEFAULT_PLOT_SETTINGS = {
        'figsize' : (16,8)
    }
    def plot(self, classical_channels = [], initial_states = [], info_for_bar_plot = dict([]), histo_output_instead_of_plot = False):
            """ Function to plot a bar graph for a collection of states. The bars indicate the probability to go from an initial state
                to an outcome in the classical channels. The circuit has to include measurement to the classical channels. 
                Optionally the classical channels to use can be specified as well as a selection of initial states. The 

                The histogram returned will be of the form {'initial_state1': [  {'output_state': '1010', 'probability': 1.0},
                                                                                {'output_state': '0101', 'probability': 0.5}
                                                                                ]}

                If classical_channels is the empty list all channels will be used (this is the default value)
                If initial_states is the empty list all initial states will be used (this is the default value)
                                                                                                                                                                                                                                        ]}
            Args:
                classical_channels : classical channels which are used in the bar plot. 
                                    Typically these are channels to which the measurement 
                                    results for the circuit have been written.
                initial_states: list of initial states to limit the plot in case 'result' 
                                    contains more states than should be plotted. Default all
                                    initial states in 'result' are used.
                info_for_bar_plot: optional information for the bar plot. info_for_bar_plot.get['title']
                                    sets the title for the graph.
                histo_output_instead_of_plot: if this bool is set to True the function will return a dict 
                                    as histogram instead of creating a plot
            """        
            # if no classical channels given use all classical channels
            if classical_channels == []:
                classical_channels = [channel_no for channel_no in range(self._no_of_classical_channels)]
            plt.rcParams['figure.figsize'] = [15,6]
            dict_for_plotting = dict([])
            #dict_for_tracking_coincidences = dict([])
            # output_states = []
            # create a dictionary with the initial state as key and a list of outcomes as value. Outcomes are the values in classical channel
            # and the probability to get to that outcome from an initial state
            # dict_for_plotting is {initial_state1 : [{'output_state':'1101', 'probability': 0.5}, {'output_state':'3000', 'probability': 0.5}] }
            map_outcome_to_classical_value = dict([])

            # check if the collection has photon resolution
            if self.is_photon_resolved():
                collection_to_plot = self._pre_plot()
            else:
                collection_to_plot = self

            for state in collection_to_plot._collection_of_states.values():
                if len(initial_states) > 0 and state.initial_state not in initial_states:
                    continue
                if state.initial_state not in dict_for_plotting.keys():
                    dict_for_plotting[state.initial_state] = []     

                probability = state.cumulative_probability
                # to generate a string for the outcome we use the same function that is used for optical channels. Note that every classical channel value will
                # be mapped on an integer. 
                outcome = self._create_state_name_from_list_of_photon_numbers([state.classical_channel_values[index] for index in classical_channels])
                map_outcome_to_classical_value.update({outcome: [state.classical_channel_values[index] for index in classical_channels]})

                for outcomes_for_this_initial_state in dict_for_plotting[state.initial_state]:
                    # if the outcome already exists add the probability
                    if outcome == outcomes_for_this_initial_state['output_state']:
                        outcomes_for_this_initial_state['probability'] += probability
                        break
                else:
                    # if the outcome does not exist create a new entry in the list
                    dict_for_plotting[state.initial_state].append({'output_state': outcome, 'probability': probability})

            # create a list of all possible outcomes across all initial states
            output_states = []
            for initial_state, list_of_outcomes in dict_for_plotting.items():
                for output_probability in list_of_outcomes:
                    outcome = output_probability['output_state']
                    if outcome not in output_states:
                        output_states.append(outcome)

            # for states who do not lead to a certain outcome add the outcome with probability zero
            # this enables easier plotting of graphs. If the outcome is absent it will need to be corrected
            # later, so better to add the outcome with probability zero
            for initial_state, list_of_outcomes in dict_for_plotting.items():
                outcomes_for_this_initial_state = []
                for output_probability in list_of_outcomes:
                    outcomes_for_this_initial_state.append(output_probability['output_state'])
                for outcome in list(set(output_states).difference(outcomes_for_this_initial_state)):
                    dict_for_plotting[initial_state].append({'output_state': outcome, 'probability': 0})

            if histo_output_instead_of_plot:
                return dict_for_plotting

            no_initial_states = len(dict_for_plotting)
            no_output_states = len(output_states)
            width = 0.8/max(no_output_states,1) # spread the bars over 80% of the distance between ticks on x -axis
            mid = no_output_states//2
            # cycle through standard color list 
            cycle = info_for_bar_plot.get('colors', list(matplotlib.colors.TABLEAU_COLORS))
            greys = ['whitesmoke','whitesmoke']
        

            fig, ax = plt.subplots(1, 1,figsize = Plot._DEFAULT_PLOT_SETTINGS['figsize'])

            for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
                x = list(dict_for_plotting.keys()).index(initial_state)
                for i in range(no_output_states):
                    ax.bar(x+(i-mid)*width, 
                            1.2,
                            color = greys[i%len(greys)],
                            width = width
                            )
            for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
                for outcome in outcomes_for_that_initial_state:
                    x = list(dict_for_plotting.keys()).index(initial_state)
                    i = output_states.index(outcome['output_state'])
                    ax.bar(x+(i-mid)*width, 
                            outcome['probability'],
                            color = cycle[i%len(cycle)],
                            width = width
                            )
            custom_lines = [matplotlib.lines.Line2D([0], [0], color = cycle[i%len(cycle)], lw=4) for i in range(len(output_states))]
            ax.legend(custom_lines, [outcome for outcome in  output_states])
            plt.xticks(rotation=90)
            plt.xticks([x for x in range(no_initial_states)], list(dict_for_plotting.keys()))
            plt.ylabel(info_for_bar_plot.get('ylabel', 'Probability'))
            text = info_for_bar_plot.get('title', 'Probabilities for going from input state to measurement result')
            plt.title(text) 
            plt.show()
            return
    
    def _pre_plot(self):
        """ Prepare a collection of states with photon resolution for plotting"""
        # step 0: Make full copy
        resulting_collection = self.copy()
        resulting_collection.clear()

        # step 1: Same as above for full outcomes
        coll_of_cols = CollectionOfStateColumns(collection_of_states=self)
        coll_of_cols.split()
        
        # step 2: Per group consolidate in new state without photon resolution
        for column in coll_of_cols.by_column():
            state = column.generate_single_state_from_column()
            resulting_collection.add_state(state=state)
        # step 3
        return resulting_collection

    
    def _create_state_name_from_list_of_photon_numbers(self,state: list) -> str:
        """ For a list of photon numbers generate a string which can serve as name for the state or a component in the state.
            Example: state [0,1,3] would become '310' with 0 representing the photon number in channel 0. If we use reversed
            state notation state [0,1,3] would become '013' (reversed or regular state notation is set in initialization of the 
            FockStateCircuit). If we allow per channel photon numbers which require more digits (e.g., 10) the format of the string will be adjusted.
            Example [10,1,3] would become '100103'

        Args:
            state (List): state as a list of values with channel 'n' at index 'n'. e.g., [0,1,3]

        Returns:
            str: name of the state of component derived from photon number per channel. e.g., '013'
        """        
        string_format_in_state_as_word = "{:0"+str(len(str(self._length_of_fock_state-1)))+ "d}"

        if self._channel_0_left_in_state_name == True:
            name = ''.join([string_format_in_state_as_word.format(int(number)) for number in state])              
        else: #self.state_least_significant_digit_left == False:
            name = ''.join([string_format_in_state_as_word.format(int(number)) for number in state[::-1]]) 

        return name
    
    def _create_list_of_values_from_name(self, name: str) -> list:
        """ Create a list of values from a name string"""
        values = []
        start, end = 0, 0
        while end < len(name):
            end = start + len(str(self._length_of_fock_state-1))
            values.append(int(name[start:end]))
            start = end
        return values
    

    def _correlation_from_collection(self, channel_combis_for_correlation: list) -> dict:
        """ Determine correlations between channels. If correlation is 1, the outcomes in the channels are always the same,
            if correlation is -1 the outcome is always different.

            Format for channel combinations is like: [(2,3),(4,5),(2,4),(3,5)]
            Results in format: {'initial_state1': [0.7,1.0,0.7,-0.7], 
                                'initial_state2': [0.5,0.5,0.5,-0.5]}

        Args:
            channel_combis_for_correlation (list): List of channels combinations (as tuples)

        Returns:
            dict: correlations per initial state
        """
        correlations_per_initial_state = dict([])
        histogram = self.plot(histo_output_instead_of_plot=True)
        for initial_state, list_of_outcomes in histogram.items():
            correlations = []
            for combi in channel_combis_for_correlation:
                corr = 0
                count = 0
                for res_as_dict in list_of_outcomes:
                    values = self._create_list_of_values_from_name(res_as_dict['output_state'])
                    if (values[combi[0]] == values[combi[1]]):
                        corr += res_as_dict['probability']
                    else:
                        corr -= res_as_dict['probability']
                    count += res_as_dict['probability']
                correlations.append(corr/count)
            correlations_per_initial_state.update({initial_state:correlations})
        return correlations_per_initial_state
    
    def plot_correlations(self, 
                          channel_combis_for_correlation : list, 
                          info_for_bar_plot = dict([]), 
                          correlation_output_instead_of_plot = False,
                          initial_states_to_assess = []):    
        """ Determine correlations between channels. If correlation is 1, the outcomes in the channels are always the same,
            if correlation is -1 the outcome is always different.

            Format for channel combinations is like: [(2,3),(4,5),(2,4),(3,5)]

            If correlation_output_instead_of_plot is set to True function will return correlation values instead of 
            creating a plot. Format for returned correlations is:
            Results in format: {'initial_state1': [0.7,1.0,0.7,-0.7], 
                                'initial_state2': [0.5,0.5,0.5,-0.5]}

        Args:

            channel_combis_for_correlation (list): List of tuples to identify channels combinations between which
                                correlations are to be calculated.
            info_for_bar_plot: optional information for the bar plot. info_for_bar_plot.get['title']
                                sets the title for the graph.
            correlation_output_instead_of_plot: if this bool is set to True the function will return a dict 
                                with correlations instead of creating a plot
            initial_states_to_assess : list of initial states to limit the plot in case 'result' 
                                contains more states than should be plotted. Default all
                                initial states are used.

        """
        
        # calculate the desired correlations
        correlations_per_initial_state = self._correlation_from_collection(channel_combis_for_correlation)

        # only leave initial_states from the parameter list_of_initial_states and remove others.
        remove_keys = []
        if len(initial_states_to_assess) > 0:
            for initial_state in correlations_per_initial_state.keys():
                if initial_state not in initial_states_to_assess:
                    remove_keys.append(initial_state)
        for name in remove_keys:
            del correlations_per_initial_state[name]

        if correlation_output_instead_of_plot:
            return correlations_per_initial_state

        no_initial_states = len(correlations_per_initial_state)
        no_output_states = len(channel_combis_for_correlation)
        width = 0.8/no_output_states # spread the bars over 80% of the distance between ticks on x -axis
        mid = no_output_states//2
        # cycle through standard color list 
        cycle = info_for_bar_plot.get('colors', list(matplotlib.colors.TABLEAU_COLORS))
        greys = ['whitesmoke','whitesmoke']
        
        fig, ax = plt.subplots(1, 1,figsize = Plot._DEFAULT_PLOT_SETTINGS['figsize'])

        for initial_state, outcomes_for_that_initial_state in correlations_per_initial_state.items():
            x = list(correlations_per_initial_state.keys()).index(initial_state)
            for i in range(no_output_states):
                ax.bar(x+(i-mid)*width, 
                        1.2,
                        color = greys[i%len(greys)],
                        width = width
                        )
                ax.bar(x+(i-mid)*width, 
                        -1.2,
                        color = greys[i%len(greys)],
                        width = width
                        )
        for initial_state, correlations_for_that_initial_state in correlations_per_initial_state.items():
            for i,correlation in enumerate(correlations_for_that_initial_state):
                x = list(correlations_per_initial_state.keys()).index(initial_state)
                ax.bar(x+(i-mid)*width, 
                        correlation,
                        color = cycle[i%len(cycle)],
                        width = width
                        )
        custom_lines = [matplotlib.lines.Line2D([0], [0], color = cycle[i%len(cycle)], lw=4) for i in range(len(channel_combis_for_correlation))]
        
        ax.legend(custom_lines, [combi for combi in  channel_combis_for_correlation])
        plt.xticks(rotation=90)
        plt.xticks([x for x in range(no_initial_states)], list(correlations_per_initial_state.keys()))
        plt.ylabel(info_for_bar_plot.get('ylabel', 'Correlation'))
        text = info_for_bar_plot.get('title', 'correlations')
        plt.title(text) 
        plt.show()
        return

def plot_correlations_from_dict(dict_for_plotting):
    """ Plot an histogram of correlations. 

        The input dictionary should be of form 
        {initial_state1 : [{'output_state':'1101', 'correlation': 0.5}, {'output_state':'3000', 'correlation': 0.5}] }
        So for every initial state there should be a list of possible outcomes. The elements in the list are of the form
        {'output_state':'1101', 'correlation': 0.5}. The value for 'correlation' should be between 1 and -1 

        NOTE: For backwards compatibility the function also accepts 'probability' instead of 'correlation' as key in the outcomes.
    
        Args:
            dict_for_plotting (dict): {initial_state1 : [{'output_state':'1101', 'correlation': 0.5}, {'output_state':'3000', 'correlation': 0.5}] }
    """

    _DEFAULT_PLOT_SETTINGS = {
        'figsize' : (16,8)
    }
    plt.rcParams['figure.figsize'] = [15,6]
    # create a list of all possible outcomes across all initial states
    output_states = []
    for initial_state, list_of_outcomes in dict_for_plotting.items():
        for output_probability in list_of_outcomes:
            outcome = output_probability['output_state']
            if outcome not in output_states:
                output_states.append(outcome)
    info_for_bar_plot = dict([])
    no_initial_states = len(dict_for_plotting)
    no_output_states = len(output_states)
    width = 0.8/no_output_states # spread the bars over 80% of the distance between ticks on x -axis
    mid = no_output_states//2
    # cycle through standard color list 
    cycle = info_for_bar_plot.get('colors', list(matplotlib.colors.TABLEAU_COLORS))
    greys = ['whitesmoke','whitesmoke']


    fig, ax = plt.subplots(1, 1,figsize = _DEFAULT_PLOT_SETTINGS['figsize'])

    for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
        x = list(dict_for_plotting.keys()).index(initial_state)
        for i in range(no_output_states):
            ax.bar(x+(i-mid)*width, 
                    1.2,
                    color = greys[i%len(greys)],
                    width = width
                    )
            ax.bar(x+(i-mid)*width, 
                    -1.2,
                    color = greys[i%len(greys)],
                    width = width
                    )
    for initial_state, outcomes_for_that_initial_state in dict_for_plotting.items():
        for outcome in outcomes_for_that_initial_state:
            if 'probability' in outcome.keys():
                value_to_plot = outcome['probability']
            else:
                value_to_plot = outcome['correlation']
            x = list(dict_for_plotting.keys()).index(initial_state)
            i = output_states.index(outcome['output_state'])
            ax.bar(x+(i-mid)*width, 
                    value_to_plot,
                    color = cycle[i%len(cycle)],
                    width = width
                    )
    custom_lines = [matplotlib.lines.Line2D([0], [0], color = cycle[i%len(cycle)], lw=4) for i in range(len(output_states))]
    ax.legend(custom_lines, [outcome for outcome in  output_states])
    plt.xticks(rotation=90)
    plt.xticks([x for x in range(no_initial_states)], list(dict_for_plotting.keys()))
    plt.ylabel(info_for_bar_plot.get('ylabel', 'Probability'))
    text = info_for_bar_plot.get('title', 'Probabilities for going from input state to measurement result')
    plt.title(text) 
    plt.show()
    return
