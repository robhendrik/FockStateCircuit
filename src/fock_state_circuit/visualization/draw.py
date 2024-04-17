import math
import matplotlib.pyplot as plt
from matplotlib.patches  import Rectangle
from copy import deepcopy

class Draw(): 
    """ draw(self, 
                    print_defaults: bool = False, 
                    settings_for_drawing_circuit: dict = None
                    ) -> None

                    Draw the optical circuit. self. settings_for_drawing_circuit is the dict with settings for circuit drawing. 
                    If this dict is not defined default values will be taken. If the boolean print_defaults is set to True the function 
                    will print out the default values to console
        
        conventions(self) 
                    Function to print return the conventions used in this class as a string

        Last modified: April 16th, 2024
    """
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
            'channels_optical' : [],
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
            'combined_gate': 'single',
            'connections_to_other_stations' : {'classical' : [], 'optical': [], 'optical_and_classical' : []}
        }
    def draw(self, 
             print_defaults: bool = False, 
             settings_for_drawing_circuit: dict = None, 
             compound_circuit_settings: dict = None,
             ) -> None:
        """ Draw the optical circuit. self.settings_for_drawing_circuit is the dict with settings for circuit drawing. 
            If this dict is not defined default values will be taken. If the boolean print_defaults is set to True the function will print
            out the default values to console. The parameter compound_circuit_settings is used when drawing a circuit with bridges to other circuits 
            (a.k.a. compound circuit).
        
        Args:
            print_defaults (bool): If 'True' function will print default settings to console.
            settings_for_drawing_circuit (dict): dict with settings for circuit drawing. If none given default will be taken.
            compound_circuit_settings (dict): only used for drawing compound circuits.
            station_channels (dict): used when circuit is distributed over multiple stations

        Returns:
            nothing
        
        Raises:
            nothing
        """

        if print_defaults: self._print_default_setting() 

        circuit_draw_settings_dict = self._create_circuit_draw_settings_dict(settings_for_drawing_circuit)

        if compound_circuit_settings is None:
            compound_circuit_settings  = self._determine_settings_for_drawing_compound_circuit(circuit_draw_settings_dict)
# ! improve, make all preparationin the function _create... and draw title when creating the page
            # create the plot, use two times expected number of pages as size. left over pages
            # will be deleted at the end
            # the better solution would be to dynamically add subplots when starting a new page.
            number_of_pages = compound_circuit_settings['number_of_pages']
            canvas_height = compound_circuit_settings['canvas_height']
            canvas_width = compound_circuit_settings['canvas_width']
            figure_size_in_inches = (circuit_draw_settings_dict['figure_width_in_inches'], 
                                        circuit_draw_settings_dict['figure_width_in_inches']*number_of_pages*canvas_height/canvas_width
                                        )
            fig, axs = plt.subplots(nrows= 2*number_of_pages, ncols=1, squeeze = False, figsize = figure_size_in_inches)
            

            # give each page of the plot a title
            for page in range(number_of_pages):
                axs[page][0].set_title(circuit_draw_settings_dict['compound_circuit_title'],
                                    fontsize=circuit_draw_settings_dict['compound_plot_title_font_size']
                                    )
            compound_circuit_settings.update({'plot_axs':axs, 'figure': fig})
        if len(self.node_list) == 0:
            raise Exception('Error drawing empty circuit. Circuits need to contain at least one node')
        
        for node_index, node in enumerate(self.node_list):
            # for backward compatibility:
            if node['node_info'] is not None:
                if 'channels' in node['node_info'].keys():
                    node['node_info'].update({'channels_optical' : node['node_info']['channels']})
                    del node['node_info']['channels']

            # if specific node information is given use that, otherwise use default for everything or per item that is not specified
            if node['node_info'] is not None:
                current_node_info = Draw._NODE_DRAW_DEFAULT_SETTINGS | node['node_info']
            else:
                current_node_info = Draw._NODE_DRAW_DEFAULT_SETTINGS

            for key in current_node_info.keys():
                if key not in Draw._NODE_DRAW_DEFAULT_SETTINGS.keys():
                    print("Unrecognized key in parameter node_info: ", key, " Run circuit.draw(print_defaults = True) to get a list of recognized keys.")
            
            # for some items we need a list if you want to mark the node different per channel (i.e., a target and a control channel)
            # if there is no list given we artificially make the list by adding the 0th element to the end
            for item_needing_list in current_node_info.keys():
                if type(Draw._NODE_DRAW_DEFAULT_SETTINGS[item_needing_list]) == type([]) and len(Draw._NODE_DRAW_DEFAULT_SETTINGS[item_needing_list])>0:
                    maximum_needed_list_length = len(current_node_info['channels_optical']) + len(current_node_info['channels_classical'])
                    if type(current_node_info[item_needing_list]) == type([]) and len(current_node_info[item_needing_list]) == 0 :
                        current_node_info[item_needing_list] = [Draw._NODE_DRAW_DEFAULT_SETTINGS[item_needing_list][0]]*maximum_needed_list_length
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
            node_has_optical_channel = ((len(current_node_info['channels_optical']) != 0))

            # determine whether this node has a connection to another station
            node_has_optical_connection_to_other_circuit = (len(current_node_info['connections_to_other_stations']['optical']) != 0)
            node_has_classical_connection_to_other_circuit = (len(current_node_info['connections_to_other_stations']['classical']) != 0)
            node_has_optical_and_classical_to_other_circuit = (len(current_node_info['connections_to_other_stations']['optical_and_classical']) != 0)
            node_has_connection_to_other_circuit = any([node_has_optical_connection_to_other_circuit,
                                                        node_has_classical_connection_to_other_circuit,
                                                        node_has_optical_and_classical_to_other_circuit])

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

            # create a number which in binary format has a 1 for occupied channels and 0 otherwise, this is used to determine
            # whether we can draw nodes on the same x-location (if the nodes do not overlap this makes the circuit drawing more
            # condensed)
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
                        optical_channels = node['node_info'].get('channels_optical',[])
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
                bitlist_optical = ['1' if (channel in current_node_info['channels_optical']) else '0' for channel in range(len(compound_circuit_settings['line_y_values_optical']))]
                bitlist_classical = ['1' if (channel in current_node_info['channels_classical']) else '0' for channel in range(len(compound_circuit_settings['line_y_values_classical']))]
                bitnumber_used = int(''.join(bitlist_optical + bitlist_classical), 2)
                # make another bitstring filled up between channels 
                memory, bitnumber_occupied_channels = 0, 0
                for bit_index in range(len(compound_circuit_settings['line_y_values_optical']) + len(compound_circuit_settings['line_y_values_classical'])):
                    bit_value_occupied = int((bitnumber_used & (1 << bit_index )) != 0)
                    memory |= bit_value_occupied
                    memory &= int((bitnumber_used >> bit_index) != 0)
                    bitnumber_occupied_channels = bitnumber_occupied_channels | (memory << bit_index )

                # if there is a connection to another station we draw a line to the bottom of the chart to indicate the connection
                # to avoid overlap with other nodes we have to full up that space
                if node_has_connection_to_other_circuit:
                    if bitnumber_occupied_channels != 0:
                        bit_index = 0
                        while (bitnumber_occupied_channels & 1 << bit_index) == 0:
                            bit_index += 1
                        mask = (2**bit_index) -1
                        bitnumber_occupied_channels = bitnumber_occupied_channels | mask

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

                # add page number
                axis.annotate(               
                        'Page ' + str(page_number+1), 
                        (0,0),
                        horizontalalignment = 'right',
                        verticalalignment =  'center'
                        )

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
                node_y_values_optical = [compound_circuit_settings['line_y_values_optical'][channel] for channel in current_node_info['channels_optical']]
                node_y_values_classical = [compound_circuit_settings['line_y_values_classical'][channel] for channel in current_node_info['channels_classical']]
                lowest_y_value = min(node_y_values_optical + node_y_values_classical)
                highest_y_value = max(node_y_values_optical + node_y_values_classical)

                if node_has_connection_to_other_circuit:
                    # draw vertical line to bottom
                    # plot a wide white vertical line
                    axis.plot(
                        [node_x,node_x],
                        [0,lowest_y_value],
                        linestyle = 'solid',
                        linewidth = current_node_info['connection_linewidth'] * 2,
                        color = 'white',
                        alpha=1
                        )
                    # plot the connection line
                    line_offset = 0
                    if node_has_optical_and_classical_to_other_circuit:
                        marker_sizes = [circuit_draw_settings_dict['bridge_marker_size']*1.2, circuit_draw_settings_dict['bridge_marker_size']*0.8]
                        line_widths = [current_node_info['connection_linewidth']*2,current_node_info['connection_linewidth']]
                        line_offsets = [0,0]
                        line_colors = [current_node_info['connection_linecolor_optical'],current_node_info['connection_linecolor_classical']]
                    else:
                        marker_sizes = [circuit_draw_settings_dict['bridge_marker_size']]
                        line_widths = [current_node_info['connection_linewidth']]
                        line_offsets = [0]
                        if node_has_classical_connection_to_other_circuit:
                            line_colors = [current_node_info['connection_linecolor_classical']]
                        else:
                            line_colors = [current_node_info['connection_linecolor_optical']]
                    for line_width, line_offset, line_color, marker_size in zip(line_widths, line_offsets, line_colors, marker_sizes):    
                        axis.plot(
                            [node_x+line_offset,node_x+line_offset],
                            [0,lowest_y_value],
                            linestyle = current_node_info['connection_linestyle'],
                            linewidth = line_width,
                            marker = 'none',
                            color = line_color,
                            alpha=1
                            )
                        axis.plot(
                            [node_x],
                            [0],
                            markeredgewidth = circuit_draw_settings_dict['bridge_markeredgewidth'],
                            marker = circuit_draw_settings_dict['bridge_marker'],
                            markersize = marker_size,
                            color = line_color,
                            markerfacecolor = 'white',
                            fillstyle='full',
                            alpha=1
                            )
                    axis.plot(
                        [node_x],
                        [0],
                        linestyle = 'none',
                        markeredgewidth = 0.3,
                        marker = current_node_info['connections_to_other_stations'].get('identifier',''),
                        markersize = circuit_draw_settings_dict['node_label_font_size'],
                        color = 'black',
                        markerfacecolor = 'black',
                        fillstyle='full',
                        alpha=1
                        )

                    axis.annotate(                  
                        current_node_info['connections_to_other_stations'].get('label_text',''),
                        (node_x+0.2*compound_circuit_settings['spacing_between_nodes'], 0),
                        fontsize=circuit_draw_settings_dict['node_label_font_size']
                        )


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
                    for index in range(len(current_node_info['channels_optical'])):
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
    
    def _print_default_settings(self) -> None:
        """ Print the default settings"""
        print(Draw._CIRCUIT_DRAW_DEFAULT_SETTINGS)
        print(Draw._NODE_DRAW_DEFAULT_SETTINGS)
        return
    
    def _create_circuit_draw_settings_dict(self, settings_for_drawing_circuit) -> dict:
        """ Merge the settings for drawing with the default settings. Overwrite the settings passed as paramater,
        and when a setting is not passed use the default value"""
        # default settings for circuit drawing (can be overwritten when calling function)
        if settings_for_drawing_circuit is not None:
            circuit_draw_settings_dict = dict(Draw._CIRCUIT_DRAW_DEFAULT_SETTINGS | settings_for_drawing_circuit)
        else:
            circuit_draw_settings_dict = dict(Draw._CIRCUIT_DRAW_DEFAULT_SETTINGS)
        return circuit_draw_settings_dict
    
    def _determine_settings_for_drawing_compound_circuit(self, circuit_draw_settings_dict) -> dict:
        """ Determine the paramaters for drawing the circuit. For compound circuits (i.e., circuits connected together 
        with nodes), this is only done once for the full compound circuit"""
            
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
    
        # gather all parameters in a dictionary which can be passed on in the recursive call to next circuits
        compound_circuit_settings = {
            'compound_circuit_name_list' : compound_circuit_names,
            'number_of_pages' : number_of_pages,
            'canvas_height' : canvas_height,
            'canvas_width' : canvas_width,
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
        return compound_circuit_settings

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
    
    def draw_station(self, 
                    stations, 
                    settings_for_drawing_circuit: dict = None) -> None:
        """ Draw the circuit by station. A station is a set of channels locally present (for instance 
            channels for 'Alice', 'Bob' and 'Charlie). The parameter 'stations' passes the information on the stations,
            including which station(s) is/are to be drawn.
            
            The format for the parameter 'stations' is:
            stations = {'station_to_draw' : 'alice',
                        'station_channels': { 'bob': {'optical_channels': [0,1], 'classical_channels': [0,1]}, 
                                                'alice': {'optical_channels': [2,3], 'classical_channels': [2,3]}
                                                }}
            If 'stations_to_draw' is omitted all stations are drawn. 'stations_to_draw' can be a single station ('Bob'),
            or can be a list (['Bob','Alice']

        Args:
            stations (dict): Dictionary with information on the channels per station, as well as what stations to draw.
            settings_for_drawing_circuit (dict, optional): Dictionary with settings for circuit drawing. If none given 
                        default will be taken.
        """

        if not stations:
            self.draw()
            return
        
        # make a list of the stations to be drawn
        if 'station_to_draw' not in stations.keys():
            station_list = list(stations['station_channels'].keys())
        elif not isinstance(stations['station_to_draw'],list):
            station_list = [stations['station_to_draw']]
        else: 
            station_list = list(stations['station_channels'].keys())

        # create the settings for drawing stations for this circuit
        settings_for_drawing_station = self.create_settings_for_drawing_stations(stations,settings_for_drawing_circuit)

        # loop through the stations 
        for station in station_list:
            # create a 'new_circuit' representing the local station
            new_circuit = self._filter_on_station( station_to_draw = station,
                                    station_channels = stations['station_channels'])
            # drawn the circuit for this station
            new_circuit.draw(settings_for_drawing_circuit = settings_for_drawing_station[station])        
        return
    
    def create_settings_for_drawing_stations(self,
                                             stations,
                                             settings_for_drawing_circuit: dict = dict([])
                                             ) -> dict:
        """ Create settings for each stations, where the title and the channel labels are specific for the circuit. 
            The function returns a dictionary with stations names as keys. Also the dictionary has the key 'total_circuit'
            which contains settings for drawing the total circuit with channel labels indicating to which station the 
            channel belongs.

        Args:
            stations (dict): Dictionary with information on the channels per station, as well as what stations to draw.
            settings_for_drawing_circuit (dict, optional): Dictionary with settings for circuit drawing. If none given 
                        default will be taken.

        Raises:
            Exception: If 'total_circuit' is the name of a station.

        Returns:
            dict: Dictionary stations as keys and settings for that specific station as values.
        """
        # make a list of all stations in the circuit
        station_list = list(stations['station_channels'].keys())
        
        # we use total_circuit as a key to give information on the overall circuit, so it cannot be a name for a station
        if 'total_circuit' in station_list:
            raise Exception('Label for station is a reserved name: total_circuit. Select another label for the station')
        
        # create the overall settings for drawing the circuit
        circuit_draw_settings_dict = self._create_circuit_draw_settings_dict(settings_for_drawing_circuit)
        
        # if the user already created channel labels we will not overwrite them
        channel_labels_already_defined = 'channel_labels_optical' in circuit_draw_settings_dict.keys()

        # determine the more detailed settings for drawing the circuit, or the compound circuit
        compound_circuit_settings  = self._determine_settings_for_drawing_compound_circuit(circuit_draw_settings_dict)
     
        # create new channel labels representing the stations in the circuit
        if not channel_labels_already_defined:
            for station in station_list:
                for local_number,channel_number in enumerate(stations['station_channels'][station].get('optical_channels',[])):
                    compound_circuit_settings['channel_labels_optical'][channel_number] = station + ', Opt.' + str(local_number)
                    circuit_draw_settings_dict['channel_labels_optical'] = compound_circuit_settings['channel_labels_optical'].copy()
                for local_number,channel_number in enumerate(stations['station_channels'][station].get('classical_channels',[])):
                    compound_circuit_settings['channel_labels_classical'][channel_number] = station + ', Clas.' + str(local_number)
                    circuit_draw_settings_dict['channel_labels_classical'] = compound_circuit_settings['channel_labels_classical'].copy()     
        
        # create a dictionary which per stations gives the title and channel labels
        settings_for_drawing_station = dict([])
        for station in station_list:
            settings_for_this_station = circuit_draw_settings_dict.copy()
            settings_for_this_station['compound_circuit_title'] += ', station: ' + station
            settings_for_this_station['channel_labels_optical'] = [compound_circuit_settings['channel_labels_optical'][index] for index in stations['station_channels'][station].get('optical_channels',[])]
            settings_for_this_station['channel_labels_classical'] = [compound_circuit_settings['channel_labels_classical'][index] for index in stations['station_channels'][station].get('classical_channels',[])]
            settings_for_drawing_station.update({station:settings_for_this_station})

        # add also settings for drawing the ovrall circuit, this contains the channel labels that are consistent with the ones used per station
        settings_for_drawing_station.update({'total_circuit':circuit_draw_settings_dict})
        return settings_for_drawing_station
    
    def _filter_on_station(self,station_to_draw: int, station_channels: dict) -> any:
        """ Create a new (reduced) circuit containing only the channels for a single station, and the connections to other 
            stations in the global circuit. If the original circuit has a bridge to a next circuit the returned circuit will 
            also bridge to a new circuit.

            Important Note: The returned circuit is not executable, the sole purpose is to visualize the stations individually. 
            For execution only the global circuit can be used.

            Paramater station_channels should be of form:                 
                    { 'bob': {'optical_channels': [0,1], 'classical_channels': [0,1]}, 
                      'alice': {'optical_channels': [2,3], 'classical_channels': [2,3]}  } 
        Args:
            station_to_draw (str): name of the station to draw, e.g. 'alice'
            station_channels (dict): dictionary with station names and keys and channel numbers as values, e.g., 

        Returns:
            fsc.FockStateCircuit: (reduced) circuit containing only channels belonging to a single station
        """

        # make sure that we at least have an empty list for optical and classical channels
        for station, channels in station_channels.copy().items():
                channels = {'optical_channels': [], 'classical_channels': []} | channels
                station_channels.update({station:channels})

        # these are the channels relevant for this specific station
        optical_channels_for_this_station = station_channels[station_to_draw].get('optical_channels',[])
        classical_channels_for_this_station = station_channels[station_to_draw].get('classical_channels',[])
      
        # make a circuit specific for this station (self is the total circuit)  
        # this is the circuit that will be returned by the function
        circuit_for_this_station = deepcopy(self)
        circuit_for_this_station.node_list = []
        circuit_for_this_station._length_of_fock_state= self._length_of_fock_state
        circuit_for_this_station._no_of_optical_channels= len(set(optical_channels_for_this_station).intersection(set([i for i in range(self._no_of_optical_channels)])))
        circuit_for_this_station._no_of_classical_channels= len(set(classical_channels_for_this_station).intersection(set([i for i in range(self._no_of_classical_channels)])))
        circuit_for_this_station._name= self._circuit_name + '/' + station_to_draw

        # iterate through the list with all the nodes in the total circuit (self is the total circuit)  
        # modify the 'original node' from the total circuit to become a 'local node' in the circuit for the local station
        for node_index, original_node in enumerate(self.node_list):
            # get the original channels (the numbers in the total circuit) for this node
            original_optical_channels = original_node['node_info'].get('channels_optical',[])
            original_classical_channels = original_node['node_info'].get('channels_classical',[])

            # determine the channel numbers for the node in the circuit for the local station
            new_optical_channels = [index for index, number in enumerate(optical_channels_for_this_station) if number in original_optical_channels]
            new_classical_channels = [index for index, number in enumerate(classical_channels_for_this_station) if number in original_classical_channels]
            
            # create node_info for the local node in the circuit for the local station
            local_node_info = deepcopy(original_node['node_info'])
            local_node_info.update({'channels_optical' : new_optical_channels, 'channels_classical' : new_classical_channels})
            local_node_info.update({'connections_to_other_stations' : {'classical' : [], 'optical': [], 'optical_and_classical' : []}})

            # split up the different cases
            if original_node['node_type'] == 'bridge':
                pass

            elif len(new_optical_channels) == 0 and len(new_classical_channels) == 0:
                # if the original_node has no presence in the local circuit skip the node completely and do not even add
                # it to the node list for the local circuit_for_this_station
                continue

            elif original_node['node_type'] == 'measurement':
                # special case for measurement node, check if the measured and the written channel belong to the same station
                # for each pair (if the node measures channel [0,1,2] to write channels [0,1,2] and 'alice' has channels [0,1], 
                # 'bob' has channel 2 this is still a local node, even though we 'define' it on the global circuit)
                # loop through all other stations
                for optical_channel, classical_channel in zip(original_optical_channels,original_classical_channels):
                    present_in_this_station = (optical_channel in optical_channels_for_this_station)
                    present_in_this_station = present_in_this_station or (classical_channel in classical_channels_for_this_station)
                    present_in_other_station = False
                    for other_station, channels_in_other_station in station_channels.items():
                        if other_station == station_to_draw:
                            continue
                        present_in_other_station = (optical_channel in channels_in_other_station['optical_channels'])
                        present_in_other_station = present_in_other_station or (classical_channel in channels_in_other_station['classical_channels'])
                        if present_in_this_station and present_in_other_station:
                            local_node_info['connections_to_other_stations']['classical'].append(other_station)               

            else:
                # this is the general case for a node that is not a bridge or a measurement node
                # we want to draw an 'optical' connection if for this node in this station we have optical which connnects
                # to pure optical in another station.
                # if this station only has classical, or the other stations only have classical connection we draw a classical line
                # in other case we draw a double line

                # determine the nature of connections in this station
                optical_in_this_station = (len(new_optical_channels) != 0)
                classical_in_this_station = (len(new_classical_channels) != 0)
                if not (optical_in_this_station or classical_in_this_station):
                    continue

                # iterature through the other stations and determine the nature of connections there.
                for other_station, other_channels in station_channels.items():
                    optical_in_other_station, classical_in_other_station = False, False

                    # skip this station, we only want to check other stations
                    if other_station == station_to_draw:
                        continue                  

                    # determine the nature of connections in the other station
                    connections = (len(set(other_channels['optical_channels']).intersection(set(original_optical_channels))),
                                len(set(other_channels['classical_channels']).intersection(set(original_classical_channels))))
                    if connections == (0,0):
                        continue
                    elif connections[0] == 0:
                        classical_in_other_station = True
                    elif connections[1] == 0:  
                        optical_in_other_station = True   
                    else:
                        optical_in_other_station = True
                        classical_in_other_station = True
                    

                    # apply the logic
                    if not (optical_in_other_station or classical_in_other_station):
                        continue
                    elif classical_in_this_station and not optical_in_this_station:
                        # only classical in this station, connection can only be classical
                        local_node_info['connections_to_other_stations']['classical'].append(other_station)

                    elif classical_in_other_station and not optical_in_other_station:
                        # only classical in other station, connection can only be classical
                        local_node_info['connections_to_other_stations']['classical'].append(other_station)

                    elif ((optical_in_this_station and not classical_in_this_station) and 
                                                (optical_in_other_station and not classical_in_other_station)):
                        # pure optical at both sides, connection can only be optical
                        local_node_info['connections_to_other_stations']['optical'].append(other_station)    

                    else:
                        # at least one side is double (optical and classical)
                        local_node_info['connections_to_other_stations']['optical_and_classical'].append(other_station)


            node_has_optical_connection_to_other_circuit = (len(local_node_info['connections_to_other_stations']['optical']) != 0)
            node_has_classical_connection_to_other_circuit = (len(local_node_info['connections_to_other_stations']['classical']) != 0)
            node_has_optical_and_classical_to_other_circuit = (len(local_node_info['connections_to_other_stations']['optical_and_classical']) != 0)
            
            if node_has_optical_and_classical_to_other_circuit or (node_has_optical_connection_to_other_circuit and node_has_classical_connection_to_other_circuit):
                local_node_info['connections_to_other_stations']['optical_and_classical'] += local_node_info['connections_to_other_stations']['classical']
                local_node_info['connections_to_other_stations']['optical_and_classical'] += local_node_info['connections_to_other_stations']['optical']
                node_has_classical_connection_to_other_circuit = False
                node_has_optical_connection_to_other_circuit = False
                node_has_optical_and_classical_to_other_circuit = True

            if any([node_has_optical_connection_to_other_circuit,
                    node_has_classical_connection_to_other_circuit,
                    node_has_optical_and_classical_to_other_circuit]):
                label_text_for_connection_to_other_circuit = 'To station(s):\n'
                if node_has_optical_connection_to_other_circuit:
                    label_text_for_connection_to_other_circuit += ','.join( sorted(set(local_node_info['connections_to_other_stations']['optical'])) )
                if node_has_classical_connection_to_other_circuit:
                    label_text_for_connection_to_other_circuit += ','.join( sorted(set(local_node_info['connections_to_other_stations']['classical'])) )
                if node_has_optical_and_classical_to_other_circuit:
                    label_text_for_connection_to_other_circuit += ','.join( sorted(set(local_node_info['connections_to_other_stations']['optical_and_classical'])) )
                local_node_info['connections_to_other_stations'].update({'label_text':label_text_for_connection_to_other_circuit})
                local_node_info['connections_to_other_stations'].update({'identifier':'$'+chr(1+16*((node_index+30)//9)+node_index%10)+'$'})
            
            circuit_for_this_station.node_list.append({'node_info':local_node_info, 'node_type' : original_node['node_type']})
            if 'next_fock_state_circuit' in original_node.keys():
                new_next_circuit = original_node['next_fock_state_circuit']._filter_on_station(station_to_draw, station_channels)
                circuit_for_this_station.node_list[-1].update({'next_fock_state_circuit':new_next_circuit})

        return circuit_for_this_station