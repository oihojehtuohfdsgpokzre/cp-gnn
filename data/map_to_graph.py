import os
import numpy as np
from numpy import linalg as LA
import networkx as nx
import xml.etree.ElementTree as ET
import sys
import json
import matplotlib.pyplot as plt

def compressGraph(graph):

    def tuple_in_links(start, end, links):
        for src, dst, _ in links:
            if src == start and dst == end:
                return True
            if dst == start and src == end: 
                return True            
        return False

    # Build immediate neibourgh list for every node
    node_neibourgh = dict()

    for edge in graph['graph_edges']:
        src_id = edge['src']
        dst_id = edge['dst']
        distance = edge['distance']
        
        if src_id not in node_neibourgh.keys():
            node_neibourgh[src_id] = []
            
        if not any(dst_id == node[0] for node in node_neibourgh[src_id]):
            node_neibourgh[src_id].append((dst_id, distance))
        
        if dst_id not in node_neibourgh.keys():
            node_neibourgh[dst_id] = []
            
        if not any(src_id == node[0] for node in node_neibourgh[dst_id]):
            node_neibourgh[dst_id].append((src_id, distance))


    '''
    Starting from every dead-end road (node with only 1 connexion)
        Q: What about round roads ?
    Reach the nearest intersection (node with more than 2 connexsions)
    Explore each road starting from intersection (just not the road which was used to find the intersection)
        Skip intersection from which exploration is already scheduled.
    All this implemented as 'dynamic programming': there is a list of roads to explore(Init node, and 1st connected node to use).
    '''

    nodes_to_explore = []
    explored_intersections = []
    
    hyper_connexions = []
    intersection_nodes = []
    
    # Find every dead end (end of road)
    for node in node_neibourgh.keys():
        if len(node_neibourgh[node]) == 1:
            nodes_to_explore.append((node, node_neibourgh[node][0][0], node_neibourgh[node][0][1]))

    # process every road to explore
    while (len(nodes_to_explore) > 0):
        (starting_node, next_node, distance) = nodes_to_explore.pop()
        
        path_lenght = distance
        previous_node = starting_node
        while (len(node_neibourgh[next_node]) == 2):    # Progress untill an intersection (or a dead end) is reached.
            for linked_node, distance in node_neibourgh[next_node]: # There is only 2 connexion
                if linked_node != previous_node: # Find the one different from previous_node
                    path_lenght += distance
                    previous_node = next_node   # and progress along the road
                    next_node = linked_node
                    break
        # When number of neibourgh is different from 2, a dead-end or an intersection was reached
        if not tuple_in_links(starting_node, next_node, hyper_connexions):
            hyper_connexions.append((starting_node, next_node, path_lenght))
            hyper_connexions.append((next_node, starting_node, path_lenght))
            if starting_node not in intersection_nodes:
                intersection_nodes.append(starting_node)
            if next_node not in intersection_nodes:
                intersection_nodes.append(next_node)                

        if next_node not in explored_intersections:
            for linked_node, distance in node_neibourgh[next_node]:
                if linked_node != previous_node:
                    nodes_to_explore.append((next_node, linked_node, distance))
        explored_intersections.append(next_node)
        
    return hyper_connexions, intersection_nodes
    
def main():

    print('converting osm to map files')
    
    for folder, dirs, files in os.walk(os.path.join('.', 'maps'), topdown=True):
        for name in files:
            if '.json' in name:
            
                city = name.split('_')[1].split('.')[0]
                print('--- processing --- ', city)
                
                map_file = open(os.path.join(folder, name), 'r')
                map_data = json.load(map_file)
                map_file.close() 

                number_of_nodes = len(map_data['graph_nodes'])    
                print('raw number of nodes', number_of_nodes)

                hyper_connexions, nodes = compressGraph(map_data)

                number_of_nodes = len(nodes)
                print('number of hyper nodes', number_of_nodes)


                # generate indexes of nodes : continuous and starting from 0
                node_id_to_nb = dict()
                unreached_nodes = []
                positions = []
                for index , node in enumerate(nodes):
                    node_id_to_nb[node] = index
                    unreached_nodes.append(index)
                    positions.append(map_data['graph_nodes'][node])

                adj_matrix = np.zeros((number_of_nodes, number_of_nodes))

                for source, dest, distance in hyper_connexions:
                    src = node_id_to_nb[source]
                    dst = node_id_to_nb[dest]
                    adj_matrix[src, dst] = distance

                G = nx.from_numpy_matrix(adj_matrix)

                node_sets = []
                bigger_subset = 0
                while(len(unreached_nodes) > 0):
                    sub_set = []
                    init_node = unreached_nodes[0]
                    sub_set.append(init_node)
                    new_unreached_nodes = []
                    for node in unreached_nodes[1:]:
                        try:
                            shortest_dist = nx.dijkstra_path_length(G, init_node, node)
                        except:
                            new_unreached_nodes.append(node)
                            continue
                        sub_set.append(node)
                    node_sets.append(sub_set)
                    if len(sub_set) > len(node_sets[bigger_subset]):
                        bigger_subset = len(node_sets)-1
                    unreached_nodes = new_unreached_nodes
                    
                print('number of subgraph', len(node_sets))
                print('bigger subset has ', len(node_sets[bigger_subset]), ' nodes')

                connected_adj_matrix = np.zeros((len(node_sets[bigger_subset]), len(node_sets[bigger_subset])))

                for id1, node1 in enumerate(node_sets[bigger_subset]):
                    for id2, node2 in enumerate(node_sets[bigger_subset]):
                        connected_adj_matrix[id1, id2] = adj_matrix[node1, node2]
                # print(connected_adj_matrix)

                G2 = nx.from_numpy_matrix(connected_adj_matrix)
                shortest_costs = np.zeros((connected_adj_matrix.shape[0],connected_adj_matrix.shape[0]))

                count = 0
                for i in range(connected_adj_matrix.shape[0]):
                    for j in range(connected_adj_matrix.shape[0]):
                        shortest_costs[i][j] = nx.dijkstra_path_length(G2, i, j)

                data = {'adj': connected_adj_matrix.tolist(), 'number_of_nodes': len(connected_adj_matrix), 'shortest_costs': shortest_costs.tolist()}
                
                graph_file = open(os.path.join('.', 'graphs', 'graph_'+city+'.json'), 'w')
                json.dump(data, graph_file, indent=2)
                graph_file.close() 

main()