import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
import random
import numpy as np
import copy
import json
from minizinc import Instance, Model, Solver, Status
import datetime
import matplotlib.pyplot as plt
import tikzplotlib

# Chargement graph data
def loadData(path):

    file = open(path, 'rb')

    data = json.load(file)
  
    file.close()

    adj_matrix = data['adj']
    number_of_nodes = int(data['number_of_nodes'])
    shortest_costs = data['shortest_costs']
    
    print('Data loaded')
    
    return adj_matrix, number_of_nodes, shortest_costs

class GCN_in_G(torch.nn.Module):
    def __init__(self, features, last_layer = 10):
        super().__init__()
    
        out_sizes = range(10, 2000, 15)
        self.core_features = out_sizes[last_layer]
    
        self.convs = torch.nn.ModuleList()
        
        prev_size = features
        for depth, size in enumerate(out_sizes[:last_layer+1]):
            self.convs.append(GraphConv(prev_size, size))
            prev_size = size

        self.relu = nn.ReLU()
        self.bn_node  = nn.BatchNorm1d(out_sizes[last_layer], momentum = 0.1, track_running_stats = False)

        hidden_size = 2048

        self.sigm = nn.Sigmoid()
        
        self.drop = nn.Dropout(p=0.2)
        self.edge_fc1 = nn.Linear(self.core_features*2, hidden_size)

        self.edge_fc2 = nn.Linear(hidden_size, 100)
        self.pooling = nn.MaxPool1d(100)


    def forward(self, data, waypoints):
        x, edge_index, edge_w = data.x, data.edge_index, data.edge_attr

        for unit_conv in self.convs:
            x = unit_conv(x, edge_index, edge_w)
            x = self.sigm(x)

        x = self.bn_node(x)
        nb_waypoints = len(waypoints)
    
        selected = torch.index_select(x, 0, waypoints)
        
        row = selected.repeat(1, nb_waypoints)
        row = row.view(nb_waypoints,nb_waypoints,self.core_features)
        col = selected.view(1,nb_waypoints*self.core_features)
        col = col.repeat(1, nb_waypoints, 1)
        col = col.view(nb_waypoints,nb_waypoints,self.core_features)
        vect = torch.stack((row, col),2)
        vect = vect.view(nb_waypoints,nb_waypoints,self.core_features*2)
        vect = self.edge_fc1(vect)
        vect = self.sigm(vect)
        vect = self.drop(vect)
        vect = self.edge_fc2(vect)
        vect = self.pooling(vect)
        vect = self.sigm(vect).clone()

        for ind in range(nb_waypoints):
            vect[ind][ind] = torch.tensor([0.0]).to(torch.device('cuda'))
        
        vect = vect.transpose(0,2)        
        return vect
        
        
def read_problems(path, max_len):
    '''
    Read information from database file.
    File structure dependencies should be kepts in this function.
    '''
    problem_list = []
    
    db_file = open(path, 'r')

    line_counter = 0
    for line in db_file.readlines():
        elems = line.split(';')
        optimal = float(elems[0])
        if '.' in elems[1]:
            start, end = int(elems[2]),int(elems[3])
            sequence = [int(x) for x in elems[4:]]
        else:
            start, end = int(elems[1]),int(elems[2])
            sequence = [int(x) for x in elems[3:]]
        problem_list.append((start, end, sequence, optimal))
        
        if max_len != None and line_counter >= max_len-1:
            break
        line_counter += 1
    
    db_file.close()
    
    return problem_list
    

def gen_edges(number_of_nodes, adj_matrix):
    edge_list = []
    edge_val = []
    
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if adj_matrix[i][j] > 0:
                edge_list.append((i,j))
                edge_list.append((j,i))
                edge_val.append(adj_matrix[i][j])
                edge_val.append(adj_matrix[i][j])
                
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    edge_attr = torch.tensor(edge_val, dtype=torch.float)

    return edge_index, edge_attr
    
    
def gen_database(path, number_of_nodes, shortest_costs, edge_index, edge_attr, max_len = None):
    problems = read_problems(path, max_len)
    
    database = []
        
    for start, end, truth_order, optimal in problems:
        wps = copy.copy(truth_order)
        random.shuffle(wps) # in place shuffle, we do not want GNN to process already ordered waypoints

        node_list = []
        for ind in range(number_of_nodes):
            if ind == start:
                node_list.append((1.0, 0.0, 0.0, 1.0, shortest_costs[ind][end] ))
                continue
            if ind == end:
                node_list.append((0.0, 1.0, 0.0, shortest_costs[start][ind], 1.0))
                continue
            if ind in wps:           
                node_list.append((0.0, 0.0, 1.0, shortest_costs[start][ind], shortest_costs[ind][end]))
                continue
            node_list.append((0.0, 0.0, 0.0, shortest_costs[start][ind], shortest_costs[ind][end]))
        
        x = torch.tensor(node_list, dtype=torch.float)

        cur_graph = Data(x = x, edge_index=edge_index, edge_attr = edge_attr)
        
        # We need truth matrix to have same data order as graph processed by GNN
        edge_matrix = torch.zeros(len(wps), len(wps), dtype = torch.float32)
        for ind1, node1 in enumerate(wps):
            for ind2, node2 in enumerate(wps):
                if truth_order.index(node1) == truth_order.index(node2) + 1:
                    edge_matrix[ind1][ind2] = 1.0
                    edge_matrix[ind2][ind1] = 1.0
        
        edge_matrix = torch.unsqueeze(edge_matrix, 0)
        
        database.append((cur_graph, wps, edge_matrix, start, end, optimal))
        
    return database
    
    
def read_intermediate_values(path, optimal, needed_time):
    f = open(path, 'r')
    timings = []
    values = []
    for line in f.readlines():
        if 'UNKNOWN' in line:
            return timings, values
        if 'sum_path' in line:
            # sum_path = 163337.0;
            cur_length = float(line.split('=')[1][:-2]) / 10000
        if 'elapsed' in line:
            # % time elapsed: 0.90 s
            cur_time = float(line.split(':')[1][:-3])
            timings.append(cur_time+needed_time)
            values.append(cur_length / optimal - 1)
    f.close()
    return timings, values
    
def solve_with_minizinc(start, end, allpoints, shortest_costs, edge_prediction, optimal, city, indice):

    # Prepare json datas
    size = len(allpoints)
    
    matrix = np.zeros((size,size), dtype = int)
    edges = np.zeros((size,size), dtype = int)
    ind_wp = []
    ind_to_wp = dict()
    wp_to_ind = dict()
    for ind1, wp1 in enumerate(allpoints):
        if wp1 == start:
            ind_start = ind1+1
            ind_to_wp[ind_start] = start
            wp_to_ind[start] = ind_start
        elif wp1 == end:
            ind_end = ind1+1
            ind_to_wp[ind_end] = end
            wp_to_ind[end] = ind_end
        else:
            ind_wp.append(ind1+1)
            ind_to_wp[ind1+1] = wp1
            wp_to_ind[wp1] = ind1+1
        
        for ind2, wp2 in enumerate(allpoints):
            matrix[ind1][ind2] = int(10000 * shortest_costs[wp1][wp2])


    for ind1, wp1 in enumerate(allpoints):
        for ind2, wp2 in enumerate(allpoints):
            max_edge_pred = max(edge_prediction[ind1][ind2], edge_prediction[ind2][ind1]) # make the edge matrix symetric
            edges[ind1][ind2] = 100 - int(100 * max_edge_pred)

    data = {'matrix': matrix.tolist(), 'number_of_nodes': size - 2, 'start':ind_start, 'end':ind_end, 'edge_prediction': edges.tolist()}
    
    with open('tmp.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)        

    os.system('minizinc -i find_treshold.mzn tmp.json --output-time --time-limit 10000 > tmp_res.txt')
    
    f = open('tmp_res.txt', 'r')
    for line in f.readlines():
        if 'UNKNOWN' in line:
            minimum_TH = 0
            print('unknown min treshold, setting 0')
        if 'min_treshold' in line:
            # min_treshold = 80;
            minimum_TH = int(line.split('=')[1][:-2])
        if 'elapsed' in line:
            # % time elapsed: 4.17 s
            needed_time = float(line.split(':')[1][:-2])
            print(minimum_TH, 'found in', needed_time)
        if '==========' in line:
            break
    f.close()

    data['minimum_TH'] = minimum_TH
    
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)        


    os.system('minizinc -i guided_solver.mzn data.json --output-time --time-limit 50000 > guided_res.txt')
    guided_timings, guided_values = read_intermediate_values('guided_res.txt', optimal, needed_time)
    os.system('minizinc -i unguided_solver.mzn data.json --output-time  --time-limit 50000 > unguided_res.txt')
    unguided_timings, unguided_values  = read_intermediate_values('unguided_res.txt', optimal, 0.0)
    return guided_timings, guided_values, unguided_timings, unguided_values
    

def main():
    # Parameters:
    city = 'angers'
    problem_lenght = 34

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('inference on', device)
    
    adj_matrix, number_of_nodes, shortest_costs = loadData(os.path.join('..', 'data', 'graphs', 'graph_'+city+'.json')) 
    print('city', city, 'graph order', number_of_nodes)
    
    edge_index, edge_attr = gen_edges(number_of_nodes, adj_matrix)

    database_validation = gen_database(os.path.join('..', 'datasets', 'test', 'DB_'+city+'_len_'+str(problem_lenght)+'.csv'), number_of_nodes, shortest_costs, edge_index, edge_attr)

    print('data base loaded', len(database_validation))
    
    model = GCN_in_G(5).to(device)
    print(model)
    
    model.load_state_dict(torch.load(os.path.join('..', 'learning', 'W', city+'.bin'),  map_location=torch.device(device)))

        
    model.eval()
    
    plt.figure(1)
        
    for ind, (myGraph, wps, truth_matrix, start, end, optimal) in enumerate(database_validation):
        print('solving problem', ind)
    
        myGraphGpu = myGraph.to(device)
        
        out = model(myGraphGpu, torch.tensor(wps).to(device))

        edge_prediction = out[0].cpu().detach().numpy()
        
        guided_timings, guided_values, unguided_timings, unguided_values = solve_with_minizinc(start, end, wps, shortest_costs, edge_prediction, optimal, city, ind)
        
        
        if len(guided_values) > 0:
            best_guided = guided_values[-1]
        else:
            best_guided = 'unknow'
        if len(unguided_values) > 0:
            best_unguided = unguided_values[-1]
        else:
            best_unguided = 'unknow'
        
        print('best unguided', best_unguided, 'best guided', best_guided)
        log_file = open('inference_results_'+city+'_'+str(problem_lenght)+'.txt', 'a')
        print('problem:', ind, file = log_file)
        for ind in range(len(guided_timings)):
            print('guided:t:', guided_timings[ind], ':v:', guided_values[ind], file = log_file)
        for ind in range(len(unguided_timings)):
            print('unguided:t:', unguided_timings[ind], ':v:', unguided_values[ind], file = log_file)
        log_file.close()
        plt.plot(unguided_timings,unguided_values, 'k')
        plt.plot(guided_timings,guided_values, 'g')
        
    plt.ylabel("solution gap")
    plt.xlabel("solving time (s)")
    plt.draw()        
    plt.show()
    tikzplotlib.save('inference_reykjavik_'+city+'_'+str(problem_lenght)+'.tex')


main()