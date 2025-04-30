#!/usr/bin/env python
# coding: utf-8

"""
Keras implementation of DNGR model. Generate embeddings for NG3, NG6 and NG9   
of 20NewsGroup dataset. Evaluate with F1-score from MNB classifier and NMI score.
Also visualizing embeddings with t-SNE.

Author: Apoorva Vinod Gorur
"""

import sys
import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import warnings
import DNGR_utils as ut
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import networkx as nx
import keras

#Stage 1 -  Random Surfing
@ut.timer("Random_Surfing")
def random_surf(cosine_sim_matrix, num_hops, alpha):

    # Get number of nodes
    num_nodes = len(cosine_sim_matrix)
    
    adj_matrix = ut.scale_sim_matrix(cosine_sim_matrix)
    P0 = np.eye(num_nodes, dtype='float32')
    P = np.eye(num_nodes, dtype='float32')
    A = np.zeros((num_nodes,num_nodes),dtype='float32')
    
    for i in range(num_hops):
        P = (alpha*np.dot(P,adj_matrix)) + ((1-alpha)*P0)
        A = A + P

    return A



#Stage 2 - PPMI Matrix
@ut.timer("Generating PPMI matrix")
def PPMI_matrix(A):
    
    num_nodes = len(A)
    A = ut.scale_sim_matrix(A)
    
    row_sum = np.sum(A, axis=1).reshape(num_nodes,1)
    col_sum = np.sum(A, axis=0).reshape(1,num_nodes)
    
    D = np.sum(col_sum)
    PPMI = np.log(np.divide(np.multiply(D,A),np.dot(row_sum,col_sum)))
    #Gotta use numpy for division, else it runs into divide by zero error, now it'll store inf or -inf
    #All Diag elements will have either inf or -inf.
    #Get PPMI by making negative values to 0
    PPMI[np.isinf(PPMI)] = 0.0
    PPMI[np.isneginf(PPMI)] = 0.0
    PPMI[PPMI<0.0] = 0.0
    
    return PPMI

#Stage 3 - AutoEncoders
@ut.timer("Generating embeddings with AutoEncoders")
def sdae(PPMI, hidden_neurons):

    #Input layer. Corrupt with Gaussian Noise. 
    inp = keras.Input(shape=(PPMI.shape[1],))
    enc = keras.layers.GaussianNoise(0.2)(inp)
    
    #Encoding layers. Last layer is the bottle neck
    for neurons in hidden_neurons:
        enc = keras.layers.Dense(neurons, activation = 'relu')(enc)
    
    #Decoding layers
    dec = keras.layers.Dense(hidden_neurons[-2], activation = 'relu')(enc)
    for neurons in hidden_neurons[:-3][::-1]:
        dec = keras.layers.Dense(neurons, activation = 'relu')(dec)
    dec = keras.layers.Dense(PPMI.shape[1], activation='relu')(dec)
    
    #Train
    auto_enc = keras.Model(inputs=inp, outputs=dec)
    auto_enc.compile(optimizer='adam', loss='mse')
    
    auto_enc.fit(x=PPMI, y=PPMI, batch_size=10, epochs=5)
    
    encoder = keras.Model(inputs=inp, outputs=enc)
    encoder.compile(optimizer='adam', loss='mse')
    embeddings = encoder.predict(PPMI)
    
    return embeddings


@ut.timer("the whole process")
def process(args):
    graph_dir = args.graph_dir
    num_hops = args.hops
    alpha = args.alpha
    hidden_neurons = args.hidden_neurons
    
    if num_hops < 1:
        sys.exit("DNGR: error: argument --hops: Max hops should be a positive whole number")
        
    if alpha < 0.0 or alpha > 1.0:
        sys.exit("DNGR: error: argument --alpha: Alpha's range is 0-1")
    
    if len(hidden_neurons) < 3:
        sys.exit("DNGR: error: argument --hidden_neurons: Need a minimum of 3 hidden layers")

    # TODO: Assumes all graphs are written in .graphml format, which isn't very resilient
    for file in os.listdir(args.graph_dir):
        file_abs = os.path.join(graph_dir, file)
        # If this isn't a valid file, ignore it
        if not os.path.isfile(file_abs):
            pass
        else:
            graph = (nx.read_graphml(file_abs)).to_undirected()

            # TODO: A compressed version would be better, this is a bit scary, but it'll probably work
            adj_matrix = nx.to_numpy_array(graph)
    
            #Stage 1 - Compute Transition Matrix A by random surfing model
            A = random_surf(adj_matrix, num_hops, alpha)

            #Stage 2 - Compute PPMI matrix
            PPMI = PPMI_matrix(A)

            #Stage 3 - Generate Embeddings using Auto-Encoder
            embeddings = sdae(PPMI, hidden_neurons)

            # Stage 4 - Serialize embeddings in .txt format expected by MUSE
            # Sort nodes by degree
            deg = list(graph.degree)
            # https://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value
            deg = sorted(deg, key=lambda x: x[1], reverse=True)

            os.makedirs("embeddings", exist_ok=True)

            name, _ = os.path.splitext(file)

            # Save embeddings to .txt, sorted by degree
            with open(os.path.join(os.getcwd(), "embeddings", name + ".txt"), 'w') as f:
                for v, d in deg:
                    line = graph.nodes[v]["word"] + " " + " ".join(np.vectorize(str)(embeddings[int(v)])) + "\n"
                    f.write(line)

            #Evaluation
            # ut.compute_metrics(embeddings, target)

            #Visualize embeddings using t-SNE
            # ut.visualize_TSNE(embeddings, target)
            # plt.show()
    
    return



def main():
    
    parser = ArgumentParser('DNGR',description="This is a Keras implementaion of DNGR evaluating the 20NewsGroup dataset.")

    parser.add_argument('--graph_dir', type=str,
                        default="",
                        help='Path to directory containing dictionary graphs to embed.')

    parser.add_argument('--hops', default=2, type=int, 
                       help='Maximum number of hops for Transition Matrix in Random surfing')

    parser.add_argument('--alpha', default=0.98,
                       help='Probability of (alpha) that surfing will go to next vertex, probability of (1-alpha) that surfing  will return to original vertex and restart. Range 0-1')
    
    parser.add_argument('--hidden_neurons', default=[512,256,128], type=int, nargs = '+',
                       help='Eg: \'512 256 128\' or \'256 128 64 32\'.  Number of hidden neurons in auto-encoder layers. Make sure there are 3 or more layers')
    
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    
    process(args)




if __name__ == '__main__':
    main()

