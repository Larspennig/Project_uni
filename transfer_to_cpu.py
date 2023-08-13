import os
import torch
import pickle

with open('layer_scores.pickle', 'rb') as f:
    layer_scores = pickle.load(f)

layer_scores_cpu = layer_scores.to('cpu')

with open('cpu_layer_scores.pickle', 'wb') as g:
    pickle.dump(layer_scores_cpu, g)