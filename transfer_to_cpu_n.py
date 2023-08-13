import os
import torch
import pickle

with open('layer_scores.pkl', 'rb') as f:
    layer_scores = pickle.load(f)

with open('layer_magnitude.pkl', 'rb') as h:
    layer_magnitude = pickle.load(h)

layers = ['layer1', 'layer2', 'layer3', 'avgpool']
concept_pairs = []

for i in range(8):
    concept_pairs.append(f'{i+1}-0')

layer_scores_cpu = {concept_pair: {layer: {'scores': []}
                                   for layer in layers} for concept_pair in concept_pairs}
layer_magnitude_cpu = {concept_pair: {layer: {'scores': []}
                                      for layer in layers} for concept_pair in concept_pairs}

for layer in layers:
    for pair in concept_pairs:
        for i in range(len(layer_scores[pair][layer]['scores'])):
            layer_scores_cpu[pair][layer]['scores'].append(
                layer_scores[pair][layer]['scores'][i].to('cpu'))

        layer_scores_cpu[pair][layer]['total'] = layer_scores[pair][layer]['total'].to(
            'cpu')

        for i in range(len(layer_magnitude[pair][layer]['scores'])):
            layer_magnitude_cpu[pair][layer]['scores'].append(
                layer_magnitude[pair][layer]['scores'][i].to('cpu'))
        layer_magnitude_cpu[pair][layer]['total'] = layer_magnitude[pair][layer]['total'].to(
            'cpu')


with open('cpu_layer_scores.pickle', 'wb') as g:
    pickle.dump(layer_scores_cpu, g)

with open('cpu_layer_magnitude.pickle', 'wb') as z:
    pickle.dump(layer_magnitude_cpu, z)
