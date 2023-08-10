import os
import shutil
import pickle

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from codebase.pt_funcs.models import LBMBaselineModel
from torch.types import Device
from torch.utils.data import DataLoader

from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
from captum.concept import TCAV

from codebase_TCAV.utils_CAVs import assemble_concept
from captum.concept._utils.common import concepts_to_str
from codebase_TCAV.dataloader import TCAV_dataset
from codebase_TCAV.Custom_Classifier import EfficientClassifier, Classifier, DefaultClassifier

import datetime
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

out_folder = 'TCAV_out/'+run_time+'/'
os.mkdir('TCAV_out/'+run_time)


if torch.cuda.is_available():
    device = 'cuda'
    print('Device: cuda available')
else:
    device = 'cpu'
    print('Device: cuda not available, using cpu')

# Setting up Concepts

concept_path = 'does_not_matter'

concept_random = assemble_concept(
    'random_class_2', 0, concept_path, device=device)
concept_random_2 = assemble_concept(
    'impervious_surface', 2, concept_path, device=device)
concept_1 = assemble_concept(
    'vegetation_wo10', 1, concept_path, device=device)
concept_3 = assemble_concept(
    'city_dense', 3, concept_path, device=device)
concept_4 = assemble_concept(
    'city_medium', 4, concept_path, device=device)
concept_5 = assemble_concept(
    'city_sparse', 5, concept_path, device=device)
concept_6 = assemble_concept(
    'agriculture', 6, concept_path, device=device)
concept_7 = assemble_concept(
    'original_water', 7, concept_path, device=device)
concept_8 = assemble_concept(
    'parking_spaces', 8, concept_path, device=device)


experiment_setup = [[concept_1, concept_random], [concept_random_2, concept_random], [concept_3, concept_random], [concept_4, concept_random], [
    concept_5, concept_random], [concept_6, concept_random], [concept_7, concept_random], [concept_8, concept_random]]

# Define Setup
setup_list = []
for concept_pair in experiment_setup:
    setup_list.append(f'{concept_pair[0].id}-{concept_pair[1].id}')

# Define Layers to evaluate
layers = ['layer1', 'layer2', 'layer3', 'avgpool']


# Set up baseline model
label_info = {}
label_info['dim_scores'] = {}
label_info['lbm_score'] = {}
label_info['lbm_score']['ylims'] = [-1.5, 1.5]
checkpoint_path = 'codebase_TCAV/epoch=14-val_lbm_mse=0.03.ckpt'
model = LBMBaselineModel('outputs_tcav/',
                         'baseline', label_info, splits=['val', 'test', 'train'])
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))[
    'state_dict']
model.load_state_dict(state_dict)
model.eval()
if device == 'cuda':
    model.model.model.to(device)


# Load inputs via dataloader
data_path = 'data/source/test_data_dict.pkl'
test_dataset = TCAV_dataset(data_path, transform=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=3)


# define TCAV Vectors for concepts
TCAV_0 = TCAV(model=model.model.model,
              layers=layers,
              layer_attr_method=LayerIntegratedGradients(
                  model, None,
                  multiply_by_inputs=False),
              classifier=EfficientClassifier(epochs=1, device=device))


# TCAV_0 = TCAV(model=model.model.model,layers=layers,classifier=EfficientClassifier(epochs=1, device=device))


# compute Classifier and save for accuracies and CAV weights
cavs_computed = TCAV_0.compute_cavs(experiment_setup)
for concept_pair in setup_list:
    for idx, layer in enumerate(layers):
        accs_score = cavs_computed[concept_pair][f"{layer}"].stats['accs']
        print(
            f'Accs score for concept pair {concept_pair} in {layer} is {accs_score}')

# shutil.rmtree('cav/')


# set up list for iteration over dataset
layer_scores = {concept_pair: {layer: {'scores': [], 'total': []}
                               for layer in layers} for concept_pair in setup_list}
layer_magnitude = {concept_pair: {layer: {'scores': [], 'total': []}
                                  for layer in layers} for concept_pair in setup_list}

counter = 0

# iterate over dataset and compute TCAV scores
num_batches = len(test_dataloader)
for idx, batch in enumerate(test_dataloader):
    print(f'Compute TCAV Scores for batch {idx} out of {num_batches}')
    batch[0].to(device)
    tcav_scores = TCAV_0.interpret(
        inputs=batch[0],
        experimental_sets=experiment_setup,
        n_steps=5)
    for concept_pair in setup_list:
        for layer in layers:
            layer_scores[concept_pair][layer]['scores'].append(tcav_scores[concept_pair]
                                                               [layer]['sign_count'])
            layer_magnitude[concept_pair][layer]['scores'].append(tcav_scores[concept_pair]
                                                                  [layer]['magnitude'])
    counter += 1
    # checkpoints
    if counter == 1000:
        with open(out_folder+f'layer_scores{idx}.pkl', 'wb') as f:
            pickle.dump(layer_scores, f)

        with open(out_folder+f'layer_magnitude{idx}.pkl', 'wb') as f:
            pickle.dump(layer_magnitude, f)

        counter = 0


# Compute overall score per layer and normalize
for concept_pair in setup_list:
    for layer in layers:
        layer_scores[concept_pair][layer]['total'] = torch.mean(
            torch.vstack(layer_scores[concept_pair][layer]['scores']), dim=0)
        layer_magnitude[concept_pair][layer]['total'] = torch.mean(
            torch.vstack(layer_magnitude[concept_pair][layer]['scores']), dim=0)
        curr_layer_score = layer_scores[concept_pair][layer]['total']
        curr_layer_magnitude = layer_magnitude[concept_pair][layer]['total']

        print(
            f'TCAV score for concept-pair {concept_pair} in {layer} is {curr_layer_score} with magnitude {curr_layer_magnitude}')


# save layer_scores and layer_magnitude
with open(out_folder+'layer_scores.pkl', 'wb') as f:
    pickle.dump(layer_scores, f)

with open(out_folder+'layer_magnitude.pkl', 'wb') as f:
    pickle.dump(layer_magnitude, f)
