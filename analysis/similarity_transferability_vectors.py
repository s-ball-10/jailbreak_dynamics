import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from src.components.helpers import (
    get_vector, ALL_WRAPPERS, get_figures_dir)


def prepare_non_diff_activations(activations, add_harmful_direction=None):
        
        vec_per_wrapper_layer = {}
        
        for wrapper, layers in activations.items():
            vec_per_wrapper_layer[wrapper] = {}
            for layer, activation in layers.items():
                stacked_activations = torch.stack(activation).mean(dim=0) 
                vec_per_wrapper_layer[wrapper][layer] = stacked_activations
        
        if add_harmful_direction is not None:
            harmfulness_dict = {}
            harmfulness_dict["harmfulness_direction"] = {}

            vec_harm_19 = get_vector(19, "vicuna-13b-v1.5", "harmfulness_direction")
            vec_harm_39 = get_vector(39, "vicuna-13b-v1.5", "harmfulness_direction")

            harmfulness_dict["harmfulness_direction"][19]=vec_harm_19
            harmfulness_dict["harmfulness_direction"][39]=vec_harm_39

            merged_dict = {**vec_per_wrapper_layer, **harmfulness_dict}
            merged_dict_lower = {key.lower(): value for key, value in merged_dict.items()}
    
            return merged_dict_lower
        
        vec_per_wrapper_layer_lower = {key.lower(): value for key, value in vec_per_wrapper_layer.items()}
              
        return vec_per_wrapper_layer_lower

# Function to calculate cosine similarity between two vectors
def calculate_cosine_similarity(vec1, vec2):
    similarity = cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return similarity.item()
    

# Calculate and store cosine similarity scores
def get_similarity_matrix(layers, wrapper_list, variable_element=None, input_dict_path = None):
    cosine_similarity_scores = {}
    
    if input_dict_path is not None: 
        input_dictionary = torch.load(input_dict_path)
        input_dict = prepare_non_diff_activations(input_dictionary, add_harmful_direction=True)
        
        
    for layer in layers:
        for i in range(len(wrapper_list)):
            for j in range(len(wrapper_list)):
                wrapper1 = wrapper_list[i]
                wrapper2 = wrapper_list[j]
                if input_dict_path is not None:
                    base_vec = input_dict[wrapper1][layer]
                    comparison_vec = input_dict[wrapper2][layer]
                else:
                    base_vec = get_vector(layer, model_name_path="vicuna-13b-v1.5", wrapper=wrapper1, variable_element=variable_element)
                    comparison_vec = get_vector(layer, model_name_path="vicuna-13b-v1.5", wrapper=wrapper2, variable_element=variable_element)
                similarity_score = calculate_cosine_similarity(base_vec, comparison_vec)
                cosine_similarity_scores[(wrapper1, wrapper2, layer)] = similarity_score

    return cosine_similarity_scores

# Plot single similarity score comparison (bar plot)
def plot_similarity_scores_single(cosine_similarity_scores, target_wrapper, wrapper_list, specific_wrapper='none', save_add=None):
    layers = sorted(set(k[2] for k in cosine_similarity_scores.keys()))
    
    for layer in layers:
        scores = []
        wrappers = []
        target_score = None
        
        for wrapper2 in wrapper_list:
            if wrapper2 == target_wrapper:
                continue
            if (target_wrapper, wrapper2, layer) in cosine_similarity_scores:
                score = cosine_similarity_scores[(target_wrapper, wrapper2, layer)]
                if wrapper2 == specific_wrapper:
                    target_score = score
                else:
                    scores.append(score)
                    wrappers.append(wrapper2)
        
        # Sort the scores and wrappers
        sorted_indices = np.argsort(scores)[::-1]
        scores = np.array(scores)[sorted_indices]
        wrappers = np.array(wrappers)[sorted_indices]

        # Plot bar graph
        plt.figure(figsize=(10, 6))
        plt.bar(wrappers, scores, align='center')
        if target_score is not None:
            plt.axhline(y=target_score, color='r', linestyle='--', label=f'Score with avg. harmf. prompt activations: {target_score:.2f}')
        plt.xlabel('Jailbreaks')
        plt.ylabel('Cosine similarity score')
        #plt.title(f'Cosine similarity scores for {target_wrapper} at Layer {layer}')
        plt.legend()
        plt.xticks(range(len(wrappers)), wrappers, rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    save_to = os.path.join(get_figures_dir(), f'cosine_comparison_{layer}_{save_add}.png')
    plt.savefig(save_to)
    plt.close()


# Create a heatmap using the cosine similarity scores
def plot_heatmap(cosine_similarity_scores, layer, wrapper_list, save_add):
    # Initialize an empty heatmap data array
    heatmap_data = np.zeros((len(wrapper_list), len(wrapper_list)))
    
    # Populate the heatmap data array with cosine similarity scores
    for i, wrapper1 in enumerate(wrapper_list):
        for j, wrapper2 in enumerate(wrapper_list):
            if (wrapper1, wrapper2, layer) in cosine_similarity_scores:
                heatmap_data[i, j] = cosine_similarity_scores[(wrapper1, wrapper2, layer)]
            else:
                heatmap_data[i, j] = cosine_similarity_scores[(wrapper2, wrapper1, layer)]  # Use the reverse key if not found
    
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest', extent=[-0.5, len(wrapper_list)-0.5, -0.5, len(wrapper_list)-0.5], vmin=-1, vmax=1)
    
    # For masked heatmap uncomment this:
    #masked_data = np.where(np.tri(len(wrapper_list), k=0) == 1, heatmap_data, np.nan)
    #plt.imshow(masked_data, cmap='viridis', interpolation='nearest', extent=[-0.5, len(wrapper_list)-0.5, -0.5, len(wrapper_list)-0.5])
    
    # Add labels and ticks
    #plt.title(f'Cosine similarity heatmap for layer {layer +1 }')
    plt.xlabel('Jailbreak')
    plt.ylabel('Jailbreak')
    plt.xticks(range(len(wrapper_list)), wrapper_list, rotation=90)
    plt.yticks(range(len(wrapper_list)), reversed(wrapper_list))
    plt.colorbar(label='Cosine similarity')
    
    plt.tight_layout()
    
    # Save the plot as an image
    save_to = os.path.join(get_figures_dir(), f'heatmap_similarity_layer_{layer}_{save_add}.png')
    plt.savefig(save_to)
    plt.close()
    
    
def plot_heatmap_bounds(cosine_similarity_scores, layer, wrapper_list, save_add):
    # Initialize an empty heatmap data array
    heatmap_data = np.zeros((len(wrapper_list), len(wrapper_list)))
    
    # Populate the heatmap data array with cosine similarity scores
    for i, wrapper1 in enumerate(wrapper_list):
        for j, wrapper2 in enumerate(wrapper_list):
            if (wrapper1, wrapper2, layer) in cosine_similarity_scores:
                heatmap_data[i, j] = cosine_similarity_scores[(wrapper1, wrapper2, layer)]
            else:
                heatmap_data[i, j] = cosine_similarity_scores[(wrapper2, wrapper1, layer)]  # Use the reverse key if not found
    
    # Define custom colormap with discrete colors for different ranges
    bounds = [-1, 0, 0.4, 0.6, 0.8, 1]
    cmap = plt.cm.get_cmap('viridis', len(bounds) - 1)
    #cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot the heatmap with custom colormap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap=cmap, norm=norm, interpolation='nearest', extent=[-0.5, len(wrapper_list)-0.5, -0.5, len(wrapper_list)-0.5])
    
    # Add labels and ticks
    plt.title(f'Cosine similarity heatmap for layer {layer+1}')
    plt.xlabel('Wrapper')
    plt.ylabel('Wrapper')
    plt.xticks(range(len(wrapper_list)), wrapper_list, rotation=90)
    plt.yticks(range(len(wrapper_list)), reversed(wrapper_list))
    plt.colorbar(label='Cosine similarity')
    
    plt.tight_layout()
    
    # Save the plot as an image
    save_to = os.path.join(get_figures_dir(), f'batched_heatmap_similarity_layer_{layer +1}_{save_add}.png')
    plt.savefig(save_to)
    plt.close()
    
    
if __name__=="__main__": 

    layers = [19]
  
    # Heatmap
    wraps = ['refusal_suppression','style_injection_short','AIM','dev_mode_v2', 
            'evil_confidant', 'prefix_injection', 'poems','distractors', 
            'payload_split','few_shot_json', 'wikipedia_with_title','style_injection_json']
    wrapper_list = [item.lower() for item in wraps]
    cosine_similarity_scores = get_similarity_matrix(layers, wrapper_list, variable_element="none_jail")
    for layer in layers: 
        plot_heatmap(cosine_similarity_scores, layer, wrapper_list, "none_jail_subset_flow")
    
    
    # Bar chart
    wraps = ALL_WRAPPERS
    wrapper_list = [item.lower() for item in wraps]
    wrapper_list.append("none")
    wrapper_list.append("harmfulness_direction")
    cosine_similarity_scores = get_similarity_matrix(layers, wrapper_list, input_dict_path="./results/activations/all_activations_none_jail.pt")
    print(cosine_similarity_scores)
    plot_similarity_scores_single(cosine_similarity_scores, target_wrapper="harmfulness_direction", wrapper_list=wrapper_list, specific_wrapper='none', save_add=None)
    
    
    
   
    
    
        
        
        
        
    

    

