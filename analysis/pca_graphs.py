import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from src.components.helpers import get_figures_dir, POTENT_WRAPPERS, MIDDLE_WRAPPERS



def recenter(x, mean=None):
    x = np.array(x)
    if x.size == 0:
        raise ValueError("Input array is empty.")
    if mean is None:
        mean = np.mean(x, axis=0, keepdims=True)
    return x - mean

def perform_pca(layers, activations_by_wrapper, n_components=2):
    pca_results = {}
    for layer in layers:
        all_activations = []
        
        # Concatenate activations
        for wrapper, activations_per_layer in activations_by_wrapper.items():
            if layer in activations_per_layer:
                all_activations.extend(activations_per_layer[layer])
        
        if not all_activations:
            raise ValueError(f"No activations found for layer {layer}")
        
        all_activations = recenter(all_activations)
        pca_model = PCA(n_components=n_components, whiten=False).fit(all_activations)
        components = pca_model.transform(all_activations)
        
        # Check if all components have the same length
        component_lengths = [len(component) for component in components.T]
        if len(set(component_lengths)) != 1:
            raise ValueError("Not all components have the same length.")
        
        explained_variance = pca_model.explained_variance_ratio_
        pca_results[layer] = (components, explained_variance)
        print("I'm done doing PCA")
    return pca_results

       
def plot_pca_results(pca_results, activations_by_wrapper, add_save, combine_layers=False):
    # Define the existing colormaps
    cmap1 = plt.cm.Set1
    cmap2 = plt.cm.Pastel1
    cmap3 = plt.cm.Dark2
    
    # Extract colors from each colormap
    colors1 = cmap1(np.linspace(0, 1, 9))
    colors2 = cmap2(np.linspace(0, 1, 9))
    colors3 = cmap3(np.linspace(0, 1, 8))
    
    # Combine the colors
    combined_colors = np.concatenate((colors1, colors2, colors3))
    
    # Create a new ListedColormap
    custom_cmap = ListedColormap(combined_colors)
    
    wrapper_colors = {}  # Dictionary to store assigned colors for each wrapper
    all_wrappers = set()  # Set to store all unique wrappers

    if combine_layers:
        # Create a single figure with subplots for combined layers
        fig, axes = plt.subplots(1, len(pca_results), figsize=(14, 6), sharey=True)
        layers_in_plot = []
        if len(pca_results) == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one subplot
    
    for idx, (layer, (components, explained_variance)) in enumerate(pca_results.items()):
        if not combine_layers:
            # Create an individual figure for each layer
            plt.figure(figsize=(10, 6))
        
        # Create an extended list of wrappers and corresponding colors
        extended_wrappers = []
        for wrapper, activations_per_layer in activations_by_wrapper.items():
            if layer in activations_per_layer:
                activations = activations_per_layer[layer]
                extended_wrappers.extend([wrapper] * len(activations))  # Repeat wrapper for each activation
                all_wrappers.add(wrapper)  # Add wrapper to set of unique wrappers
                if wrapper not in wrapper_colors:
                    wrapper_colors[wrapper] = custom_cmap(len(wrapper_colors) % custom_cmap.N)
        
        # Plot scatter plot with points colored by wrapper
        if combine_layers:
            ax = axes[idx]
        else:
            ax = plt.gca()
        
        for i, (x, y) in enumerate(components):
            wrapper = extended_wrappers[i]
            color = wrapper_colors[wrapper]
            ax.scatter(x, y, color=color)
        
        if not combine_layers:
            # Create legend with unique wrappers and their colors
            legend_handles = [plt.scatter([], [], color=wrapper_colors[wrapper], label=wrapper) for wrapper in all_wrappers]
            plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1))  # Adjust legend location
            
            #plt.title(f'PCA results for layer {layer+1}')
            plt.xlabel('Principal component 1')
            plt.ylabel('Principal component 2')
            plt.tight_layout()  # Adjust layout to prevent overlap
            
            plt.show()
            # Save the plot as an image
            save_to = os.path.join(get_figures_dir(), f'pca_layer_{layer+1}_{add_save}.png')
            plt.savefig(save_to)
            plt.close()
            print(f"Finished plotting layer {layer+1}")
        else:
            ax.set_title(f'PCA layer {layer+1}')
            ax.set_xlabel('Principal component 1')
            if idx == 0:
                ax.set_ylabel('Principal component 2')
            layers_in_plot.append(layer + 1)  # Store the layer number (1-based index)
    
    if combine_layers:
        # Create shared legend with unique wrappers and their colors
        legend_handles = [plt.scatter([], [], color=wrapper_colors[wrapper], label=wrapper) for wrapper in all_wrappers]
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(wrapper_colors))  # Adjust legend location
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to provide space for legend
        
        plt.show()
        # Save the combined plot as an image with the layer numbers included in the filename
        layers_str = "_".join(map(str, layers_in_plot))
        save_to = os.path.join(get_figures_dir(), f'pca_layers_{layers_str}_{add_save}.png')
        plt.savefig(save_to)
        plt.close()
        print("Finished plotting combined layers")       
        
def filter_activations_by_selected_wrappers(activations_by_wrapper, selected_wrappers):
    filtered_activations = {}
    for wrapper in selected_wrappers:
        if wrapper in activations_by_wrapper:
            filtered_activations[wrapper] = activations_by_wrapper[wrapper]
    return filtered_activations

def combine_activation_dicts(data_paths: list):
    combined_dict = {}
    for file_path in data_paths:
        data = torch.load(file_path)
        combined_dict.update(data)
    return combined_dict
        
if __name__ == "__main__":

    layers = [19,39] 
    
    # PCA plots jailbreak activations layer 20 and 40
    activations_by_wrapper = torch.load("./vectors/without_norm_vectors/all_diff_activations_none_jail.pt")
    wrappers = POTENT_WRAPPERS + MIDDLE_WRAPPERS
    activations_by_wrapper = filter_activations_by_selected_wrappers(activations_by_wrapper, wrappers)
    pca_results = perform_pca(layers, activations_by_wrapper, n_components=2)
    plot_pca_results(pca_results, activations_by_wrapper, "diff_none_jail_EOS_token_potent_middle")
    
    # PCA plot harmful harmless question
    activations_by_wrapper = torch.load("./results/activations/all_activations_harmful.pt")
    pca_results = perform_pca(layers, activations_by_wrapper, n_components=2)
    plot_pca_results(pca_results, activations_by_wrapper, "harmful_harmless")


    

    

