import os 
import sys

sys.path.append(os.path.abspath(os.curdir))

import torch

from src.components.helpers import get_vector, get_vector_path, POTENT_WRAPPERS, MIDDLE_WRAPPERS

def normalize_vectors(wrappers: list, layers: list, variable_element=None):
    for layer in layers:
        print(layer)
        norms = {}
        vecs = {}
        new_paths = {}
        for wrapper in wrappers:
            vec_path = get_vector_path(layer, model_name_path="lymsys/vicuna-13b-v1.5", wrapper=wrapper, variable_element=variable_element)
            vec = torch.load(vec_path)
            norm = vec.norm().item()
            vecs[wrapper] = vec
            norms[wrapper] = norm
            new_path = vec_path.replace("without_norm_vectors", "normalized_vectors")
            new_paths[wrapper] = new_path
        print(norms)
        mean_norm = torch.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for wrapper in wrappers:
            vecs[wrapper] = vecs[wrapper] * mean_norm / norms[wrapper]
        # save the normalized vectors
        for wrapper in wrappers:
            if not os.path.exists(os.path.dirname(new_paths[wrapper])):
                os.makedirs(os.path.dirname(new_paths[wrapper]))
            torch.save(vecs[wrapper], new_paths[wrapper])
            
           
if __name__== "__main__":
    
    # Normalize jailbreak vectors with respect to each other
    wrappers = POTENT_WRAPPERS + MIDDLE_WRAPPERS
    wrappers = [item.lower() for item in wrappers]
    wrappers.append("italian")
    wrappers.append("adverserial_suffix")

    normalize_vectors(wrappers = wrappers, layers=[19], variable_element="none_jail")

    # Inspect results and imported random vector
    for wrapper in wrappers: 
        vec_1 = get_vector(layer=19, model_name_path="lymsys/vicuna-13b-v1.5", 
                   wrapper=wrapper, normalized=True, variable_element="none_jail")
        print(wrapper)
        print(vec_1.norm().item())
        print(vec_1.size())

    vec_random = torch.load("./vectors/normalized_vectors/vec_random_layer_19_vicuna-13b-v1.5_none_jail.pt")
    print(vec_random.norm().item())
    print(vec_random.size())

    '''
    Logic for generating random vectors

    # Desired dimension and norm
    dimension = 5120
    desired_norm =48.875

    # Step 1: Generate a random vector with the specified dimension
    random_vector = torch.randn(dimension)

    # Step 2: Normalize the vector to have unit norm
    unit_norm_vector = random_vector / torch.norm(random_vector, p=2)

    # Step 3: Scale the normalized vector to have the desired norm
    scaled_vector = unit_norm_vector * desired_norm

    # Verify the norm of the resulting vector
    resulting_norm = torch.norm(scaled_vector, p=2)
    
    vec_path = get_vector_path(19, model_name_path="lymsys/vicuna-13b-v1.5", wrapper="random", normalized=True, variable_element="none_jail")
    print(vec_path)
    
    torch.save(scaled_vector, vec_path)
            
    print(f"Desired Norm: {desired_norm}")
    print(f"Resulting Norm: {resulting_norm}")
    print(f"Vector: {scaled_vector}")

    '''
    
    
    
    
    
    