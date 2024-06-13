import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import json
import torch 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.components.model_wrapper import ModelWrapper
from src.components.helpers import tokenize_vicuna, get_activations_dir, get_vector_path, get_vector_dir
from typing import List
from tqdm import tqdm
from src.utils.settings import Settings

SETTINGS = Settings()

hf_api_token = SETTINGS.models["general"]["api_token"]

class ExtractionDataset(Dataset):
    def __init__(self, token, data_path, choice:bool):
        self.data_path = data_path
        with open(self.data_path, 'r') as f:
            self.dataset = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            SETTINGS.models["tokenizer"]["model_name"], token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.choice = choice
        
                
    def prompt_to_tokens(self, instruction, model_output=None):
        tokens = tokenize_vicuna(
            self.tokenizer, 
            user_input = instruction, 
            model_output = model_output
        )
        return torch.tensor(tokens).unsqueeze(0)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        wrapper = item["wrapper"]
        inst_end_position = None
        
        if self.choice == "none_jail": 
            j_text = item["jailbreak"]  
            none_text = item["none"] 
            j_tokens = self.prompt_to_tokens(j_text)
            none_tokens = self.prompt_to_tokens(none_text)
            
        
        if self.choice == "harm":
            j_text= item["jailbreak"]
            none_text=item["jailbreak_h"]
            
            j_tokens = self.prompt_to_tokens(j_text)
            none_tokens = self.prompt_to_tokens(none_text)
            
        if self.choice == "help":
            
            q_text = item["jailbreak_h"]  
            j_text = item["completion_h_short"]
            none_text = item["refusal_h_short"]
            
            j_tokens = self.prompt_to_tokens(q_text, j_text)
            none_tokens = self.prompt_to_tokens(q_text, none_text)
            
            q_tokens_single = self.prompt_to_tokens(q_text)
            inst_end_position = q_tokens_single.numel()
            return none_tokens, j_tokens, wrapper, inst_end_position
            
        if self.choice == "evolution":
            q_text = item["jailbreak"] 
            none_text = item["jailbreak"] 
            j_text = item['response']
            none_tokens = self.prompt_to_tokens(none_text)
            j_tokens = self.prompt_to_tokens(q_text, j_text)
        
        return none_tokens, j_tokens, wrapper, inst_end_position
    
    

def get_activations(
    layers: List[int],
    model_name: str,
    data_path: str,
    variable_element: str = None,
    save_activations: bool = True,
    choice: str = "none_jail",
):
    if save_activations and not os.path.exists(get_activations_dir()):
        os.makedirs(get_activations_dir())

    model = ModelWrapper(model_name)
    model.set_save_internal_decodings(False)
    model.reset_all()
    
    diff_activations_by_wrapper = {}  
    activations_by_wrapper = {}

    if choice == "harm":
        activations_by_wrapper["harmful_question"] = {layer: [] for layer in layers}
        activations_by_wrapper["harmless_question"] = {layer: [] for layer in layers}
        
    
    dataset = ExtractionDataset(
        hf_api_token,
        data_path= data_path,
        choice = choice
    )
    
    for none_tokens, j_tokens, wrapper, inst_end_position in tqdm(dataset, desc="Processing prompts"):
        none_tokens = none_tokens.to(model.device)
        j_tokens = j_tokens.to(model.device)
        
        model.reset_all()
        model.get_logits(none_tokens)
        
        none_activations = {}  
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            if inst_end_position is not None:
                n_activations = get_average_activation(n_activations, inst_end_position)
            
            else:    
                n_activations = n_activations[0, -1, :].detach().cpu()
            none_activations[layer] = n_activations
            
        model.reset_all()
        model.get_logits(j_tokens)
        
        jail_activations = {}  
        for layer in layers:
            j_activations = model.get_last_activations(layer)
            if inst_end_position is not None:
                j_activations = get_average_activation(j_activations, inst_end_position)
            
            else:
                j_activations = j_activations[0, -1, :].detach().cpu() 
            jail_activations[layer] = j_activations
        
        # Calculate the difference between none and jail activations for each layer
        diff_activations = {}
        for layer in layers:
            diff_activations[layer] = jail_activations[layer] - none_activations[layer]  
            
        # Store the difference activations by wrapper
        if wrapper not in diff_activations_by_wrapper:
            diff_activations_by_wrapper[wrapper] = {layer: [] for layer in layers}
            if choice!="harm":
                activations_by_wrapper[wrapper] = {layer: [] for layer in layers}
        
            
        for layer in layers:
            diff_activations_by_wrapper[wrapper][layer].append(diff_activations[layer])
            if choice == "harm":
                activations_by_wrapper["harmless_question"][layer].append(none_activations[layer])
                activations_by_wrapper["harmful_question"][layer].append(jail_activations[layer])

            elif choice != "harm":
                if wrapper == "none":
                    activations_by_wrapper["none"][layer].append(none_activations[layer])
                else:
                    activations_by_wrapper[wrapper][layer].append(jail_activations[layer])
        

    # Save difference in activations
    for wrapper, wrapper_activations in diff_activations_by_wrapper.items():
        for layer, activations in wrapper_activations.items():
            vec_per_wrap_layer = torch.stack(activations).mean(dim=0) 
            if choice == "harm":
                vec_name_harm = f'vec_harmfulness_direction_layer_{layer}_{model.model_name.split("/")[-1]}.pt'
                vec_path_harm = os.path.join("vectors", "without_norm_vectors", vec_name_harm)
                torch.save(vec_per_wrap_layer,
                           vec_path_harm)
            if choice == "help":
                vec_name_help = f'vec_helpfulness_direction_layer_{layer}_{model.model_name.split("/")[-1]}.pt'
                vec_path_help =  os.path.join("vectors", "without_norm_vectors", vec_name_help)
                torch.save(vec_per_wrap_layer,
                           vec_path_help)
            else:
                torch.save(
                    vec_per_wrap_layer,
                    get_vector_path(layer, model.model_name, wrapper.lower(), variable_element=variable_element)
                )
           
    # Store the entire dictionaries
    torch.save(
        activations_by_wrapper,
        os.path.join(get_activations_dir(), f"all_activations_{variable_element}.pt")
    )

    torch.save(
        diff_activations_by_wrapper,
        os.path.join(get_vector_dir(), f"all_diff_activations_{variable_element}.pt")
    )
    
def get_average_activation(activations, inst_end_position):
    print("I am in the loop")
    all_activations_from_inst_end = activations[0, inst_end_position :, :].detach().cpu()
    mean_activations_from_inst_end = all_activations_from_inst_end.mean(dim=0)
    return mean_activations_from_inst_end


             
if __name__ == "__main__":
    
    get_activations(
        layers = [19,39],
        model_name= "lmsys/vicuna-13b-v1.5",
        data_path = "./data/train_vector.json",
        variable_element="none_jail",
        choice = "none_jail"
    )

    get_activations(
        layers = [19,39],
        model_name= "lmsys/vicuna-13b-v1.5",
        data_path = "./data/train_harmful_harmless_combined.json",
        variable_element="harmful",
        choice = "harm"
    )

    get_activations(
        layers = [19,39],
        model_name= "lmsys/vicuna-13b-v1.5",
        data_path = "./data/train_harmful_harmless_combined.json",
        variable_element="helpful",
        choice = "help"
    )

  

    



   
    
   