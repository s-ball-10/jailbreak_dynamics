# Code from Nina Rimsky: https://github.com/nrimsky/CAA

import os
import sys

sys.path.append(os.path.abspath(os.curdir)) 

import torch as t
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils.settings import Settings
from src.components.helpers import ( 
    tokenize_vicuna,
    tokenize_llama_chat,
    add_vector_after_position,
    find_instruction_end_position,
    ADD_AFTER_POS_CHAT,
)


SETTINGS = Settings()

hf_api_token = SETTINGS.models["general"]["api_token"]
t.cuda.empty_cache()

class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output
    

class BlockOutputWrapper(t.nn.Module):
 
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block 
        self.unembed_matrix = unembed_matrix 
        self.norm = norm 
        self.tokenizer = tokenizer
        
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm
        
        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None
    
        self.activations = None
        self.add_activations = None
        self.after_position = None
        self.save_internal_decodings = False
        
        
    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
           
        if self.add_activations is not None: 
            augmented_output = add_vector_after_position(
                matrix = output[0],
                vector = self.add_activations,
                position_ids = kwargs["position_ids"], 
                after = self.after_position
            )
            output = (augmented_output,) + output[1:]
            
        if not self.save_internal_decodings: 
            return output
        
        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output
    
    def add(self, activations):
            self.add_activations = activations    
        
    def reset(self):
        self.activations = None
        self.block.self_attn.activations = None
        self.add_activations = None
        self.after_position = None
  

    
class ModelWrapper: 
    def __init__(self, model_name: str, hf_token: str = hf_api_token):
        self.model_name = model_name
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token = hf_token)

        self.model = self.model.half() ## for memory efficiency
        self.model = self.model.to(self.device)
        self.END_STR = t.tensor(self.tokenizer.encode(ADD_AFTER_POS_CHAT)[1:]).to(self.device)   
        
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm, self.tokenizer) 
            
                
    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value
            
    def set_after_positions(self, pos:int):
        for layer in self.model.model.layers:
            layer.after_position = pos
                            
    def generate(self, tokens, max_new_tokens = SETTINGS.models["general"]["max_new_tokens"]):
        with t.no_grad():
            instr_pos = find_instruction_end_position(tokens[0], self.END_STR) 
            self.set_after_positions(instr_pos) 
            generated = self.model.generate(
                inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
            )
            return self.tokenizer.batch_decode(generated)[0] 
            
    def generate_text(self, user_input: str, model_output:Optional[str] = None, 
                      with_sys_prompt: Optional[str] = None, 
                      max_new_tokens:int = SETTINGS.models["general"]["max_new_tokens"],
                      ) -> str:
        '''
        tokens = tokenize_llama_chat(
            tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt
            ) ## This gives a special format to the user input
        '''
        if self.model_name == "lmsys/vicuna-13b-v1.5":
            tokens = tokenize_vicuna(
                tokenizer=self.tokenizer, user_input=user_input, model_output=model_output
                )
        else:
            tokens = tokenize_llama_chat(
                tokenizer=self.tokenizer, user_input=user_input, model_output=model_output
                )
        
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)
        
        
    def get_logits(self, tokens):
        with t.no_grad():
            instr_pos = find_instruction_end_position(tokens[0], self.END_STR)
            self.set_after_positions(instr_pos)
            logits = self.model(tokens).logits
            return logits 
        
    def get_logits_from_text(self, user_input: str, model_output: Optional[str] = None, with_sys_prompt: Optional[str] = True) -> t.Tensor:
        if self.model_name == "lmsys/vicuna-13b-v1.5":
            tokens = tokenize_vicuna(
                tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, with_sys_prompt=with_sys_prompt
                )
        else:
            tokens = tokenize_llama_chat(
                tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=None
                )
            
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)
        
    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)
            
    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()
            
            

    
    

          

            
            
            
            
        
            
            
        
    
        
    
        
    




