'''
Entry point for the decompilation procedure. General outline of the steps:

1. 

'''

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from data.AdditionDataset import AdditionDataset
from data.CustomCollator import CustomCollator
from model.CustomTokenizer import CustomTokenizer

BASE_CONFIG = {'batch_size_for_pruning': 120, 'device': 'cuda','find_graph_method': 'pruning', 'length_range': [0, 150], 'num_iterations': 100, 'num_repeat_for_pruning': 12, 'path_to_saved_model': '', 'period_for_data': 3, 'prune_inputs_to_mlps_and_lm_head': True, 'seed': 0, 
               'training_steps0_for_pruning': 1000, 'training_steps1_for_pruning': 500, 'training_steps2_for_pruning': 500, 'split_mlps': True, 'train_new_attn_heads': False, 'find_attention_primitives': False, 'find_logits_primitives': False, 'find_primitives': False,
               'lr_LN_var_for_pruning': 0.1, 'lr_MLP_for_pruning': 0.001, 'lr_oa_for_pruning': 0.002, 'lr_sampler_for_pruning': 0.1, 'max_test_length': 150,}

def main():
    # Select device for model inference
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    
    # Load transformer model to be decompiled
    model_path = "models/addition-@4l2h64d3lr00drop"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    print("Model Loaded Successfully")
    
    # Attempt to generate some text from the pretrained model
    tokenizer = CustomTokenizer(["0", "1", "+"])
    dataset = AdditionDataset(tokenizer, BASE_CONFIG['length_range'], BASE_CONFIG['max_test_length'])
    collator = CustomCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        print(batch)
        
        break

if __name__ == '__main__':
    main()