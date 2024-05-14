
from model_definitions import MoELoraModel, DTIDataset, BigModel

from transformers import AutoConfig,AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from torch.cuda.amp import GradScaler, autocast
from tdc.multi_pred import DTI
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.nn as nn
from peft import LoraModel, LoraConfig
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import profiler
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from tdc.utils import convert_back_log
import torch.distributed as dist

import deepspeed
import mpi4py
import sys
import subprocess
def trace_backprop(tensor):
    """
    Prints the backpropagation graph of a tensor.
    """
    def _trace(tensor, depth=0):
        if tensor is None:
            print("  " * depth + "None")
        elif hasattr(tensor, 'grad_fn'):
            if tensor.grad_fn is not None:
                print("  " * depth + str(tensor.grad_fn))
                for next_tensor in tensor.grad_fn.next_functions:
                    if next_tensor[0] is not None:
                        _trace(next_tensor[0], depth + 1)
                    else:
                        print("  " * depth + 1 * " " + "None")
            else:
                print("  " * depth + "No grad_fn, could be a leaf tensor or a detached tensor")
        else:
            print("  " * depth + "Object does not have a grad_fn attribute")

    _trace(tensor)


# Load the model
big_model_dir = "checkpoints/concat/bigModel/2/mp_rank_00_model_states.pt"

# Load the model
data = DTI(name = 'BindingDB_patent',path='downloads/datasets/')
data.convert_to_log(form='binding')
split = data.get_split()

train_dataset = DTIDataset(split, 'train')
val_dataset = DTIDataset(split, 'valid')
val_num_batches = (len(val_dataset)+1)//2

batch_size=2
# train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, prefetch_factor=2, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, prefetch_factor=2,  shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
drug_model_dir = "downloads/pretrained_models/ChemBERTaLM/models--gokceuludogan--ChemBERTaLM/snapshots/33199b39d6f4844644d436da9ae9399dcb7b505f"
drug_tokenizer = RobertaTokenizer.from_pretrained(drug_model_dir)
pretrained_drug_model = RobertaModel.from_pretrained(drug_model_dir, output_hidden_states=True)

print(f"Dowloaded drug model",flush=True)

esm650 = "downloads/pretrained_models/esm2_t33_650M_UR50D/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
esm150 = "downloads/pretrained_models/esm2_t30_150M_UR50D/models--facebook--esm2_t30_150M_UR50D/snapshots/a695f6045e2e32885fa60af20c13cb35398ce30c"

target_model_dir = esm150
config = AutoConfig.from_pretrained(target_model_dir, output_hidden_states=True)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_dir)

pretrained_target_model = AutoModelForSequenceClassification.from_pretrained(target_model_dir,config=config)
pretrained_target_model.gradient_checkpointing_enable()

print(f"Dowloaded target model",flush=True)

target_modules = ["query","value"]
peft_config = LoraConfig(
task_type="SEQ_CLS",
target_modules=target_modules,
inference_mode=False,
r=8,
lora_alpha=16,
lora_dropout=0.05,
)

num_experts=8

drug_moe_model = MoELoraModel(pretrained_drug_model, peft_config, num_experts, embedding_dim=768)
target_moe_model = MoELoraModel(pretrained_target_model, peft_config, num_experts, embedding_dim=640)

regressor = nn.Sequential(nn.Linear(1408,512),nn.ReLU(),nn.Dropout(0.12), nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1))

# make regressor half
regressor = regressor.half()

model = BigModel(drug_moe_model, target_moe_model, regressor)
# train only regressor
params_to_train = list(model.regressor.parameters())

model.load_state_dict(torch.load(big_model_dir)['module'])

model.to(device)

print(f"Loaded model",flush=True)

# go over the validation dataset, and get the output of the routing network
# for each drug and target
drug_strings = []
drug_embeddings = []
target_embeddings = []
drug_routing_logits = []
target_routing_logits = []

model.eval()

summed_drug_loads = torch.zeros((8,))
summed_target_loads = torch.zeros((8,))


loss_fn = nn.MSELoss()

for i, input in enumerate(val_data_loader):
    drug_smiles = input["drug"]
    target_seq = input["target"]
    drug_input = drug_tokenizer(drug_smiles, return_tensors="pt", padding=True, truncation=True,max_length=1900).to(device)
    target_input = target_tokenizer(target_seq, return_tensors="pt", padding=True, truncation=True,max_length=1900).to(device)

    with torch.no_grad():
        # out, drug_routing_probabilities, target_routing_probabilities, drug_expert_load, target_expert_load  = model(drug_input,target_input)

        drug_strings.append(drug_smiles)
        drug_embed = drug_moe_model.original_embedding(**drug_input)
        target_embed = target_moe_model.original_embedding(**target_input)
        drug_embeddings.append(drug_embed)
        target_embeddings.append(target_embed)

        # convert logits to probabilities
        # drug_routing_logits.append(torch.softmax(drug_moe_model.routing_network(drug_embed),dim=1))
        # target_routing_logits.append(torch.softmax(target_moe_model.routing_network(target_embed),dim=1))

        # drug_top2, drug_chosen_experts = torch.topk(drug_moe_model.routing_network(drug_embed),2)
        # drug_top2 = torch.softmax(drug_top2,dim=1)
        # target_top2, target_chosen_experts = torch.topk(target_moe_model.routing_network(target_embed),2)
        # target_top2 = torch.softmax(target_top2,dim=1)

        # print expert logits
        # print(f"Drug experts weights {drug_top2} for experts {drug_chosen_experts}")
        # print(f"Target experts weights {target_top2} for experts {target_chosen_experts}")

        
        out, drug_routing_probabilities, target_routing_probabilities, drug_load, target_load  = model(drug_input,target_input)

        summed_drug_loads += drug_load.detach().cpu()
        summed_target_loads += target_load.detach().cpu()

    if i%100 == 0 and i > 0:
        print(f"Drug load {summed_drug_loads/(2*i)}")
        print(f"Target load: {summed_target_loads/(2*i)}")

        plt.clf()
        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # despine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # summed drug loads is an array of size 8. Plot it as a histogram, each with label Expert 0, Expert 1, etc.
        # Use seaborn to plot the histogram
        # plot the bar charts side to side
        x = np.arange(8)
        width = 0.35
        plt.bar(x, 100*summed_drug_loads/(2*i), width, label='Drug',color='#bae1ff')
        plt.bar(x + width, 100*summed_target_loads/(2*i), width, label='Target',color='#ffb3ba')

        # move the ticks to the center
        plt.xticks(x + width/2, [f"Expert {i}" for i in range(8)])
        # make the y ticks have %
        plt.yticks(np.arange(0, 101, 10))
        plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
        plt.ylabel('Percentage Load')
        

        plt.legend(loc='upper right')
        plt.savefig(f"expert_loads.pdf")
        

print(f"Drug load {summed_drug_loads/(2*i)}")
print(f"Target load: {summed_target_loads/(2*i)}")            