
from model_definitions import MoELoraModel, RoutingNetworkFromTransformer, get_embeddings, DTIDataset, MoERegressor, BigModel

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


# Load the model
big_model_dir = "checkpoints/concat/bigModel/3/mp_rank_00_model_states.pt"

# Load the model
data = DTI(name = 'BindingDB_patent',path='downloads/datasets/')
data.convert_to_log(form='binding')
split = data.get_split()

train_dataset = DTIDataset(split, 'train')
val_dataset = DTIDataset(split, 'valid')
val_num_batches = (len(val_dataset)+1)//2

# train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, prefetch_factor=2, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, prefetch_factor=2,  shuffle=True)

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

for i, input in enumerate(val_data_loader):
    drug_smiles = input["drug"]
    target_seq = input["target"]
    drug_input = drug_tokenizer(drug_smiles, return_tensors="pt", padding=True, truncation=True,max_length=1900).to(device)   
    target_input = target_tokenizer(target_seq, return_tensors="pt", padding=True, truncation=True,max_length=1900).to(device)

    with torch.no_grad():
        drug_strings.append(drug_smiles)
        drug_embed = drug_moe_model.original_embedding(**drug_input)
        target_embed = target_moe_model.original_embedding(**target_input)
        drug_embeddings.append(drug_embed)
        target_embeddings.append(target_embed)

        # convert logits to probabilities
        drug_routing_logits.append(torch.softmax(drug_moe_model.routing_network(drug_embed),dim=1))
        target_routing_logits.append(torch.softmax(target_moe_model.routing_network(target_embed),dim=1))

        # print the drug smiles if the max routing probability is not the 7th expert
        if drug_routing_logits[-1].argmax() != 6:
            print(f"Drug: {drug_smiles}")
            print(f"Drug routing probabilities: {drug_routing_logits[-1]}")
            print("\n",flush=True)
            