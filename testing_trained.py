
from model_definitions import MoELoraModel, RoutingNetworkFromTransformer, get_embeddings, DavisDataset, MoERegressor

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

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def print_computation_graph(tensor, indent=""):
    # Print tensor details
    print(indent, tensor)
    if hasattr(tensor, 'grad_fn') and tensor.grad_fn is not None:
        print(indent + "  Grad_fn:", tensor.grad_fn)
        for next_tensor, _ in tensor.grad_fn.next_functions:
            if next_tensor is not None:
                print_computation_graph(next_tensor, indent + "    ")


def train(
    drug_model,
    target_model,
    regressor,
    drug_tokenizer,
    target_tokenizer,
    optimizer,
    optim_scheduler,
    train_data_loader,
    val_data_loader,
    get_embeddings_fn,
    loss_fn,
    device,
    num_epochs,
    accumulation_steps=4,
    report_interval=100,
):
    scaler = GradScaler()
    print("Starting training", flush=True)
    losses_arr = []

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()  # Initialize gradient accumulation

        for i, input in enumerate(train_data_loader):
            drug_smiles = input['drug']
            target_seq = input['target']
            target_affinity = torch.tensor(input['affinity'], dtype=torch.float).to(device)

            with autocast():
                drug_embeddings = get_embeddings_fn(drug_tokenizer, drug_model, drug_smiles, device)
                target_embeddings = get_embeddings_fn(target_tokenizer, target_model, target_seq, device)

                all_embeds = torch.cat((drug_embeddings, target_embeddings), dim=1)
                predicted_affinity = regressor(all_embeds)
                # print(f"Predicted affinity: {predicted_affinity}", flush=True)
                # print(f"Target affinity: {target_affinity}", flush=True)
                loss = loss_fn(predicted_affinity, target_affinity)
                total_loss += loss.detach().item() / accumulation_steps  # Adjust loss reporting
                losses_arr.append(loss.detach().cpu().numpy())

            scaler.scale(loss / accumulation_steps).backward()  # Scale loss for gradient accumulation

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Reset gradients for the next accumulation
                optim_scheduler.step()

            if i % report_interval == 0:
                print(f"Loss at step {i}: {loss.item()}", flush=True)

            del drug_embeddings, target_embeddings, all_embeds, predicted_affinity, loss, target_affinity
            torch.cuda.empty_cache() 


        print(f"Epoch {epoch} completed", flush=True)
        total_loss = total_loss / len(train_data_loader)
        print(f"Average loss: {total_loss}", flush=True)

        # save model
        torch.save(drug_model.state_dict(), f"checkpoints/drug_model_{epoch}.pt")
        torch.save(target_model.state_dict(), f"checkpoints/target_model_{epoch}.pt")
        torch.save(regressor.state_dict(), f"checkpoints/regressor_{epoch}.pt")

        # plot loss with gaussian smoothing
        plt.plot(gaussian_filter1d(losses_arr, sigma=10))

        plt.savefig(f"plots/loss_plot_{epoch}.png")

        # save loss array
        np.save(f"checkpoints/losses.npy", losses_arr)

data = DTI(name = 'DAVIS',path='downloads/datasets/')
split = data.get_split()

train_dataset = DavisDataset(split, 'train')
val_dataset = DavisDataset(split, 'valid')

train_data_loader = DataLoader(train_dataset, batch_size=2, num_workers=4, prefetch_factor=2, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=2, num_workers=4, prefetch_factor=2,  shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
drug_model_dir = "/home/gridsan/cgeorgiou/bio/downloads/pretrained_models/ChemBERTaLM/models--gokceuludogan--ChemBERTaLM/snapshots/33199b39d6f4844644d436da9ae9399dcb7b505f"
drug_tokenizer = RobertaTokenizer.from_pretrained(drug_model_dir)
pretrained_drug_model = RobertaModel.from_pretrained(drug_model_dir, output_hidden_states=True)

print(f"Dowloaded drug model",flush=True)

target_model_dir = "/home/gridsan/cgeorgiou/bio/downloads/pretrained_models/esm2_t33_650M_UR50D/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
config = AutoConfig.from_pretrained(target_model_dir, output_hidden_states=True)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_dir)

pretrained_target_model = AutoModelForSequenceClassification.from_pretrained(target_model_dir,config=config)

print(f"Dowloaded target model",flush=True)

target_modules = ["value"]
peft_config = LoraConfig(
  task_type="SEQ_CLS",
  target_modules=target_modules,
  inference_mode=False,
  r=8,
  lora_alpha=16,
  lora_dropout=0.05,
)

num_experts=4

drug_moe_model = MoELoraModel(pretrained_drug_model, peft_config, num_experts, embedding_dim=768)
target_moe_model = MoELoraModel(pretrained_target_model, peft_config, num_experts, embedding_dim=1280)

regressor = nn.Sequential(nn.Linear(2048,512),nn.ReLU(),nn.Dropout(0.3), nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1))

drug_moe_model.expert_model.set_adapter([f"expert_{i}" for i in range(num_experts)])  # we need to do this to pass all params to the optimizer
params_to_train = list(drug_moe_model.parameters()) + list(target_moe_model.parameters()) + list(regressor.parameters())
grad_accum_steps = 4

optimizer = torch.optim.AdamW(params_to_train, lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3*len(train_data_loader)//grad_accum_steps, eta_min=1e-6)

# Run one ineference
drug_moe_model.to(device)
target_moe_model.to(device)
regressor.to(device)
loss = nn.MSELoss()

train(
    drug_moe_model,
    target_moe_model,
    regressor,
    drug_tokenizer,
    target_tokenizer,
    optimizer,
    scheduler,
    train_data_loader,
    val_data_loader,
    get_embeddings,
    loss,
    device,
    1000,
    accumulation_steps=grad_accum_steps,
    report_interval=10
)