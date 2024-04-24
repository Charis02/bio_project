
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
from torch.autograd import profiler

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from tdc.utils import convert_back_log

def debug( 
    drug_model,
    target_model,
    regressor,
    drug_tokenizer,
    target_tokenizer,
    train_data_loader,
    val_data_loader,
    get_embeddings_fn,
    loss_fn,
    device,
):
    # print a sample from the training dataloader
    train_sample = next(iter(train_data_loader))
    drug_smiles = train_sample['drug']
    target_seq = train_sample['target']
    target_affinity = torch.tensor(train_sample['affinity'], dtype=torch.float).to(device)

    drug_embeddings = get_embeddings_fn(drug_tokenizer, drug_model, drug_smiles, device)
    target_embeddings = get_embeddings_fn(target_tokenizer, target_model, target_seq, device)

    all_embeds = torch.cat((drug_embeddings, target_embeddings), dim=1)
    predicted_affinity = regressor(all_embeds)
    # print(f"Predicted affinity: {predicted_affinity}", flush=True)
    # print(f"Target affinity: {target_affinity}", flush=True)
    loss = loss_fn(predicted_affinity, target_affinity)

    print(f"Input drug SMILES: {drug_smiles}", flush=True)
    print(f"Input target sequence: {target_seq}", flush=True)

    print(f"Sample training loss: {loss.item()}", flush=True)

    # print a sample from the validation dataloader
    val_sample = next(iter(val_data_loader))
    drug_smiles = val_sample['drug']
    target_seq = val_sample['target']
    target_affinity = torch.tensor(val_sample['affinity'], dtype=torch.float).to(device)

    drug_embeddings = get_embeddings_fn(drug_tokenizer, drug_model, drug_smiles, device)
    target_embeddings = get_embeddings_fn(target_tokenizer, target_model, target_seq, device)

    all_embeds = torch.cat((drug_embeddings, target_embeddings), dim=1)
    predicted_affinity = regressor(all_embeds)
    # print(f"Predicted affinity: {predicted_affinity}", flush=True)
    # print(f"Target affinity: {target_affinity}", flush=True)
    loss = loss_fn(predicted_affinity, target_affinity)

    print(f"Input drug SMILES: {drug_smiles}", flush=True)
    print(f"Input target sequence: {target_seq}", flush=True)

    print(f"Sample validation loss: {loss.item()}", flush=True)



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
            #with profiler.profile(profile_memory=True, use_cuda=True) as prof:
  
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

            if i % report_interval == 0:
                print(f"Loss at step {i}: {loss.item()}", flush=True)

            del drug_embeddings, target_embeddings, all_embeds, predicted_affinity, loss, target_affinity
            torch.cuda.empty_cache() 

            if i > 1000:
                break
        
        debug(
            drug_model,
            target_model,
            regressor,
            drug_tokenizer,
            target_tokenizer,
            train_data_loader,
            val_data_loader,
            get_embeddings_fn,
            loss_fn,
            device,
        )



        print(f"Epoch {epoch} completed", flush=True)
        total_loss = total_loss / len(train_data_loader)
        print(f"Average loss: {total_loss}", flush=True)

        # save model
        torch.save(drug_model.state_dict(), f"checkpoints/drug_model_{epoch}.pt")
        torch.save(target_model.state_dict(), f"checkpoints/target_model_{epoch}.pt")
        torch.save(regressor.state_dict(), f"checkpoints/regressor_{epoch}.pt")

        # plot loss vs iterations with gaussian smoothing in red
        plt.clf()
        plt.plot(range(len(losses_arr)),gaussian_filter1d(losses_arr, sigma=2), label="Train Loss vs Iterations", color='red')
        plt.xlabel("Iterations")
        plt.ylabel("Training Loss")

        # drop top and right axis
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(f"plots/loss_plot_{epoch}.png")

        # save loss array
        np.save(f"checkpoints/losses.npy", losses_arr)
        optim_scheduler.step()

        # calculate pearson correlation on validation set
        drug_model.eval()
        target_model.eval()
        regressor.eval()
        
        val_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for i, input in enumerate(val_data_loader):
                drug_smiles = input['drug']
                target_seq = input['target']
                target_affinity = torch.tensor(input['affinity'], dtype=torch.float).to(device)

                drug_embeddings = get_embeddings_fn(drug_tokenizer, drug_model, drug_smiles, device)
                target_embeddings = get_embeddings_fn(target_tokenizer, target_model, target_seq, device)

                all_embeds = torch.cat((drug_embeddings, target_embeddings), dim=1)
                predicted_affinity = regressor(all_embeds)
                loss = loss_fn(predicted_affinity, target_affinity)
                val_loss += loss.detach().item()

                # Collect all predictions and actual targets for Pearson correlation calculation
                # Values are in log10 space, so we need to convert them back to original space
                predictions.append(convert_back_log(predicted_affinity.cpu().numpy()))
                targets.append(convert_back_log(target_affinity.cpu().numpy()))
                

            val_loss = val_loss / len(val_data_loader)
            print(f"Validation loss: {val_loss}", flush=True)

            # Calculate Pearson correlation coefficient
            predictions = np.concatenate(predictions).squeeze()
            targets = np.concatenate(targets)
            pcc, _ = pearsonr(predictions, targets)
            print(f"Pearson correlation coefficient: {pcc}", flush=True)

        drug_model.train()
        target_model.train()
        regressor.train()



data = DTI(name = 'DAVIS',path='downloads/datasets/')
data.convert_to_log(form='binding')
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

esm650 = "/home/gridsan/cgeorgiou/bio/downloads/pretrained_models/esm2_t33_650M_UR50D/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
esm150 = "/home/gridsan/cgeorgiou/bio/downloads/pretrained_models/esm2_t30_150M_UR50D/models--facebook--esm2_t30_150M_UR50D/snapshots/a695f6045e2e32885fa60af20c13cb35398ce30c"

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

regressor = nn.Sequential(nn.Linear(1408,512),nn.ReLU(),nn.Dropout(0.3), nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1))

drug_moe_model.expert_model.set_adapter([f"expert_{i}" for i in range(num_experts)])  # we need to do this to pass all params to the optimizer
params_to_train = list(drug_moe_model.parameters()) + list(target_moe_model.parameters()) + list(regressor.parameters())
grad_accum_steps = 16

optimizer = torch.optim.AdamW(params_to_train, lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=1e-6)

# print available devices
print("Available devices:")
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(device)

# Run one ineference
drug_moe_model.to(device)
target_moe_model.to(device)
regressor.to(device)
loss = nn.MSELoss()

allocated_memory = torch.cuda.memory_allocated(device)
print(f"Allocated Memory: {allocated_memory} bytes")

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
    20,
    accumulation_steps=grad_accum_steps,
    report_interval=100
)