
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

def inverse_log(x):
    return (10**(9-x) - 0.1)


def gather_numpy_arrays(array, root=0):
    # Convert numpy array to tensor
    tensor = torch.from_numpy(array).float().cuda()
    # Gather tensors on root process
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.gather(tensor, gather_list=gathered_tensors if dist.get_rank() == root else None, dst=root)
    if dist.get_rank() == root:
        # Convert tensors back to numpy
        gathered_arrays = [t.cpu().numpy() for t in gathered_tensors]
        return np.concatenate(gathered_arrays)
    else:
        return None

def validate(model, val_data_loader,total_val_batches, loss_fn, device, root=0):
    rank = torch.distributed.get_rank()
    model.eval()
    
    val_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for i, input in enumerate(val_data_loader):
            # only get 2 out of 16 batches in input, corresponding to your rank
            if i % dist.get_world_size() != rank:
                continue

            drug_smiles = input['drug']
            target_seq = input['target']
            target_affinity = torch.tensor(input['affinity'], dtype=torch.float).to(device)

            drug_input = drug_tokenizer(drug_smiles, return_tensors="pt", padding=True, truncation=True,max_length=1900).to(model.device)   
            target_input = target_tokenizer(target_seq, return_tensors="pt", padding=True, truncation=True,max_length=1900).to(model.device)

            predicted_affinity = model(drug_input, target_input)

            targets.append(inverse_log(target_affinity.double().detach().cpu().numpy()))
            predictions.append(inverse_log(predicted_affinity.double().detach().cpu().numpy()))
            # print(f"Predicted affinity: {predicted_affinity}", flush=True)
            # print(f"Target affinity: {target_affinity}", flush=True)
            loss = loss_fn(predicted_affinity, target_affinity)
            val_loss += loss.detach().item()  # Adjust loss reporting

        gathered_loss = torch.tensor(val_loss).cuda()
        dist.reduce(gathered_loss,root)
        gathered_loss = gathered_loss.item()
        gathered_loss = gathered_loss / total_val_batches

        # Calculate Pearson correlation coefficient
        predictions = np.concatenate(predictions).squeeze()
        targets = np.concatenate(targets)

        # COllect all predictions and targets
        gathered_targets = gather_numpy_arrays(targets, root=root)
        gathered_predictions = gather_numpy_arrays(predictions, root=root)

        if rank == root:
            print(f"Validation loss: {gathered_loss}", flush=True)
            pcc, _ = pearsonr(gathered_predictions, gathered_targets)
            print(f"Validation Pearson correlation coefficient: {pcc}", flush=True)
            val_loss = gathered_loss
        else:
            pcc = None

    model.train()

    return val_loss, pcc

def train(
    model,
    drug_tokenizer,
    target_tokenizer,
    train_data_loader,
    val_data_loader,
    total_train_batches,
    total_val_batches,
    loss_fn,
    device,
    num_epochs,
    report_interval=100,
):
    root = 0
    rank = dist.get_rank()
    print(f"Starting training for rank {rank}", flush=True)

    losses_arr = []
    val_losses = []
    val_pcc = []
    pcc_arr = []

    for epoch in range(num_epochs):
        total_loss = 0
        predictions = []
        targets = []

        for i, input in enumerate(train_data_loader):
            #with profiler.profile(profile_memory=True, use_cuda=True) as prof:
            drug_smiles = input['drug']
            target_seq = input['target']
            target_affinity = torch.tensor(input['affinity'], dtype=torch.float).to(device)

            drug_input = drug_tokenizer(drug_smiles, return_tensors="pt", padding=True, truncation=True,max_length=1900).to(model.device)   
            target_input = target_tokenizer(target_seq, return_tensors="pt", padding=True, truncation=True,max_length=1900).to(model.device)

            predicted_affinity = model(drug_input, target_input)

            predictions.append(inverse_log(predicted_affinity.double().detach().cpu().numpy()))
            targets.append(inverse_log(target_affinity.double().detach().cpu().numpy()))
            # print(f"Predicted affinity: {predicted_affinity}", flush=True)
            # print(f"Target affinity: {target_affinity}", flush=True)
            with torch.cuda.amp.autocast(cache_enabled=False):
                loss = loss_fn(predicted_affinity, target_affinity)
                total_loss += loss.detach().item()  # Adjust loss reporting
                losses_arr.append(loss.detach().cpu().numpy())
                
                model.backward(loss)  # Scale loss for gradient accumulation
            model.step()  # Update weights

            # only do this if you are rank 0
            if i % report_interval == 0:
                np_predictions = np.concatenate(predictions).squeeze()
                np_targets = np.concatenate(targets)
                
                gathered_targets = gather_numpy_arrays(np_targets, root=root)
                gathered_predictions = gather_numpy_arrays(np_predictions, root=root)

                if rank == root:
                    # reduce loss and calculate average
                    print(f"Sampled loss at step {i}: {loss.item()}", flush=True)

                    if i > 0 and i % (report_interval * 10) == 0:
                        pcc, _ = pearsonr(gathered_predictions, gathered_targets)
                        print(f"Pearson correlation coefficient: {pcc}", flush=True)
                        pcc_arr.append(pcc)

                predictions = []
                targets = []


        total_loss = total_loss
        gathered_loss = torch.tensor(total_loss).cuda()
        dist.reduce(gathered_loss,root)
        gathered_loss = gathered_loss.item()
        gathered_loss = gathered_loss / total_train_batches
        # plot loss vs iterations with gaussian smoothing in red
        if rank == 0:
            print(f"Epoch {epoch} completed", flush=True)
            print(f"Average loss: {gathered_loss}", flush=True)
            
            np.save(f"checkpoints/losses.npy", losses_arr)
            np.save(f"checkpoints/pcc.npy", pcc_arr)

        # save model checkpoint
        model.save_checkpoint(f"checkpoints/bigModel",tag=epoch)
        # calculate pearson correlation on validation set

        if rank == 0:
            print(f"Starting validation for epoch {epoch}", flush=True)

        val_loss, pcc = validate(model, val_data_loader, total_val_batches, loss_fn, device)

        if rank == 0:
            val_losses.append(val_loss)
            val_pcc.append(pcc)
            print(f"Finished validation for epoch {epoch}", flush=True)
            np.save(f"checkpoints/val_losses.npy", val_losses)
            np.save(f"checkpoints/val_pcc.npy", val_pcc)

if __name__ == "__main__":
    # run the cmd "alias mpirun=my_mpirun_wrapper"
    os.environ['PATH'] = "/home/gridsan/cgeorgiou/bio:" + os.environ['PATH']  
    
    data = DTI(name = 'BindingDB_patent',path='downloads/datasets/')
    data.convert_to_log(form='binding')
    split = data.get_split()

    train_dataset = DTIDataset(split, 'train')
    val_dataset = DTIDataset(split, 'valid')

    train_num_batches = (len(train_dataset)+1)//2
    val_num_batches = (len(val_dataset)+1)//2

    # train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, prefetch_factor=2, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=2, num_workers=4, prefetch_factor=2,  shuffle=True)

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

    model.drug_model.expert_model.set_adapter([f"expert_{i}" for i in range(num_experts)])  # we need to do this to pass all params to the optimizer
    model.target_model.expert_model.set_adapter([f"expert_{i}" for i in range(num_experts)])  # we need to do this to pass all params to the optimizer

    params_to_train = list(model.parameters())

    deepspeed.init_distributed(dist_backend="nccl",auto_mpi_discovery=True,verbose=False)
    
    ds_model, optimizer, train_data_loader, _ = deepspeed.initialize(config="deepspeed_config.json",
                                                        model=model,
                                                        # optimizer=optimizer,
                                                        model_parameters=params_to_train,
                                                        training_data=train_dataset,)
    # optimizer = torch.optim.AdamW(params_to_train, lr=3e-4)

    # print available devices
    print("Available devices:")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(device)

    # Run one ineference
    loss = nn.MSELoss()

    allocated_memory = torch.cuda.memory_allocated(device)
    print(f"Allocated Memory: {allocated_memory} bytes")

    train(
        ds_model,
        drug_tokenizer,
        target_tokenizer,
        train_data_loader,
        val_data_loader,
        train_num_batches,
        val_num_batches,
        loss,
        device,
        100,
        report_interval=500
    )