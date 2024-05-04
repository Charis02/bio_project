
from model_definitions import MoELoraModel, RoutingNetworkFromTransformer, get_embeddings, DTIDataset, MoERegressor

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
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
import torch.distributed as dist

import deepspeed

def inverse_log(x):
    return (10**(9-x) - 0.1)

def train(
    model,
    drug_tokenizer,
    target_tokenizer,
    train_data_loader,
    val_data_loader,
    loss_fn,
    device,
    num_epochs,
    report_interval=100,
):
    rank = dist.get_rank()
    print(f"Starting training for rank {rank}", flush=True)

    for epoch in range(num_epochs):
        total_loss = 0
        predictions = []
        targets = []

        if epoch == 0:
            model.save_checkpoint(f"checkpoints/bigModel",tag=epoch)

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

                # losses_arr.append(loss.detach().cpu().numpy())
                
                model.backward(loss)  # Scale loss for gradient accumulation
            model.step()  # Update weights

            # only do this if you are rank 0
            if i % report_interval == 0 and rank==0:
                print(f"Loss at step {i}: {loss.item()}", flush=True)
                np_predictions = np.concatenate(predictions).squeeze()
                np_targets = np.concatenate(targets)
                if i > 0 and i % (report_interval * 10) == 0:
                    pcc, _ = pearsonr(np_predictions, np_targets)
                    print(f"Pearson correlation coefficient: {pcc}", flush=True)
                predictions = []
                targets = []
                
            del predicted_affinity, loss, target_affinity
            torch.cuda.empty_cache() 

        print(f"Epoch {epoch} completed", flush=True)
        total_loss = total_loss / len(train_data_loader)
        print(f"Average loss: {total_loss}", flush=True)

        # save model checkpoint
        model.save_checkpoint(f"checkpoints/bigModel",tag=epoch)

        # plot loss vs iterations with gaussian smoothing in red
        # plt.clf()
        # plt.plot(range(len(losses_arr)),gaussian_filter1d(losses_arr, sigma=2), label="Train Loss vs Iterations", color='red')
        # plt.xlabel("Iterations")
        # plt.ylabel("Training Loss")

        # # drop top and right axis
        # ax = plt.gca()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # plt.savefig(f"plots/loss_plot_{epoch}.png")

        # save loss array
        # np.save(f"checkpoints/losses.npy", losses_arr)
        # calculate pearson correlation on validation set
        model.eval()
        
        val_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for i, input in enumerate(val_data_loader):
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
                    val_loss += loss.detach().item()  # Adjust loss reporting

            val_loss = val_loss / len(val_data_loader)
            print(f"Validation loss: {val_loss}", flush=True)

            # Calculate Pearson correlation coefficient
            predictions = np.concatenate(predictions).squeeze()
            targets = np.concatenate(targets)
            pcc, _ = pearsonr(predictions, targets)
            print(f"Validation Pearson correlation coefficient: {pcc}", flush=True)

        model.train()

if __name__ == "__main__":
    data = DTI(name = 'BindingDB_patent',path='downloads/datasets/')
    data.convert_to_log(form='binding')
    split = data.get_split()

    train_dataset = DTIDataset(split, 'train')
    val_dataset = DTIDataset(split, 'valid')

    # train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, prefetch_factor=2, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=4, num_workers=4, prefetch_factor=2,  shuffle=True)

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

    regressor = nn.Sequential(nn.Linear(1408,512),nn.ReLU(),nn.Dropout(0.12), nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1))

    model = BigModel(drug_moe_model, target_moe_model, regressor)

    model.drug_model.expert_model.set_adapter([f"expert_{i}" for i in range(num_experts)])  # we need to do this to pass all params to the optimizer
    model.target_model.expert_model.set_adapter([f"expert_{i}" for i in range(num_experts)])  # we need to do this to pass all params to the optimizer

    params_to_train = list(model.parameters())

    deepspeed.init_distributed(dist_backend="nccl",auto_mpi_discovery=False,verbose=True)
    
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
        loss,
        device,
        20,
        report_interval=500
    )