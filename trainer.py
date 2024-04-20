from transformers import RobertaModel, RobertaTokenizer
import torch
from peft import LoraModel, LoraConfig
import torch.nn as nn

from tqdm import tqdm
from tdc.multi_pred import DTI
from torch.utils.data import Dataset, DataLoader

from model_definitions import MoELoraModel, RoutingNetworkFromTransformer, get_embeddings
from transformers import AutoConfig,AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from torch.cuda.amp import GradScaler, autocast

import xlora


def train(
    drug_model,
    target_model,
    regressor,
    drug_tokenizer,
    target_tokenizer,
    optimizer,
    train_data_loader,
    val_data_loader,
    get_embeddings_fn,
    loss_fn,
    device,
    num_epochs,
    report_interval=100
):
  scaler = GradScaler()
  
  for epoch in range(num_epochs):
    i = 0
    for input in train_data_loader:
      drug_smiles = input['drug']
      target_seq = input['target']
      target_affinity = torch.tensor(input['affinity'],dtype=torch.float).to(device)

      optimizer.zero_grad()
      with autocast():
        drug_embeddings = get_embeddings_fn(drug_tokenizer,drug_model,drug_smiles,device)
        target_embeddings = get_embeddings_fn(target_tokenizer,target_model,target_seq,device)

        all_embeds = torch.cat((drug_embeddings, target_embeddings), dim=1)

        predicted_affinity = regressor(all_embeds)

        loss = loss_fn(predicted_affinity,target_affinity)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

      if i % report_interval == 0:
        print(f"Loss: {loss.item()}",flush=True)

      del drug_embeddings, target_embeddings, all_embeds, predicted_affinity, loss
      torch.cuda.empty_cache()
    
      i = i + 1
    
    print(f"Epoch {epoch} completed",flush=True)
    # Validate
    with torch.no_grad():
        val_loss = 0
        for input in val_data_loader:
            drug_smiles = input['drug']
            target_seq = input['target']
            target_affinity = torch.tensor(input['affinity'],dtype=torch.float).to(device)
    
            drug_embeddings = get_embeddings_fn(drug_tokenizer,drug_model,drug_smiles,device)
            target_embeddings = get_embeddings_fn(target_tokenizer,target_model,target_seq,device)
    
            all_embeds = torch.cat((drug_embeddings, target_embeddings), dim=1)
    
            predicted_affinity = regressor(all_embeds)
    
            loss = loss_fn(predicted_affinity,target_affinity)
            val_loss += loss.item()
    
            del drug_embeddings, target_embeddings, all_embeds, predicted_affinity, loss
            torch.cuda.empty_cache()
        
        if epoch%10 == 0:
           # Save the model
            torch.save(drug_model.state_dict(), f"drug_model_{epoch}.pt")
            torch.save(target_model.state_dict(), f"target_model_{epoch}.pt")
            torch.save(regressor.state_dict(), f"regressor_{epoch}.pt")
        
        avg_val_loss = val_loss / len(val_data_loader)
        print(f"Validation loss: {avg_val_loss}",flush=True)

def print_gpu_memory_distribution(model):
    total_memory = 0
    for name, param in model.named_parameters():
        # Calculate memory usage in bytes
        param_memory = param.numel() * param.element_size()
        total_memory += param_memory
        print(f"{name}: {param_memory / (1024 ** 2):.2f} MB")  # Convert bytes to megabytes
        
    print(f"Total memory usage by parameters: {total_memory / (1024 ** 2):.2f} MB")


class DavisDataset(Dataset):
    def __init__(self, data, split='train'):
        # Assuming data is a dictionary with 'train', 'valid', 'test' splits
        # Concatenate training, validation, and test sets if needed
        # Or you can adjust the code to use only one of the splits
        self.data = data[split]
        self.drug = self.data['Drug']
        self.target = self.data['Target']
        self.affinity = self.data['Y']

        # Here, additional preprocessing can be done (e.g., tokenization)

    def __len__(self):
        return len(self.affinity)

    def __getitem__(self, idx):
        return {
            "drug": self.drug[idx],
            "target": self.target[idx],
            "affinity": torch.tensor(self.affinity[idx], dtype=torch.float)
        }
    
print(f"Declared all functions and classes",flush=True)
    
drug_model_dir = "/home/gridsan/cgeorgiou/bio/downloads/pretrained_models/ChemBERTaLM/models--gokceuludogan--ChemBERTaLM/snapshots/33199b39d6f4844644d436da9ae9399dcb7b505f"
drug_tokenizer = RobertaTokenizer.from_pretrained(drug_model_dir)
pretrained_drug_model = RobertaModel.from_pretrained(drug_model_dir, output_hidden_states=True)

print(f"Dowloaded drug model",flush=True)

target_model_dir = "/home/gridsan/cgeorgiou/bio/downloads/pretrained_models/esm2_t33_650M_UR50D/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
config = AutoConfig.from_pretrained(target_model_dir, output_hidden_states=True)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_dir)

pretrained_target_model = AutoModelForSequenceClassification.from_pretrained(target_model_dir,config=config)

print(f"Dowloaded target model",flush=True)

num_lora_experts = 8

# lora_config = LoraConfig(
#     task_type="SEQ_CLS",
#     target_modules=["query","value"],
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.01,
# )

drug_model = xlora.add_xlora_to_model(
    model=pretrained_drug_model,
    xlora_config=xlora.xLoRAConfig(
        pretrained_drug_model.config.hidden_size,
        base_model_id="ChemBERTaLM/models--gokceuludogan--ChemBERTaLM",
        xlora_depth=8,
        device=torch.device("cuda"),
    ),
    verbose=True,
)

print(f"Added xLora to drug model",flush=True)
print(f"Drug model architecture is: {drug_model}",flush=True)
# drug_router = RoutingNetworkFromTransformer(pretrained_drug_model, num_lora_experts, embedding_dim=768)
# target_router = RoutingNetworkFromTransformer(pretrained_target_model, num_lora_experts, embedding_dim=1280)

# drug_model = MoELoraModel(pretrained_drug_model, drug_router, lora_config, num_lora_experts)
# target_model = MoELoraModel(pretrained_target_model, target_router, lora_config, num_lora_experts)
regressor = nn.Sequential(nn.Linear(2048,128),nn.ReLU(),nn.Linear(128,1))

combined_parameters = list(drug_model.parameters()) + list(target_model.parameters()) + list(regressor.parameters())
optimizer = torch.optim.AdamW(combined_parameters, lr=1e-4)

data = DTI(name = 'DAVIS',path='downloads/datasets/')
split = data.get_split()
mse_loss_fn = nn.MSELoss()
train_dataset = DavisDataset(split, 'train')
val_dataset = DavisDataset(split, 'valid')
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

drug_model.to(device)
target_model.to(device)
regressor.to(device)

train(
    drug_model,
    target_model,
    regressor,
    drug_tokenizer,
    target_tokenizer,
    optimizer,
    train_dataloader,
    val_dataloader,
    get_embeddings,
    mse_loss_fn,
    device,
    5
)