
from model_definitions import MoELoraModel, RoutingNetworkFromTransformer, get_embeddings, DavisDataset, MoERegressor

from transformers import AutoConfig,AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from torch.cuda.amp import GradScaler, autocast
from tdc.multi_pred import DTI
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.nn as nn
from peft import LoraModel, LoraConfig


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
  print(f"Starting training",flush=True)
  for epoch in range(num_epochs):
    i = 0
    total_loss = 0
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
        total_loss += loss.item()

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      if i % report_interval == 0:
        print(f"Loss: {loss.item()}",flush=True)

      del drug_embeddings, target_embeddings, all_embeds, predicted_affinity, loss
      torch.cuda.empty_cache()
    
      i = i + 1
    
    print(f"Epoch {epoch} completed",flush=True)
    total_loss = total_loss / len(train_data_loader)
    print(f"Average loss: {total_loss}",flush=True)

data = DTI(name = 'DAVIS',path='downloads/datasets/')
split = data.get_split()

train_dataset = DavisDataset(split, 'train')
val_dataset = DavisDataset(split, 'valid')

train_data_loader = DataLoader(train_dataset, batch_size=12, num_workers=8, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=12, num_workers=8, shuffle=True)

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

num_lora_experts = 1

lora_config = LoraConfig(
    task_type="SEQ_CLS",
    target_modules=["query","value"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.01,
)

drug_lora_model = LoraModel(pretrained_drug_model, lora_config, "default")
target_lora_model = LoraModel(pretrained_target_model, lora_config, "default")
regressor = MoERegressor(num_experts=4,hidden_size=2048,output_size=1)

optimizer = torch.optim.AdamW(list(drug_lora_model.parameters()) + list(target_lora_model.parameters()) + list(regressor.parameters()), lr=1e-4)

# drug_model.load_state_dict(torch.load('drug_model_0.pt'))
# target_model.load_state_dict(torch.load('target_model_0.pt'))
# regressor.load_state_dict(torch.load('regressor_0.pt'))

# Run one ineference
drug_lora_model.to(device)
target_lora_model.to(device)
regressor.to(device)
loss = nn.MSELoss()

train(
    drug_lora_model,
    target_lora_model,
    regressor,
    drug_tokenizer,
    target_tokenizer,
    optimizer,
    train_data_loader,
    val_data_loader,
    get_embeddings,
    loss,
    device,
    1000,
    report_interval=1000
)