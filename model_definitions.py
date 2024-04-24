import torch.nn as nn
import torch
from peft import LoraModel, LoraConfig
from torch.utils.data import Dataset, DataLoader
# from moelora.src import molora,routing
# import jax.random as random

class MoELoraModel(nn.Module):
  def __init__(self, model, config, num_experts, embedding_dim=768):
    super().__init__()
    self.num_experts = num_experts
    self.expert_model = LoraModel(model, config, f"expert_0")
      
    for i in range(1,num_experts):
      self.expert_model.add_adapter(config,f"expert_{i}")

    self.routing_network = nn.Sequential(
        nn.Linear(embedding_dim, num_experts),
    )
    self.num_experts_topk = 1

    self.batch_norm_layer = nn.BatchNorm1d(embedding_dim)

  def forward(self, router_inputs, **inputs):
    logits = self.routing_network(router_inputs)  # logits should be shape (num_experts, )
    weights, chosen_experts = torch.topk(logits, self.num_experts_topk)
    weights = torch.softmax(weights, dim=1)

    # Create a prediction with the shape [batch, hidden_size]
    prediction = torch.zeros(inputs["input_ids"].shape[0],self.expert_model.config.hidden_size, device=next(self.parameters()).device)

    for current_expert in range(self.num_experts):
      batch_idx, nth_expert = torch.where(chosen_experts == current_expert)
      routed_inputs = {k: v[batch_idx] for k, v in inputs.items()}

      if len(batch_idx) > 0:
        # enable only the expert adapter
        # self.expert_model.disable_adapter_layers()
        self.expert_model.set_adapter(f"expert_{current_expert}")
        # get the embeddings
        expert_embed = self.expert_model(**routed_inputs)


        # Extract the hidden states
        if hasattr(expert_embed, "last_hidden_state"): # Depends on the pretrained model backbone
          hidden_states = expert_embed.last_hidden_state
        else:
          hidden_states = expert_embed.hidden_states[-1]

        # Aggregate hidden states to get a single vector representation (e.g., mean pooling)
        hidden_states = torch.mean(hidden_states, dim=1)

        w = weights[batch_idx, nth_expert, None].unsqueeze(1)
        w = w.view(-1, 1)  # Reshape w to have shape [sub_batch_size, 1]

        prediction[batch_idx] += w * hidden_states

    if prediction.shape[0] > 1:
      prediction = self.batch_norm_layer(prediction)
    
    return prediction

  def original_embedding(self, **inputs):
    # self.expert_model.disable_adapter_layers()
    self.expert_model.set_adapter([])
    original_outputs = self.expert_model(**inputs)
    
    # Extract the hidden states
    if hasattr(original_outputs, "last_hidden_state"): # Depends on the pretrained model backbone
      original_embedding = original_outputs.last_hidden_state
    else:
      original_embedding = original_outputs.hidden_states[-1]

    # Aggregate hidden states to get a single vector representation (e.g., mean pooling)
    original_embedding = torch.mean(original_embedding, dim=1)

    return original_embedding

  def to(self, *args, **kwargs):
    self.routing_network.to(*args, **kwargs)
    self.expert_model.to(*args, **kwargs)
    return super().to(*args, **kwargs)

class RoutingNetworkFromTransformer(nn.Module):
  def __init__(self, model, num_experts, embedding_dim=384):
    super().__init__()
    self.num_experts = num_experts
    self.last_layer = nn.Sequential(nn.Linear(embedding_dim, num_experts), nn.Softmax())
    self.model = model

  def forward(self, **inputs):
    outputs = self.model(**inputs)

    # Extract the hidden states
    if hasattr(outputs, "last_hidden_state"): # Depends on the pretrained model backbone
      hidden_states = outputs.last_hidden_state
    else:
      hidden_states = outputs.hidden_states[-1]

    # Aggregate hidden states to get a single vector representation (e.g., mean pooling)
    embeddings = torch.mean(hidden_states, dim=1)
    return self.last_layer(embeddings)
  

def get_embeddings(tokenizer, model, input,device):
    # Encode the SMILES sequence
    # print(len(input[0]))
    # print(len(input[1]))
    encoded_input = tokenizer(input, return_tensors="pt", padding=True, truncation=True,max_length=1900).to(device)   
    with torch.no_grad():
      original_embedding = model.original_embedding(**encoded_input)
    embeddings = model(original_embedding,**encoded_input)

    return embeddings

class DavisDataset(Dataset):
    def __init__(self, data, split='train'):
        # Assuming data is a dictionary with 'train', 'valid', 'test' splits
        # Concatenate training, validation, and test sets if needed
        # Or you can adjust the code to use only one of the splits
        self.data = data[split]
        self.drug = self.data['Drug']
        self.target = self.data['Target']
        self.affinity = self.data['Y']

        # self.normalize_data()
        # self.logarithm_data()

    def __len__(self):
        return len(self.affinity)

    def __getitem__(self, idx):
        return {
            "drug": self.drug[idx],
            "target": self.target[idx],
            "affinity": torch.tensor(self.affinity[idx], dtype=torch.float)
        }

    def normalize_data(self):
      pass
      # self.affinity_mean = self.affinity.mean()
      # self.affinity_std = self.affinity.std()
      # self.affinity = (self.affinity - self.affinity_mean) / self.affinity_std
    
    def logarithm_data(self):
      self.affinity = torch.log(self.affinity)
    
class MoERegressor(nn.Module):
  """
  Given an embedding, the MoERegressor predicts the affinity score.
  Each expert is a feedforward neural network.
  The router is a feedforward neural network that predicts the expert weights.
  To find the routing weights, we use topk.
  """
  def __init__(self, num_experts, hidden_size, output_size):
    super().__init__()
    self.num_experts = num_experts
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.k = 2

    self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)) for _ in range(num_experts)])
    self.router = nn.Linear(hidden_size, num_experts)

  def forward(self, embeddings):
     """
     Find the top k experts for each batch and their weights
     Send the sub-batch to each expert
     """
     routing_logits = self.router(embeddings)
     expert_weights, expert_indices = torch.topk(routing_logits, self.k, dim=-1)

     expert_weights = torch.softmax(expert_weights, dim=-1)
  
     # Create a prediction with the shape [batch, output dim]
     prediction = torch.zeros(embeddings.shape[0], self.output_size, device=next(self.parameters()).device)
  
     for expert_idx in range(self.num_experts):
      batch_indices = torch.where(expert_indices == expert_idx)
      if len(batch_indices) > 0:
        expert_input = embeddings[batch_indices[0]]
        expert_output = self.experts[expert_idx](expert_input)
        prediction_contribution = expert_weights[batch_indices].unsqueeze(-1)*expert_output
        prediction[batch_indices[0]] += prediction_contribution

  
     return prediction