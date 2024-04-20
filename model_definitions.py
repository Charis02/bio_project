import torch.nn as nn
import torch
from peft import LoraModel, LoraConfig
from torch.utils.data import Dataset, DataLoader

class MoELoraModel(nn.Module):
  def __init__(self, model, routing_network, config, num_experts):
    super().__init__()
    self.num_experts = num_experts
    # self.experts_model = LoraModel(model, config, "default")
    # self.experts_list = []

    self.expert_models = [LoraModel(model, config, f"expert_{i}") for i in range(num_experts)]

    self.routing_network = routing_network
    self.num_experts_topk = 2

  def forward(self, inputs_dict):
    return self.route(inputs_dict)
    return self.experts_model(**model_inputs)

  def route(self, inputs_dict):
    model_inputs = inputs_dict["model_inputs"]
    routing_network_inputs = inputs_dict["routing_network_inputs"]
    logits = self.routing_network(**routing_network_inputs)  # logits should be shape (num_experts, )
    print(f"Logits: {logits}",flush=True)
    weights, chosen_experts = torch.topk(logits, self.num_experts_topk)
    weights = torch.softmax(weights, dim=1)

    # Create a prediction with the shape [batch, hidden_state_dim]
    prediction = torch.zeros(model_inputs["input_ids"].shape[0], self.expert_models[0].config.hidden_size, device=next(self.parameters()).device)

    print(f"Chosen experts: {chosen_experts}",flush=True)
    print(f"Weights: {weights}",flush=True)

    for current_expert in range(self.num_experts):
      batch_idx, nth_expert = torch.where(chosen_experts == current_expert)
      print(f"Batch going into expert {current_expert}: {batch_idx} and model inputs are {model_inputs}",flush=True)
      routed_inputs = {k: v[batch_idx] for k, v in model_inputs.items()}
      print(f"Routed inputs: {routed_inputs}",flush=True)
      print(f"Bug in: {routed_inputs['input_ids'][:, [-1, 0]]}")

      print("What is nth expert",nth_expert)

      if len(batch_idx) > 0:
        expert_embed = self.expert_models[current_expert](**routed_inputs)
        print(f"Architecture of expert {current_expert}: {self.expert_models[current_expert]}",flush=True)

        # Extract the hidden states
        if hasattr(expert_embed, "last_hidden_state"): # Depends on the pretrained model backbone
          hidden_states = expert_embed.last_hidden_state
        else:
          hidden_states = expert_embed.hidden_states[-1]

        w = weights[batch_idx, nth_expert, None].unsqueeze(-1)
        print(f"Hidden states: {hidden_states.shape}",flush=True)
        print(f"Weight {w.shape}",flush=True)
        print(f"Prediction: {prediction[batch_idx].shape}",flush=True)

        prediction[batch_idx] += w * hidden_states

    return prediction

  # def topk_with_softmax(self, logits):
  #   values, indices = torch.topk(logits,2)
  #   ret = torch.zeros(self.num_experts,device=next(self.parameters()).device)
  #   values = values/torch.norm(values)  # todo: probably remove
  #   expert_weights = torch.softmax(values.float(), dim=1)
  #   ret[indices] = expert_weights
  #   return ret

  # def choose_expert(self,expert_num):
  #   self.experts_model.disable_adapter_layers()
  #   self.experts_model.set_adapter(f"expert_{expert_num}")

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
  

def get_embeddings(tokenizer,model, input,device):
    # Encode the SMILES sequence
    encoded_input = tokenizer(input, return_tensors="pt", padding=True).to(device)
    # Get model outputs   
    outputs = model(**encoded_input)

    # Extract the hidden states
    if hasattr(outputs, "last_hidden_state"): # Depends on the pretrained model backbone
      hidden_states = outputs.last_hidden_state
    else:
      hidden_states = outputs.hidden_states[-1]

    # Aggregate hidden states to get a single vector representation (e.g., mean pooling)
    embeddings = torch.mean(hidden_states, dim=1)
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

        # Here, additional preprocessing can be done (e.g., tokenization)

    def __len__(self):
        return len(self.affinity)

    def __getitem__(self, idx):
        return {
            "drug": self.drug[idx],
            "target": self.target[idx],
            "affinity": torch.tensor(self.affinity[idx], dtype=torch.float)
        }
    
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
  
     
  