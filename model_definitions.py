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

    # self.routing_network = nn.Sequential(
    #     nn.Linear(embedding_dim, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, num_experts),
    # )

    self.routing_network = nn.Sequential(
        nn.Linear(embedding_dim, num_experts),
    )
    self.num_experts_topk = 2

    self.exploration_epsilon = 0

    # self.batch_norm_layer = nn.BatchNorm1d(embedding_dim)

  def forward(self, router_inputs, **inputs):
    logits = self.routing_network(router_inputs)  # logits should be shape (num_experts, )
    # with epsilon probability, choose a random expert if the model is in training mode
    if False:#torch.rand(1) < self.exploration_epsilon and self.training:
      chosen_experts = torch.randint(0, self.num_experts, (inputs["input_ids"].shape[0], self.num_experts_topk), device=next(self.parameters()).device)
      exp_weights = torch.randn_like(chosen_experts, dtype=torch.float)
    else:
      exp_weights, chosen_experts = torch.topk(logits, self.num_experts_topk)
    
    weights = torch.softmax(exp_weights, dim=1)

    expert_load = torch.zeros(self.num_experts, device=next(self.parameters()).device)

    # Create a prediction with the shape [batch, hidden_size]
    prediction = torch.zeros(inputs["input_ids"].shape[0],self.expert_model.config.hidden_size, device=next(self.parameters()).device)

    for current_expert in range(self.num_experts):
      batch_idx, nth_expert = torch.where(chosen_experts == current_expert)
      routed_inputs = {k: v[batch_idx] for k, v in inputs.items()}
      ones = torch.ones_like(batch_idx, dtype=torch.float)
      expert_load[current_expert] = ones.sum()

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

    # if prediction.shape[0] > 1:
    #   prediction = self.batch_norm_layer(prediction)
    return prediction, torch.softmax(logits,dim=1), expert_load

  def reduce_exploration(self):
    self.exploration_epsilon = max(0, self.exploration_epsilon - 0.1)

  def original_embedding(self, **inputs):
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

class DTIDataset(Dataset):
    def __init__(self, data, split='train',drug_label='Drug',target_label='Target',affinity_label='Y'):
        # Assuming data is a dictionary with 'train', 'valid', 'test' splits
        # Concatenate training, validation, and test sets if needed
        # Or you can adjust the code to use only one of the splits
        self.data = data[split]
        self.drug = self.data[drug_label]
        self.target = self.data[target_label]
        self.affinity = self.data[affinity_label]

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

class BigModel(nn.Module):
    def __init__(self, drug_model, target_model, regressor):
        super().__init__()
        self.drug_model = drug_model
        self.target_model = target_model
        self.regressor = regressor
        # self.drug_to_common = nn.Linear(768, 640)
        # self.target_to_common = nn.Linear(640, 640)
        # self.cross_attention_drug = nn.MultiheadAttention(640, 8,dtype=torch.float16)
        # self.cross_attention_target = nn.MultiheadAttention(640, 8,dtype=torch.float16)


    def forward(self, drug_input, target_input):
        with torch.no_grad():
            original_drug_embedding = self.drug_model.original_embedding(**drug_input)
            original_target_embedding = self.target_model.original_embedding(**target_input)

        drug_embeddings, drug_routing_probabilities, drug_expert_load = self.drug_model(original_drug_embedding,**drug_input)
        target_embeddings, target_routing_probabilities, target_expert_load = self.target_model(original_target_embedding,**target_input)

        all_embeds = torch.cat([drug_embeddings, target_embeddings], dim=1).half()
        # drug_embeddings = self.drug_to_common(drug_embeddings)
        # target_embeddings = self.target_to_common(target_embeddings)

        # cross_attended_drug, _ = self.cross_attention_drug(drug_embeddings, target_embeddings, target_embeddings)
        # cross_attended_target, _ = self.cross_attention_target(target_embeddings, drug_embeddings, drug_embeddings)
        
        # all_embeds = torch.cat([cross_attended_drug, cross_attended_target], dim=1).half()
        
        return self.regressor(all_embeds), drug_routing_probabilities, target_routing_probabilities, drug_expert_load, target_expert_load

    def reduce_exploration(self):
      self.drug_model.reduce_exploration()
      self.target_model.reduce_exploration()