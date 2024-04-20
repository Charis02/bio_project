from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


# The name of the model you want to download
model_name = "gokceuludogan/ChemBERTaLM"

# Specify the directory where you want to save the model and tokenizer
save_directory = "./pretrained_models/ChemBERTaLM/"

# Download and save the tokenizer
# tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=save_directory)

# Download and save the model
# model = RobertaModel.from_pretrained(model_name, cache_dir=save_directory)

# Model identifier
model_name = "facebook/esm2_t33_650M_UR50D"

# Specify the directory where you want to save the model and tokenizer
save_directory = "./pretrained_models/esm2_t33_650M_UR50D"

# Download and save the configuration with output_hidden_states set to True
config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, cache_dir=save_directory)
print(f"Downloaded config",flush=True)
config.save_pretrained(save_directory)

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_directory)
print(f"Downloaded tokenizer",flush=True)
tokenizer.save_pretrained(save_directory)

# Since you are using from_config to create the model, it does not automatically download weights.
# First, download and save the model using from_pretrained, then save the configuration as well.
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, cache_dir=save_directory)
print(f"Downloaded model",flush=True)
model.save_pretrained(save_directory)