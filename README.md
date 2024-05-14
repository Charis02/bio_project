# MoELoRA-DTI: An MoE-based Parameter Efficient Fine-Tuning Method for Drug-Target Interaction Prediction

This repository contains the implementation of MoELoRA-DTI, a novel approach that combines Mixture of Experts (MoE) and Parameter Efficient Fine-Tuning (PEFT) to predict drug-target interactions (DTIs) effectively. Our model leverages pre-trained transformer-based language models to generate meaningful drug and target embeddings, enhancing the prediction of binding affinity scores.

## Overview

MoELoRA-DTI aims to address the challenges posed by the complexity of biological interactions and the limited availability of high-quality DTI datasets. By employing the MoE framework, the model tailors its predictions to specific protein characteristics, improving both accuracy and interpretability.

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
  - [Data Download](#data-download)
  - [Model Download](#model-download)
  - [Training](#training)
  - [Expert Analysis](#expert-analysis)

## Introduction

The prediction of drug-target interactions (DTIs) is a critical step in the drug discovery process. Traditional methods often involve extensive and costly experimental procedures, motivating the need for efficient computational approaches. MoELoRA-DTI is designed to enhance the prediction accuracy of DTIs by leveraging state-of-the-art transformer-based language models through a combination of Mixture of Experts (MoE) and Parameter Efficient Fine-Tuning (PEFT) techniques.

### Key Features

- **Mixture of Experts (MoE) Framework**: This allows the model to specialize in different aspects of the protein and drug interaction landscape, improving interpretability and performance.
- **Parameter Efficient Fine-Tuning (PEFT)**: Utilizing LoRA (Low-Rank Adaptation), our model fine-tunes pre-trained language models efficiently, reducing the computational resources required.
- **Pre-trained Models**: The model employs ChemBERTaLM for drug embeddings and ESM2 for protein embeddings, both of which are pre-trained on large datasets to capture intricate biochemical properties.
- **Scalable Training**: By integrating DeepSpeed for distributed training across multiple GPUs, the model can handle large-scale datasets and complex architectures efficiently.

### Background

MoELoRA-DTI builds upon recent advancements in the use of pre-trained protein language models (PLMs) and introduces a novel MoE-based architecture for DTI prediction. This approach not only improves prediction accuracy but also provides insights into the interaction mechanisms between drugs and targets, which are often obscured in traditional black-box models.

The methodology and experimental results demonstrate the model's ability to predict binding affinity scores with high precision, making it a valuable tool in the early stages of drug discovery. For a comprehensive understanding of the model architecture and its performance, please refer to our [paper](path/to/6_8710_Final_Project.pdf).

## Usage

### Downloads
First, download the necessary data using the ```download_data.py``` script:

```
python download_data.py
```

Next, download the pre-trained models using the ```download_models.py``` script:

```
python download_models.py
```

### Training

To train the model, use the ```deepspeed_train.py``` script:

```
deepspeed deepspeed_train.py --deepspeed_config deepspeed_config.json
```

### Expert Analysis

After training, you can analyze the performance of the experts using the ```expert_analysis.py``` script:

```
python expert_analysis.py
```
