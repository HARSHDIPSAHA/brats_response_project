# brats_response_project
# RECAP-Net: RANO Ensemble for Classification of Active Progression

[cite_start]This repository contains the official implementation for **RECAP-Net**, an end-to-end deep learning pipeline for classifying glioblastoma treatment response from longitudinal MRI scans[cite: 9].  
[cite_start]This work was developed for the BraTS 2025 Tumor Progression Challenge and uses an ensemble of 3D CNNs to assess tumor progression based on the Response Assessment in Neuro-Oncology (RANO) criteria[cite: 8, 16].

---

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [References](#references)
- [License](#license)

---

## Features

- **End-to-End Pipeline**: A complete deep learning pipeline for classifying glioblastoma treatment response from MRI scans[cite: 9].
- **Custom Segmentation**: Employs a fine-tuned Swin UNETR model for accurate tumor segmentation, which is then used as an input channel for the classifier[cite: 9, 60, 124].
- **Ensemble Learning**: Integrates three distinct 3D CNN architectures (ResNet-18, DenseNet-121, EfficientNet-B0) to improve prediction robustness and accuracy[cite: 9, 85].
- **Temporal Change Highlighting**: Uses channel augmentation by creating voxel-wise difference maps between baseline and follow-up scans to explicitly guide the model's focus on areas of change[cite: 89, 98].
- **Class Imbalance Handling**: Utilizes a conditional 3D SN-GAN for synthetic data augmentation to effectively handle class imbalance in the training dataset[cite: 9, 58, 75].

---

## Folder Structure
brats_response_project/
├── Dockerfile # For building the Docker container
├── README.md # This file
├── requirements.txt # Python dependencies
├── run_inference.sh # Script to execute inference
├── entrypoints/
│ └── infer.py # Main inference script
├── input/
│ ├── data/ # Location for input MRI scans
│ └── model/ # Location for pre-trained model weights
├── output/ # Directory for model outputs (e.g., predictions)
│ └── .gitkeep
└── src/
├── data/ # Data loading and preprocessing scripts
├── eval/ # Evaluation metric scripts
├── train/ # Model training scripts
├── configEnsemble.py # Configuration for the ensemble
└── mainEnsemble.py # Main training script for the ensemble


### Detailed File Descriptions

#### `run_inference.sh`
A shell script that orchestrates the inference process by running the `infer.py` entrypoint.

---

#### `entrypoints/`
Contains the primary script for running model inference.
- **infer.py**: Loads a trained model and processes new data from the `input/` directory to generate predictions.

---

#### `input/`
This directory serves as the main folder for all input data and models.
- **data/**: Intended for storing pre-processed MRI scans and related data for training or inference.
- **model/**: Used to store the pre-trained model checkpoints.

---

#### `output/`
This directory is designated for storing all generated outputs, such as prediction files, evaluation metrics, and logs.  
It contains a `.gitkeep` placeholder to ensure the directory is tracked by Git even when empty.

---

#### `src/`
Contains all the core source code for the project.
- **mainEnsemble.py**: The main script used to start the training process for the ensemble model.
- **configEnsemble.py**: A configuration file to manage hyperparameters, model paths, and other settings for the ensemble.
- **data/**, **eval/**, **train/**: Subdirectories containing modular scripts for data handling, model evaluation, and the training logic, respectively.

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Deepaksn19/brats_response_project.git
   cd brats_response_project
Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate


Install the required dependencies from requirements.txt:

pip install -r requirements.txt

Usage
Training

Place your pre-processed training data (following the BraTS directory structure) into the input/data/ folder.

Adjust any hyperparameters or paths in the src/configEnsemble.py file.

Run the main training script:

python src/mainEnsemble.py

Inference

Place your pre-trained model weights in the input/model/ directory and the data to be processed in input/data/.

Execute the inference script. The script is designed to automatically find the models and data from the input/ directory.

bash run_inference.sh


The output predictions will be saved in the output/ directory.

Dependencies

Ensure you have the following key Python libraries installed. You can install all of them using the requirements.txt file.

torch

monai

torchio

numpy

Install all dependencies via pip:

pip install -r requirements.txt

References

RECAP-Net: Our model, an ensemble of ResNet-18, DenseNet-121, and EfficientNet-B0, designed for RANO classification.

Swin UNETR: Transformer-based architecture used for 3D medical image segmentation.

MONAI: An open-source PyTorch-based framework for deep learning in healthcare imaging.

LUMIERE Dataset: The dataset used for training and evaluation, curated for the BraTS-PRO 2025 challenge.

License

This project is licensed under the MIT License. See the LICENSE file for details.

