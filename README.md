# RECAP-Net: RANO Ensemble for Classification of Active Progression

[![MICCAI 2025](https://img.shields.io/badge/MICCAI-2025-blue)](https://miccai.org/)
[![BraTS 2025](https://img.shields.io/badge/BraTS-2025-green)](https://www.synapse.org/)
[![Rank](https://img.shields.io/badge/Rank-3rd%20Place-gold)](https://www.synapse.org/)

**Official implementation of RECAP-Net for the BraTS 2025 Tumor Progression Challenge**

This repository contains the official implementation for **RECAP-Net**, an end-to-end deep learning pipeline for classifying glioblastoma treatment response from longitudinal MRI scans. This work was developed for the BraTS 2025 Tumor Progression Challenge and achieved **World Rank 3**, with the paper accepted at **MICCAI 2025, South Korea**.

## 🏆 Achievements

- **World Rank 3** in BraTS 2025 Tumor Progression Challenge
- **Accepted** at MICCAI 2025, South Korea
- Ensemble-based approach using RANO (Response Assessment in Neuro-Oncology) criteria

## 📋 Overview

RECAP-Net uses an ensemble of 3D CNNs to assess tumor progression based on the Response Assessment in Neuro-Oncology (RANO) criteria. The system integrates multiple deep learning components to achieve robust and accurate classification of active progression in glioblastoma patients.

## ✨ Key Features

- **End-to-End Pipeline**: Complete deep learning pipeline for classifying glioblastoma treatment response from MRI scans
- **Custom Segmentation**: Fine-tuned Swin UNETR model for accurate tumor segmentation, used as an input channel for the classifier
- **Ensemble Learning**: Integrates three distinct 3D CNN architectures (ResNet-18, DenseNet-121, EfficientNet-B0) for improved robustness and accuracy
- **Temporal Change Highlighting**: Channel augmentation using voxel-wise difference maps between baseline and follow-up scans to guide model focus on areas of change
- **Class Imbalance Handling**: Conditional 3D SN-GAN for synthetic data augmentation to effectively handle class imbalance

## 🐳 Docker Support

This project includes a Docker container for easy deployment and inference. The Docker image is based on NVIDIA CUDA 11.8 and includes all necessary dependencies.

### Docker Image Details

- **Base Image**: Ubuntu 22.04 with NVIDIA CUDA 11.8
- **Python**: Python 3 with pip
- **CUDA Version**: 11.8.0
- **Dependencies**: All packages from `requirements.txt` pre-installed (~5.63 GB)
- **Working Directory**: `/workspace`
- **Pre-installed Components**:
  - Segmentation models in `/workspace/segmentation/`
  - Pre-trained ensemble models in `/workspace/models/`
  - Model checkpoints in `/workspace/checkpoints/`
  - Inference scripts: `inference.py`, `build_tensor.py`, `run_inference.sh`

### Building the Docker Image

```bash
docker build -t recap-net:latest .
```

**Note**: The build process will:
- Install CUDA 11.8 runtime and compatibility packages
- Install Python 3 and pip
- Install all dependencies from `requirements.txt` (this may take several minutes)
- Copy segmentation models, ensemble models, and checkpoints
- Set up the inference scripts

### Running Inference with Docker

The Docker container expects two arguments: test data directory and output directory.

```bash
docker run --gpus all \
  -v /path/to/test/data:/workspace/test_data \
  -v /path/to/output:/workspace/output \
  recap-net:latest \
  /workspace/run_inference.sh /workspace/test_data /workspace/output
```

**Parameters**:
- `--gpus all`: Enables GPU access (required for inference)
- `-v /path/to/test/data:/workspace/test_data`: Mounts your test data directory
- `-v /path/to/output:/workspace/output`: Mounts output directory for predictions
- `/workspace/run_inference.sh`: The inference script (automatically executed)
- Arguments: `TEST_DATA_DIR` and `PRED_DIR` paths inside the container

**Example**:
```bash
docker run --gpus all \
  -v /home/user/brats_test_data:/workspace/test_data \
  -v /home/user/predictions:/workspace/output \
  recap-net:latest \
  /workspace/run_inference.sh /workspace/test_data /workspace/output
```

**Requirements**:
- NVIDIA Docker runtime installed
- NVIDIA drivers >= 470 (for CUDA 11.8)
- GPU with compute capability 3.5 or higher

## 📁 Project Structure

```
brats_response_project/
├── Dockerfile              # Docker container configuration
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── run_inference.sh        # Inference execution script
├── entrypoints/
│   └── infer.py           # Main inference entrypoint
├── input/
│   ├── data/              # Input MRI scans directory
│   └── model/             # Pre-trained model weights directory
├── output/                # Model outputs (predictions, logs)
└── src/
    ├── configEnsemble.py  # Ensemble configuration
    ├── mainEnsemble.py    # Main training script
    ├── data/              # Data loading and preprocessing
    ├── eval/              # Evaluation scripts
    └── train/             # Training scripts
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HARSHDIPSAHA/brats_response_project.git
   cd brats_response_project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Training

1. Place your pre-processed training data (following the BraTS directory structure) into the `input/data/` folder.

2. Adjust hyperparameters or paths in `src/configEnsemble.py` if needed.

3. Run the training script:
   ```bash
   python src/mainEnsemble.py
   ```

### Inference

1. Place your pre-trained model weights in `input/model/` directory.

2. Place the data to be processed in `input/data/`.

3. Execute the inference script:
   ```bash
   bash run_inference.sh /path/to/test/data /path/to/output
   ```

   Or using the entrypoint directly:
   ```bash
   python entrypoints/infer.py --test_data_dir /path/to/test/data --pred_dir /path/to/output
   ```

4. Predictions will be saved in the specified output directory.

## 📦 Dependencies

Key dependencies include:

- `torch` - PyTorch deep learning framework
- `monai` - Medical imaging AI framework
- `torchio` - Medical image preprocessing
- `numpy` - Numerical computing

For the complete list, see `requirements.txt`. Install all dependencies with:

```bash
pip install -r requirements.txt
```

## 🏗️ Architecture

### System Overview

RECAP-Net is an end-to-end deep learning pipeline that processes longitudinal MRI scans to classify glioblastoma treatment response according to RANO criteria. The architecture consists of four main components working in tandem.

### Input Processing

**Multi-Modal Input Preparation:**
- **Input Modalities**: 4 baseline MRI sequences (T1, T1ce, T2, FLAIR) + 4 follow-up MRI sequences
- **Segmentation Masks**: 2 masks (baseline and follow-up) from Swin UNETR segmentation
- **Total Channels**: 10-channel tensor (8 MRI modalities + 2 segmentation masks)
- **Spatial Resolution**: Standardized to 256×256×256 voxels

**Preprocessing Pipeline:**
1. **Intensity Normalization**: Z-score normalization per modality
   ```
   I_norm(x) = (I(x) - μ) / σ
   ```
   where μ and σ are computed over brain voxels only

2. **Spatial Standardization**:
   - Center-cropping for volumes larger than 256×256×256
   - Symmetric zero-padding for volumes smaller than 256×256×256
   - Ensures uniform input dimensions for batch processing

### Component 1: Swin UNETR Segmentation

**Purpose**: Generate accurate tumor segmentation masks for both baseline and follow-up scans

**Architecture**:
- **Base Model**: Swin UNETR (Swin Transformer-based U-Net for 3D medical image segmentation)
- **Output Classes**: 
  - Background (0)
  - Peritumoral edema (1)
  - Enhancing tumor core (2)
- **Fine-tuning**: Pre-trained on medical imaging datasets, fine-tuned on LUMIERE dataset
- **Integration**: Segmentation masks are concatenated as additional input channels to guide the classifier

### Component 2: Conditional 3D SN-GAN (Data Augmentation)

**Purpose**: Address class imbalance (3:2:1:1 distribution across RANO classes) through synthetic data generation

**Architecture**:
- **Type**: Conditional 3D Spectral Normalized Generative Adversarial Network
- **Input**: 8-channel temporal MRI volumes (4 baseline + 4 follow-up modalities)
- **Training**: Trained on complete longitudinal scan data to learn realistic tumor progression patterns
- **Output**: Synthetic 8-channel MRI volumes representing plausible tumor progressions
- **Segmentation Integration**: Generated scans are passed through Swin UNETR to produce corresponding segmentation masks
- **Result**: Each synthetic sample includes 10-channel tensor (8 MRI + 2 masks) ready for classification

### Component 3: Ensemble of 3D CNNs

**Architecture**: Three independent 3D convolutional neural networks:

1. **ResNet-18 (3D)**
   - Residual connections for gradient flow
   - 18-layer deep architecture adapted for 3D medical imaging
   - Processes 10-channel input tensors

2. **DenseNet-121 (3D)**
   - Dense connectivity pattern for feature reuse
   - 121-layer architecture with dense blocks
   - Efficient parameter utilization

3. **EfficientNet-B0 (3D)**
   - Compound scaling for optimal efficiency
   - Balanced depth, width, and resolution
   - Lightweight yet powerful architecture

**Input Format**: All three models receive the same 10-channel tensor (256×256×256)

**Output**: Each model produces probability distributions over 4 RANO classes:
- Complete Response (CR) = 0
- Partial Response (PR) = 1
- Stable Disease (SD) = 2
- Progressive Disease (PD) = 3

### Component 4: Soft Voting Ensemble

**Combination Strategy**: Soft voting aggregates predictions from all three models

**Process**:
1. Each model outputs class probabilities: [P(CR), P(PR), P(SD), P(PD)]
2. Probabilities are averaged across all three models
3. Final prediction: Class with highest average probability

**Advantages**:
- Reduces overfitting by combining diverse architectures
- Improves robustness through model diversity
- Better generalization to unseen data

### Temporal Change Highlighting

**Feature Enhancement**: 
- Voxel-wise difference maps computed between baseline and follow-up scans
- Highlights regions of change (growth, shrinkage, enhancement)
- Guides model attention to clinically relevant areas
- Integrated into the input tensor through channel augmentation

## 🔄 Workflow

### Training Pipeline

1. **Data Preprocessing**
   - Load baseline and follow-up MRI volumes
   - Apply z-score normalization per modality
   - Standardize spatial dimensions to 256×256×256
   - Concatenate into 10-channel tensors

2. **GAN Training** (if using synthetic augmentation)
   - Train conditional 3D SN-GAN on longitudinal MRI data
   - Generate synthetic samples for underrepresented classes
   - Generate segmentation masks for synthetic data using Swin UNETR

3. **Segmentation Model Training**
   - Fine-tune Swin UNETR on LUMIERE dataset
   - Generate segmentation masks for all training samples

4. **Ensemble Model Training**
   - Train ResNet-18, DenseNet-121, and EfficientNet-B0 independently
   - Each model learns from 10-channel input tensors
   - Optimize using cross-entropy loss with class weighting

5. **Validation**
   - Evaluate each model separately
   - Test ensemble performance with soft voting
   - Select best model checkpoints

### Inference Pipeline

1. **Input Preparation**
   - Load baseline and follow-up MRI volumes
   - Apply same preprocessing as training (normalization, resizing)

2. **Segmentation**
   - Generate tumor masks using pre-trained Swin UNETR
   - Concatenate masks with MRI volumes (10-channel tensor)

3. **Classification**
   - Pass 10-channel tensor through all three ensemble models
   - Collect probability distributions from each model

4. **Ensemble Prediction**
   - Average probabilities across all three models
   - Select class with highest probability
   - Output RANO response category

## 📊 Performance

On the augmented LUMIERE dataset, RECAP-Net achieves:

| Metric | Score |
|--------|-------|
| **Balanced Accuracy** | 0.9400 |
| **F1 Score** | 0.9460 |
| **True Positive Rate (TPR)** | 0.9510 |
| **True Negative Rate (TNR)** | 0.9550 |
| **Precision** | 0.9420 |
| **AUROC** | 0.9600 |

The ensemble approach significantly outperforms individual models, demonstrating the effectiveness of combining multiple architectures for robust classification.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- BraTS 2025 Challenge organizers
- MICCAI 2025 conference
- LUMIERE Dataset contributors
- MONAI framework developers

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This implementation was developed for the BraTS 2025 Tumor Progression Challenge and achieved World Rank 3. The paper was accepted at MICCAI 2025, South Korea.
