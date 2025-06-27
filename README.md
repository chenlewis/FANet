# FANet
![Method Overview](figure/image1.png)


# Forensicability Assessment: Not All Samples Qualify for Recapture Detection

## Introduction

This repository contains the official implementation for the paper **"Forensicability Assessment: Not All Samples Qualify for Recapture Detection"**. The code is designed to assess the "forensicability" of an image. It filters out samples with weak forensic cues before they are passed to more computationally expensive forensic tasks like Face Anti-Spoofing (FAS) or Document Presentation Attack Detection (DPAD).

## Environment and Dependencies

This project consists of Python and MATLAB components.

### Python

The code is compatible with **PyTorch 1.6.0**. Please install the required libraries based on the `import` statements in the Python scripts. Common dependencies include:

- `torch`
    
- `torchvision`
    
- `numpy`
    
- `pandas` 
    
- `opencv-python`
    

### MATLAB

**Required**: The Image Quality Feature (IQF) extraction part of this project depends on MATLAB. Please ensure you have a working MATLAB environment to run this stage.

## Project Structure
```
.
├── Forensicability_Feature_Extraction
│   ├── FF  (Lightweight Forensic Features)
│   │   ├── main.py
│   │   ├── nets.py
│   │   └── resnet.py
│   ├── IQF (Image Quality Features)
│   │   ├── main.py
│   │   ├── Data.py
│   │   ├── nets.py
│   │   └── utils
│   │       ├── BIQI_release/     (MATLAB code)
│   │       ├── BRISQUE_release/  (MATLAB code)
│   │       └── GM-LOG-BIQA/      (MATLAB code)
│   └── define_class.py
└── Forensicability_Quantification
    ├── data
    │   └── train.csv
    ├── utils
    │   ├── datasets_csv.py
    │   └── duq.py
    ├── compute_forensicability.py
    ├── test.py
    └── train.py
```

## Usage Workflow

The entire process is divided into two main parts: Feature Extraction, followed by FANet Training and Evaluation.

### Part 1: Feature Extraction

You need to extract two types of features from your image dataset.

#### Step 1.1: Extract Image Quality Features (IQF)

This step requires **MATLAB**.

- **Path**: `Forensicability_Feature_Extraction/IQF/`
    
- **Instructions**: The `utils` sub-directory contains the MATLAB implementations for `BIQI`, `BRISQUE`, and `GM-LOG-BIQA`. You need to run these MATLAB scripts to generate the image quality features for each image. According to the paper, this results in a 94-dimensional feature vector. The `main.py` script likely coordinates this process.
    

#### Step 1.2: Extract Lightweight Forensic Features (FF)

- **Path**: `Forensicability_Feature_Extraction/FF/`
    
- **Instructions**: Run the `main.py` script in this directory. It uses a CNN model defined in `nets.py` and `resnet.py` to extract features. According to the paper, this is a 128-dimensional feature vector.
    

#### Step 1.3: Combine Features

Combine the features generated from the two steps above into a single CSV file.

- **Output File**: `Forensicability_Quantification/data/train.csv`
    
- **Format**: Each row represents one sample. Combine the IQF features ($x_Q$​) and FF features ($x_F$) into a single vector. The first 222 dimensions should represent these combined features.
    

### Part 2: Train and Evaluate FANet

Once the `train.csv` feature file is ready, proceed to the `Forensicability_Quantification` directory to execute the core FANet workflow.

- **Path**: `Forensicability_Quantification/`
    

This process is divided into three scripts that should be run in order:

1. **Train the Model (`train.py`)**
    
    - This script trains the FANet model and learns the optimal class center locations ($e_c​$).
        
2. **Predict Scores (`test.py`)**
    
    - Using the trained model from the previous step, this script predicts the forensicability score pair ($\hat{\boldsymbol{y}}$​) for each test sample.
        
3. **Compute Final Forensicability (`compute_forensicability.py`)**
    
    - This script takes the predicted scores ($\hat{\boldsymbol{y}}$​) and the learned class centers ($\hat{e}_c​$) to calculate the final forensicability score ($F$) for each sample. A lower score F indicates weaker forensicability.
    ## Citation

If you use this code or the ideas from the paper in your research, please cite:

```
@inproceedings{chen2025forensicability,
  title={Forensicability Assessment: Not All Samples Qualify for Recapture Detection},
  author={Yongqi Chen and Lin Zhao and Rizhao Cai and Zitong Yu and Changsheng Chen and Bin Li},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)}, 
  year={2025}
}
```
