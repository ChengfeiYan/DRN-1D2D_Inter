# DRN-1D2D_Inter
inter-protein contact prediction from sequences of interacting proteins:
![image](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/data/main_fig.jpg)
## Requirements
- #### python3.8
  1. [pytorch1.9](https://pytorch.org/)  
  2. [Biopython](https://biopython.org/)
  3. [esm](https://github.com/facebookresearch/esm)
  4. [numpy](https://numpy.org/)
  
  **Please note**: To implement protein language models (ESM-1b and ESM-MSA-1b in this study) in [esm](https://github.com/facebookresearch/esm), model weights of these protein language models should be downloaded first from the links provided in the **"Available Models and Datasets"** table of [esm github](https://github.com/facebookresearch/esm). The paths of these model weights need to be set in [predict.py](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/predict.py#L30) later. Besides, the contact regression parameter files of ESM-1b:  [esm1b_t33_650M_UR50S-contact-regression.pt](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/data/regression/esm1b_t33_650M_UR50S-contact-regression.pt) and ESM-MSA-1b: [esm_msa1b_t12_100M_UR50S-contact-regression.pt](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/data/regression/esm_msa1b_t12_100M_UR50S-contact-regression.pt) should be stored in the same directory with the model parameter files.
- #### other packages
  1. [alnstats](https://github.com/psipred/metapsicov/tree/master/src/alnstats) (directly download the executable file, and change its mode to be executable)
  2. [fasta2aln](https://github.com/kad-ecoli/hhsuite2/blob/master/bin/fasta2aln) (directly dowload the executable file, and change its mode to be executable)
  3. [hh-suite](https://github.com/soedinglab/hh-suite)
  4. [CCMpred](https://github.com/soedinglab/CCMpred)

## Installation
### 1. Install DRN-1D2D_Inter
    git clone https://github.com/ChengfeiYan/DRN-1D2D_Inter.git
### 2. Modify the path of each tool (CCMpred, alnstats ...) and the paths of the model weights of the protien lanuage models (ESM-1b and EMS-MSA-1b) in [predict.py](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/predict.py#L22)
### 3. Copy the [esm1b_t33_650M_UR50S-contact-regression.pt](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/data/regression/esm1b_t33_650M_UR50S-contact-regression.pt) from /data/regression to the location of [ESM-1b's model weights](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/predict.py#L30);  Copy the [esm_msa1b_t12_100M_UR50S-contact-regression.pt](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/data/regression/esm_msa1b_t12_100M_UR50S-contact-regression.pt) from /data/regression to the location of [ESM-MSA-1b's model weights](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/predict.py#L31);
  
### 4. Download the trained models
   Download the trained models from  [trained models](https://drive.google.com/file/d/1ICqJSNc01E2cGYhVj1IxzIkmnS-FMT2C/view?usp=sharing), then unzip it into the folder named "model".

## Usage
    python predict.py sequenceA msaA sequenceB msaB result_path device
    1.  sequenceA: fasta file corresponding to target A.
    2.  msaA: a3m file corresponding to target A (multiple sequence alignment).
    3.  sequenceB: fasta file corresponding to target B (multiple sequence alignment).
    4.  msaB: a3m file corresponding to target B.
    5.  result_path: [a directory for the output]
    6.  device: cpu, cuda:0, cuda:1, ...
    
   Where MSA should be derived from Uniref90 or Uniref100 database.

## Example
    python predict.py ./example/1GL1_A.fasta ./example/1GL1_A_uniref100.a3m ./example/1GL1_I.fasta ./example/1GL1_I_uniref100.a3m ./example/result cpu

## The output of exmaple(1GL1)
![image](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/data/drn.jpg)

It should be noted, we downsampled the MSAs of the example target due to the file size limiation of github. The real performance of DRN-1D2D_Inter for the provided example should be better in real practice.

## Train
The script used to train DRN-1D2D_Inter is [train.py](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/train.py), which contains all the details of training DRN-1D2D_Inter, including how to choose the best model, how to calculate the loss, etc.

## Reference  
Please cite: Yunda Si, Chengfei Yan, Improved inter-protein contact prediction using dimensional hybrid residual networks and protein language models, Briefings in Bioinformatics, 2023, bbad039, https://doi.org/10.1093/bib/bbad039

If you meet any problem in installing or running the program, please contact chengfeiyan@hust.edu.cn.
