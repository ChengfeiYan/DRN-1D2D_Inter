# DRN-1D2D_Inter
inter-protein contact prediction from sequence.
![image](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/data/main_fig.jpg)
## Requirements
- #### python3.8
  1. [pytorch1.9](https://pytorch.org/)  
  2. [Biopython](https://biopython.org/)
  3. [esm](https://github.com/facebookresearch/esm)
  4. [numpy](https://numpy.org/)
- #### other packages
  1. [alnstats](https://github.com/psipred/metapsicov/tree/master/src)
  2. [fasta2aln](https://github.com/kad-ecoli/hhsuite2/blob/master/bin/fasta2aln)
  3. [hh-suite](https://github.com/soedinglab/hh-suite)
  4. [CCMpred](https://github.com/soedinglab/CCMpred)

## Installation
### 1. Install DRN-1D2D_Inter
    git clone https://github.com/ChengfeiYan/DRN-1D2D_Inter.git
### 2. Modify the path of each tool (CCMpred, alnstats ...) in predict.py
  
### 3. Download the trained models
   Download the trained models from  [trained models](https://drive.google.com/file/d/1ICqJSNc01E2cGYhVj1IxzIkmnS-FMT2C/view?usp=sharing), then unzip it into the folder named "model".

## Usage
    python predict.py sequenceA msaA sequenceB msaB result_path device
    1.  sequenceA: fasta file corresponding to target A.
    2.  msaA: a3m file corresponding to target A.
    3.  sequenceB: fasta file corresponding to target B.
    4.  msaB: a3m file corresponding to target B.
    5.  result_path: /
    6.  device: cpu, cuda:0, cuda:1, ...
    
   Where MSA should be derived from Uniref90 or Uniref100 database.

## Example
    python predict.py ./example/1GL1_A.fasta ./example/1GL1_A_uniref100.a3m ./example/1GL1_I.fasta ./example/1GL1_I_uniref100.a3m ./example/result cpu

## The output of exmaple(1GL1)
![image](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/data/drn.jpg)

## Reference  
Please cite: Improved inter-protein contact prediction using dimensional hybrid residual networks and protein language models
Yunda Si, Chengfei Yan
bioRxiv 2022.08.04.502748; doi: https://doi.org/10.1101/2022.08.04.502748.

If you meet any problem in installing or running the program, please contact chengfeiyan@hust.edu.cn.
