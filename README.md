# DRN-1D2D_Inter
inter-protein contact prediction from sequence.
![image](https://github.com/ChengfeiYan/DRN-1D2D_Inter/blob/main/data/main_fig.png)
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
    python generate_feature.py sequenceA msaA sequenceB msaB result_path device
   Where MSA should be derived from Uniref90 or Uniref100 database.

## Example
    python generate_feature.py ./example/1GL1_A.fasta ./example/1GL1_A_uniref100.a3m ./example/1GL1_I.fasta ./example/1GL1_I_uniref100.a3m ./example/result cpu
