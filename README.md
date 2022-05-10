
### Environment Setup

> base environment: python 3.7, cuda 10.2, pytorch 1.6, torchvision 0.7.0, tensorboardX 2.1 \
pytorch-geometric: \
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html \
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html \
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html \
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html \
pip install torch-geometric 

### Training

Training codes in gnn_train.py, and the run script in run.py.


#### Dataset Download:

STRING(we use Homo sapiens subset): 
- PPI: https://stringdb-static.org/download/protein.actions.v11.0/9606.protein.actions.v11.0.txt.gz 
- Protein sequence: https://stringdb-static.org/download/protein.sequences.v11.0/9606.protein.sequences.v11.0.fa.gz 
Dataset：
SHS27k and SHS148k: 
- http://yellowstone.cs.ucla.edu/~muhao/pipr/SHS_ppi_beta.zip

