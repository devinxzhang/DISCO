# DISCO

## Environment
### Requirements
- The requirements can be installed with:
  
  ```bash
  conda create -n mfuser python=3.9 numpy=1.26.4
  conda activate mfuser
  conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  pip install xformers==0.0.20
  pip install mmcv-full==1.5.1 
  pip install mamba_ssm==2.2.2
  pip install causal_conv1d==1.4.0
  ```

## Datasets
- To set up datasets, please follow [the official **DELIVER** repo]([https://github.com/ssssshwan/TLDR/tree/main?tab=readme-ov-file#setup-datasets](https://github.com/jamycheung/DELIVER)).
