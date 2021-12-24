# Python setup

## Conda setup

```bash
brew install miniforge
conda init zsh
```

Restart your terminal to have conda available.

## Create virtual env for xai

```bash
conda create --name xai python=3.9
conda activate xai
conda install -c pytorch "pytorch==1.9.1" "torchvision==0.10.0"
pip install -r ./mac-m1-requirements.txt
```

## Running application

```bash
conda activate xai
python -m xaitk_demo
```