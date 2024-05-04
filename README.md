# cs226-debiasing-vl-embeddings

Repository for Harvard CS 226R (Topics in Theory for Society: The Theory of Algorithmic Fairness) final project - Evaluating Efficacy of Using Adversarial Learning to Debias Word-Embeddings

## Description

This repository contains the code for the final project of Harvard CS 226R. The project focuses on evaluating the efficacy of using adversarial learning to debias word embeddings based on [Zhang et al. (2018)](https://arxiv.org/pdf/1801.07593). The code is written in Python 3.8 and uses the PyTorch library.

## Installation

To install and run the code, follow these steps:

1. Clone the repository: `git clone https://github.com/Kev-Y-Huang/cs226-debiasing-vl-embeddings.git`
2. Change into the project directory: `cd cs226-debiasing-vl-embeddings`
3. Install the required dependencies: `pip install -r requirements.txt`

## Data

To train the embeddings yourself, you will need to download the data from the following [google drive link](https://drive.google.com/drive/folders/1pXL31TU0LPHw9J9p3BXep0O1p2A_rrDZ?usp=drive_link). The code is currently set to use the `orig_glove` embeddings but you can modify the code to use `orig_w2v` embeddings instead (provided you have enough memory).

## Analysis

To view the analysis notebook, you can access it at the following [google colab link](https://colab.research.google.com/drive/1a0qT0HPKquYq03m_p7OYfyIblAV87r2D?usp=sharing).

## Usage

To use code and train embeddings, run the main script `main.py` with necessary arguments. For example:

```bash
python main.py --embedding_file test --model_type simple --debiased --n_epochs 250
```
