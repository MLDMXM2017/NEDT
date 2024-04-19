## NEDT
The source code of article, "NEDT: a novel feature pair-based neural network embedded decision tree for anti-tumor synergistic drug combination prediction".

## Requirements

```bash
Python==3.7.9
torch==1.7.0+cu101
torchvision==0.8.1+cu101
pandas
tqdm
sklearn
matplotlib
```

## Usage

The preprocessing of the dataset and the training of NEDT are encapsulated in a file named main.py, which you can run directly to get the experimental results.

```bash
python main.py
```

The Results are stored in the results directory, with a csv file for each dataset that stores the results of 5-fold cross-validation.


