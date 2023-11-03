## NEDT
The source code of paper, "An Entity Pair-Based Neural Network Embedded Decision Tree for Biomedical Interaction Prediction".

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
cd data
python python.py
```

The Results are stored in the results directory, with a csv file for each dataset that stores the results of the three-fold 5-fold cross-validation.


