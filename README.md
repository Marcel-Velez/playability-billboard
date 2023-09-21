# Quantifying the Ease of Playing Song Chords on the Guitar
_Still under construction_
This repository contains the code and data associated with the paper "Quantifying the Ease of Playing Song Chords on the Guitar" In this paper, we focus on rhythm guitar, a popular instrument among amateur musicians, and propose a rubric-based "playability" metric to formalize the spectrum of difficulty.
- We propose an interpretable guitar playability metric;
- an extension of the Billboard dataset of 200 playability annotated songs tested for reliability;
- a rule-based baseline for our playability metric.

<img src="media/playability.png" width="500">

## Contents
1. `data/`: Contains the dataset of 200 songs from the McGill Billboard dataset, along with the playability annotations, and the code for the billboard and custom data dataloaders in pytorch.
2. `models/`: Includes the code for the LSTM, and GRU models, used to predict the rubric categories automatically.
3. `rule_models`: Contains the fitted rule-based model pickle files.
4. `trained_models`: Contains the trained model pth files.
5. `utils/`: Contains the rule-based model functions, helper functions for the ML models, and the evaluation metrics.
6. `main.py`: Contains the code for training the models on the billboard playability dataset and evaluating them.
7. `custom_data.py`: Contains the code for predicting playability scores for new data.
8. `rule`
## Installation
```bash
pip install -r requirements.txt
```

## Usage of code

### training on the billboard dataset

Train an lstm on predicting the weighted total playability score for all songs in the dataset
```bash
python main.py --chord_encoding char --target weighted_total --model lstm --accelerator cpu 
```
Training on a Single Category other than the default values (see configuration for more info on arguments)
```bash
python main.py --batch_size 64 --chord_encoding char --target weighted_total --k_fold 0 --learning_rate 0.002 --model lstm --dropout 0.5 --max_epochs 20 --num_workers 0 --accelerator cpu --file_path ./data/Annotations.csv
```
Training on All Categories
```bash
python main.py --batch_size 64 --chord_encoding char --target all --k_fold 0 --learning_rate 0.002 --model lstm --dropout 0.5 --max_epochs 20 --num_workers 0 --accelerator cpu --file_path ./data/Annotations.csv
```

### predicting playability scores for new data
predicting new values for the data in ./data/
note that the data should be in the same format as the billboard dataset and for the guitardiagram one has to implement a mapping to guitarfingers like A:maj: ['x', 'o', '2:1', '2:2', '2:3', 'o'].
```bash
python custom_data.py --chord_encoding char --target weighted_total --model lstm --accelerator cpu --file_path ./data/Annotations.csv --data_path ./data/Annotations.csv
```

## Configuration

Explain the configuration options available to users. Describe each command-line argument or configuration file option and what it does.

    --batch_size: Specify the batch size for training. 
        (_default_: 64)
    --chord_encoding: Choose the chord encoding method.
        _default_: 'char', 
        options:['char','dotsplit','guitardiagram']).
    --target: Specify the target category or 'all' to train on all categories
        _default_: 'weighted_total'   
        options: ['CFP', 'CFD', 'UC', 'RHC', 'CPT', 'BD', 'R', 'weighted_total']
    --k_fold: Specify the k-fold index for cross-validation.
    --learning_rate: Set the learning rate for training.
    --model: Choose the model architecture 
        default: 'lstm'
        options: ['lstm','gru']
    --dropout: Set the dropout rate.
    --max_epochs: Specify the maximum number of training epochs.
    --num_workers: Set the number of data loading workers.
    --accelerator: Choose between 'cpu', 'mps, and 'gpu' for hardware acceleration.
    --file_path: Specify the path to the playability data file.
        _default_: './data/Annotations.csv'

## The rubric

|   Criterion    | Weight | Very difficult (3 points) | Difficult (2 points) | Easy (1 point) | Very Easy (0 points) |
| :------------: | :----: | :----------------------: | :------------------: | :------------: | :------------------: |
| Uncommonness of chord |   3    |  A lot of uncommon chords  |  Some uncommon chords  | Few uncommon chords | No uncommon chords |
| Chord finger positioning |   3    |  Very cramped or very wide fingerspread | Uncomfortable or spread out fingers | Slightly uncomfortable or spread out fingers | Comfortable hand and finger position |
| Chord fingering difficulty |   2    |  Mostly chords that require four fingers or barre chords | Some chords require four fingers to be played or are barre chords (not A or E) | Most chords require three fingers or are A or E barre chords | Most chords can be played with two or three fingers |
| Repetitiveness |   2    | No repeated chord progressions | A few repeated chord progressions | Quite a bit of repetition of chord progressions | A lot of repetition of chord progressions |
| Right-hand complexity |   2    | For some chords multiple inner strings are not strummed | For some chords one inner string is not strummed | For some of the chords one or more outer strings are not strummed | For the chords all strings are strummed |
| Chord progression time |   1    | Very quick chord transitions | Quick chord transitions | Slow chord transitions | Very slow chord transitions |
| Beat difficulty (syncopes/ghostnotes) |   0    | A lot of syncopes or ghostnotes | Some syncopes or ghostnotes | A few syncopes or ghostnotes | No syncopes or ghostnotes |

Table: Proposed rubric for human annotators evaluating the difficulty of playing the chords of a song on the guitar. Although the rubric functions acceptably using the raw scores from the table header, it has even better predictive power when weighting the criteria according to the factor in the weight column. Note that the beat difficulty criterion provides so little extra information that we recommend omitting it (i.e., setting its weight to zero).


If you have any questions, please feel free to contact us.

Citation:
```
@article{velezvasquez2023,
  title   = {Quantifying the Ease of Playing Song Chords on the Guitar},
  author  = {Vélez Vásquez, M.A. and Baelemans, M.C.E. and Driedger, J. and Zuidema, W. and Burgoyne, J.A.},
  booktitle = {Proceedings of the 24th International Society for Music Information Retrieval Conference},
  year    = {2023},
}
```
