# Quantifying the Ease of Playing Song Chords on the Guitar

This repository contains the code and data associated with the paper "Quantifying the Ease of Playing Song Chords on the Guitar" In this paper, we focus on rhythm guitar, a popular instrument among amateur musicians, and propose a rubric-based "playability" metric to formalize the spectrum of difficulty.
- We propose an interpretable guitar playability metric;
- an extension of the Billboard dataset of 200 playability annotated songs tested for reliability;
- a rule-based baseline for our playability metric.

<img src="playability.png" width="500">

## Contents
1. `data/`: Contains the dataset of 200 songs from the McGill Billboard dataset, along with the playability annotations.
2. `code/`: Includes the code for the rule-based baseline, LSTM, and GRU models, used to predict the rubric categories automatically.

## Usage
1. **Data**: The dataset of 200 songs and their corresponding playability annotations can be found in the `data/` directory. Each song is labeled according to the rubric-based metric, capturing various aspects of playability.
2. **Code**: The `code/` directory contains the implementation of the rule-based baseline, LSTM, and GRU models. These models utilize chord symbols and textual representations of guitar tablature to predict the rubric categories automatically. Instructions for running and evaluating the models can be found in the code documentation.

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
