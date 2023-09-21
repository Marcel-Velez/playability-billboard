import os

import numpy as np
import pandas as pd
import torch
import argparse

from data import retrieve
from tqdm import tqdm
import pickle as pkl
from utils import get_data, convert_to_seq_lab, retrieve_model, retrieve_encode_dict




def main(args):
    if args['accelerator'] == 'gpu':
        device = torch.device('cuda')
    elif args['accelerator'] == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # takes care of loading csv and possibly skipping rows and batching of csv in case of large dataset
    custom_data = get_data(args)

    print("loaded data")
    only_seq, only_lab = convert_to_seq_lab(custom_data)

    print("extracted lab files, chord sequences and applied transpositions")

    data = retrieve('billboard_salami')
    custom = retrieve('custom')

    print("make models")
    # for p in ['CFP']:  # ,'CFD','UC','RHC','CPT','BD','R','weighted_total']:
    #     all_data = data(
    #         batch_size=2,  # config['batch_size'],
    #         num_workers=0,  # config['num_workers'],
    #         chord_encoding=args['chord_encoding'],
    #         target=p,
    #         target_data_path='./data/Annotations.csv'
    #
    #     )
    #     all_data.custom_setup(1)
    # for i in range(10):
    #     all_data.custom_setup(i)
    #     dict_fold = all_data.encode_dict
    #     with open(f"{args['chord_encoding']}_encode_dict_fold_{i}.pkl", "wb") as f:
    #         pkl.dump(dict_fold, f)
    # exit()
    print("converted data")

    cat_difficulties = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

    for cat in tqdm(range(8)):
        fold_scores = []

        for fold in tqdm(range(10), leave=False):

            encode_dict = retrieve_encode_dict(args['chord_encoding'], fold)
            train_loader = custom(only_seq,
                                  chord_encoding=args['chord_encoding'],
                                  encode_dict=encode_dict).loader(batch_size=args['batch_size'])

            model = retrieve_model(args['model_type'], input_size=(len(encode_dict) + 2), cat=cat, fold=fold, chord_encoding=args['chord_encoding'], device=device)
            data_set_scores = []

            for i, batch in tqdm(enumerate(train_loader), leave=False):
                batch, batch_len = batch
                batch = batch.to(device)
                output = model(batch, batch_len)

                if i == 0:
                    data_set_scores = output.argmax(1).cpu().numpy()
                else:
                    data_set_scores = np.concatenate((data_set_scores, output.argmax(1).cpu().numpy()))

            fold_scores.append(data_set_scores)

        fold_scores = np.asarray(fold_scores)
        cat_difficulties[cat] += list(fold_scores.mean(0))
        with open(args['save_file'], 'wb') as f:
            pkl.dump(cat_difficulties, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument("--skiprows", type=int, default=-1)
    parser.add_argument('--nrows', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--chord_encoding', type=str, default="char")
    parser.add_argument('--file_path', type=str, default="data/custom_data.csv")
    parser.add_argument('--save_file', type=str, default="predictions.pkl")
    parser.add_argument('--model_type', type=str, default="DeepGRU")
    parser.add_argument('--accelerator', type=str, choices=['gpu', 'mps'])
    parser.add_argument('--delimiter', type=str, default="\t")
    args = vars(parser.parse_args())

    main(args)