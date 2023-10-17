import pandas as pd
import os
import torch
from models import DeepGRU, LSTM
import pickle as pkl

letter2num = {
    'C': 0,
    'C#': 1,
    'Db': 1,
    'D': 2,
    'D#': 3,
    'Eb': 3,
    'E': 4,
    'E#': 5,
    'Fb': 4,
    'F': 5,
    'F#': 6,
    'Gb': 6,
    'G': 7,
    'G#': 8,
    'Ab': 8,
    'A': 9,
    'A#': 10,
    'Bb': 10,
    'B': 11,
    'Cb': 11,
}
num2letter = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'Eb',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'Ab',
    9: 'A',
    10: 'Bb',
    11: 'B',
}


def retrieve_model(model_type, input_size, cat, learning_rate=0.002, chord_encoding='char', device=torch.device('cpu'), fold=-1, max_epochs=20, train=False):
    if model_type == "DeepGRU" or model_type == "gru":
        model = DeepGRU(
            input_size=input_size,
            output_size=4 if cat != 7 else 39,
            learning_rate=learning_rate,
        ).to(device)
        file_name = f"v2_gru_target_{cat + 1}_fold_{fold}_{max_epochs}_{chord_encoding}.pth"
    elif model_type == "lstm":
        model = LSTM(
            input_size=input_size,
            output_size=4 if cat != 7 else 39,
            learning_rate=0.002,
        ).to(device)
        file_name = f"new_lstm_target_{cat + 1}_fold_{fold}_{max_epochs}_{chord_encoding}.pth"
    if train:
        return model
    path = os.path.join('./trained_models/', file_name)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def retrieve_encode_dict(chord_encoding, fold, dir="./trained_models/"):
    file = f"{chord_encoding}_encode_dict_fold_{fold}.pkl"
    path = os.path.join(dir, file)
    with open(path, 'rb') as f:
        encode_dict = pkl.load(f)
    return encode_dict


def get_data(args):
    if args['nrows'] != -1 and args['skiprows'] != -1:
        custom_data = pd.read_csv(args['file_path'], delimiter=args['delimiter'],
                                  skiprows=range(1, args['skiprows']), nrows=args['nrows'])
    elif args['nrows'] != -1 and args['skiprows'] == -1:
        custom_data = pd.read_csv(args['file_path'], delimiter=args['delimiter'],
                                  nrows=args['nrows'])
    else:
        custom_data = pd.read_csv(args['file_path'], delimiter=args['delimiter'])
    return custom_data


def convert_to_seq_lab(data, transpose=True, columns=["chords", "transposed_amount"]):
    """
    extract the chord and transpose amount
    :param data:
    :param columns:
    :return: only_seq: string containing only the symbols, without timing information
    :return: only_lab: string containing lines in the form of: starting_time ending_time chord\n
    """
    only_seq = []
    only_lab = []
    for i, row in data[columns].iterrows():
        better_chords = row[columns[0]].split('\\n')
        better_chords.remove('')
        if transpose and len(columns) == 2:
            x = extract_lab_seq(better_chords, row[columns[1]])
        else:
            x = extract_lab_seq(better_chords, 0)
        only_seq.append(x[0].split('PHRASE'))
        only_lab.append(x[1])

    return only_seq, only_lab


def transpose(chord, transposition):
    if transposition == 0:
        return chord
    if chord == 'N':
        return chord
    root, rest = chord.split(':')
    root_number = letter2num[root]
    root_transposed = (root_number + transposition) % 12
    root_transposed = num2letter[root_transposed]
    return root_transposed + ':' + rest


def extract_lab_seq(chordify_data, transposition):
    start = True
    phrase_counter = 0
    cleaned_sequence = ''
    lab_sequence = ''
    prev = 0
    cur = 0
    for chord in chordify_data:
        chord_info = chord.split(';')
        prev = cur
        cur = int(chord_info[0])
        if start:
            start = False
            cleaned_sequence += f'{chord_info[2]}\t| '
        if cur < prev:
            if phrase_counter >= 4:
                cleaned_sequence += f'|\nPHRASE{chord_info[2]}\t'
                phrase_counter = 0
            cleaned_sequence += '| '
            phrase_counter += 1

        cleaned_sequence += transpose(chord_info[1], transposition) + ' '

        lab_sequence += chord_info[2] + '\t' + chord_info[3] + '\t' + chord_info[1] + '\n'
    cleaned_sequence += '|'
    return cleaned_sequence, lab_sequence
