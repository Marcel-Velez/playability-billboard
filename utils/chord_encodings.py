from chordify_json_extension import all_chords, convert2chordify
from utils import process_song


def encode_character(line):
    possible_chars = []
    for char in line:
        possible_chars.append(char)
    return possible_chars


def encode_dotsplit(line, create_dict=False):
    if create_dict:
        line = [symbol for symbol in line.split(' ') if symbol != '']
    converted_chords = []
    for cho in line:
        if cho.find(':') != -1:
            converted_chords += cho.split(':')
        else:
            converted_chords.append(cho)

    return converted_chords


def encode_guitardiagram(line, create_dict=False):
    if create_dict:
        line = [anno for anno in line.split(' ') if anno != '']

    converted_chords = ''
    for symbol in line:
        if symbol in ['|', '\n', 'N', '*', '']:
            converted_chords += symbol + ' '
        else:
            try:
                converted = convert2chordify(symbol)
                to_add = all_chords[converted[0]]['guitar']
                converted_chords += str(to_add)
            except:
                print("could not convert: ", symbol)

    if create_dict:
        converted_chords = [char for char in converted_chords]

    return converted_chords


UNK_PAD_OFFSET = 2
N_FINGERS = 5
N_STRINGS = 6
FINGER_STRING_INDICES_OFFSET = N_FINGERS * N_STRINGS
FRETS_PER_STRING_OFFSET = 26

def process_guitar_chord(chord):
    converted_chord = []
    for nth_string, string in enumerate(chord):
        if string.find(":") != -1:
            finger_index = UNK_PAD_OFFSET + (nth_string * (N_FINGERS + FRETS_PER_STRING_OFFSET)) + int(string.split(':')[1]) -1 # adjust for 0 indexing, finger indices range from 1-5, but we reserve 0-4 for the fingers
            string_fret_index = UNK_PAD_OFFSET + ((nth_string+1) * N_FINGERS) + (nth_string * FRETS_PER_STRING_OFFSET) + int(string.split(':')[0]) + 1 # 0th index is reserved for not playing the string and 1st index is reserved for playing the string open
            converted_chord.append(string_fret_index)
            converted_chord.append(finger_index)
        elif string == 'o':
            string_fret_index = UNK_PAD_OFFSET + ((nth_string+1) * N_FINGERS) + (nth_string * FRETS_PER_STRING_OFFSET) + 1
            converted_chord.append(string_fret_index)
        elif string == 'x':
            string_fret_index = UNK_PAD_OFFSET + ((nth_string+1) * N_FINGERS) + (nth_string * FRETS_PER_STRING_OFFSET)
            converted_chord.append(string_fret_index)
        else:
            raise ValueError("weird value in guitar chord: ", string)

    return converted_chord


def default_return_val():
    """the default value for unknown symbols in the encoding dictionary.
       as a function because lambda functions are not pickleable."""
    return 1
from collections import defaultdict
import torch.nn.functional as F
import torch
def encode_guitar_test(line, create_dict=False):
    if create_dict:
        line = [anno for anno in line.split(' ') if anno != '']

    possible_symbols = guitar_test_symbols()
    encode_dict = {k: i + 2 for i, k in enumerate(set(possible_symbols))}
    encode_dict = defaultdict(default_return_val, encode_dict)
    encode_dict_size = len(encode_dict) + 2

    converted_chords = []
    for symbol in line:
        if symbol in ['|', '\n', 'N', '*']:
            symbol_int = encode_dict[symbol]
            ohe_symbol = F.one_hot(torch.tensor(symbol_int).to(torch.int64), encode_dict_size).float()
            # ohe_symbol = [0] * len(encode_dict)
            converted_chords.append(ohe_symbol)
        elif symbol == '':
            continue
        else:
            # try:
            converted = convert2chordify(symbol)
            to_add = all_chords[converted[0]]['guitar']
            processed_chord = process_guitar_chord(to_add)

            chord_ints = [encode_dict[symbol] if isinstance(symbol, str) else symbol for symbol in processed_chord]
            ohe_chord = F.one_hot(torch.tensor(chord_ints).to(torch.int64), encode_dict_size).float()
            max_ohe_chord = torch.max(ohe_chord, dim=0)[0]
            converted_chords.append(max_ohe_chord)
            # except:
            #     print("could not convert: ", symbol)

    # if create_dict:
    #     converted_chords = [char for char in converted_chords]
    #     print(converted_chords)

    return torch.stack(converted_chords)


def guitar_test_symbols():
    possible_symbols = []
    possible_symbols += list(range(FINGER_STRING_INDICES_OFFSET)) # for the fingers
    possible_symbols += list(range(FINGER_STRING_INDICES_OFFSET, ((N_STRINGS * FRETS_PER_STRING_OFFSET) + FINGER_STRING_INDICES_OFFSET)))  # for the frets
    possible_symbols += ['|', '\n', 'N', '*']  # for the rest of the annotations
    return possible_symbols

def extract_possible_symbols(dataset, chord_encoding):
    possible_symbols = []
    if chord_encoding == 'guitar_test':
        possible_symbols = guitar_test_symbols()
        return possible_symbols
    for file_path in dataset:
        with open(file_path, 'r') as f:
            annotations = f.readlines()
        cleaned_song = process_song(annotations, remove_nl=False)
        if chord_encoding == 'dotsplit':
            possible_symbols += encode_dotsplit(cleaned_song, True)
        elif chord_encoding == 'char':
            possible_symbols += encode_character(cleaned_song)
        elif chord_encoding == 'guitardiagram':
            possible_symbols += encode_guitardiagram(cleaned_song, True)
        elif chord_encoding == 'guitar_test':
            # possible_symbols += encode_guitar_test(cleaned_song, True)
            # possible_symbols = guitar_test_symbols()
            pass
        else:
            raise ValueError(f"Unknown chord encoding: {chord_encoding}")
    return possible_symbols

def convert_to_encoding(song, chord_encoding, encoding_dict):
    if chord_encoding != 'char':
        song = [symbol for symbol in song.split(' ') if symbol != '']
        if chord_encoding == 'dotsplit':
            song = encode_dotsplit(song)
        elif chord_encoding == 'guitardiagram':
            song = encode_guitardiagram(song)
        elif chord_encoding == 'guitar_test':
            song = encode_guitar_test(song)
            return song

    encoded_song = [encoding_dict[symbol] for symbol in song]

    return encoded_song
