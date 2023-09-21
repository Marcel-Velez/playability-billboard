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


def extract_possible_symbols(dataset, chord_encoding):
    possible_symbols = []
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

    encoded_song = [encoding_dict[symbol] for symbol in song]

    return encoded_song
