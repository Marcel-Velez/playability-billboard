from .song_functions import process_song, get_idf_dict
from .billboard_utils import get_chords_and_annotations, extract_chord_path_and_annotations, extract_indices_subset
from .chord_encodings import extract_possible_symbols, convert_to_encoding
from .ngrams import extract_n_grams, simplify_grams
from .clean_up import update_dict
from .custom_data_functions import convert_to_seq_lab, get_data, retrieve_model, retrieve_encode_dict

split_1_ind = [88, 71, 4, 171, 33, 56, 105, 72, 139, 64, 26, 117, 151, 183, 80, 153, 121, 63, 176, 97]
split_2_ind = [132, 182, 3, 168, 39, 165, 147, 109, 114, 0, 177, 133, 160, 198, 20, 75, 32, 123, 157, 193]
split_3_ind = [167, 57, 100, 195, 27, 118, 128, 190, 103, 141, 162, 186, 13, 81, 55, 8, 36, 135, 50, 2]
split_4_ind = [172, 79, 10, 148, 73, 92, 164, 34, 40, 47, 11, 25, 98, 85, 136, 5, 142, 99, 161, 126]
split_5_ind = [102, 62, 23, 74, 178, 170, 112, 24, 106, 95, 152, 90, 101, 138, 21, 197, 7, 150, 77, 42]
split_6_ind = [35, 187, 52, 194, 115, 16, 116, 6, 53, 45, 15, 67, 131, 185, 191, 1, 96, 28, 19, 66]
split_7_ind = [83, 9, 59, 175, 76, 22, 51, 69, 14, 94, 38, 46, 154, 91, 104, 173, 93, 108, 48, 12]
split_8_ind = [180, 86, 137, 119, 43, 120, 84, 49, 58, 18, 89, 179, 155, 166, 111, 129, 192, 143, 134, 124]
split_9_ind = [17, 159, 82, 61, 41, 54, 184, 130, 199, 188, 181, 78, 60, 169, 174, 37, 29, 140, 145, 163]
split_10_ind = [144, 87, 70, 127, 68, 44, 196, 158, 31, 125, 113, 146, 122, 65, 156, 30, 149, 189, 107, 110]
split_dict = {
    1: split_1_ind,
    2: split_2_ind,
    3: split_3_ind,
    4: split_4_ind,
    5: split_5_ind,
    6: split_6_ind,
    7: split_7_ind,
    8: split_8_ind,
    9: split_9_ind,
    0: split_10_ind,
}


