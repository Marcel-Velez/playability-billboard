import os
import pandas as pd


def extract_chord_path_and_annotations(csv_path, cols2extract, delimiter=";"):
    csv_file = pd.read_csv(csv_path, delimiter=delimiter)
    extracted_rows = []
    try:
        for (track_obj, cat1, cat2, cat3, cat4, cat5, cat6, cat7, total) in csv_file[cols2extract].values:
            extracted_rows.append(
                (
                    os.path.join(track_obj),
                    os.path.join((track_obj.split('_')[0] + '_full.lab').replace('chords','lab_files')),
                    (cat1, cat2, cat3, cat4, cat5, cat6, cat7),
                    total)
            )
    except:
        raise ValueError("Error in extracting rows from csv file, available cols: ", csv_file.columns, " cols2extract: ", cols2extract)
    return extracted_rows


def get_chords_and_annotations(chord_paths_and_annotations, no_duplicates=False):
    chords_and_annotations = []

    for filename in chord_paths_and_annotations:
        f = filename[0] # 0th index is file_path

        if os.path.isfile(os.path.join(f)):
            with open(os.path.join(f), 'r', encoding="utf-8") as opened_file:
                chords = opened_file.readlines()
            chords_and_annotations.append((chords, *filename[1:]))

    if no_duplicates:
        unique_song_names = list(set([''.join(x for x in filen[0][:5]) for filen in chords_and_annotations]))
        clean_files = []
        for filen in chords_and_annotations:
            if ''.join(x for x in filen[0][:5]) in unique_song_names:
                clean_files.append(filen)
                unique_song_names.remove(''.join(x for x in filen[0][:5]))
        chords_and_annotations = clean_files

    return chords_and_annotations


def extract_indices_subset(indices, chords_and_annotations):
    subset = []
    for index, chords_and_stuff in enumerate(chords_and_annotations):
        if index in indices:
            subset.append(chords_and_stuff)
    return subset