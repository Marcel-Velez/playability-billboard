from .ngrams import extract_n_grams, simplify_grams

from .clean_up import extract_chords_from_line, remove_tempo, add_repeated_chords, remove_inconsistencies


def get_idf_dict(chords_list, n_gram=1, simplified=True):
    idf_dict = {}
    for file in chords_list:
        long_string = process_song(file[0])
        n_gram_dict = extract_n_grams(long_string, n_gram=n_gram)

        if simplified:
            n_gram_dict = simplify_grams(n_gram_dict)

        for key in n_gram_dict.keys():
            if key in idf_dict.keys():
                idf_dict[key] += 1
            else:
                idf_dict[key] = 1

    return idf_dict


def process_song(unprocessed_chords_file, remove_nl=True):
    processed_list = []

    for line in unprocessed_chords_file:
        if not line[0].isnumeric():
            continue
        if line.find('\t') != -1:
            line = line.split('\t')[1]

        line = extract_chords_from_line(line)

        if line.find('\n') != -1 and remove_nl:
            line = line.replace('\n', '')

        elif line.find('\n') == -1 and not remove_nl:
            line += ' \n '

        line = remove_tempo(line)  # some songs have tempo changes which are removed
        line = add_repeated_chords(line)  # billboard corpus replaces repeated chords with the '.' character

        if line[-1] != '|':
            if line.find('x') != -1:
                x_index = line.find('x')
                for _ in range(int(line[x_index + 1:])):
                    if remove_nl:
                        processed_list.append(line[:x_index])
                    else:
                        processed_list.append(line[:x_index] + ' \n ')
            elif line.find('->') != -1:
                processed_list.append(line.replace('->',''))
            elif line in ["silence\n", "end\n", "Z\n", "Z \n", "silence", "end", "Z", "Z "]:
                pass
            elif line[:9] in ["# tonic: ", "# metre: ", "Z, applau", "Z, talkin", "Z', talki"]:
                pass
            elif not remove_nl and line[-2:] == '\n ':
                processed_list.append(line)
            elif not remove_nl and line[-1] == '\n':
                processed_list.append(line + ' ')
            else:
                print(f"new case: '{line}'")
        elif line.find('x') != -1:
            print(line)
        else:
            processed_list.append(line)

    one_long_string = remove_inconsistencies(processed_list, remove_nl)

    return one_long_string

