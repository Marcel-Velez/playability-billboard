import re

def update_dict(dict_a, dict_b):
    for key, value in dict_a.items():
        if key in dict_b.keys():
            dict_b[key] += value
        else:
            dict_b[key] = value
    return dict_b


def extract_chords_from_line(line):
    if line.find(',') != -1:
        not_already_line = True
        for section in line.split(', '):
            if section.find('|') != -1:
                if not_already_line:
                    not_already_line = False
                    line = section
                else:
                    print(line, section)
    return line


def remove_tempo(line):
    if re.findall("\([0-9]/[0-9]\) ", line):
        tempos = re.findall(" \([0-9]/[0-9]\)", line)
        for tempo in tempos:
            line = line.replace(tempo, '')
    return line


def add_repeated_chords(line):
    if line.find(' . ') != -1:
        split_line = line.split(' ')
        for i in range(len(split_line)):
            if split_line[i] == '.':
                repeat_chord = split_line[i - 1]
                split_line[i] = repeat_chord
        line = split_line[0] + ' '.join(x for x in split_line[0:])
    return line


def remove_inconsistencies(song_list, remove_nl):
    if remove_nl:
        one_long_string = '|' + ''.join(x[1:] for x in song_list[0:])
        one_long_string = one_long_string.replace('||','|')
    else:
        one_long_string = ''.join(x for x in song_list[0:])
        one_long_string = one_long_string.replace('||','|').replace('\n|','\n |').replace('|\n', '| \n')
        if one_long_string[-3:] == '   ':
            one_long_string = one_long_string[:-3]
        if one_long_string[-2:] == '  ' or one_long_string[-2:] == ' \n':
            one_long_string = one_long_string[:-2]
        if one_long_string[-1] == ' ' or one_long_string[-1] == '\n':
            one_long_string = one_long_string[:-1]

    one_long_string = one_long_string.replace('  ', ' ')
    return one_long_string
