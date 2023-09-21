def ngram_extractor(chopped, n):
    without_bars = [n for n in chopped if n not in ['|', 'N', '.', '*', '->', '']]
    n_gram_count_dict = {}

    for i in range(len(without_bars) - n + 1):
        cur_tuple = tuple(without_bars[i:i + n])
        if len(cur_tuple) < n:
            print(f'too short tuple: "{cur_tuple}" while n: {n}')
        if cur_tuple in n_gram_count_dict:
            n_gram_count_dict[cur_tuple] += 1
        else:
            n_gram_count_dict[cur_tuple] = 1

    return n_gram_count_dict


def extract_n_grams(long_string, n_gram=None):
    count_dict = {}
    chopped_up = long_string.split(' ')
    for token in chopped_up:
        if token == '':
            continue
        if token[0] == '(':
            continue
        if token in count_dict:
            count_dict[token] += 1
        else:
            count_dict[token] = 1
    if n_gram:
        return ngram_extractor(chopped_up, n=n_gram)
    else:
        uni_gram = ngram_extractor(chopped_up, n=1)
        bi_gram = ngram_extractor(chopped_up, n=2)
        tri_gram = ngram_extractor(chopped_up, n=3)
        quad_gram = ngram_extractor(chopped_up, n=4)
        return uni_gram, bi_gram, tri_gram, quad_gram



def simplify_grams(n_grams_count):
    # simplify chords in the n-grams in case we did not have the chord finger positions
    clean_count = {}
    for chords, value in n_grams_count.items():
        cur_chord = []
        for single_chord in chords:
            if single_chord[0] not in ['&', '(', '']:
                cur_chord.append(simplify_chord(single_chord))
            elif single_chord[0] == '&':
                cur_chord.append(single_chord)
        if cur_chord:
            cur_chord = tuple(cur_chord)
            if cur_chord in clean_count.keys():
                clean_count[cur_chord] += value
            else:
                clean_count[cur_chord] = value

    return clean_count


def simplify_chord(chord):
    split = chord.split(':')
    try:
        root, stripped = split[0], split[1]
    except:
        print(f"[simplify_chord]Could not process: '{chord}'")
        pass

    if stripped.find('/') != -1:
        split_back = stripped.split('/')
        stripped = split_back[0]

    if stripped.find('(') != -1:
        split_back = stripped.split('(')
        stripped = split_back[0]

    return root + ':' + stripped


def simplify_chord(chord):
    split = chord.split(':')
    try:
        root, stripped = split[0], split[1]
    except:
        print(f"[simplify_chord]Could not process: '{chord}'")
        pass

    if stripped.find('/') != -1:
        split_back = stripped.split('/')
        stripped = split_back[0]

    if stripped.find('(') != -1:
        split_back = stripped.split('(')
        stripped = split_back[0]


    return root + ':' + stripped


