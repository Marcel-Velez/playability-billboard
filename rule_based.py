from chordify_json_extension import all_chords, convert2chordify
from utils.thresholding_and_grouping import *
from collections import Counter, defaultdict
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import os
import math
from utils import process_song, extract_n_grams, simplify_grams, update_dict

def guitar2distance(guitardiagram):
    """
    used for criteria 1 chord finger positioning
    Get the number of frets between the lowest and highest played fret given a guitardiagram
    Guitardiagram example: A:major = ['x','o','2:1','2:2','2:3','o']
    :returns integer of fret distance
    """
    distance = 0

    lowest_fret = 9999
    highest_fret = 0

    for string in guitardiagram:
        if string == 'x':
            pass
        elif string == 'o':
            distance += 0
        else:
            fret, finger = string.split(':')
            fret, finger = int(fret), int(finger)
            if fret < lowest_fret:
                lowest_fret = fret
            if fret > highest_fret:
                highest_fret = fret
    if lowest_fret == 9999:
        lowest_fret = 0

    return highest_fret - lowest_fret


def get_finger_distance_dict(gram_dict, simplify_factor=.25):
    distance_dict = {}

    for key, value in gram_dict.items():
        new_chord, simplified = convert2chordify(key[0])
        guitar_diagram = all_chords[new_chord]['guitar']

        finger_distance = guitar2distance(guitar_diagram)

        distance_dict[key] = (1 + simplified * simplify_factor) * finger_distance

    return distance_dict


def guitar2fingers(guitardiagram):
    fingers = {1: 0, 2: 0, 3: 0, 4: 0}
    bar = False
    finger_used = 0

    for string in guitardiagram:
        if string == 'x':
            pass
        elif string == 'o':
            pass
        else:
            fret, finger = string.split(':')
            fret, finger = int(fret), int(finger)
            fingers[finger] += 1

    for key, value in fingers.items():
        if value > 1:
            bar = True
        if value >= 1:
            finger_used += 1
    return finger_used, bar


def get_chord_fingering_difficulty_dict(gram_dict, finger_factor=.5, bar_factor=.5, simplify_factor=.25, ):
    distance_dict = {}
    for key, value in gram_dict.items():
        new_chord, simplified = convert2chordify(key[0])
        guitar_diagram = all_chords[new_chord]['guitar']

        fingers_used, bar = guitar2fingers(guitar_diagram)

        distance_dict[key] = (1 + simplified * simplify_factor) * (fingers_used * finger_factor + bar * bar_factor)

    return distance_dict


def guitar2strings(guitardiagram):
    difficulties = {}
    X = True
    O = False

    guitardiagram = [True if x == 'x' else False for x in guitardiagram]

    difficulties[(X, X, X, X, X, X)] = 0  # none are struck
    difficulties[(O, O, O, O, O, O)] = 0  # ALL ARE STRUCK

    difficulties[(X, O, O, O, O, O)] = 1
    difficulties[(O, O, O, O, O, X)] = 1
    difficulties[(X, X, O, O, O, O)] = 1
    difficulties[(O, O, O, O, X, X)] = 1
    difficulties[(X, X, X, O, O, O)] = 1
    difficulties[(O, O, O, X, X, X)] = 1
    difficulties[(X, X, X, X, O, O)] = 1
    difficulties[(O, O, X, X, X, X)] = 1
    difficulties[(X, X, X, X, X, O)] = 1
    difficulties[(O, X, X, X, X, X)] = 1

    difficulties[(X, O, O, O, O, X)] = 1
    difficulties[(X, O, O, O, X, X)] = 1
    difficulties[(X, O, O, X, X, X)] = 1
    difficulties[(X, O, X, X, X, X)] = 1

    difficulties[(X, X, O, O, O, X)] = 1
    difficulties[(X, X, O, O, X, X)] = 1
    difficulties[(X, X, O, X, X, X)] = 1

    difficulties[(X, X, X, O, O, X)] = 1
    difficulties[(X, X, X, O, X, X)] = 1

    difficulties[(X, X, X, X, O, X)] = 1

    difficulties[(O, X, O, O, O, O)] = 2
    difficulties[(O, O, X, O, O, O)] = 2
    difficulties[(O, O, O, X, O, O)] = 2
    difficulties[(O, O, O, O, X, O)] = 2

    difficulties[(X, O, X, O, O, O)] = 2
    difficulties[(X, O, O, X, O, O)] = 2
    difficulties[(X, O, O, O, X, O)] = 2

    difficulties[(X, X, O, X, O, O)] = 2
    difficulties[(X, X, O, O, X, O)] = 2
    difficulties[(X, X, X, O, X, O)] = 2

    difficulties[(O, O, O, X, O, X)] = 2
    difficulties[(O, O, X, O, O, X)] = 2
    difficulties[(O, X, O, O, O, X)] = 2

    difficulties[(O, O, X, O, X, X)] = 2
    difficulties[(O, X, O, O, X, X)] = 2
    difficulties[(O, X, O, X, X, X)] = 2

    # ONE OR MORE INNER STRINGS NOT STRUCK

    difficulties[(O, X, X, O, O, O)] = 3
    difficulties[(O, X, O, X, O, O)] = 3
    difficulties[(O, X, O, O, X, O)] = 3

    difficulties[(O, O, X, X, O, O)] = 3
    difficulties[(O, O, X, O, X, O)] = 3
    difficulties[(O, O, O, X, X, O)] = 3

    difficulties[(O, O, O, X, X, O)] = 3

    difficulties[(X, O, X, X, O, O)] = 3
    difficulties[(X, O, X, O, X, O)] = 3
    difficulties[(X, O, O, X, X, O)] = 3

    difficulties[(X, X, O, X, X, O)] = 3

    return difficulties[tuple(guitardiagram)]


def score_cat_generic(gram_dict, idf_dict, metric='average', distance_dict=None):
    gram_dict = copy.copy(gram_dict)
    distance_metric = 0
    total_chords = 0

    for key, value in gram_dict.items():
        total_chords += value

    if metric == 'average':
        weighted_dict = {}
        for key, value in gram_dict.items():
            weighted_dict[key] = value / total_chords

        for key, value in weighted_dict.items():
            if distance_dict == None:
                distance_metric += idf_dict[key]
            else:
                distance_metric += value * distance_dict[key]
    elif metric == 'max':
        max_dist = 0
        for key, value in gram_dict.items():
            if distance_dict == None:
                cur_dist = idf_dict[key]
            else:
                cur_dist = distance_dict[key]

            if cur_dist > max_dist:
                max_dist = cur_dist
        distance_metric = max_dist

    elif metric == 'tfidf':
        weighted_dict = {}
        for key, value in gram_dict.items():
            weighted_dict[key] = value / total_chords

        for key, value in weighted_dict.items():
            if distance_dict == None:
                distance_metric += value * idf_dict[key]
            else:
                distance_metric += value * idf_dict[key] * distance_dict[key]

    return distance_metric


def score_cat_5(lab_file, idf_dict, cat_one_metric, tf_idf=True):
    timings = []
    if not os.path.isfile(lab_file):
        return -1
    with open(lab_file, 'r') as lab:
        lab_inhoud = lab.readlines()
    for line in lab_inhoud:
        if line == '\n':
            continue
        begin, end, rest = line.split('\t',2)
        if rest == 'N\n' or rest == 'X\n':
            continue
        begin, end = float(begin), float(end)
        difference = end-begin
        if tf_idf:
            if rest == 'F:1/1\n':
                rest = 'F:1'
            elif rest == "F:(b5,11)\n":
                rest = "F:1(b5,11)"
            elif rest == "B:1/1\n":
                rest = "B:1"
            elif rest == "E:1/1\n":
                rest = "E:1"
            elif rest == "F#:1/1\n":
                rest = "F#:1"
            elif rest == "G:1/1\n":
                rest = "G:1"
            elif rest == "D:1/1\n":
                rest = "D:1"
            elif rest == "C:1/1\n":
                rest = "C:1"
            elif rest == "A:1/1\n":
                rest = "A:1"
            elif rest == "Bb:1/1\n":
                rest = "Bb:1"
            elif rest == "Eb:1/1\n":
                rest = "Eb:1"
            elif rest == "Ab:1/1\n":
                rest = "Ab:1"
            elif rest == "C#:1/1\n":
                rest = "C#:1"
            elif rest == "Gb:1/1\n":
                rest = "Gb:1"
            elif rest == "A:(11)\n":
                rest = "A:1(11)"
            elif rest == "Bb:(3)\n":
                rest = "Bb:1(3)"
            elif rest == "D:(#5)\n":
                rest = "D:1(#5)"
            elif rest == "C#:(b3,b7,11,9)\n":
                rest = "C#:1(b3,b7,11,9)"
            elif rest == "A:(11,9)\n":
                rest = "A:1(11,9)"
            elif rest == "E:(b5,b7,3)/b5\n":
                rest = "E:1(b5,b7,3)/b5"
            elif rest == "Db:1/1\n":
                rest = "Db:1"
            elif rest == "A:(b7)/b7\n":
                rest = "A:1(b7)/b7"
            elif rest == "A:(13)/6\n":
                rest = "A:1(13)/6"
            elif rest == "F:(3)\n":
                rest = "F:1(3)"
            elif rest == "Bb:(b5,b7,3)\n":
                rest = "Bb:1(b5,b7,3)"

            timings.append(difference * idf_dict[tuple([rest.replace('\n',''),])])
        else:
            timings.append(difference)

    timings = np.asarray(timings)
    if cat_one_metric == "mean":
        return timings.mean()
    elif cat_one_metric == "median":
        return np.median(timings)
    elif cat_one_metric == "var":
        return timings.var()
    elif cat_one_metric == "min":
        return timings.min()
    else:
        return timings.max()


def score_cat_6(grams, lab_file, idf_dict, cat_one_metric, tf_idf=True):
    var = score_cat_5(lab_file, idf_dict, "var", tf_idf=False)
    mean = score_cat_5(lab_file, idf_dict, "mean", tf_idf=False)
    same_as_median = 0
    total = 0

    if not os.path.isfile(lab_file):
        return -1

    with open(lab_file, 'r') as lab:
        lab_content = lab.readlines()
    for line in lab_content:
        if line == '\n':
            continue
        begin, end, rest = line.split('\t', 2)
        if rest == 'N\n' or rest == 'X\n':
            continue

        if rest[0] == '&':
            print("hallo")
        begin, end = float(begin), float(end)
        difference = end - begin
        if (difference - mean) >= var:
            same_as_median += 1
        total += 1

    return same_as_median / total


def score_cat_7(big_string_new_lines):
    split = big_string_new_lines.split('\n')
    try:
        split.remove('')
    except:
        pass
    counted = Counter(split)
    return len(counted)



def get_right_hand_complexity_dict(gram_dict, string_factor=.5, simplify_factor=.25, ):
    distance_dict = {}
    for key, value in gram_dict.items():
        new_chord, simplified = convert2chordify(key[0])
        guitar_diagram = all_chords[new_chord]['guitar']

        string_difficulty = guitar2strings(guitar_diagram)

        distance_dict[key] = (1 + simplified * simplify_factor) * (string_difficulty * string_factor)

    return distance_dict

def score_cat_one(cat_num,
                  files_to_iterate,
                  idf_dict,
                  cat_1_distance,
                  cat_2_distance,
                  cat_4_distance,
                  cat_strategy='average',
                  cat_5_strat='max',
                  cat_5_tfidf=False,
                  cat_6_strat='max',
                  cat_6_tfidf=False,
                  clean=True,
                  ):
    first_cat_scores = {0: [], 1: [], 2: [], 3: []}

    for (chords, lab_file, sub_scores, total) in files_to_iterate:

        long_string = process_song(chords)
        long_string_nl = process_song(chords, remove_nl=False)

        new_uni, new_bi, new_tri, new_quad = extract_n_grams(long_string)

        if clean:
            new_uni = simplify_grams(new_uni)

        if cat_num == 1:
            cat_1_score = score_cat_generic(new_uni, idf_dict, metric=cat_strategy, distance_dict=cat_1_distance)
        elif cat_num == 2:
            cat_1_score = score_cat_generic(new_uni, idf_dict, metric=cat_strategy, distance_dict=cat_2_distance)
        elif cat_num == 3:
            cat_1_score = score_cat_generic(new_uni, idf_dict, metric=cat_strategy)
        elif cat_num == 4:
            cat_1_score = score_cat_generic(new_uni, idf_dict, metric=cat_strategy, distance_dict=cat_4_distance)
        elif cat_num == 5:
            cat_1_score = score_cat_5(lab_file, idf_dict, cat_5_strat, cat_5_tfidf)  # default max is best
        elif cat_num == 6:
            cat_1_score = score_cat_6(new_uni, lab_file, idf_dict, cat_6_strat, cat_6_tfidf)  # wonky
        elif cat_num == 7:
            cat_1_score = score_cat_7(long_string_nl)
        else:
            raise ValueError("category number doesnot exist")

        first_cat_scores[sub_scores[cat_num - 1]].append(cat_1_score)

    return first_cat_scores


def evaluate_score_cat_1(chords_and_annotations, all_idf_dict, all_uni_gram):
    best_simpl_cat_1 = 0
    best_simpl_cat_1_factor = 0
    best_bounds_cat_1 = None

    for i in range(1, 5, ):
        cat_1_dict = get_finger_distance_dict(all_uni_gram, simplify_factor=i)
        scores = score_cat_one(
            1,
            chords_and_annotations,
            all_idf_dict,
            cat_1_dict,
            None,
            None,
            cat_strategy="tfidf",
            cat_5_strat='max',
            cat_5_tfidf=False,
            cat_6_strat='max',
            cat_6_tfidf=False,
            clean=False,
        )

        cur_cat_bounds = get_boundaries(scores, 'brute')
        cur_score, cur_grid = create_grid(scores, cur_cat_bounds, False, False)
        if cur_score > best_simpl_cat_1:
            best_simpl_cat_1 = cur_score
            best_simpl_cat_1_factor = i
            best_bounds_cat_1 = cur_cat_bounds

    return best_bounds_cat_1, best_simpl_cat_1_factor


def evaluate_score_cat_2(chords_and_annotations, all_idf_dict, all_gram):
    best_simpl_cat_2 = 0
    best_simpl_cat_2_factor = 0
    best_bar_cat_2_factor = 0
    best_finger_cat_2_factor = 0
    best_bounds_cat_2 = None

    for i in range(1, 5, ):
        for j in range(1, 5, ):
            for k in np.arange(.25, 1, .25):
                cat_2_dict = get_chord_fingering_difficulty_dict(all_gram, finger_factor=k, bar_factor=j,
                                                                 simplify_factor=i, )
                scores = score_cat_one(
                    2,
                    chords_and_annotations,
                    all_idf_dict,
                    None,
                    cat_2_dict,
                    None,
                    cat_strategy="tfidf",
                    cat_5_strat='max',
                    cat_5_tfidf=False,
                    cat_6_strat='max',
                    cat_6_tfidf=False,
                    clean=False,
                )

                cur_cat_bounds = get_boundaries(scores, 'brute')
                cur_score, cur_grid = create_grid(scores, cur_cat_bounds, False, False)
                if cur_score > best_simpl_cat_2:
                    best_simpl_cat_2 = cur_score
                    best_simpl_cat_2_factor = i
                    best_bar_cat_2_factor = j
                    best_finger_cat_2_factor = k
                    best_bounds_cat_2 = cur_cat_bounds

    return best_bounds_cat_2, best_simpl_cat_2_factor, best_bar_cat_2_factor, best_finger_cat_2_factor


def evaluate_score_cat_3(chords_and_annotations, all_idf_dict):
    scores = score_cat_one(
        3,
        chords_and_annotations,
        all_idf_dict,
        None,
        None,
        None,
        cat_strategy="tfidf",
        cat_5_strat='max',
        cat_5_tfidf=False,
        cat_6_strat='max',
        cat_6_tfidf=False,
        clean=False,
    )

    cur_cat_bounds = get_boundaries(scores, 'brute')
    cur_score, cur_grid = create_grid(scores, cur_cat_bounds, False, False)

    return cur_cat_bounds


def evaluate_score_cat_4(chords_and_annotations, all_idf_dict, all_gram):
    best_simpl_cat_4 = 0
    best_simpl_cat_4_factor = 0
    best_string_cat_4_factor = 0
    best_bounds_cat_4 = 0
    for i in np.arange(.25, 2, .25):
        for j in np.arange(.25, 2, .25):
            cat_4_dict = get_right_hand_complexity_dict(all_gram, string_factor=j, simplify_factor=i, )
            scores = score_cat_one(
                4,
                chords_and_annotations,
                all_idf_dict,
                None,
                None,
                cat_4_dict,
                cat_strategy="tfidf",
                cat_5_strat='max',
                cat_5_tfidf=False,
                cat_6_strat='max',
                cat_6_tfidf=False,
                clean=False,
            )

            cur_cat_bounds = get_boundaries(scores, 'brute')
            cur_score, cur_grid = create_grid(scores, cur_cat_bounds, False, False)
            if cur_score > best_simpl_cat_4:
                best_simpl_cat_4 = cur_score
                best_simpl_cat_4_factor = i
                best_string_cat_4_factor = j
                best_bounds_cat_4 = cur_cat_bounds

    return best_bounds_cat_4, best_simpl_cat_4_factor, best_string_cat_4_factor


def evaluate_score_cat_5(chords_and_annotations, all_idf_dict):
    ## get best cat_5 settings
    best_5_score = 0
    best_5_strat = None
    best_5_bool = None
    best_5_bounds = None
    for strat in ['max', 'min', 'mean', 'median', 'var']:
        for tf_bool in [True, False]:
            scores = score_cat_one(
                5,
                chords_and_annotations,
                all_idf_dict,
                None,
                None,
                None,
                cat_strategy="tfidf",
                cat_5_strat=strat,
                cat_5_tfidf=tf_bool,
                cat_6_strat='max',
                cat_6_tfidf=False,
                clean=False,

            )
            cur_cat_bounds = get_boundaries(scores, 'brute')
            cur_score, cur_grid = create_grid(scores, cur_cat_bounds, False, False)
            if cur_score > best_5_score:
                best_5_score = cur_score
                best_5_strat = strat
                best_5_bool = tf_bool
                best_5_bounds = cur_cat_bounds

    return best_5_bounds, best_5_bool, best_5_strat


def evaluate_score_cat_6(chords_and_annotations, all_idf_dict):
    ## cat 6 score
    scores = score_cat_one(
        6,
        chords_and_annotations,
        all_idf_dict,
        None,
        None,
        None,
        cat_strategy="tfidf",
        cat_5_strat=None,
        cat_5_tfidf=None,
        cat_6_strat='',
        cat_6_tfidf=False,
        clean=False,

    )
    cur_cat_bounds = get_boundaries(scores, 'brute')
    cur_score, cur_grid = create_grid(scores, cur_cat_bounds, False, False)

    return cur_cat_bounds


def evaluate_score_cat_7(chords_and_annotations, all_idf_dict):
    ## cat 7 score
    scores = score_cat_one(
        7,
        chords_and_annotations,
        all_idf_dict,
        None,
        None,
        None,
        cat_strategy="tfidf",
        cat_5_strat=None,
        cat_5_tfidf=None,
        cat_6_strat=None,
        cat_6_tfidf=None,
        clean=False,

    )
    cur_cat_bounds = get_boundaries(scores, 'brute')
    cur_score, cur_grid = create_grid(scores, cur_cat_bounds, False, False)

    return cur_cat_bounds


def train_all(chords_and_annotations, all_idf_dict, all_gram):
    cat_1_bounds, cat_1_simplify_factor = evaluate_score_cat_1(chords_and_annotations, all_idf_dict, all_gram)
    cat_2_bounds, cat_2_simplify_factor, cat_2_bar_factor, cat_2_finger_factor = evaluate_score_cat_2(
        chords_and_annotations, all_idf_dict, all_gram)
    cat_3_bounds = evaluate_score_cat_3(chords_and_annotations, all_idf_dict)
    cat_4_bounds, cat_4_simplify_factor, cat_4_string_factor = evaluate_score_cat_4(chords_and_annotations,
                                                                                    all_idf_dict, all_gram)
    cat_5_bounds, cat_5_tfidf, cat_5_strat = evaluate_score_cat_5(chords_and_annotations, all_idf_dict)
    cat_6_bounds = evaluate_score_cat_6(chords_and_annotations, all_idf_dict)
    cat_7_bounds = evaluate_score_cat_7(chords_and_annotations, all_idf_dict)

    cfg = {
        'cat_1_bounds': cat_1_bounds,
        'cat_2_bounds': cat_2_bounds,
        'cat_3_bounds': cat_3_bounds,
        'cat_4_bounds': cat_4_bounds,
        'cat_5_bounds': cat_5_bounds,
        'cat_6_bounds': cat_6_bounds,
        'cat_7_bounds': cat_7_bounds,

        'cat_1_simplify_factor': cat_1_simplify_factor,

        'cat_2_simplify_factor': cat_2_simplify_factor,
        'cat_2_bar_factor': cat_2_bar_factor,
        'cat_2_finger_factor': cat_2_finger_factor,

        'cat_4_simplify_factor': cat_4_simplify_factor,
        'cat_4_string_factor': cat_4_string_factor,

        'cat_5_tfidf': cat_5_tfidf,
        'cat_5_strat': cat_5_strat,

    }
    return cfg


class RuleModel:
    def __init__(self, cfg, all_uni_gram, idf_dict):
        self.cat_thresholds = {
            1: cfg['cat_1_bounds'],
            2: cfg['cat_2_bounds'],
            3: cfg['cat_3_bounds'],
            4: cfg['cat_4_bounds'],
            5: cfg['cat_5_bounds'],
            6: cfg['cat_6_bounds'],
            7: cfg['cat_7_bounds']
        }

        self.cat_1_simplify_factor = cfg['cat_1_simplify_factor']
        cat_1_dict = get_finger_distance_dict(
            all_uni_gram,
            simplify_factor=self.cat_1_simplify_factor
        )
        self.cat_1_dict = defaultdict(self.cat_1_dict_mean, cat_1_dict)

        self.cat_2_simplify_factor = cfg['cat_2_simplify_factor']
        self.cat_2_bar_factor = cfg['cat_2_bar_factor']
        self.cat_2_finger_factor = cfg['cat_2_finger_factor']
        cat_2_dict = get_chord_fingering_difficulty_dict(
            all_uni_gram,
            finger_factor=self.cat_2_finger_factor,
            bar_factor=self.cat_2_bar_factor,
            simplify_factor=self.cat_2_simplify_factor,
        )
        self.cat_2_dict = defaultdict(self.cat_2_dict_mean, cat_2_dict)

        self.cat_4_simplify_factor = cfg['cat_4_simplify_factor']
        self.cat_4_string_factor = cfg['cat_4_string_factor']
        cat_4_dict = get_right_hand_complexity_dict(
            all_uni_gram,
            string_factor=self.cat_4_string_factor,
            simplify_factor=self.cat_4_simplify_factor,
        )
        self.cat_4_dict = defaultdict(self.cat_4_dict_mean, cat_4_dict)

        self.cat_5_tfidf = cfg['cat_5_tfidf']
        self.cat_5_strat = cfg['cat_5_strat']

        self.idf_dict = copy.deepcopy(idf_dict)

    def cat_1_dict_mean(self):
        return np.mean([value for key, value in self.cat_1_dict.items()])

    def cat_2_dict_mean(self):
        return np.mean([value for key, value in self.cat_2_dict.items()])

    def cat_4_dict_mean(self):
        return np.mean([value for key, value in self.cat_4_dict.items()])

    def mean_defaultdict(self):
        return np.mean([value for key, value in dict.items()])

    def which_class(self, score, category):
        thresholds = self.cat_thresholds[category]
        if score <= thresholds[0]:
            playability = 0
        elif score <= thresholds[1] and score > thresholds[0]:
            playability = 1
        elif score <= thresholds[2] and score > thresholds[1]:
            playability = 2
        elif score > thresholds[2]:
            playability = 3
        else:
            raise ValueError(f'score: {score} wrong, thresholds: {thresholds}')
        return playability

    def classification(self, pred, true):
        predicted = np.asarray(pred)
        y = np.asarray(true)
        zero_good = ((predicted == y) & (0 == y)).sum() / (0 == y).sum()
        one_good = ((predicted == y) & (1 == y)).sum() / (1 == y).sum()
        two_good = ((predicted == y) & (2 == y)).sum() / (2 == y).sum()
        three_good = ((predicted == y) & (3 == y)).sum() / (3 == y).sum()
        all_good = (predicted == y).sum() / y.size
        return zero_good, one_good, two_good, three_good, all_good

    def predict(self, chords, lab_file, partial=False, start_ind=0, len_ind=0):

        long_string = process_song(chords)
        long_string_nl = process_song(chords, remove_nl=False)
        if partial:
            long_string = long_string[start_ind:start_ind+len_ind].split(' ',1)[1].rsplit(' ',1)[0]
            long_string_nl = long_string_nl[start_ind:start_ind+len_ind].split(' ',1)[1].rsplit(' ',1)[0]

        new_uni = extract_n_grams(long_string, n_gram=1)

        cat_1_score = score_cat_generic(new_uni, self.idf_dict, metric="tfidf", distance_dict=self.cat_1_dict)
        cat_2_score = score_cat_generic(new_uni, self.idf_dict, metric="tfidf", distance_dict=self.cat_2_dict)
        cat_3_score = score_cat_generic(new_uni, self.idf_dict, metric="tfidf")
        cat_4_score = score_cat_generic(new_uni, self.idf_dict, metric="tfidf", distance_dict=self.cat_4_dict)
        if not partial:
            cat_5_score = score_cat_5(lab_file, self.idf_dict, self.cat_5_strat, self.cat_5_tfidf)  # default max is best
            cat_6_score = score_cat_6(new_uni, lab_file, self.idf_dict, "", "")  # wonky
        else:
            cat_5_score = -1
            cat_6_score = -1
        cat_7_score = score_cat_7(long_string_nl)

        cat_1_class = self.which_class(cat_1_score, 1)
        cat_2_class = self.which_class(cat_2_score, 2)
        cat_3_class = self.which_class(cat_3_score, 3)
        cat_4_class = self.which_class(cat_4_score, 4)
        cat_5_class = self.which_class(cat_5_score, 5)
        cat_6_class = self.which_class(cat_6_score, 6)
        cat_7_class = self.which_class(cat_7_score, 7)

        if not partial:
            total = cat_1_class * 3 + cat_2_class * 2 + cat_3_class * 3 + cat_4_class * 2 + cat_5_class * 1 + \
                    cat_6_class * 0 + cat_7_class * 2
        else:
            total = -1

        return cat_1_class, cat_2_class, cat_3_class, cat_4_class, cat_5_class, cat_6_class, cat_7_class, total

def boundary_loss(input_torch, target, class_total=False):
    input_torch = nn.Softmax()(input_torch)
    if class_total:
        for index in range(input_torch.size(1)):
            if index == 0:
                losses = input_torch[:,index] * abs(target - index)
            else:
                losses += input_torch[:,index] * abs(target - index)

        return torch.mean(losses)
    else:
        return torch.mean(input_torch[:,0] * target + input_torch[:,1] * abs(target-1) + input_torch[:,2] * abs(target-2) + input_torch[:,3] * abs(target-3))


def evaluate_test_set(model, test_set, classify=False, custom_loss=True):
    NUM_CRIT = 8 # 7 criteria + the weighted sum
    WITHOUT_TOTAL = 7
    pred_scores = [[] for _ in range(NUM_CRIT)]
    true_scores = [[] for _ in range(NUM_CRIT)]

    for chords, lab_file, sub_scores, total in test_set:
        all_preds = model.predict(chords, lab_file)
        for cat_index, pred in enumerate(all_preds):
            pred_scores[cat_index].append(pred)

        for cat_index, true_score in enumerate(sub_scores):
            true_scores[cat_index].append(true_score)
        true_scores[-1].append(total)

    if classify:
        mse = [None for _ in range(NUM_CRIT)]
        rmse = [model.classification(pred_scores[i], true_scores[i]) for i in range(NUM_CRIT)]

    elif custom_loss:
        mse = [None for _ in range(NUM_CRIT)]
        rmse = [boundary_loss(F.one_hot(torch.tensor(pred_scores[i]).to(torch.int64),4).float(),
                              torch.tensor(true_scores[i])) for i in range(WITHOUT_TOTAL)]
        rmse.append(sum(rmse))
    else:
        mse = []
        rmse = []
        for i in range(NUM_CRIT):
            mse.append(np.square(np.subtract(true_scores[i], pred_scores[i])).mean())
            rmse.append(math.sqrt(mse[i]))

    for i in range(NUM_CRIT):
        if i == NUM_CRIT-1:
            print(f"Total Root Mean Square Error: {rmse[i]}  Mean Square Error: {mse[i]}\n")
        else:
            print(f"Criteria: [{i+1}/{NUM_CRIT}] Root Mean Square Error: {rmse[i]}  Mean Square Error: {mse[i]}\n")
    return tuple(rmse[:])


def get_uni_gram(chords_and_annotations):
    all_uni_gram = {}
    for (chords, _, sub_scores, total) in chords_and_annotations:
        long_string = process_song(chords,  remove_nl=True)
        if long_string.find('||') != -1:
            print(chords, '\n\n', long_string)
            break
        new_uni = extract_n_grams(long_string, n_gram=1)
        all_uni_gram = update_dict(new_uni, all_uni_gram)
    return all_uni_gram

