import numpy as np

def get_boundaries(cat_dict, method='brute'):
    boundaries = []

    if method == 'all':
        sorted_all = list(sorted(cat_dict[1] + cat_dict[2] + cat_dict[3] + cat_dict[4]))
        best_classified = 0
        best_thresh = []

        for x_1 in range(len(sorted_all)):
            for x_2 in range(x_1, len(sorted_all)):
                for x_3 in range(x_2, len(sorted_all)):

                    thresh_1 = sorted_all[x_1]
                    thresh_2 = sorted_all[x_2]
                    thresh_3 = sorted_all[x_3]

                    cor_0 = sum(np.asarray(cat_dict[1]) <= thresh_1)
                    cor_1 = len(
                        np.where((np.asarray(cat_dict[2]) <= thresh_2) & (np.asarray(cat_dict[2]) > thresh_1))[0])
                    cor_2 = len(
                        np.where((np.asarray(cat_dict[3]) <= thresh_3) & (np.asarray(cat_dict[3]) > thresh_2))[0])
                    cor_3 = sum(np.asarray(cat_dict[4]) > thresh_3)
                    total = cor_0 + cor_1 + cor_2 + cor_3

                    if total > best_classified:
                        best_classified = total
                        best_thresh = [thresh_1, thresh_2, thresh_3]
        boundaries = best_thresh

    for i in range(1, len(cat_dict.keys())):
        if method == 'brute':
            best_classified = 0
            best_thresh = 0
            for x in list(sorted(cat_dict[i + 1] + cat_dict[i])):
                correct_class = sum(np.asarray(cat_dict[i]) <= x) + sum(np.asarray(cat_dict[i + 1]) > x)
                if correct_class > best_classified:
                    best_classified = correct_class
                    best_thresh = x
            boundaries.append(best_thresh)
        elif method == 'mean':
            bound = (np.asarray(cat_dict[i + 1]).mean() - np.asarray(cat_dict[i]).mean()) * .5 + np.asarray(
                cat_dict[i]).mean()
            boundaries.append(bound)

    return boundaries


def test_thresholds(thresholds, print_all=True):
    for x in range(len(thresholds)):
        for y in range(x, len(thresholds)):
            if thresholds[y] < thresholds[x]:
                if print_all:
                    print(thresholds[y], thresholds[x])
                thresholds[y] = thresholds[x]
    return thresholds


def create_grid(cat_dict, thresholds, print_grid=True, print_test_threshold=True):
    thresholds = test_thresholds(thresholds, print_test_threshold)
    grid = np.zeros((4, 4))
    for i_index, cat in enumerate(cat_dict.keys()):
        cur_values = np.asarray(cat_dict[cat])

        grid[i_index, 0] = sum(cur_values <= thresholds[0])
        grid[i_index, 1] = len(np.where((cur_values <= thresholds[1]) & (cur_values > thresholds[0]))[0])
        grid[i_index, 2] = len(np.where((cur_values <= thresholds[2]) & (cur_values > thresholds[1]))[0])
        grid[i_index, 3] = sum(cur_values > thresholds[2])

    total_classifications = grid.sum().sum()

    if total_classifications > 200:
        print(f"cannot have more classifications than songs, " +
              f"max is 200, found: {total_classifications} with thresholds: {thresholds}")
    if print_grid:
        correct_ones = np.trace(grid) / grid.sum().sum()
        print(f"correctly classified: {correct_ones}, total classifications: {total_classifications}")
        print(grid)
    return np.trace(grid) / grid.sum().sum(), grid

