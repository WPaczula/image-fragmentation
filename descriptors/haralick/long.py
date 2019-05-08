import numpy as np
import itertools as it

def get_long_feature(glcm, levels, texture_feature):
    glcm = glcm[:, :, 0, 0]
    if texture_feature == 'sum_of_square_variance':
        i_raw = np.empty_like(glcm)
        i_raw[...] = np.arange(glcm.shape[0])
        i_raw = np.transpose(i_raw)
        i_minus_mean = (i_raw - glcm.mean()) ** 2
        res = np.apply_over_axes(np.sum, i_minus_mean * glcm, axes=(0, 1))[0][0]
    elif texture_feature == 'inverse_difference_moment':
        j_cols = np.empty_like(glcm)
        j_cols[...] = np.arange(glcm.shape[1])
        i_minus_j = ((j_cols - np.transpose(j_cols)) ** 2) + 1
        res = np.apply_over_axes(np.sum, glcm / i_minus_j, axes=(0, 1))[0][0]
    elif texture_feature == 'sum_average':
        # Slow
        tuple_array = np.array(
            list(it.product(list(range(levels)), list(range(levels)))),
            dtype=(int, 2))
        index = [list(map(tuple, tuple_array[tuple_array.sum(axis=1) == x])) for x in
                    range(levels)]
        p_x_y = [glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))]
        res = np.array(p_x_y * np.array(range(len(index)))).sum()
    elif texture_feature == 'sum_variance':
        # Slow
        tuple_array = np.array(
            list(it.product(list(range(levels)), list(range(levels)))),
            dtype=(int, 2))
        index = [list(map(tuple, tuple_array[tuple_array.sum(axis=1) == x])) for x in
                    range(levels)]
        p_x_y = [glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))]
        sum_average = np.array(p_x_y * np.array(range(len(index)))).sum()
        res = ((np.array(range(len(index))) - sum_average) ** 2).sum()
    elif texture_feature == 'sum_entropy':
        # Slow
        tuple_array = np.array(
            list(it.product(list(range(levels)), list(range(levels)))),
            dtype=(int, 2))
        index = [list(map(tuple, tuple_array[tuple_array.sum(axis=1) == x])) for x in
                    range(levels)]
        p_x_y = [glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))]
        res = (p_x_y * np.log(p_x_y + np.finfo(float).eps)).sum() * -1.
    elif texture_feature == 'difference_variance':
        # Slow
        tuple_array = np.array(
            list(it.product(list(range(levels)), list(np.asarray(range(levels)) * -1))),
            dtype=(int, 2))
        index = [list(map(tuple, tuple_array[np.abs(tuple_array.sum(axis=1)) == x])) for x in
                    range(levels)]
        p_x_y = [glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))]
        sum_average = np.array(p_x_y * np.array(range(len(index)))).sum()
        res = ((np.array(range(len(index))) - sum_average) ** 2).sum()
    else:
        # texture_feature == 'difference_entropy':
        # Slow
        tuple_array = np.array(
            list(it.product(list(range(levels)), list(np.asarray(range(levels)) * -1))),
            dtype=(int, 2))
        index = [list(map(tuple, tuple_array[np.abs(tuple_array.sum(axis=1)) == x])) for x in
                    range(levels)]
        p_x_y = [glcm[tuple(np.moveaxis(index[y], -1, 0))].sum() for y in range(len(index))]
        res = (p_x_y * np.log(p_x_y + np.finfo(float).eps)).sum() * -1.
    return res
