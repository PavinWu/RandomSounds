import numpy as np

import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import pickle
import os.path

def generate_all_contours(c_len):
    all_contours = [[] for note in range(7+1)]
    for init_note in range(7+1):
        # li = len(all_contours) - 1
        prev_max = 7-init_note   # max contour change
        prev_min = -init_note    # min contour change
        prev_choice = 0
        pos_contours = get_pos_con(prev_max, prev_min, prev_choice)
        _gen_aux(all_contours[init_note], [], pos_contours, c_len)
    return all_contours

def _gen_aux(all_note_contours, partial_contours, pos_contours, c_len):
    if len(partial_contours) < c_len:
        prev_max = pos_contours[-1]     # max(pos_contours)
        prev_min = pos_contours[0]      # min(pos_contours)
        for contour in pos_contours:
            x = list(partial_contours)
            x.append(contour)
            prev_choice = contour
            new_pos_contours = get_pos_con(prev_max, prev_min, prev_choice)
            _gen_aux(all_note_contours, x, new_pos_contours, c_len)         
    else:
        all_note_contours.append(partial_contours)
            
def get_pos_con(prev_max, prev_min, prev_choice):
    """ get all possible contours """
    new_min, new_max = prev_min-prev_choice, prev_max-prev_choice
    return range(new_min, new_max+1)

# def get_init_notes(ci_ind, cj_ind, num_contours):
#     found_note_i, found_note_j = False, False
#     init_note_i = ci_ind // num_contours
#     init_note_j = cj_ind // num_contours

#     return init_note_i, init_note_j

# def get_all_dtwdist(flat_all_contours, num_contours):
#     # cum_nums_contours = [0]
#     # for num in nums_contours:
#     #     cum_nums_contours.append(num+cum_nums_contours[-1])

#     ln = len(flat_all_contours)
#     distmat = np.zeros((ln, ln))
#     for ci_ind in range(ln):
#         for cj_ind in range(ln):
#             distmat[ci_ind, cj_ind], path = fastdtw(
#                                 np.array(flat_all_contours[ci_ind]),
#                                 np.array(flat_all_contours[cj_ind]),
#                                 dist=euclidean)
#             # init_note_i, init_note_j = get_init_notes(ci_ind, cj_ind, cum_nums_contours)
#             init_note_i, init_note_j = get_init_notes(ci_ind, cj_ind, num_contours)
#             distmat[ci_ind, cj_ind] += abs(init_note_i - init_note_j)
#             print(ci_ind, cj_ind)
#     return distmat

def get_repeat_threshold(c_len):
    thres = 0
    for i in range(c_len, 0, -1):
        thres += 8**(i-1)
    return thres, 8**c_len

def get_all_dtwdist_eff(flat_all_contours, c_len):
    """ more efficient version (much less repeated calculation) """
    ln = len(flat_all_contours)
    distmat = np.zeros((ln, ln))
    thres, num_contours = get_repeat_threshold(c_len)
    print(thres, num_contours)

    for cj_ind in range(ln):
        print(cj_ind)
        for ci_ind in range(cj_ind, ln):  # lower triangular matrix
            # init_note_i, init_note_j = get_init_notes(ci_ind, cj_ind, num_contours)

            if (ci_ind < thres + num_contours or ci_ind % 8 == 0) and \
                (cj_ind < thres + num_contours or cj_ind % 8 == 0):
                distmat[ci_ind, cj_ind], path = fastdtw(
                                    np.array(flat_all_contours[ci_ind]),
                                    np.array(flat_all_contours[cj_ind]),
                                    dist=euclidean)
            elif ci_ind > thres + num_contours and ci_ind % 8 != 0:
                distmat[ci_ind, cj_ind] = distmat[ci_ind-thres, cj_ind] + 1           
                # + 1 since current dist is one note apart
            else:
                distmat[ci_ind, cj_ind] = distmat[ci_ind-thres, cj_ind-thres]
                # not + 1 since current dist is same dist from prev note
                # as entry in referred position
    
    for cj_ind in range(ln):    # make symmetric matrix
        for ci_ind in range(cj_ind):
            distmat[ci_ind, cj_ind] = distmat[cj_ind, ci_ind]

    return distmat

# def plot_surface3d(X, Y, Z):
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#     fig.colorbar(surf, shrink=0.5, aspect=5)

if __name__ == "__main__":
    # len of contour (== number of notes in a tune - 1)
    c_len = 4
    all_contours = generate_all_contours(c_len)

    flat_all_contours = []
    for note_contours in all_contours:
        flat_all_contours.extend(note_contours)
    
    filename = "distmat0.pkl"
    # filename = "distmat.pkl"
    if not os.path.isfile(filename):
        distmat = get_all_dtwdist_eff(flat_all_contours[0:8**c_len], c_len)
        # distmat = get_all_dtwdist_eff(flat_all_contours, c_len)
        with open(filename, 'wb') as f:
            pickle.dump(distmat, f)
    else:
        with open(filename, 'rb') as f:
            distmat = pickle.load(f)

    print("mean:", distmat.mean())
    print("max:", distmat.max(), "at [", distmat.argmax()//4096, ",", distmat.argmax()%4096, "]")
    print("min:", distmat.min())
    
    flat_distmat = distmat.flatten().astype(int)
    count_arr = np.bincount(flat_distmat)
    print("count:", count_arr)
    plt.hist(flat_distmat, len(count_arr))
    plt.show()

#   mean: 11.6362603903
#   max: 35.0 with [0,0,0,7] and [7, -7, 7, -7]
#   min: 0.0 
# count: [   4612   27904   97154  217294  398202  598996  825796 1019350 1200876
#  1328470 1415334 1435846 1400560 1312706 1176240 1022238  845916  685576
#   524940  397150  281224  200174  132560   89830   56746   36172   20762
#    12092    6286    3330    1558     856     306     124      26      10]