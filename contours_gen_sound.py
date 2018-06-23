import numpy as np
import numpy.random as rand
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from midiutil.MidiFile import MIDIFile

# EXP_DIR = "D:\Downloads\cssh_contours_sounds\c"
EXP_DIR = "/home/wu079/Dropbox/CSIRO/cognitive_smarthomes_auralisations/exp/cont1/"


###### Naive approach
def pass_dtwdist_thres(all_init_notes, all_contours, threshold):
    l = len(all_contours)
    distmat = np.zeros((l, l))
    for c_j in range(l):
        for c_i in range(c_j+1, l):   # lower triangular matrix WITHOUT repeat
            dist, path = fastdtw(np.array(all_contours[c_i]),
                                    np.array(all_contours[c_j]),
                                    dist=euclidean)
            dist = dist + abs(all_init_notes[c_i] - all_init_notes[c_j])
            distmat[c_i, c_j] = dist
            if dist <= threshold:
                return False

    # for c_j in range(l):    # fill in upper triangular part - NO NEED
    #     for c_i in range(c_j):
    #         distmat[c_i, c_j] = distmat[c_j, c_i]
    return True
    # return True

def naive_generate_all_contours(num, c_len, threshold, iter_num):
    """ naively generate num number of contours with each c_len contours long """
    print("Contours generation iteration:", iter_num)

    all_init_notes = []
    all_contours = []
    for i in range(num):
        init_note, contours = generate_random_one_contours(c_len)
        all_init_notes.append(init_note)
        all_contours.append(contours)
    print(all_init_notes)
    print(all_contours)
    if not pass_dtwdist_thres(all_init_notes, all_contours, threshold):
        all_init_notes, all_contours = generate_all_contours(num, c_len, threshold, iter_num + 1)
        # this will stop when reached recursion stack limit
    return all_init_notes, all_contours
#############

###### 'Re-randomise worst' approach
def generate_random_one_contours(c_len):
    init_note = rand.random_integers(0, 7)  # from discrete uniform dist
    prev_max = 7-init_note   # max contour change
    prev_min = -init_note    # min contour change
    prev_choice = 0
    
    contours = []
    for con_id in range(c_len):
        pos_contour = get_pos_con(prev_max, prev_min, prev_choice)  
        choice = pos_contour[rand.choice(pos_contour)]
        contours.append(choice)

        prev_choice = choice
        prev_max = max(pos_contour)
        prev_min = min(pos_contour)

    return init_note, contours

def get_pos_con(prev_max, prev_min, prev_choice):
    """ get all possible contours """
    new_min, new_max = prev_min-prev_choice, prev_max-prev_choice
    return range(new_min, new_max+1)

def min_below_thres(all_init_notes, all_contours, threshold):
    """ find contours with lowest average distance which also has a distance 
    that's lower than threshold. Return integer of contour index. 
    Return None if not found. """

    l = len(all_contours)

    below_ids = set()   # set of contour id with dist lower than thres
    sum_dists = [0 for _ in range(l)]

    # distmat = np.zeros((l, l))
    for c_j in range(l):
        for c_i in range(c_j+1, l):   # lower triangular matrix WITHOUT repeat
            dist, path = fastdtw(np.array(all_contours[c_i]),
                                np.array(all_contours[c_j]),
                                dist=euclidean)
            dist = dist + abs(all_init_notes[c_i] - all_init_notes[c_j])
            # distmat[c_i, c_j] = dist
            if dist <= threshold:
                below_ids.add(c_i)
                below_ids.add(c_j)
            sum_dists[c_i] += dist
            sum_dists[c_j] += dist

    if not below_ids:    # if empty
        return None
    sum_dists = [dist/(l-1) for dist in sum_dists]
    sum_dists_below = [(sum_dists[i], i) for i in below_ids]
    sum_dists_below.sort()
    return sum_dists_below[0][1]    # return id of least dist contours    

def generate_all_contours(num, c_len, threshold):
    """ generate num number of contours with each c_len contours long """
    iter_num = 0

    all_init_notes = []
    all_contours = []
    for i in range(num):
        init_note, contours = generate_random_one_contours(c_len)
        all_init_notes.append(init_note)
        all_contours.append(contours)

    below_id = min_below_thres(all_init_notes, all_contours, threshold)
    while below_id is not None:
        init_note, contours = generate_random_one_contours(c_len)
        all_init_notes[below_id] = init_note
        all_contours[below_id] = contours
        below_id = min_below_thres(all_init_notes, all_contours, threshold)

        iter_num += 1
        print(iter_num)

    return all_init_notes, all_contours
#############

###### Sounds creation
def get_rel_scale(mood):
    # len of moods must be same as other lists! NOT CHECKED!
    # ALSO not checked value!
    # mood: 1 - happy, 0 - neutral, -1 - sad, -2 - sudden sad
    # right now: 1 == 0 and -1 == -2
    rel_major = [0, 2, 4, 5, 7, 9, 11, 12]
    rel_minor = [0, 2, 3, 5, 7, 8, 10, 12]
    
    if mood == -1 or mood == 0:
        return rel_major
    else:
        return rel_minor

def get_notes(init_note, contours, rel_scale, key):
    # NOT CHECKED len of lists!
    rel_notes = [init_note]
    for cont in contours:
        rel_notes.append(rel_notes[-1] + cont)
    notes = [rel_scale[rn]+key for rn in rel_notes]
    return notes

def contours_to_sounds(all_init_notes, all_contours, keys, moods):
    # len of moods must be same as other lists! NOT CHECKED!
    # ALSO not checked value!
    # mood: 1 - happy, 0 - neutral, -1 - sad, -2 - sudden sad
    # right now: 1 == 0 and -1 == -2
    
    l = len(all_init_notes)

    duration = 1
    tempo = 120
    base_volumn = 100
    time = 0
    track = 0
    channel = 0
    vol = 100

    for i in range(l):
        outfile_name = EXP_DIR + str(i) + '.mid'        
        # generate dir if not exists! TODO

        init_note = all_init_notes[i]
        contours = all_contours[i]
        rel_scale = get_rel_scale(moods[i])
        key = keys[i]
        notes = get_notes(init_note, contours, rel_scale, key)

        MyMIDI = MIDIFile(1, adjust_origin=None)
        MyMIDI.addTempo(track, time, tempo)

        # MyMIDI.addProgramChange(track=0, channel=0, time=time, program=40)
        MyMIDI.addProgramChange(track=0, channel=0, time=time, program=1)

        t_inc = 0
        for note in notes:
            t_inc += 1
            MyMIDI.addNote(track, channel, note, time + t_inc, duration, vol)

        with open(outfile_name, 'wb') as output_file:
            MyMIDI.writeFile(output_file) 


if __name__ == "__main__":

    num = 10
    c_len = 4   # c_len + 1 notes
    threshold = 13  # a little greater than mean dist for 0 note

    # all_init_notes, all_contours = generate_all_contours(num, c_len, threshold)
    # print('all_init_notes =', all_init_notes)
    # print('all_contours =', all_contours)
    
    # 0
    # num = 10, c_len = 7, thres = 20. At iteration 210:
    all_init_notes = [1, 0, 7, 1, 1, 0, 3, 7, 7, 5]
    all_contours = [[4, -5, 7, -7, 6, -5, 2], [7, -2, 0, -3, -1, 6, -4], [-7, 3, 3, 1, -1, -3, 4], [0, 1, 2, 3, 0, -7, 7], [-1, 6, -5, -1, 6, -1, -1], [3, 3, 1, -7, 6, -6, 7], [1, 1, -4, 5, -6, 7, -6], [-2, -1, -2, -1, 6, -6, 4], [-6, 1, -2, 1, 1, 5, -7], [-3, 5, -7, 7, -6, 0, 0]]
    # moods = [1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1]
    moods = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

    # 1
    # num = 10, c_len = 4, thres = 13, At iteration 198 (might get higher):
    # all_init_notes = [1, 1, 2, 7, 0, 4, 6, 7, 7, 4]
    # all_contours = [[4, 2, 0, -6], [6, -7, 4, -2], [-2, 4, -4, 0], [-1, -1, -4, 6], [5, -2, -3, 6], [-4, 2, 1, 3], [-6, 7, -7, 3], [-4, -1, 5, -6], [-7, 5, 1, -5], [3, -7, 0, 7]]
    # # moods = [1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1]
    # # moods = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # moods = [1 for _ in range(10)]

    # 
    # all_init_notes = [6, 2, 2, 1, 1, 5, 6, 6]
    # all_contours = [[-5, 1, 3, -3, 1, -2, 5], [5, -6, 0, -1, 5, 0, -4], [0, 2, 2, -3, -1, -1, 0], [4, -2, 3, -4, 2, 2, -6], [3, -4, 1, 6, -7, 1, 6], [-2, 0, -3, 7, -4, -3, 6], [-4, -2, 7, -1, -6, 5, 0], [1, -5, 2, -1, -1, 1, 1]]
    # moods = [1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1]

    keys = [60 for _ in range(len(all_contours))]    # key of all sounds
    # keys = [24+i*7 for i in range(len(all_contours))]   # may be change key to depend on mood
    # keys = [60, 67, 60, 53, 60, 48, 60, 36, 72, 65]
    contours_to_sounds(all_init_notes, all_contours, keys, moods)