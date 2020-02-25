import copy

def isPowerOfTwo(x): 
    return (x and (not(x & (x - 1))) )

def get_pyramid_ranges(seqlen):
    S_new = int(seqlen/2)
    ranges = []
    prev_ranges = []
    lvl = 1
    while S_new >= 1:
        cur_ranges = []
        if lvl==1:
            for i in range(S_new):
                cur_ranges.append([2*i, 2*i+1])
        else:
            for i in range(0, len(prev_ranges), 2):
                r = copy.copy(prev_ranges[i])
                r.extend(prev_ranges[i+1])
                cur_ranges.append(r) 
        ranges.extend(cur_ranges)
        prev_ranges = copy.copy(cur_ranges)
        S_new = int(S_new/2)
        lvl+=1
    return ranges