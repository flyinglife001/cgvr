
from subprocess import Popen, PIPE
import numpy as np
from sklearn import metrics

def run(command):
    process = Popen(command, stdout=PIPE, shell=True)
    while True:
        line = process.stdout.readline()
        if not line:
            break
        yield line.strip()

def grid_generate(valid_dict):
    """
    :param valid_dict: it's form is params with str items.
    :return: paras dict with str items. 
    """

    #idx is reaching to the top of list or not.
    #print 'idx',idx_L

    idx_L = [0]*len(valid_dict)

    while True:
        is_finished = True
        comb_tuple = {}
        max_add_idx = -1

        i = 0
        for k,S in valid_dict.items():
            if idx_L[i] == len(S):
                max_add_idx = i
            elif idx_L[i] != len(S) - 1:
                is_finished = False
                comb_tuple[k] = S[idx_L[i]]
            else:
                comb_tuple[k] = S[idx_L[i]]

            i += 1

        #compute the next combination
        for i in range(max_add_idx + 1):
            idx_L[i] = 0
        idx_L[max_add_idx + 1] += 1

        if len(comb_tuple) == len(valid_dict):
            yield comb_tuple

        if is_finished:
            break



def estimate_auc(pred, Y):
    n = Y.shape[0]
    seq = zip(pred,Y)
    seq = sorted(seq,key=lambda x:x[0])
    #print seq
    ranks = []
    last_v = -1e8

    #[b_idx, e_idx)
    b_idx = 0
    e_idx = 0
    i = 0
    for v,y in seq:
        ranks.append(i + 1)
        if abs(v - last_v) > 1e-6:
            e_idx = i
            base_value = ranks[b_idx]
            for k in xrange(b_idx,e_idx):
                ranks[k] = base_value + (e_idx - b_idx - 1)/2.0
            b_idx = e_idx

        last_v = v
        i += 1

    #process the last range
    e_idx = n
    base_value = ranks[b_idx]
    for k in xrange(b_idx,e_idx):
        ranks[k] = base_value + (e_idx - b_idx - 1)/2.0

    #print ranks

    sum_pos_ranks = 0.0
    n_pos = 0
    #compute the accumulate value of positive values.
    for i in range(n):
        label = seq[n - 1 - i][1]
        if int(label) == 1:
            #print n - 1 - i, ranks[n - 1 - i]
            sum_pos_ranks += ranks[n - 1 - i]
            n_pos += 1

    if n_pos == 0 or n_pos == n:
        return 1

    return (sum_pos_ranks - (n_pos + 1.0)*n_pos/2)/(n_pos*(n - n_pos))



# pred = np.array([0.2,0.3,0.3,0.1])
# y = np.array([1,0,1,0])
# print estimate_auc(pred,y)
# fpr, tpr, thresholds = metrics.roc_curve(y, pred)
# print metrics.auc(fpr, tpr)


# y = np.array([0, 0, 1, 1])
# pred = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(y, pred)
# print metrics.auc(fpr, tpr)
# print estimate_auc(pred,y)






