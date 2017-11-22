
from sklearn.datasets import load_svmlight_file,dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np

"""
split  the original traning into train and validation sets.
"""

dt_root = '/home/vision/data/cgvr/'


def split(fname):
    X, Y = load_svmlight_file(dt_root + fname)

    #print np.unique(Y)

    Y = Y.reshape(-1,1)#convert it into 2-D array

    #preprocess the datasets
    scaler = MinMaxScaler((-1,1))
    X = X.todense()
    X = scaler.fit_transform(X)
    #print X
    Y = scaler.fit_transform(Y)
    #print Y
    #print np.unique(Y)

    f_va = open(dt_root + fname + '.v','w')
    f_tr = open(dt_root + fname + '.r','w')
    f_te = open(dt_root + fname + '.t','w')

    i = 0
    for x,y in zip(X,Y):
        if i%3 == 0:
            f = f_te
        elif i%5 == 0:
            f = f_va
        else:
            f = f_tr

        x = x.reshape(1,-1)

        dump_svmlight_file(x,y,f,zero_based=False)

        i += 1

    f_te.close()
    f_va.close()
    f_tr.close()

# datasets = []
# for i in range(1,10):
#     datasets.append('a' + str(i) + 'a')
#
# for i in range(1,9):
#     datasets.append('w' + str(i) + 'a')

#datasets = ['covtype']
#datasets = ['a9a']
datasets = ['a9a','covtype','w8a','ijcnn1','SUSY','HIGGS']

for d in datasets:
    print d
    split(d)
