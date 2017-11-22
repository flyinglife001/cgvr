
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC,LinearSVC
import json
import numpy as np
from run_utils import grid_generate,run,estimate_auc
import time
from sklearn import metrics
import sys

dt_root = '/home/vision/data/cgvr/'

def scikit_exp(dname,lbd_set, f, is_linear = False):
    """
    :param dname: 
    :param dim: 
    :param lbd_set: lambda parameter, C = 1/(2 lambda)
    :return: 
    """
    trainX, trainY = load_svmlight_file(dt_root + dname + '.r')
    validX, validY = load_svmlight_file(dt_root + dname + '.v')
    testX, testY = load_svmlight_file(dt_root + dname + '.t')
    print 'load ',dname,trainX.shape,validX.shape,testX.shape

    for l in lbd_set:
        c = 1/(2*l)
        #print 'c:',c

        start = time.time()
        if is_linear:
            clf = LinearSVC(random_state=0,loss = 'hinge',C = c)
            print 'use linear svc for classification'
        else:
            clf = SVC(random_state=0,kernel='linear',C = c)
            print 'use libsvm for classification ..'
        clf.fit(trainX,trainY)
        elapse = time.time() - start

        config = {}

        pred = clf.predict(validX)
        fpr, tpr, thresholds = metrics.roc_curve(validY, pred)
        config['valid'] = metrics.auc(fpr, tpr)
        #print estimate_auc(pred,validY),config['valid']

        pred = clf.predict(testX)
        fpr, tpr, thresholds = metrics.roc_curve(testY, pred)
        config['test'] = metrics.auc(fpr, tpr)

        config['name'] = 'svm'
        config['loss'] = 'hinge'
        config['dt'] = dname
        config['lambda'] = l
        config['time'] = elapse
        str = json.dumps(config)
        print str
        f.write(str + '\n')

def libsvm_exp(dname,lbd_set, f, is_linear = False):

    config = {}
    dt = dt_root + dname
    config['loss'] = 'hinge'
    config['name'] = 'svm'
    config['dt'] = dname
    for l in lbd_set:
        c = 1/(2*float(l))
        config['lambda'] = l

        #training process, output probability
        if is_linear:
            #primal l2-regularization square l1-loss (2) or l2-regularization logistic regression(0)
            cmd_str = './liblinear/build/lin-train -s 1 -c %s %s.r model'%(c,dt)
        else:
            cmd_str = './build/svm-train -h 0 -c %s %s.r model'%(c,dt)

        result = run(cmd_str)
        last_l = ''
        for l in result:
            print l
            last_l = l
        config['time'] = last_l

        #valid process
        if is_linear:
            cmd_str = './liblinear/build/lin-predict %s.v model output'%dt
        else:
            cmd_str = './build/svm-predict %s.v model output'%dt
        result = run(cmd_str)
        for l in result:
            print l
            last_l = l

        config['valid'] = last_l

        # fpr, tpr, thresholds = metrics.roc_curve(validY, pred)
        # print 'scikit',metrics.auc(fpr, tpr)
        # print config['valid']

        #test process
        if is_linear:
            cmd_str = './liblinear/build/lin-predict %s.t model output'%dt
        else:
            cmd_str = './build/svm-predict %s.t model output'%dt
        result = run(cmd_str)
        for l in result:
            print l
            last_l = l
        config['test'] = last_l


        str = json.dumps(config)
        print str
        f.write(str + '\n')


if __name__ == '__main__':
    linear = False
    if len(sys.argv) > 1:
        linear = True

    # datasets = []
    # for i in range(1,10):
    #     datasets.append('a' + str(i) + 'a')
    # for i in range(1,9):
    #     datasets.append('w' + str(i) + 'a')

    #datasets = ['ijcnn1']
    #datasets = ['a1a']
    datasets = ['a9a','covtype','w8a','ijcnn1','SUSY','HIGGS']

    lbd_set = ['1e-1','5e-2','1e-2','8e-3','5e-3']
    #lbd_set = [1e2,1e1,1e0,1e-1,1e-2]
    if linear:
        f = open('svm_results-lindual','w',0)
    else:
        f = open('svm_results-libsvm','w',0)

    dim = 0
    for d in datasets:
        libsvm_exp(d,lbd_set,f,linear)
        #scikit_exp(d,lbd_set,f,linear)

    f.close()
