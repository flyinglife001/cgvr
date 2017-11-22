#coding:utf-8

from run_utils import grid_generate,run
import sys
import time

#generate the parameter config

#dt = /home/vision/data/svrg/w8a
#name = cg
#lr = 0.01
#max_iter = 20
#seed = 13
#M = 3
#q = 4
#L = 1
#m = 50


dt_root = '/home/vision/data/cgvr/'

def converge(dname, loss, l_value):
    #dname = 'ijcnn1'
    #dname = 'covtype'
    #print l_value
    fname = '%s/%s-%s-test.txt' %(loss,dname,l_value[-1])
    result_f = open(fname,'w',0)
    max_iter = 25

    cfg_dict = {}

    #sgd config file
    # config = {
    #     'max_iter':[str(max_iter)],
    #     'lr':['1e-3','1e-4','1e-5'],
    #     'seed':['1'],
    #     'loss':[loss],
    #     'lambda':[l_value]
    # }
    # cfg_dict['sgd'] = config
    #
    # config = {
    #     'max_iter':[str(max_iter)],
    #     'seed':['1'],
    #     'loss':[loss],
    #     'lambda':[l_value]
    # }
    # cfg_dict['cg'] = config
    #
    # config = {
    #     'max_iter':[str(max_iter)],
    #     'lr':['1e-3','1e-4','1e-5'],
    #     'seed':['1'],
    #     #'m':['50'], # empricial settings
    #     'm':['5','10','50','100','150'],
    #     'loss':[loss],
    #     'lambda':[l_value]
    # }
    # cfg_dict['svrg'] = config
    #
    # config = {
    #     'max_iter':[str(max_iter)],
    #     'lr':['1e-3','1e-4','1e-5'],
    #     'seed':['1'],
    #     #'m':['50'],
    #     'm':['5','10','50','100','150'],
    #     'L':['10'],
    #     'loss':[loss],
    #     'lambda':[l_value]
    # }
    # cfg_dict['s_lbfgs'] = config

    config = {
        'max_iter':[str(max_iter)],
        'seed':['1'],
        #'m':['50'],
        'm':['5','10','50','100','150'],
        'lr':['1e-3'],
        'loss':[loss],
        'L':['1'],
        'lambda':[l_value]
    }
    cfg_dict['cgvr'] = config

    for name,config in cfg_dict.items()[:1]:
        for input in grid_generate(config):
            input['dt'] = dname
            input['name'] = name
            input['task'] = 'converge'

            print input

            cmd_str = '/home/vision/cppfile/cgvr/main '
            for k,v in input.items():
                cmd_str += str(k) + ' ' + str(v) + ' '

            print cmd_str

            result = run(cmd_str)
            last_l = ''
            for l in result:
                print l
                last_l = l

            #read the running results
            result_f.write(last_l + '\n')

    result_f.close()

def generalization(loss,dname):
    #dname = 'ijcnn1'
    #dname = 'covtype'
    #print l_value
    fname = '%s/%s-acc-test.txt' %(loss,dname)
    result_f = open(fname,'w',0)
    max_iter = 25

    cfg_dict = {}

    #sgd config file
    config = {
        'max_iter':[str(max_iter)],
        'lr':['1e-3','1e-4','1e-5'],
        'seed':['1'],
        'loss':[loss],
        #'lambda':['1e2','1e1','1','1e-1','1e-2']
        'lambda':['1e-1','5e-2','1e-2','8e-3','5e-3']
    }
    cfg_dict['sgd'] = config

    config = {
        'max_iter':[str(max_iter)],
        'seed':['1'],
        'loss':[loss],
        #'lambda':['1e2','1e1','1','1e-1','1e-2']
        'lambda':['1e-1','5e-2','1e-2','8e-3','5e-3']
    }
    cfg_dict['cg'] = config

    config = {
        'max_iter':[str(max_iter)],
        'lr':['1e-3','1e-4','1e-5'],
        'seed':['1'],
        'm':['50'], # empricial settings
        #'m':['5','10','50','100','150'],
        'loss':[loss],
        #'lambda':['1e2','1e1','1','1e-1','1e-2']
        'lambda':['1e-1','5e-2','1e-2','8e-3','5e-3']
    }
    cfg_dict['svrg'] = config

    config = {
        'max_iter':[str(max_iter)],
        'lr':['1e-3','1e-4','1e-5'],
        'seed':['1'],
        'm':['50'],
        #'m':['5','10','50','100','150'],
        'L':['10'],
        'loss':[loss],
        #'lambda':['1e2','1e1','1','1e-1','1e-2']
        'lambda':['1e-1','5e-2','1e-2','8e-3','5e-3']
    }
    cfg_dict['s_lbfgs'] = config

    config = {
        'max_iter':[str(max_iter)],
        #'max_iter':['5','10','15','20','25'],
        'seed':['1'],
        #'m':['10'],
        'm':['50'],
        'lr':['1e-3'],
        'loss':[loss],
        'L':['1'],
        #'lambda':['1e2','1e1','1','1e-1','1e-2']
        'lambda':['1e-1','5e-2','1e-2','8e-3','5e-3']
    }
    cfg_dict['cgvr'] = config

    for name,config in cfg_dict.items():
        for input in grid_generate(config):
            input['name'] = name
            input['dt'] = dname
            # if dname.startswith('a'):
            #     input['dim'] = 123
            #
            # if dname.startswith('w'):
            #     input['dim'] = 300

            input['task'] = 'generalization'

            print input

            cmd_str = '/home/vision/cppfile/cgvr/main '
            for k,v in input.items():
                cmd_str += str(k) + ' ' + str(v) + ' '

            print cmd_str

            result = run(cmd_str)
            last_l = ''
            for l in result:
                print l
                last_l = l

            #read the running results
            result_f.write(last_l + '\n')

    result_f.close()


#is_classified = True
is_classified = False

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'no loss param ...'
        exit(-1)

    if not is_classified:
        #datasets = ['gisette']
        #datasets = ['w8a']
        datasets = ['a9a']
        #datasets = ['HIGGS']
        #datasets = ['breast-cancer']
        #datasets = ['ijcnn1']
        #loss = ['hinge']
        #loss = ['ridge']
        loss = [sys.argv[1]]
        #l_value = ['0']
        #l_value = ['1e2'] #lambda value
        #datasets = ['a9a','covtype','w8a','ijcnn1','SUSY','HIGGS']
        #loss = ['ridge','logistic','hinge']
        l_value = ['1e-4','1e-6'] #lambda value

        for l in loss:
            for v in l_value:
                for d in datasets:
                    converge(d,l,v)

    else:
        #datasets = ['HIGGS']
        #datasets = ['covtype']
        #datasets = ['ijcnn1']
        #datasets = ['breast-cancer']
        loss = [sys.argv[1]]
        #loss = ['ridge']
        #loss = ['logistic']

        #loss = ['ridge','logistic','hinge']

        # datasets = []
        # for i in range(1,10):
        #     datasets.append('a' + str(i) + 'a')
        #
        # for i in range(1,9):
        #     datasets.append('w' + str(i) + 'a')

        datasets = ['a9a','covtype','ijcnn1','w8a','SUSY','HIGGS']
        for l in loss:
            for d in datasets:
                generalization(l,d)






