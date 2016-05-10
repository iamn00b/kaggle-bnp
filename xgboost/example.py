import xgboost as xgb
import h5py

import sys
import argparse
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

parser = argparse.ArgumentParser(description='Train BNP Paribas claim management data with XGBOOST')
parser.add_argument('--data', 
                      default=CURRENT_DIR + '../data_preprocess/data_preprocessed.h5',
                      help='Data for training and testing (in h5 format)')
parser.add_argument('--output', 
                      default=CURRENT_DIR + 'output.csv',
                      help='output filename (in csv format)')
args = parser.parse_args()

print '== LOADING DATA'
print 'Loading data from file'
h5f = h5py.File(args.data, 'r')
train_input = h5f['train_input'][:]
train_label = h5f['train_label'][:]
test_input = h5f['test'][:]
test_id = h5f['test_id'][:]
h5f.close()

print 'Loading data into XGBoost DMatrix'
dtrain = xgb.DMatrix(train_input, missing=-999, label=train_label)
dtest = xgb.DMatrix(test_input, missing=-999)

print ''
print '== XGBOOST PREPARATION'
param = {'bst:max_depth':2, 'bst:eta':1, 'bst:gamma':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
evallist  = [(dtrain,'train')]

print ''
print '== XGBOOST TRAINING'
num_round = 10
bst = xgb.train( param, dtrain, num_round, evallist )

print ''
print '== XGBOOST PREDICTION'
ypred = bst.predict(dtest)
print ypred

xgb.plot_importance(bst)