import tensorflow as tf
import os
import keras.backend as K
import hickle as hkl
import numpy as np
import argparse
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='correlation')
    parser.add_argument('--train_data', '-d', type=str, help='input train data path',default = '../data/train_mat.hkl')
    parser.add_argument('--test_data', '-t', type=str, help='input test data path',default= '../data/test_mat.hkl')
    parser.add_argument('--outdir', '-o', type=str, default='./output/correlation/', help='Output path')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for repeat results')
    parser.add_argument('--save_result','-p', type = bool, default = True, help='Save test labels and predicted labels')
    

    args = parser.parse_args()
    
    seed = args.seed
    outdir = args.outdir

    # Load data and create label
    train_mat = hkl.load(args.train_data).astype(int)
    test_mat = hkl.load(args.test_data).astype(int)
    train_label = [1]*1600+[0]*1600
    test_label = [1]*400+[0]*400
    train_data,train_label = shuffle(train_mat, train_label, random_state=0)
    test_data,test_label = shuffle(test_mat, test_label, random_state=0)
    # Model
    model = KNeighborsClassifier(n_neighbors=5)
    # Train and predict
    model.fit(train_data, train_label)
    predict_label = model.predict(test_data)
    test_posterior_probability = model.predict_proba(test_data)
    test_label = np.array(test_label)
    # Save the result
    if args.save_result:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        hkl.dump(predict_label,outdir+ 'predict_label.hkl')
        hkl.dump(test_posterior_probability,outdir+'test_posterior_probability.hkl')
        hkl.dump(test_label,outdir+'test_label.hkl')