import tensorflow as tf
import os
import keras.backend as K
import hickle as hkl
import numpy as np
import argparse
from DeepSilencer import DeepSilencer
from sklearn.utils import shuffle
from Loading_data import seq_to_kspec,num2acgt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSilencer: Newly developed deep learning model to predict silencers')
    parser.add_argument('--train_data', '-d', type=str, help='input train data path',default = '../data/train_mat.hkl')
    parser.add_argument('--test_data', '-t', type=str, help='input test data path',default= '../data/test_mat.hkl')
    parser.add_argument('--model_name', '-m', type=str, default='../model/kmer_seq.h5', help='Model name to save')
    parser.add_argument('--gpu', '-g', default='0', type=str, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for repeat results')
    parser.add_argument('--learning_rate','-lr', type=float, default=1e-3,help='Learning rate for training')


    args = parser.parse_args()
    seed = args.seed
    lr = args.learning_rate
    modelname = args.model_name

    # Load data and create label
    train_mat = hkl.load(args.train_data).astype(int)
    test_mat = hkl.load(args.test_data).astype(int)
    train_label = [1]*1600+[0]*1600
    test_label = [1]*400+[0]*400
    # contact the train data and test data
    whole_mat = np.vstack(train_mat,test_mat)
    whole_label = train_label + test_label
    whole_data,whole_label = shuffle(whole_mat, whole_label, random_state=0)
    whole_data = whole_data.reshape(-1,4,200,1)
    # gpu
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Obtain kmer information of data
    K = 5
    whole_data_kmer = []
    for ind in range(whole_data.shape[0]):
        seq = ''
        for i in np.argmax(whole_data[ind].reshape(4,200),axis=0):
            seq += num2acgt[i]
        whole_data_kmer.append(seq_to_kspec(seq,K=K))
    whole_data_kmer = np.array(whole_data_kmer).reshape(-1,4**K)
    # Model
    deepsilencer = DeepSilencer()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    # train the DeepSilencer and save the model
    deepsilencer.fit(whole_data,whole_data_kmer,whole_label,filename= modelname,learning_rate=lr)
