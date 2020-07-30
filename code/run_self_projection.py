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
    parser.add_argument('--train_data', '-d', type=str, help='input train data path',default = 'train_mat.hkl')
    parser.add_argument('--test_data', '-t', type=str, help='input test data path',default= 'test_mat.hkl')
    parser.add_argument('--outdir', '-o', type=str, default=os.path.dirname(os.getcwd())+'/output/self-projection', help='Output path')
    parser.add_argument('--file_name', '-f', type=str, default='kmer_seq.h5', help='Model name to save')
    parser.add_argument('--gpu', '-g', default='0', type=str, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for repeat results')
    parser.add_argument('--learning_rate','-lr', type=float, default=1e-3,help='Learning rate for training')
    parser.add_argument('--save_result','-p', type = bool, default = True, help='Save test labels and predicted labels')

    args = parser.parse_args()
    seed = args.seed
    lr = args.learning_rate
    filename = args.file_name
    outdir = args.outdir

    # Load data and create label
    train_mat = hkl.load(args.train_data).astype(int)
    test_mat = hkl.load(args.test_data).astype(int)
    train_label = [1]*1600+[0]*1600
    test_label = [1]*400+[0]*400
    train_data,train_label = shuffle(train_mat, train_label, random_state=0)
    test_data,test_label = shuffle(test_mat, test_label, random_state=0)
    train_data = train_data.reshape(-1,4,200,1)
    test_data = test_data.reshape(-1,4,200,1)
    # gpu
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Obtain kmer information of data
    K = 5
    train_data_kmer = []
    for ind in range(train_data.shape[0]):
        seq = ''
        for i in np.argmax(train_data[ind].reshape(4,200),axis=0):
            seq += num2acgt[i]
        train_data_kmer.append(seq_to_kspec(seq,K=K))
    train_data_kmer = np.array(train_data_kmer).reshape(-1,4**K)
    test_data_kmer = []
    for ind in range(test_data.shape[0]):
        seq = ''
        for i in np.argmax(test_data[ind].reshape(4,200),axis=0):
            seq += num2acgt[i]
        test_data_kmer.append(seq_to_kspec(seq,K=K))
    test_data_kmer = np.array(test_data_kmer).reshape(-1,4**K)
    # Model
    deepsilencer = DeepSilencer()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    # train the DeepSilencer and save the model
    deepsilencer.fit(train_data,train_data_kmer,train_label,filename= filename,learning_rate=lr)
    # predict on test data
    test_posterior_probability = deepsilencer.predict(test_data,test_data_kmer)
    predict_label = np.copy(test_posterior_probability)
    predict_label[predict_label>=0.5]=1
    predict_label[predict_label<0.5]=0
    test_label = np.array(test_label)
    # save result
    if args.save_result:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        hkl.dump(predict_label,outdir+ 'predict_label.hkl')
        hkl.dump(test_posterior_probability,outdir+'test_posterior_probability.hkl')
        hkl.dump(test_label,outdir+'test_label.hkl')