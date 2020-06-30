import tensorflow as tf
import os
gpu_use = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_use
import hickle as hkl
from DeepSilencer import DeepSilencer
from sklearn.utils import shuffle
from Loading_data import seq_to_kspec
import numpy as np
train_mat = hkl.load('../data/train_mat.hkl').astype(int)
test_mat = hkl.load('../data/test_mat.hkl').astype(int)
train_label = [1]*1600+[0]*1600
test_label = [1]*400+[0]*400
train_data,train_label = shuffle(train_mat, train_label, random_state=0)
test_data,test_label = shuffle(test_mat, test_label, random_state=0)
train_data = train_data.reshape(-1,4,200,1)
test_data = test_data.reshape(-1,4,200,1)
num2acgt = {0:'A',
            1:'C',
            2:'G',
            3:'T'}
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
deepsilencer = DeepSilencer()
tf.set_random_seed(1234)
np.random.seed(1234)
deepsilencer.fit(train_data,train_data_kmer,train_label)
test_posterior_probability = deepsilencer.predict(test_data,test_data_kmer)
predict_label = np.copy(test_posterior_probability)
predict_label[predict_label>=0.5]=1
predict_label[predict_label<0.5]=0
test_label = np.array(test_label)
hkl.dump(predict_label,'../result/self_projection/predict_label.hkl')
hkl.dump(test_posterior_probability,'../result/self_projection/test_posterior_probability.hkl')
hkl.dump(test_label,'../result/self_projection/test_label.hkl')