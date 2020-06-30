from keras.layers import Input, Dense, Dropout, Flatten, Convolution2D, \
MaxPooling2D, BatchNormalization, LSTM, Bidirectional, Permute, Reshape, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

class DeepSilencer():
    def init(self,L = 200,K = 5):
        input_seq  = Input(shape=(4, L, 1))
        input_kmer = Input(shape=(4**K,))
        seq_conv1_ = Convolution2D(64, 4, 10, activation='relu',border_mode='valid',dim_ordering='tf')
        seq_conv1  = seq_conv1_(input_seq)
        seq_conv2_ = Convolution2D(32, 1, 3, activation='relu',border_mode='same')
        seq_conv2  = seq_conv2_(seq_conv1)
        seq_pool1  = MaxPooling2D(pool_size=(1, 4))(seq_conv2)
        seq_conv7_ = Convolution2D(8, 1, 3, activation='relu',border_mode='same')
        seq_conv7  = seq_conv7_(seq_pool1)
        seq_pool2  = MaxPooling2D(pool_size=(1, 2))(seq_conv7)
        x = Flatten()(seq_pool2)
        dense_seq  = Dense(32, activation='relu')(x)

        dense1_    = Dense(64, activation='relu')
        dense1     = dense1_(input_kmer)
        dense_kmer = Dense(8, activation='relu')(dense1)

        merge   = concatenate([dense_seq, dense_kmer], axis = 1)
        dense3  = Dense(8, activation='relu')(merge)
        pred_output = Dense(1, activation='sigmoid')(dense3)
        self.model = Model(input=[input_seq, input_kmer], output=[pred_output])
    def fit(self,train_data,train_label,file_name = 'kmer_seq.h5',learning_rate = 1e-3):
        
        early_stopping = EarlyStopping(monitor='val_acc', verbose=0, patience=30, mode='max')
        save_best = ModelCheckpoint(filename, save_best_only=True, save_weights_only=True)
        adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
        self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        if model_ind == 9:
            self.model.fit([train_data, train_data_kmer], train_label, batch_size=128, nb_epoch=50, validation_split=0.1,\
                      callbacks=[early_stopping, save_best])
    def load_weights(self,filename):
        self.model.load_weights(filename)
    def predict(self,test_data):
        