# coding=utf-8
import numpy as np
import keras
from keras.layers import Input, Dense, Conv1D, GRU, Dropout, Flatten, Activation
from keras.layers import concatenate, add, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Adam
import keras.backend as K
from keras.layers import Layer


class LSTNet(object):
    def __init__(self, args, dims):
        super(LSTNet, self).__init__()
        self.P = args.window  # 时间序列长度
        self.m = dims  # 特征长度
        self.hidR = args.hidRNN  # RNN out dim
        self.hidC = args.hidCNN  # CNN out dim（filter nums）
        self.hidS = args.hidSkip  # skip RNN out dim
        self.Ck = args.CNN_kernel  # filter map size
        self.skip = args.skip   # skip -》 period（step size）
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = args.highway_window # ar weight
        self.dropout = args.dropout
        self.output = args.output_fun
        self.lr = args.lr
        self.loss = args.loss
        self.clip = args.clip
        self.attention = True

    def make_model(self):

        x = Input(shape=(self.P, self.m))

        # CNN
        c = Conv1D(self.hidC, self.Ck, activation='relu')(x)
        c = Dropout(self.dropout)(c)

        # add attention
        if self.attention:
            r = GRU(self.hidR, return_sequences=True)(c)
            r = AttentionLayer()(r)
        else:
            # RNN
            r = GRU(self.hidR)(c)
            r = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r)
            r = Dropout(self.dropout)(r)
            # skip-RNN
            if self.skip > 0:
                # c: batch_size*steps*filters, steps=P-Ck
                s = Lambda(lambda k: k[:, int(-self.pt * self.skip):, :])(c)
                s = Lambda(lambda k: K.reshape(k, (-1, self.pt, self.skip, self.hidC)))(s)
                s = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1, 3)))(s)
                s = Lambda(lambda k: K.reshape(k, (-1, self.pt, self.hidC)))(s)

                s = GRU(self.hidS)(s)
                s = Lambda(lambda k: K.reshape(k, (-1, self.skip * self.hidS)))(s)
                s = Dropout(self.dropout)(s)
                r = concatenate([r, s])

        res = Dense(self.m)(r)

        # highway
        if self.hw > 0:
            z = Lambda(lambda k: k[:, -self.hw:, :])(x)
            z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.hw)))(z)
            z = Dense(1)(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.m)))(z)
            res = add([res, z])

        if self.output != 'no':
            res = Activation(self.output)(res)

        model = Model(inputs=x, outputs=res)
        model.compile(optimizer=Adam(lr=self.lr, clipnorm=self.clip), loss=self.loss)
        return model


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class LSTNet_multi_inputs(object):
    def __init__(self, args, dims):
        super(LSTNet_multi_inputs, self).__init__()
        self.P = args.window
        self.m = dims
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        # self.pt = int((self.P-self.Ck)/self.skip)
        self.pt = args.ps
        self.hw = args.highway_window
        self.dropout = args.dropout
        self.output = args.output_fun
        self.lr = args.lr
        self.loss = args.loss
        self.clip = args.clip

    def make_model(self):

        # Input1: short-term time series
        input1 = Input(shape=(self.P, self.m))
        # CNN
        conv1 = Conv1D(self.hidC, self.Ck, strides=1, activation='relu')  # for input1
        # It's a probelm that I can't find any way to use the same Conv1D layer to train the two inputs, 
        # since input2's strides should be Ck, not 1 as input1
        conv2 = Conv1D(self.hidC, self.Ck, strides=self.Ck, activation='relu')  # for input2
        conv2.set_weights(conv1.get_weights())  # at least use same weight

        c1 = conv1(input1)
        c1 = Dropout(self.dropout)(c1)
        # RNN
        r1 = GRU(self.hidR)(c1)
        # r1 = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r1)
        r1 = Dropout(self.dropout)(r1)

        # Input2: long-term time series with period
        input2 = Input(shape=(self.pt * self.Ck, self.m))
        # CNN
        c2 = conv2(input2)
        c2 = Dropout(self.dropout)(c2)
        # RNN
        r2 = GRU(self.hidS)(c2)
        # r2 = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r2)
        r2 = Dropout(self.dropout)(r2)

        r = concatenate([r1, r2])
        res = Dense(self.m)(r)

        # highway
        if self.hw > 0:
            z = Lambda(lambda k: k[:, -self.hw:, :])(input1)
            z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.hw)))(z)
            z = Dense(1)(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.m)))(z)
            res = add([res, z])

        if self.output != 'no':
            res = Activation(self.output)(res)

        model = Model(inputs=[input1, input2], outputs=res)
        model.compile(optimizer=Adam(lr=self.lr, clipnorm=self.clip), loss=self.loss)
        return model
