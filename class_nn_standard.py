# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import he_normal
from keras.layers import Input, Dense, LeakyReLU


# random seed
seed = 2023
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


class StandardNN:

    ###########################################################################################
    #                                          API                                            #
    ###########################################################################################

    ###############
    # Constructor #
    ###############

    def __init__(self, nn_params, seed=2023):

        # seed
        self.seed = seed

        # NN parameters
        self.layer_dims = nn_params['layer_dims']  # [n_x, n_h1, .., n_hL, n_y], at least one hidden layer
        self.learning_rate = nn_params['learning_rate']
        self.num_epochs = nn_params['num_epochs']
        self.minibatch_size = nn_params['minibatch_size']
        self.reg_l2 = nn_params['reg_l2']

        # fixed
        self.output_activation = 'softmax'
        self.weight_init = he_normal(seed=self.seed)
        self.loss_function = 'categorical_crossentropy'

        # model
        self.model = self.create_fc_model()
        # self.model.summary()

        # configure model for training
        self.model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=self.loss_function,
            metrics=['accuracy']
        )

    ##############
    # Prediction #
    ##############

    def predict(self, x):
        # probability prediction
        y_hat = self.model.predict(x=x, verbose=0)

        # class prediction
        y_hat_max = np.max(y_hat)
        y_hat_argmax = np.argmax(y_hat)

        return y_hat, y_hat_max, y_hat_argmax

    ############
    # Training #
    ############

    def train(self, x, y, validation_data=None, verbose=0, flag_plot=False):
        history = self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            shuffle=False,
            verbose=verbose  # 0: off, 1: full, 2: brief
        )

        flag_val = False if validation_data is None else True
        if flag_plot:
            plt.plot(history.history['loss'])
            if flag_val:
                plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

        acc = history.history['accuracy'][-1]
        loss = history.history['loss'][-1]

        if flag_val:
            val_acc = history.history['val_accuracy'][-1]
            val_loss = history.history['val_loss'][-1]
            return loss, acc, val_loss, val_acc
        else:
            return loss, acc

    ###########################################################################################
    #                                   Internal functions                                    #
    ###########################################################################################

    ############
    # FC Model #
    ############

    def create_fc_model(self):
        # Input and output dims
        n_x = self.layer_dims[0]
        n_y = self.layer_dims[-1]

        # Input layer
        x_input = Input(shape=(n_x,), name='input')

        # First hidden layer
        x = Dense(
            units=self.layer_dims[1],
            activation=None,
            kernel_initializer=self.weight_init,
            kernel_regularizer=l2(self.reg_l2),
        )(x_input)
        x = LeakyReLU(alpha=0.01)(x)

        # Other hidden layers (if any)
        for n_units in self.layer_dims[2:-1]:
            x = Dense(
                units=n_units,
                activation=None,
                kernel_initializer=self.weight_init,
                kernel_regularizer=l2(self.reg_l2),
            )(x)
            x = LeakyReLU(alpha=0.01)(x)

        # Output layer
        y_out = Dense(
            units=n_y,
            activation=self.output_activation,
            kernel_initializer=self.weight_init,
            kernel_regularizer=None,
            name='output'
        )(x)

        # Model
        return Model(inputs=x_input, outputs=y_out)
