# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import he_normal
from keras.layers import Input, Dense, Lambda, LeakyReLU

# random seed
global_seed = 2023
random.seed(global_seed)
np.random.seed(global_seed)
tf.random.set_seed(global_seed)


class SiameseSecondNN:

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
        self.layer_dims = nn_params['layer_dims']  # [n_encoding, n_hidden]
        self.learning_rate = nn_params['learning_rate']
        self.num_epochs = nn_params['num_epochs']
        self.minibatch_size = nn_params['minibatch_size']
        self.reg_l2 = nn_params['reg_l2']

        # fixed
        self.weight_init = he_normal(seed=self.seed)
        self.output_activation = 'sigmoid'
        self.loss_function = 'binary_crossentropy'

        # model
        self.model = self.create_siamese_model()

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
        y_hat = self.model.predict(x=x, verbose=0)  # probability prediction

        return y_hat

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

    #################
    # Siamese model #
    #################

    def create_siamese_model(self):
        input_left = Input(shape=(self.layer_dims[0],))
        input_right = Input(shape=(self.layer_dims[0],))

        # Add a customized layer to compute the distance between the encodings
        lambda_layer = Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]), name='abs')
        distance = lambda_layer([input_left, input_right])

        # Hidden layer
        x = None
        for n_units in self.layer_dims[1:]:
            x = Dense(
                units=n_units,
                activation=None,
                use_bias=True,
                kernel_initializer=self.weight_init,
                bias_initializer='zeros',
                kernel_regularizer=l2(self.reg_l2)
            )(distance)
            x = LeakyReLU(alpha=0.01)(x)

        # Output layer
        x_output = Dense(
            units=1,
            activation='sigmoid',
            use_bias=True,
            kernel_initializer=self.weight_init,
            bias_initializer='zeros',
            kernel_regularizer=None,
        )(x)

        # Connect the inputs with the outputs
        return Model(inputs=[input_left, input_right], outputs=x_output)
