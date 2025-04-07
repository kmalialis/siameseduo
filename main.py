# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from tqdm import tqdm
from main_inner import run
from class_nn_siamese import SiameseNN
from class_nn_standard import StandardNN
from class_nn_siamese_second import SiameseSecondNN

###############
# Random seed #
###############


seed = 2023
random.seed(seed)
np.random.seed(seed)

####################
# GPU-related code #
####################


flag_gpu = False

if flag_gpu:
    import tensorflow as tf

    # hide warnings (before importing Keras)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # choose GPU (before importing Keras)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # dynamically grow memory
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    # GPU Server 1 comment next two lines, uncomment third
    # GPU Server 2 uncomment next two lines, comment third
    from tensorflow.python.keras import backend as K
    K.set_session(sess)
    # tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

    # random seed
    tf.random.set_seed(seed)

###########################################################################################
#                                   Auxiliary: General                                    #
###########################################################################################

#######
# I/O #
#######


# Create text file
def create_file(filename):
    f = open(filename, 'w')
    f.close()


# Write array to a row in the given file
def write_to_file(filename, arr):
    with open(filename, 'a') as f:
        np.savetxt(f, [arr], delimiter=', ', fmt='%1.6f')

#################
# Safety checks #
#################


def run_safety_checks(params_env):
    if params_env['flag_learning'] not in ['supervised', 'active']:
        raise Exception('Incorrect learning paradigm entered.')

    if params_env['method'] not in ['rvus', 'actiq', 'actisiamese']:
        raise Exception('Incorrect learning method entered.')

    if params_env['data_source'] not in [
        # synthetic
        'sea', 'circles', 'blobs',
        'sea_abrupt', 'circles_abrupt', 'blobs_abrupt',
        'sea_mm_extreme', 'circles_mm_extreme', 'blobs_mm_extreme',
        'sea_abrupt_mm_severe', 'circles_abrupt_mm_severe', 'blobs_abrupt_mm_severe',
        'sea_recurrent', 'circles_recurrent', 'blobs_recurrent',

        # real
        'keystroke', 'uwave', 'gestures', 'StarLightCurves', 'digit'

    ]:
        raise Exception('Incorrect dataset entered.')

    if params_env['method'] == 'actiq' and params_env['memory'] < 1:
        raise Exception('Neural network requires memory size >= 1')

    if params_env['method'] == 'actisiamese' and params_env['memory'] < 2:
        raise Exception('Siamese network requires memory size >= 2')

    if params_env['flag_learning'] == 'active':
        if params_env['active_budget_total'] < 0.0 or params_env['active_budget_total'] > 1.0:
            raise Exception('Budget must be in [0,1].')

    if params_env['flag_da']:

        if params_env['flag_learning'] == 'supervised':
            raise Exception('Augmentation only with active learning paradigm.')

        if params_env['method'] in ['rvus', 'actiq']:
            raise Exception('Augmentation only with siamese network.')

        if not isinstance(params_env['da_method'], list):
            raise Exception('The augm. method should be a list.')

        for m in params_env['da_method']:
            if m not in ['interpolation', 'extrapolation', 'gaussian_noise']:
                raise Exception('Data augmentation method incorrect.')

        if params_env['da_n_generated'] < 1:
            raise Exception('Number of augmentation per example should be >= 1.')

        if params_env['da_beta'] <= 0 or params_env['da_beta'] >= 1:
            raise Exception('Beta parameters should be in (0,1).')

###########################################################################################
#                                   Auxiliary: Datasets                                   #
###########################################################################################


# sea, circles, blobs
def add_data_sea_circles_blobs(d_env, d_data):
    # dataset
    temp, name = (None,) * 2
    if 'sea' in d_env['data_source']:
        name = 'sea'
        temp = 'sea10'
    if 'circles' in d_env['data_source']:
        name = 'circles'
        temp = 'circles10'
    if 'blobs' in d_env['data_source']:
        name = 'blobs'
        temp = 'blobs12'
    temp += '_' + str(d_env['memory'])

    # paths
    d_data["path"] = os.path.join(os.getcwd(), 'data', 'synthetic', name)
    d_data["name_init"] = temp + '_init'

    # scenario
    temp_flag_abrupt = True if 'abrupt' in d_env['data_source'] else False
    temp_flag_mm_extreme = True if 'mm_extreme' in d_env['data_source'] else False
    temp_flag_recurrent = True if 'recurrent' in d_env['data_source'] else False

    temp += '_arriving'
    if temp_flag_abrupt and temp_flag_mm_extreme:
        temp += '_abrupt_mm_extreme'
    elif temp_flag_abrupt and not temp_flag_mm_extreme:
        temp += '_abrupt'
    elif not temp_flag_abrupt and temp_flag_mm_extreme:
        temp += '_mm_extreme'
    elif temp_flag_recurrent:
        temp += '_recurrent'

    # paths cont'd
    d_data["name_arr"] = temp


# rest of the datasets
def add_data(data_type, name_init, name_arriving, d_data):
    d_data["path"] = os.path.join(os.getcwd(), 'data', data_type, name_arriving)
    d_data["name_init"] = name_init
    d_data["name_arr"] = name_arriving

###########################################################################################
#                                   Auxiliary: Classifiers                                #
###########################################################################################

#############
# Synthetic #
#############


# sea, circles, blobs
def add_classifier_sea_circles_blobs(d_env, d_nn, d_nn_second):
    d_nn['learning_rate'] = 0.01
    d_nn['minibatch_size'] = 64
    d_nn['reg_l2'] = 0.0001
    d_nn['layer_dims'] = [d_env['num_features'], 32, 32, d_env['num_classes']]

    if d_env['method'] == 'actisiamese':
        del d_nn['layer_dims'][-1]

        if d_env['flag_da']:
            d_nn_second['layer_dims'] = [d_nn['layer_dims'][-1], 16]
            d_nn_second['learning_rate'] = 0.001
            d_nn_second['minibatch_size'] = 64
            d_nn_second['reg_l2'] = 0.0001

########
# Real #
########


# keystroke (AL budget 10%)
def add_classifier_keystroke(d_env, d_nn, d_nn_second):
    d_nn['learning_rate'] = 0.001
    d_nn['minibatch_size'] = 64
    d_nn['reg_l2'] = 0.0001
    d_nn['layer_dims'] = [d_env['num_features'], 128, 64, d_env['num_classes']]

    if d_env['method'] == 'actisiamese':
        del d_nn['layer_dims'][-1]

        if d_env['flag_da']:
            d_nn_second['layer_dims'] = [d_nn['layer_dims'][-1], 16]
            d_nn_second['learning_rate'] = 0.001
            d_nn_second['minibatch_size'] = 64
            d_nn_second['reg_l2'] = 0.0001


# uwave (AL budget 10%)
def add_classifier_uwave(d_env, d_nn, d_nn_second):
    d_nn['learning_rate'] = 0.001
    d_nn['minibatch_size'] = 64
    d_nn['reg_l2'] = 0.0001
    d_nn['layer_dims'] = [d_env['num_features'], 512, 128, d_env['num_classes']]

    if d_env['method'] == 'actisiamese':
        del d_nn['layer_dims'][-1]

        if d_env['flag_da']:
            d_nn_second['layer_dims'] = [d_nn['layer_dims'][-1], 16]
            d_nn_second['learning_rate'] = 0.001
            d_nn_second['minibatch_size'] = 64
            d_nn_second['reg_l2'] = 0.0001


# gestures (AL budget 5%)
def add_classifier_gestures(d_env, d_nn, d_nn_second):
    d_nn['learning_rate'] = 0.001
    d_nn['minibatch_size'] = 64
    d_nn['reg_l2'] = 0.0001
    d_nn['layer_dims'] = [d_env['num_features'], 512, 512, d_env['num_classes']]

    if d_env['method'] == 'actisiamese':
        del d_nn['layer_dims'][-1]

        if d_env['flag_da']:
            d_nn_second['layer_dims'] = [d_nn['layer_dims'][-1], 32]
            d_nn_second['learning_rate'] = 0.001
            d_nn_second['minibatch_size'] = 64
            d_nn_second['reg_l2'] = 0.0


# Digits (AL budget 1%)
def add_classifier_digit(d_env, d_nn, d_nn_second):
    d_nn['learning_rate'] = 0.001
    d_nn['minibatch_size'] = 64
    d_nn['reg_l2'] = 0.0001
    d_nn['layer_dims'] = [d_env['num_features'], 128, 64, d_env['num_classes']]

    if d_env['method'] == 'actisiamese':
        del d_nn['layer_dims'][-1]

        if d_env['flag_da']:
            d_nn_second['layer_dims'] = [d_nn['layer_dims'][-1], 16]
            d_nn_second['learning_rate'] = 0.001
            d_nn_second['minibatch_size'] = 64
            d_nn_second['reg_l2'] = 0.0001


# StarLightCurves (AL budget 1%)
def add_classifier_StarLightCurves(d_env, d_nn, d_nn_second):
    d_nn['learning_rate'] = 0.001
    d_nn['minibatch_size'] = 64
    d_nn['reg_l2'] = 0.0001
    d_nn['layer_dims'] = [d_env['num_features'], 128, 64, d_env['num_classes']]

    if d_env['method'] == 'actisiamese':
        del d_nn['layer_dims'][-1]

        if d_env['flag_da']:
            d_nn_second['layer_dims'] = [d_nn['layer_dims'][-1], 16]
            d_nn_second['learning_rate'] = 0.0001
            d_nn_second['minibatch_size'] = 64
            d_nn_second['reg_l2'] = 0.0001

###########################################################################################
#                                         Main                                            #
###########################################################################################


def main(params_env):

    ###########
    # General #
    ###########

    run_safety_checks(params_env)  # safety checks
    params_env['seed'] = seed  # random seed

    ##################
    # Settings: data #
    ##################

    # parameters
    params_data = {}
    if 'sea' in params_env['data_source'] or 'circles' in params_env['data_source'] or \
            'blobs' in params_env['data_source']:
        add_data_sea_circles_blobs(params_env, params_data)
    elif params_env['data_source'] == 'gestures':
        add_data('real', 'gestures_init10', 'gestures', params_data)
    elif params_env['data_source'] == 'keystroke':
        add_data('real', 'keystroke_init10', 'keystroke', params_data)
    elif params_env['data_source'] == 'uwave':
        add_data('real', 'uwave_init10', 'uwave', params_data)
    elif params_env['data_source'] == 'StarLightCurves':
        add_data('real', 'StarLightCurves_init10', 'StarLightCurves', params_data)
    elif params_env['data_source'] == 'digit':
        add_data('real', 'digit_init10', 'digit', params_data)

    # load data
    path_data = os.path.join(params_data['path'], params_data['name_arr'] + '.txt')
    path_data_init = os.path.join(params_data['path'], params_data['name_init'] + '.txt')

    params_env['data'] = np.genfromtxt(path_data, delimiter=',')
    params_env['data_init'] = np.genfromtxt(path_data_init, delimiter=',')

    del params_data

    # derived parameters
    params_env['time_steps'] = params_env['data'].shape[0]
    params_env['num_features'] = params_env['data_init'].shape[1] - 1
    params_env['num_classes'] = len(np.unique(params_env['data_init'][:, -1]))

    # extra safety check
    memory_size = int(params_env['data_init'].shape[0] / params_env['num_classes'])
    if params_env['memory'] != memory_size:
        raise Exception("Memory size incorrect")

    ####################
    # Settings: models #
    ####################

    params_nn = {'num_epochs': 1}
    params_nn_second = {'num_epochs': 1}

    # Synthetic datasets
    if 'sea' in params_env['data_source'] or 'circles' in params_env['data_source'] or \
            'blobs' in params_env['data_source']:
        add_classifier_sea_circles_blobs(params_env, params_nn, params_nn_second)
    elif params_env['data_source'] == 'gestures':
        add_classifier_gestures(params_env, params_nn, params_nn_second)
    elif params_env['data_source'] == 'keystroke':
        add_classifier_keystroke(params_env, params_nn, params_nn_second)
    elif params_env['data_source'] == 'uwave':
        add_classifier_uwave(params_env, params_nn, params_nn_second)
    elif params_env['data_source'] == 'StarLightCurves':
        add_classifier_StarLightCurves(params_env, params_nn, params_nn_second)
    elif params_env['data_source'] == 'digit':
        add_classifier_digit(params_env, params_nn, params_nn_second)

    ###################
    # Settings: fixed #
    ###################
    # NOTE: Keep these parameters fixed to replicate the paper's results

    # fixed - active learning (suggested by their authors)
    params_env['active_threshold_update'] = 0.01
    params_env['active_budget_window'] = 300
    params_env['active_budget_lambda'] = 1.0 - (1.0 / params_env['active_budget_window'])
    params_env['active_delta'] = 1.0  # N(1, delta) - no randomisation if set to 0

    # fixed - prequential evaluation
    params_env['preq_fading_factor'] = 0.99

    ################
    # Output files #
    ################

    # file directory and names
    out_dir = 'exps/'

    out_name = '{}_{}'.format(params_env['data_source'], params_env['method'])

    if params_env['method'] != 'rvus':
        out_name += '{}'.format(params_env['memory'])

    out_name += '_{}'.format(params_env['active_budget_total'])

    if params_env['flag_da']:
        out_name += '_DA' + str(params_env['da_n_generated']) + str(params_env['da_method'])

    # out_name += '_' + str(params_env['name'])

    filename_gmean = os.path.join(os.getcwd(), out_dir, out_name + '_preq_gmean.txt')
    create_file(filename_gmean)

    filename_auc = os.path.join(os.getcwd(), out_dir, out_name + '_preq_auc.txt')
    create_file(filename_auc)

    #########
    # Start #
    #########

    for r in tqdm(range(params_env['repeats'])):
        print('Repetition: ', r)

        # create classifier
        if params_env['method'] in ['rvus', 'actiq']:
            params_env['nn'] = StandardNN(params_nn, seed=params_env['seed'])
        elif params_env['method'] == 'actisiamese':
            params_env['nn'] = SiameseNN(params_nn, seed=params_env['seed'])

            if params_env['flag_da']:
                params_env['nn_second'] = SiameseSecondNN(params_nn_second, seed=params_env['seed'])

        # start
        _, preq_gmeans, preq_aucs = run(params_env)

        # store
        write_to_file(filename_gmean, preq_gmeans)
        write_to_file(filename_auc, preq_aucs)
