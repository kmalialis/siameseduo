# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
from itertools import combinations
from keras.utils import to_categorical
from aux_transformations import interpolation, extrapolation, gaussian_noise
from aux_pmauc import calculate_A_pairs, average_A_pairs, calculate_M
from plot_tsne import plot_tsne_two_dicts

###########################################################################################
#                                   Auxiliary functions                                   #
###########################################################################################

########
# Data #
########


# Get samples in dict of queues
def get_sample(params_env, time_step, flag_init):
    # get pairs
    d_xs = {}

    if flag_init:
        arr = params_env['data_init']
        for i in range(params_env['num_classes']):
            idx_cls = np.where(arr[:, -1] == i)[0]
            arr_cls = arr[idx_cls]
            q_cls = deque(maxlen=params_env['memory'])
            for j in range(len(idx_cls)):
                q_cls.append(arr_cls[j, :])
            d_xs[i] = q_cls
    else:
        arr_cls = params_env['data'][time_step, :]
        q_cls = deque(maxlen=1)
        q_cls.append(arr_cls)
        cls = int(arr_cls[-1])
        d_xs[cls] = q_cls

    return d_xs

##########################
# Prequential evaluation #
##########################


def update_preq_metric(s_prev, n_prev, correct, fading_factor):
    s = correct + fading_factor * s_prev
    n = 1.0 + fading_factor * n_prev
    metric = s / n

    return s, n, metric

########################
# Classifiers training #
########################


# Data prep for ActiSiamese training
def siamese_prep_training(d):
    # get all pairs
    pairs = [x for k, q in d.items() for x in q]

    # identical pairs
    input_left_identical = np.asarray(pairs)[:, :-1]
    input_right_identical = np.asarray(pairs)[:, :-1]

    # pairs with same & different class
    input_left_same, input_right_same = [], []
    input_left_diff, input_right_diff = [], []

    for outer_pair in combinations(pairs, 2):
        left_pair = outer_pair[0]
        right_pair = outer_pair[1]

        if left_pair[-1] == right_pair[-1]:  # same class
            input_left_same.append(left_pair[:-1])
            input_right_same.append(right_pair[:-1])
        else:  # different class
            input_left_diff.append(left_pair[:-1])
            input_right_diff.append(right_pair[:-1])

    input_left_same = np.asarray(input_left_same)
    input_right_same = np.asarray(input_right_same)

    input_left_diff = np.asarray(input_left_diff)
    input_right_diff = np.asarray(input_right_diff)

    # positive pairs
    input_left_id_same = np.vstack((input_left_identical, input_left_same))
    input_right_id_same = np.vstack((input_right_identical, input_right_same))

    # balance pairs
    size_id_same = input_left_id_same.shape[0]
    size_diff = input_left_diff.shape[0]

    if size_id_same < size_diff:  # shrink different pairs
        idx = np.random.choice(a=range(size_diff), size=size_id_same, replace=False)
        input_left_diff = input_left_diff[idx, :]
        input_right_diff = input_right_diff[idx, :]

    elif size_id_same > size_diff:  # shrink identical + same pairs
        idx = np.random.choice(a=range(size_id_same), size=size_diff, replace=False)
        input_left_id_same = input_left_id_same[idx, :]
        input_right_id_same = input_right_id_same[idx, :]

    # merge pairs
    input_left = np.vstack((input_left_id_same, input_left_diff))
    input_right = np.vstack((input_right_id_same, input_right_diff))

    # labels
    y_id_same = np.ones((input_left_id_same.shape[0], 1))
    y_diff = np.zeros((input_left_diff.shape[0], 1))
    y = np.vstack((y_id_same, y_diff))

    # shuffle data (not really needed)
    # input_left, input_right, y = unison_shuffled_copies(params_env, input_left, input_right, y)
    return [input_left, input_right], y


# Data prep for ActiQ training
def fc_prep_training(d, params_env):
    # unfold dict
    xy = [a for _, q in d.items() for a in q]
    xy = np.vstack(xy)

    # features
    x = xy[:, :-1]

    # target
    y = xy[:, -1]
    y_encoded = to_categorical(y, num_classes=params_env['num_classes'], dtype='float32')
    y = np.reshape(y, (y.shape[0], 1))

    # shuffle data (not really needed)
    # x, y, y_encoded = unison_shuffled_copies(params_env, x, y, y_encoded)

    return x, y, y_encoded


# Data prep for RVUS training
def incr_fc_prep_training(xy, params_env):
    # features
    x = xy[:-1]

    # target
    y = xy[-1]
    y_encoded = to_categorical(y, num_classes=params_env['num_classes'], dtype='float32')

    # reshape
    x = np.reshape(x, (1, x.shape[0]))
    y_encoded = np.reshape(y_encoded, (1, y_encoded.shape[0]))
    y = np.reshape(y, (1, 1))

    return x, y, y_encoded


# Train model
def prep_and_train(d, xy, params_env):
    # get x and y
    x, y = (None,) * 2
    if params_env['method'] == 'rvus':  # y is y_encoded here
        x, _, y = incr_fc_prep_training(xy, params_env)
    elif params_env['method'] in ['actiq']:  # y is y_encoded here
        x, _, y = fc_prep_training(d, params_env)
    elif params_env['method'] == 'actisiamese':
        x, y = siamese_prep_training(d)

    # train
    params_env['nn'].train(x, y, verbose=0)


# Train siamese second
def prep_and_train_second(d, params_env):
    # get x and y
    x, y = siamese_prep_training(d)

    # train
    params_env['nn_second'].train(x, y, verbose=0)

##########################
# ActiSiamese prediction #
##########################


def data_prep_for_predict(d, x):
    nn_input_xy = np.array([a for _, v in d.items() for a in v])
    nn_input_y = nn_input_xy[:, -1]
    nn_input_1_x = nn_input_xy[:, :-1]
    nn_input_2_x = np.tile(x, (nn_input_1_x.shape[0], 1))

    return nn_input_y, nn_input_1_x, nn_input_2_x

################
# Augmentation #
################


# Convert dict to arr
def convert_queue_to_arr(q):
    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in q]
    return np.concatenate(q_lst, axis=0)


# Data augmentation
def augment_data(samples, d_env):
    augmented_list = []
    augmented_samples = None

    for m in d_env['da_method']:
        if m == 'interpolation':
            augmented_samples = interpolation(samples, distance='cosine', n_generated=d_env['da_n_generated'],
                                              beta=d_env['da_beta'])
        elif m == 'extrapolation':
            augmented_samples = extrapolation(samples, n_generated=d_env['da_n_generated'], beta=d_env['da_beta'])
        elif m == 'gaussian_noise':
            augmented_samples = gaussian_noise(samples, n_generated=d_env['da_n_generated'], beta=d_env['da_beta'])

        augmented_list.append(augmented_samples)

    return np.concatenate(augmented_list, axis=0)


# Convert dict to arrays
def from_dict_to_arrs(d):
    arr_xy = [a for _, q in d.items() for a in q]
    arr_xy = np.vstack(arr_xy)

    arr_x, arr_y = arr_xy[:, :-1], arr_xy[:, -1]
    arr_y = np.reshape(arr_y, (arr_y.shape[0], 1))

    return arr_xy, arr_x, arr_y


# Convert dict to dict of Siamese encodings
def from_dict_to_dictOfEncodings(d, d_env):
    _, arr_x, arr_y = from_dict_to_arrs(d=d)
    arr_x_encodings = d_env['nn'].model_base(arr_x).numpy()

    d_enc = {}
    for c in range(d_env['num_classes']):
        idx_c, _ = np.where(arr_y == c)
        examples_c_x = arr_x_encodings[idx_c]
        examples_c_y = arr_y[idx_c]
        examples_c_xy = np.c_[examples_c_x, examples_c_y]
        d_enc[c] = deque(examples_c_xy)

    return d_enc

###########################################################################################
#                                           Run                                           #
###########################################################################################


def run(params_env):

    ################
    # Init metrics #
    ################

    # prequential accuracy per class
    keys = range(params_env['num_classes'])
    preq_class_accs = {k: [] for k in keys}
    preq_class_acc = dict(zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
    preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
    preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))

    # prequential gmean
    preq_gmeans = []

    # prequential AUC
    preq_aucs = []
    auc_window_len = 500
    auc_X_window = []
    auc_y_window = []
    auc_probs = []

    ####################
    # Init AL strategy #
    ####################

    active_threshold = 1.0
    budget_current = 0.0
    budget_u = 0.0

    ######################
    # Init training data #
    ######################

    d_xy = get_sample(params_env, -1, flag_init=True)

    #########
    # Start #
    #########

    for t in range(0, params_env['time_steps']):

        if t % 10 == 0:
            print('Time step: ', t)

        # Visualise embeddings
        if False and t == 1000:
            plot_tsne_two_dicts(d_xy_encoded, d_xy_augmented)

        #######################
        # Reset preq. metrics #
        #######################

        if 'sea' in params_env['data_source'] or 'circles' in params_env['data_source'] \
                or 'blobs' in params_env['data_source']:

            flag_reset = False

            if t == 3000:
                if 'abrupt' in params_env['data_source']:
                    flag_reset = True

            if t == 6000 or t == 9000:
                if 'recurrent' in params_env['data_source']:
                    flag_reset = True

            if flag_reset:
                preq_class_acc = dict(zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
                preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
                preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))

        ###############
        # Get example #
        ###############

        # get example
        d_temp = get_sample(params_env, t, flag_init=False)
        xy = [list(i) for i in d_temp.values()][0][0]
        xy = np.reshape(xy, (1, len(xy)))

        x = xy[0, :-1]
        y = xy[0, -1]

        # reshape here once to avoid reshaping multiple times later on
        x = np.reshape(x, (1, x.shape[0]))
        xy = np.reshape(xy, (xy.shape[1],))

        ###################
        # Predict example #
        ###################
        # Output:
        # y_pred_max: will be used by the AL strategy
        # pred_class: will be used to determine correctness (evaluation)

        y_pred_max, pred_class = (None,) * 2
        y_pred, gba = (None,) * 2

        if params_env['method'] in ['rvus', 'actiq']:
            y_pred, y_pred_max, pred_class = params_env['nn'].predict(x)

        elif params_env['method'] == 'actisiamese':
            if not params_env['flag_da']:
                nn_input_y, nn_input_1_x, nn_input_2_x = data_prep_for_predict(d_xy, x)
                y_pred = params_env['nn'].predict([nn_input_1_x, nn_input_2_x])
            else:
                x_encoded = params_env['nn'].model_base(x).numpy()
                d_xy_encoded = from_dict_to_dictOfEncodings(d_xy, params_env)
                nn_input_y, nn_input_1_x, nn_input_2_x = data_prep_for_predict(d_xy_encoded, x_encoded)
                y_pred = params_env['nn_second'].predict([nn_input_1_x, nn_input_2_x])

            y_pred = np.hstack((y_pred, nn_input_y.reshape(nn_input_y.shape[0], 1)))
            gba = np.array([[c, np.mean(y_pred[y_pred[:, 1] == c][:, 0])] for c in np.unique(nn_input_y)])
            gba_max = np.max(gba[:, 1])

            pred_class = gba[gba[:, 1] == gba_max][0][0]  # select class with highest average prediction
            arr = y_pred[y_pred[:, 1] == pred_class][:, 0]  # all predictions in selected class
            y_pred_max = np.max(arr)  # highest prediction in predicted class

        ###############
        # Correctness #
        ###############

        correct = 1 if y == pred_class else 0  # check if prediction was correct

        #######################
        # Update preq. G-mean #
        #######################

        # update preq. class accuracies
        preq_class_acc_s[y], preq_class_acc_n[y], preq_class_acc[y] = update_preq_metric(
            preq_class_acc_s[y], preq_class_acc_n[y], correct, params_env['preq_fading_factor'])

        lst = []
        for k, v in preq_class_acc.items():
            preq_class_accs[k].append(v)
            lst.append(v)

        # update preq. gmean
        gmean = np.power(np.prod(lst), 1.0 / len(lst))
        preq_gmeans.append(gmean)

        ####################
        # Update preq. AUC #
        ####################

        if t >= params_env['time_steps'] - auc_window_len:
            auc_X_window.append(x.ravel())
            auc_y_window.append(y)

            probs = y_pred
            if params_env['method'] == 'actisiamese':
                probs = np.array([p for (c, p) in gba])
                probs = np.exp(probs) / np.sum(np.exp(probs))  # softmax
            else:  # RVUS / Actiq
                probs = y_pred.ravel()
            auc_probs.append(probs)

        if t == params_env['time_steps'] - 1:
            auc_X_window = np.asarray(auc_X_window)
            auc_y_window = np.asarray(auc_y_window).astype(np.int64)
            auc_probs = np.asarray(auc_probs)

            A_pairs = calculate_A_pairs(auc_y_window, auc_probs)
            A_pairs_avg = average_A_pairs(A_pairs)
            pmauc = calculate_M(A_pairs_avg, params_env['num_classes'])
            preq_aucs.append(pmauc)

        ############
        # Training #
        ############

        # Online supervised learning (NOTE: different from AL with budget = 1.0)
        if params_env['flag_learning'] == 'supervised':
            d_xy[y].append(xy)  # append new example
            prep_and_train(d_xy, xy, params_env)  # data prep and training

        # Online active learning
        elif params_env['flag_learning'] == 'active':
            labelling = 0

            if budget_current < params_env['active_budget_total']:
                rnd = np.random.normal(1.0, params_env['active_delta'])
                threshold = active_threshold * rnd

                if y_pred_max <= threshold:
                    labelling = 1  # set flag
                    d_xy[y].append(xy)  # append to queues
                    prep_and_train(d_xy, xy, params_env)  # train

                    #####################
                    # Data augmentation #
                    #####################

                    if params_env['flag_da'] and params_env['method'] == 'actisiamese':
                        d_xy_encoded = from_dict_to_dictOfEncodings(d_xy, params_env)
                        _, arr_x, arr_y = from_dict_to_arrs(d=d_xy_encoded)

                        d_xy_augmented = {}
                        for c in range(params_env['num_classes']):
                            idx_c, _ = np.where(arr_y == c)

                            examples_c_x = arr_x[idx_c]
                            examples_c_y = arr_y[idx_c]
                            examples_c_xy = np.c_[examples_c_x, examples_c_y]

                            augmented_c_x = augment_data(examples_c_x, params_env)
                            augmented_c_y = c + np.zeros(augmented_c_x.shape[0])
                            augmented_c_xy = np.c_[augmented_c_x, augmented_c_y]

                            d_xy_augmented[c] = deque(np.r_[examples_c_xy, augmented_c_xy])

                        # data prep and training
                        prep_and_train_second(d_xy_augmented, params_env)

                    # reduce AL threshold
                    active_threshold *= (1.0 - params_env['active_threshold_update'])
                else:
                    # increase AL threshold
                    active_threshold *= (1.0 + params_env['active_threshold_update'])

            # update budget
            budget_u = labelling + budget_u * params_env['active_budget_lambda']
            budget_current = budget_u / params_env['active_budget_window']

    return preq_class_accs, preq_gmeans, preq_aucs
