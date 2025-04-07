# -*- coding: utf-8 -*-
import numpy as np


##########################
# Find nearest neighbour #
##########################
# Description:
# Auxiliary function for interpolation().
# Finds closest neighbour of "example" from all examples in "inputs".
# Note that the "example" may be included in "inputs", therefore, we need to ignore it.
#
# Input:
# - inputs: 2D-array of shape (n_examples, n_features)
# - example: 1D-array of shape (n_features,)
# - distance: metric to calculate distance (euclidean, cosine)
#
# Output:
# - s: 1D-array of shape (n_features,)

def find_nearest_neighbour(inputs, x, distance):
    n_examples, _ = inputs.shape
    x_repeated = np.tile(x, (n_examples, 1))

    if distance == 'euclidean':
        diff = np.linalg.norm(inputs - x_repeated, axis=1)
    elif distance == 'cosine':
        cosine_similarity = np.dot(inputs, x) / np.linalg.norm(inputs, axis=1) * np.linalg.norm(x_repeated, axis=1)
        diff = 1.0 - cosine_similarity
    else:
        raise Exception('Specified distance metric not supported.')

    arr_bool = np.isclose(diff, 0.0)
    idx_zero = np.where(arr_bool == True)
    diff[idx_zero] = 100

    idx_closer = np.argmin(diff, axis=0)
    closer = inputs[idx_closer, :]

    return closer


#################
# Interpolation #
#################
# Description
# Given an example "x", interpolation is defined as x = x + beta * (s - x),
# where beta is a scaling factor in (0,1), and s is the closest example to x.
#
# Input:
# - inputs: 2D-array of shape (n_examples, n_features)
# - n_generated: number of examples to be generated per *example*
# - distance: metric to calculate distance (euclidean, cosine)
#
# Output:
# - examples_generated: 2D-array of shape (n_generated, n_features)

def interpolation(inputs, n_generated, beta, distance):
    examples_generated = np.array([])
    n_examples, n_features = inputs.shape

    for i in range(n_examples):
        x = inputs[i, :]
        x_repeated = np.tile(x, (n_generated, 1))

        if beta == 'random':
            betas = np.random.uniform(low=0, high=1, size=(n_generated,))
            betas = np.reshape(betas, (betas.shape[0], 1))
        else:
            betas = beta * np.ones((n_generated, 1))

        s = find_nearest_neighbour(inputs, x, distance)
        diff = s - x
        diff_repeated = np.tile(diff, (n_generated, 1))

        x_generated = x_repeated + betas * diff_repeated
        if examples_generated.shape[0] == 0:
            examples_generated = x_generated
        else:
            examples_generated = np.vstack((examples_generated, x_generated))

    return examples_generated


#################
# Extrapolation #
#################
# Description
# Given an example "x", extrapolation is defined as x = x + beta * (x - mu),
# where beta is a scaling factor in (0,1), and mu is the class mean.
#
# Input:
# - inputs: 2D-array of shape (n_examples, n_features)
# - n_generated: number of examples to be generated per *example*
#
# Output:
# - examples_generated: 2D-array of shape (n_generated, n_features)

def extrapolation(inputs, n_generated, beta):
    mu = np.mean(inputs, axis=0)
    n_examples, n_features = inputs.shape

    examples_generated = np.array([])
    for i in range(n_examples):
        x = inputs[i, :]
        x_repeated = np.tile(x, (n_generated, 1))

        if beta == 'random':
            betas = np.random.uniform(low=0, high=1, size=(n_generated,))
            betas = np.reshape(betas, (betas.shape[0], 1))
        else:
            betas = beta * np.ones((n_generated, 1))

        diff = x - mu
        diff_repeated = np.tile(diff, (n_generated, 1))

        x_generated = x_repeated + betas * diff_repeated

        if examples_generated.shape[0] == 0:
            examples_generated = x_generated
        else:
            examples_generated = np.vstack((examples_generated, x_generated))

    return examples_generated


##################
# Gaussian noise #
##################
# Description
# Given an example "x", injecting noise is defined as x = x + beta * eta,
# where beta is a scaling factor in (0,1), and eta~N(0, sigma) is the injected noise.
#
# Input:
# - inputs: 2D-array of shape (n_examples, n_features)
# - n_generated: number of examples to be generated per *example*
#
# Output:
# - examples_generated: 2D-array of shape (n_generated, n_features)

def gaussian_noise(inputs, n_generated, beta):
    n_examples, n_features = inputs.shape

    examples_generated = np.array([])
    for i in range(n_examples):
        x = inputs[i, :]
        x_repeated = np.tile(x, (n_generated, 1))

        if beta == 'random':
            betas = np.random.uniform(low=0, high=1, size=(n_generated,))
            betas = np.reshape(betas, (betas.shape[0], 1))
        else:
            betas = beta * np.ones((n_generated, 1))

        mu = np.zeros(n_features)
        sigma = np.std(inputs, axis=0)
        cov = sigma * np.identity(n_features)
        etas = np.random.multivariate_normal(mean=mu, cov=cov, size=n_generated)

        x_generated = x_repeated + betas * etas

        if examples_generated.shape[0] == 0:
            examples_generated = x_generated
        else:
            examples_generated = np.vstack((examples_generated, x_generated))

    return examples_generated

#########
# Tests #
#########


# arr = np.random.random((10, 3))
# i = interpolation(arr, n_generated=10, distance='cosine')
# e = extrapolation(arr, n_generated=10)
# g = gaussian_noise(arr, n_generated=10)
