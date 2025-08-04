# -*- coding: utf-8 -*-
""" This module contains the model class for matrix product operator Gaussian restricted
Boltzmann machine (MPO-GRBM). It implements the tensor form of Gaussian restricted 
Boltzmann Machine using matrix product operator for the weight matrix. The decomposition
used assumes that $N_v=M_1^{(v)}*M_2^{(v)}$ and $N_h=M_1^{(h)}*M_2^{(h)}. 

Created on Sat Jul 16 18:39:29 2022

@author: Bruno Monsia

     :Implemented:
        - MPO-GRBM model

    :Version:
        1.0.0
        
    :Contact:
        bruno.monsia@umontreal.ca, monsiabruno@gmail.com
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def makebatch(dataset, batchsize):
    ''' Makes batches for a given dataset.
    

    Parameters
    ----------
    dataset : array
        DESCRIPTION. Dataset to be batched.
    batchsize : int, optional.
        DESCRIPTION. size of each batch. The default is 100.

    Returns
    -------
    batchdata : array.
        DESCRIPTION. batchdata set. 

    '''
    
    numdims = dataset.shape[1:] 
    numbatches = dataset.shape[0]//batchsize
    indexes = [i for i in range(dataset.shape[0])]
    np.random.seed(0)
    np.random.shuffle(indexes)
    batchdata = np.zeros((batchsize,) + numdims + (numbatches,))
    #batchlabels = np.zeros((batchsize, numbatches))
    for i in range(numbatches):
        batchdata[...,i] = dataset[indexes[i*batchsize:(i+1)*batchsize], ...]
    return batchdata

#Plot function during training
def plot(samples):
    ''' Plots some examples from the dataset.
    

    Parameters
    ----------
    samples : array of shape (N_sample, Dim_v). In particular, N_sample=100.
        DESCRIPTION. Sample of 100 individuals to be plotted.

    Returns
    -------
    None.

    '''
    
    plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')

# MPO decomposition #

def mpo_params_initialize(vis_shape, hid_shape, r):
    ''' This function initializes the MPO-GRBM parameters.
    

    Parameters
    ----------
    vis_shape : list of integers used to decompose the observed data dimension
    ($N_v = M_1^{(v)}*M_2^{(v)}$).
        DESCRIPTION. shape of the observed variable dimension.
    hid_shape : list of integers used to decompose the number of latent variables
    ($N_h=M_1^{(h)}*M_2^{(h)}$).
        DESCRIPTION. shape of the latent variable dimension.
    r : list of the tensor decomposition ranks.
        DESCRIPTION. ranks of the matrix product operator.

    Returns
    -------
           
    visbiases : array of shape ($M_1^{(v)}, M_2^{(v)}$).
        DESCRIPTION. bias associated with the observed variable in tensor format.
    hidbiases : array of shape ($M_1^{(h)}, M_2^{(h)}$)
        DESCRIPTION. bias associated with the latent variable in tensor format.
    cores : dictionnary. 
        DESCRIPTION. Contains cores of the matrix product operator.

    '''
    
    d = len(vis_shape)
    cores = {}
    for k in range(d):
        cores['G'+str(k)] =\
            tf.Variable(0.1*tf.random.normal(shape = (r[k], vis_shape[k], hid_shape[k], r[k+1]), stddev = 1, seed=1), name= 'G'+str(k)) 
    visbiases = tf.Variable(0.1*tf.ones( shape = vis_shape, dtype=tf.dtypes.float32), name='visbiases')
    hidbiases = tf.Variable(tf.zeros( shape = hid_shape, dtype=tf.dtypes.float32), name='hidbiasess')
    
    return visbiases, hidbiases, cores 

def mpo_gradients_initialize(vis_shape, hid_shape, r):
    ''' This function initializes the MPO-GRBM parameters.
    

    Parameters
    ----------
    vis_shape : list of integers used to decompose the observed data dimension
    ($N_v = M_1^{(v)}*M_2^{(v)}$).
        DESCRIPTION. shape of the observed variable dimension.
    hid_shape : list of integers used to decompose the number of latent variables
    ($N_h=M_1^{(h)}*M_2^{(h)}$).
        DESCRIPTION. shape of the latent variable dimension.
    r : list of the tensor decomposition ranks.
        DESCRIPTION. ranks of the matrix product operator.

    Returns
    -------
    grad_visbiases : array,
        DESCRIPTION. Contains energy's gradient with respect to the visible bias parameter.
    grad_hidbiases : array.
        DESCRIPTION. Contains energy's gradient with respect to the latent bias parameter.
    grad_cores : dictionnary.
        DESCRIPTION. Contains energy's gradient with respect to the cores of 
        the matrix product operator.

    '''
    
    d = len(vis_shape)
    grad_cores = {}
    for k in range(d):
        grad_cores['gradG'+str(k)] = tf.zeros(shape = (r[k], vis_shape[k], hid_shape[k], r[k+1]), name = 'gradG'+ str(k))
    grad_visbiases = tf.zeros( shape = vis_shape, dtype=tf.dtypes.float32, name='grad_visbiases')
    grad_hidbiases = tf.zeros( shape = hid_shape, dtype=tf.dtypes.float32, name='grad_hidbiasess')
    
    return grad_visbiases, grad_hidbiases, grad_cores

def probh_givenv(v, b, cores, sigma):
    ''' Computes conditional probability of the latent variables given 
    observed variables v.
    

    Parameters
    ----------
        
    v : array of shape ($N,M_1^{(v)},M_2^{(v)}$)
        DESCRIPTION. It is the observed dataset or a batch of dataset. Its 
        elements should be binary.
    b : array of shape $(M_1^{(h)},M_2^{(h)})$.
        DESCRIPTION. bias associated with the latent variables in tensorizing of GRBM.
    cores : dictionnary. 
        DESCRIPTION. Contains cores of the matrix product operator.
    sigma : scalar (float) or array.
        DESCRIPTION. standard deviation of v|h.

    Returns
    -------
    probs_h_given_v : array of shape ($N,M_1^{(h)},M_2^{(h)}$).
        DESCRIPTION. conditional probability of the latent variables given 
        observed variables v.
    sample_binary_hid : array of shape ($N,M_1^{(h)},M_2^{(h)}$).
        DESCRIPTION. generated binary data using conditional probability $p(v|h;\theta)$.

    '''
    
    v = v/sigma**2    
    term = tf.einsum("nbd,abfc,cdha->nfh", v, cores['G0'], cores['G1'])
    probs_h_given_v = tf.sigmoid(b + term)
    sample_binary_hid = tf.cast(probs_h_given_v > tf.random.uniform(shape = b.shape, minval=0, maxval=1), dtype=tf.float32)
    
    return probs_h_given_v, sample_binary_hid

def probv_givenh(h, a, cores, sigma):
    ''' Computes conditional probability of the observed variables v given 
    latent variables.
    

    Parameters
    ----------
    h : array of shape ($N,M_1^{(h)},M_2^{(h)}$).
        DESCRIPTION. Represents latent data.
    a : array of shape $(M_1^{(v)},M_2^{(v)})$.
        DESCRIPTION. bias associated with observed variables in tensorizing of GRBM.
    cores : dictionnary. 
        DESCRIPTION. Contains cores of matrix product operator.
    sigma : scalar (float) or array.
        DESCRIPTION. standard deviation of v|h.

    Returns
    -------        
    probs_v_given_h : array of shape ($N,M_1^{(v)},M_2^{(v)}$).
        DESCRIPTION. conditional probability of the observed variables given 
        latent variables h.
    sample_binary_vis : array of shape ($N,M_1^{(v)},M_2^{(v)}$).
        DESCRIPTION. generated data using conditional probability $p(v|h;\theta)$.

    '''
    
    term = tf.einsum("nfh,abfc,cdha->nbd", h, cores['G0'], cores['G1'])
    sample_vis = term + a # + sigma*tf.random.normal(shape=a.numpy().shape())    
    probs_v_given_h = tf.sigmoid(sample_vis)
    
    return probs_v_given_h, sample_vis
       
def reconstructor_weight(cores):
    ''' This function reconstructs the weight matrix from the cores of the
    matrix product operator.
    

    Parameters
    ----------
    cores : dictionnary.
        DESCRIPTION. Contains the cores of the matrix product operator.

    Returns
    -------
    reconst_weight : array of shape ($N_v, N_h$)
        DESCRIPTION. The original weight matrix.

    '''
    
    reconst_weight = tf.einsum("abfc,cdha->bdfh", cores['G0'], cores['G1'])
    reconst_weight = tf.squeeze(reconst_weight)
    
    return reconst_weight

def gradients_energy_cores(v, h, cores):
    ''' Computes the gradients of the energy function with respect to cores of
    the matrix product operator.
    

    Parameters
    ----------
    v : array of shape ($N,M_1^{(v)},M_2^{(v)}$)
        DESCRIPTION. It is the observed dataset.
    h : array of shape ($N,M_1^{(h)},M_2^{(h)}$).
        DESCRIPTION. Represents latent data.
    cores : dictionnary.
        DESCRIPTION. Contains the cores of the matrix product operator.

    Returns
    -------
    gradE_cores : dictionnary.
        DESCRIPTION. Contains the gradients of energy function with respect 
        to the cores.

    '''
    
    gradE_cores = {}
    gradE_cores['gradE_G' + str(0)] = tf.math.reduce_mean(tf.einsum("nbd,cdha, nfh->nabfc", v, cores['G1'],h), axis=[0])
    gradE_cores['gradE_G' + str(1)] = tf.math.reduce_mean(tf.einsum("nbd, abfc,nfh->ncdha", v, cores['G0'],h), axis =[0])
    
    return gradE_cores

def energy_function(data, h, a, b, cores, sigma):
    ''' Computes energy function of GRBM model.
    

    Parameters
    ----------
    data : array of shape ($N,M_1^{(v)},M_2^{(v)}$)
        DESCRIPTION. It is the observed dataset.
    h : array of shape ($N,M_1^{(h)},M_2^{(h)}$).
        DESCRIPTION. Represents latent data.
    a : array of shape $(M_1^{(v)},M_2^{(v)})$.
        DESCRIPTION. bias associated with observed variables in tensorizing of GRBM.
    b : array of shape $(M_1^{(h)},M_2^{(h)})$.
        DESCRIPTION. bias associated with latent variables in tensorizing of GRBM.
    cores : dictionnary.
        DESCRIPTION. Contains the cores of the matrix product operator.
    sigma : scalar (float) or array.
        DESCRIPTION. standard deviation of v|h.

    Returns
    -------
    energy: array of shape (N,).
        DESCRIPTION. Energy function.
        
    '''
    
    data_splited = data/(sigma**2)
    term = (data-a)**2/(sigma**2)
    energy = tf.einsum("nbd,nfh,abfc,cdha->n", data_splited, h, cores['G0'], cores['G1']) \
        + tf.math.reduce_sum(term, axis=(1,2)) + tf.einsum("nfh,fh->n", h, b)
    
    energy = -energy
    
    return energy

@tf.function
def gradient_energy_tf(energy, a, b, cores):
    ''' This function returns the gradients of the energy function with respect
    to RBM parameters using a tensorflow function.
    

    Parameters
    ----------
    energy : array of shape (N,).
        DESCRIPTION. Energy function.
    a : array of shape $(M_1^{(v)},M_2^{(v)})$.
        DESCRIPTION. bias associated with observed variables in tensorizing of GRBM.
    b : array of shape $(M_1^{(h)},M_2^{(h)})$.
        DESCRIPTION. bias associated with latent variables in tensorizing of GRBM.
    cores : dictionnary.
        DESCRIPTION. Contains the cores of the matrix product operator.

    Returns
    -------
    grad : array
        DESCRIPTION.

    '''
    
    grad = tf.gradients( energy, [a, b], stop_gradients=[a, b])    
    
    return grad    

def reconstructor_error(data, reconst_data ):
    ''' Computes the mean square error between the original data and the 
    model-generated data.
    

    Parameters
    ----------
    data : array of shape ($N,M_1^{(v)},M_2^{(v)}$).
        DESCRIPTION. It is the observed dataset.
    sample_data : array of shape ($N,M_1^{(v)},M_2^{(v)}$).
        DESCRIPTION. data generated by the model.

    Returns
    -------
    error : scalar (float)
        DESCRIPTION. The mean square error between the original data and the 
    model-generated data


    '''
    
    l_shape_data = len(data.numpy().shape)
    axis_ = [i+1 for i in range(l_shape_data-1)]
    individual_errors = tf.math.sqrt(tf.math.reduce_sum((data - reconst_data)**2, axis = axis_))   
    error = tf.math.reduce_mean(tf.math.sqrt(individual_errors)).numpy()
    #individual_errors_ = list(individual_errors.numpy())
    
    return error


def CD_algorithm(data, vis_bias, hid_bias, cores, vis_shape, hid_shape, K):
    ''' This function initializes the log-likelihood gradients used for 
    updating model parameters.
    

    Parameters
    ----------
    vis_shape : list.
        DESCRIPTION. Contains numbers used to decompose $N_v = \prod_{k} M_k^{(v)}$.
    hid_shape : list.
        DESCRIPTION. Contains numbers used to decompose $N_h = \prod_{k} M_k^{(h)}$.
    r : list of ranks of the tensor decomposition.
        DESCRIPTION. ranks of the matrix product operator.

    Returns
    -------
    grad_visbiases : array,
        DESCRIPTION. Contains energy's gradient with respect to the visible bias parameter.
    grad_hidbiases : array.
        DESCRIPTION. Contains energy's gradient with respect to the latent bias parameter.
    grad_cores : dictionnary.
        DESCRIPTION. Contains energy's gradient with respect to the cores
        of the matrix product operator.

    '''
    
    sample_data = data
    for k in range(K):
        pos_hid_probs = probh_givenv(sample_data, hid_bias, cores,vis_shape, hid_shape)
        sample_data = probv_givenh(pos_hid_probs,vis_bias,cores,vis_shape, hid_shape)
           
    return pos_hid_probs, sample_data







