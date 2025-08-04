# -*- coding: utf-8 -*-
""" This module contains the model class for canonical format Gaussian restricted
Boltzmann machine (CF-GRBM). It implements the tensor form of Gaussian restricted 
Boltzmann Machine using canonical format for the weight matrix. The decomposition
used assumes that $N_v=M_1^{(v)}*M_2^{(v)}$ and $N_h=M_1^{(h)}*M_2^{(h)}.

Created on Sat Jul 16 19:39:13 2022

@author: Bruno Monsia

     :Implemented:
        - CF-GRBM model

    :Version:
        1.0.0
        
    :Contact:
        bruno.monsia@umontreal.ca
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
    gs = gridspec.GridSpec(10,10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')
        

def canonical_params_initialize(vis_shape, hid_shape, r):
    ''' This function initializes the GRBM parameters in its canonical format.
    

    Parameters
    ----------
    vis_shape : list of integers used to decompose the observed data dimension
    ($N_v = M_1^{(v)}*M_2^{(v)}$).
        DESCRIPTION. shape of the observed variable dimension.
    hid_shape : list of integers used to decompose the number of latent variables
    ($N_h=M_1^{(h)}*M_2^{(h)}$).
        DESCRIPTION. shape of the latent variable dimension.
    r : r : int
        DESCRIPTION. rank of the canonical decomposition.

    Returns
    -------
           
    visbiases : array of shape ($M_1^{(v)},M_2^{(v)}$).
        DESCRIPTION. bias associated with the observed variable in tensor format.
    hidbiases : array of shape ($M_1^{(h)},M_2^{(h)}$)
        DESCRIPTION. bias associated with the latent variable in tensor format.
    cores : dictionnary. 
        DESCRIPTION. Contains the factors of the canonical decomposition.

    '''
    
    d = len(vis_shape)
    cores = {}
    for k in range(d):
        cores['G'+str(k)] =\
            tf.Variable(tf.random.normal(shape=(vis_shape[k], r, hid_shape[k]),
                                         stddev = 1, seed=1), name= 'G'+str(k))
    visbiases = tf.Variable(0.1*tf.ones(shape = vis_shape, dtype=tf.dtypes.float32), name='visbiases')
    hidbiases = tf.Variable(tf.zeros(shape = hid_shape, dtype=tf.dtypes.float32), name='hidbiasess')
    
    return cores, visbiases, hidbiases        

def canonical_grads_initialize(vis_shape, hid_shape, r):
    ''' This function initializes the GRBM parameters in its canonical format.
    

    Parameters
    ----------
    vis_shape : list of integers used to decompose the observed data dimension
    ($N_v = M_1^{(v)}*M_2^{(v)}$).
        DESCRIPTION. shape of the observed variable dimension.
    hid_shape : list of integers used to decompose the number of latent variables
    ($N_h=M_1^{(h)}*M_2^{(h)}$).
        DESCRIPTION. shape of the latent variable dimension.
    r : int
        DESCRIPTION. rank of the canonical decomposition.

    Returns
    -------
    grad_visbiases : array,
        DESCRIPTION. Contains energy's gradient with respect to the visible bias parameter.
    grad_hidbiases : array.
        DESCRIPTION. Contains energy's gradient with respect to the latent bias parameter.
    grad_cores : dictionnary.
        DESCRIPTION. Contains energy's gradient with respect to the factors
        of the canonical decomposition.

    '''
    
    total_shape = vis_shape + hid_shape
    d = len(total_shape)
    grad_cores = {}
    for k in range(d):
        grad_cores['gradG'+str(k)] = tf.zeros(shape = (total_shape[k], r), name = 'gradG'+ str(k))
    grad_visbiases = tf.zeros( shape = vis_shape, dtype=tf.dtypes.float32, name='grad_visbiases')
    grad_hidbiases = tf.zeros( shape = hid_shape, dtype=tf.dtypes.float32, name='grad_hidbiasess')
    
    return grad_cores, grad_visbiases, grad_hidbiases

def initialize_parameters(vis_shape, hid_shape, r):
    ''' This function initializes the GRBM parameters in its canonical format.
    

    Parameters
    ----------
    vis_shape : list of integers used to decompose the observed data dimension
    ($N_v = M_1^{(v)}*M_2^{(v)}$).
        DESCRIPTION. shape of the observed variable dimension.
    hid_shape : list of integers used to decompose the number of latent variables
    ($N_h=M_1^{(h)}*M_2^{(h)}$).
        DESCRIPTION. shape of the latent variable dimension.
    r : int
        DESCRIPTION. rank of the canonical decomposition.    

    Returns
    -------
        
    visbiases : array of shape ($r_{k-1}, M_k^{(v)}, r_{k}$).
        DESCRIPTION. bias associated with the observed variable in tensor format.
    hidbiases : array of shape ($r_{k-1}, M_k^{(h)}, r_{k}$)
        DESCRIPTION. bias associated with the latent variable in tensor format.
    cores : dictionnary. 
        DESCRIPTION. Contains the factors of the canonical decomposition.

    '''
    
    total_shape = vis_shape + hid_shape
    l_shape = len(total_shape)
    cores = {}
    for k in range(l_shape):
        cores['G'+str(k)] = tf.Variable(0.1*tf.random.normal(shape = (total_shape[k],r), seed=1), name= 'G'+str(k))
    visbiases = tf.Variable(0.1*tf.ones( shape=vis_shape, dtype=tf.dtypes.float32, name='visbiases'))
    hidbiases = tf.Variable(tf.zeros( shape=hid_shape, dtype=tf.dtypes.float32, name='hidbiasess'))
    
    return visbiases, hidbiases, cores   

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
        DESCRIPTION. Contains the factors of the canonical decomposition.
    sigma : scalar (float) or array.
        DESCRIPTION. standard deviation of v|h.

    Returns
    -------
    probs_h_given_v : array of shape ($N,M_1^{(h)},M_2^{(h)}$).
        DESCRIPTION. conditional probability of the latent variables given 
        observed variables v.
    sample_binary_hid : array of shape ($N, M_1^{(h)},M_2^{(h)}$).
        DESCRIPTION. generated binary data using conditional probability $p(v|h;\theta)$.

    '''
    
    v = v/sigma**2    
    term = tf.einsum("nbd,ba,da,fa,ha->nfh", v, cores['G0'], cores['G1'], cores['G2'], cores['G3'])
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
        DESCRIPTION. Contains the canonical of the canonical decomposition.
    sigma : scalar (float) or array.
        DESCRIPTION. standard deviation of v|h.

    Returns
    -------        
    probs_v_given_h : array of shape ($N, M_1^{(v)},M_2^{(v)}$).
        DESCRIPTION. conditional probability of the observed variables given 
        latent variables h.
    sample_binary_vis : array of shape ($N, M_1^{(v)},M_2^{(v)}$).
        DESCRIPTION. generated data using conditional probability $p(v|h;\theta)$.

    '''
    
    term = tf.einsum("nfh,ba,da,fa,ha->nbd", h, cores['G0'], cores['G1'], cores['G2'], cores['G3'])
    sample_vis = term + a # + sigma*tf.random.normal(shape=a.numpy().shape())    
    probs_v_given_h = tf.sigmoid(sample_vis)
    
    return probs_v_given_h, sample_vis
       
def reconstructor_weight(cores):
    ''' This function reconstructs the weight matrix from the factors of the
    canonical decomposition.
    

    Parameters
    ----------
    cores : dictionnary.
        DESCRIPTION. Contains the factors of the canonical decomposition.

    Returns
    -------
    reconst_weight : array of shape ($N_v, N_h$)
        DESCRIPTION. The original weight matrix.

    '''
    
    reconst_weight =\
        tf.einsum("abc,cde,efg,gha->bdfh", cores['G0'], cores['G1'], cores['G2'], cores['G3'])
    reconst_weight = tf.squeeze(reconst_weight)
    
    return reconst_weight

def gradients_energy_cores(v, h, cores):
    ''' Computes the gradients of the energy function with respect to factors of
    the canonical decomposition.
    

    Parameters
    ----------
    v : array of shape ($N,M_1^{(v)},M_2^{(v)}$)
        DESCRIPTION. It is the observed dataset.
    h : array of shape ($N,M_1^{(h)},M_2^{(h)}$).
        DESCRIPTION. Represents latent data.
    cores : dictionnary.
        DESCRIPTION. Contains the factors of the canonical decomposition.

    Returns
    -------
    gradE_cores : dictionnary.
        DESCRIPTION. Contains the gradients of energy function with respect 
        to the factors.

    '''
    
    gradE_cores = {}
    gradE_cores['gradE_G' + str(0)] =\
        tf.math.reduce_mean(tf.einsum("nbd,da,fa,ha,nfh->nba", v, cores['G1'],
                                      cores['G2'], cores['G3'],h), axis=0) 
    gradE_cores['gradE_G' + str(1)] =\
        tf.math.reduce_mean(tf.einsum("nbd,ba,fa,ha,nfh->nda", v, cores['G0'],
                                      cores['G2'], cores['G3'],h), axis=0) 
    gradE_cores['gradE_G' + str(2)] =\
        tf.math.reduce_mean(tf.einsum("nbd,ba,da,ha,nfh->nfa", v, cores['G0'],
                                      cores['G1'], cores['G3'],h), axis=0) 
    gradE_cores['gradE_G' + str(3)] =\
        tf.math.reduce_mean(tf.einsum("nbd,ba,da,fa,nfh->nha", v, cores['G0'],
                                      cores['G1'], cores['G2'],h), axis=0)
    
    return gradE_cores

def energy_function(data, h, a, b, cores, sigma):
    '''Computes energy function of GRBM model.
    

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
        DESCRIPTION. Contains the factors of the canonical decomposition.
    sigma : scalar (float) or array.
        DESCRIPTION. standard deviation of v|h.

    Returns
    -------
    energy: array of shape (N,).
        DESCRIPTION. Energy function.
    '''
    
    data_splited = data/(sigma**2)
    term = (data-a)**2/(sigma**2)
    energy = tf.einsum("nbd,nfh,ba,da,fa,ha->n", data_splited, h, cores['G0'],
                       cores['G1'], cores['G2'], cores['G3']) \
        + tf.math.reduce_sum(term, axis=(1,2)) + tf.einsum("nfh,fh->n", h, b)
    energy = -energy
    
    return energy

def gradient_energy_tf(energy, a, b, cores):
    ''' This function returns the gradients of the energy function with respect
    to GRBM parameters using a tensorflow function.
    

    Parameters
    ----------
    energy : array of shape (N,).
        DESCRIPTION. Energy function.
    a : array of shape $(M_1^{(v)},M_2^{(v)})$.
        DESCRIPTION. bias associated with observed variables in tensorizing of GRBM.
    b : array of shape $(M_1^{(h)},M_2^{(h)})$.
        DESCRIPTION. bias associated with latent variables in tensorizing of GRBM.
    cores : dictionnary.
        DESCRIPTION. Contains the factors of the canonical decomposition.

    Returns
    -------
    grad : array
        DESCRIPTION.

    '''
    
    grad = tf.gradients( energy, [a, b], stop_gradients=[a, b])    
    
    return grad    

def reconstructor_error(data, sample_data ):
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
    individual_errors =\
        tf.math.sqrt(tf.math.reduce_sum((data - sample_data)**2, axis = axis_))   
    error = tf.math.reduce_mean(tf.math.sqrt(individual_errors)).numpy()
    
    return error

def initialize_gradients(vis_shape, hid_shape, r):
    ''' This function initializes the log-likelihood gradients used for 
    updating model parameters.
    

    Parameters
    ----------
    vis_shape : list.
        DESCRIPTION. Contains numbers used to decompose $N_v = \prod_{k} M_k^{(v)}$.
    hid_shape : list.
        DESCRIPTION. Contains numbers used to decompose $N_h = \prod_{k} M_k^{(h)}$.
    r : int
        DESCRIPTION. rank of the canonical decomposition.

    Returns
    -------
    grad_visbiases : array,
        DESCRIPTION. Contains energy's gradient with respect to the visible bias parameter.
    grad_hidbiases : array.
        DESCRIPTION. Contains energy's gradient with respect to the latent bias parameter.
    grad_cores : dictionnary.
        DESCRIPTION. Contains energy's gradient with respect to the factors
        of the canonical decomposition.

    '''
    
    total_shape = vis_shape + hid_shape    
    d = len(total_shape)
    grad_cores = {}
    for k in range(d):
        grad_cores['gradG'+str(k)] = tf.zeros(shape = (total_shape[k], r), name = 'gradG'+ str(k))
    grad_visbiases = tf.zeros(shape = vis_shape, dtype=tf.dtypes.float32, name='grad_visbiases')
    grad_hidbiases = tf.zeros(shape = hid_shape, dtype=tf.dtypes.float32, name='grad_hidbiasess')
    
    return grad_visbiases, grad_hidbiases, grad_cores 

def CD_algorithm(data, vis_bias, hid_bias, cores, vis_shape, hid_shape, K):
    ''' This function implements the Gibbs sampler by initializing the Markov 
    chain with the observed data. It gives generated data used for contrastive 
    divergence algorithm.
    

    Parameters
    ----------        
    data : array of shape ($N,M_1^{(v)},M_2^{(v)}$).
        DESCRIPTION. It is the observed dataset.
    vis_bias : TYPE
        DESCRIPTION.
    hid_bias : TYPE
        DESCRIPTION.
    cores : dictionnary.
        DESCRIPTION. Contains the cores of the canonical decomposition.
    vis_shape :  list.
        DESCRIPTION. Contains the numbers used to decompose $N_v = \prod_{k} M_k^{(v)}$.
    hid_shape : list.
        DESCRIPTION. Contains numbers used to decompose $N_h = \prod_{k} M_k^{(h)}$.
    K : int, optional  
        DESCRIPTION. The default is 1. It is the number of iterations of the Gibbs sampler.

    Returns
    -------    
    pos_hid_probs : array of shape ($N,M_1^{(h)},M_2^{(h)}$).
        DESCRIPTION. generated data from $p(h;\theta)$ using Gibbs sampler. 
    sample_data :  array of shape ($N,M_1^{(v)},M_2^{(v)}$).
        DESCRIPTION. generated data from $p(v;\theta)$ using Gibbs sampler.

    '''
    
    sample_data = data
    for k in range(K):
        pos_hid_probs = probh_givenv(sample_data, hid_bias, cores,vis_shape, hid_shape)
        sample_data = probv_givenh(pos_hid_probs,vis_bias,cores,vis_shape, hid_shape)
           
    return pos_hid_probs, sample_data






