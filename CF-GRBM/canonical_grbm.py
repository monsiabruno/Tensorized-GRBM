# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:00:53 2022

@author: Bruno Monsia

     :Implemented:
        - CF-GRBM model

    :Version:
        1.0.0
        
    :Contact:
        bruno.monsia@umontreal.ca
"""

import os
path = "C:/Users/monsi/OneDrive - Universite de Montreal/Theme_doc_2019/Tensorization/Code_python/final codes/tgrbm"
os.chdir(path)
import tensorflow as tf
import numpy as np
import helpers_canonical_tgrbm as helpers

class CFgrbm():
    
    def __init__(self, sigma=1, hid_shape=[20,25], r=1, maxepoch=5, 
                 batchsize=100, K=1, momentum=0.7, learning_rate=0.01,                 
                 lambda_1=0.0, lambda_2=0.000, verbose=True):
        ''' This function initializes the CF-GRBM model learning algorithm. 
        

        Parameters
        ----------
        sigma : scalar (float) or array, optional
            DESCRIPTION. standard deviation of v|h. The default is 1.0.
        hid_shape : list, optional
            DESCRIPTION. The default is [20,25]. Contains the numbers used to 
            decompose $N_h = \prod_{k} N_k^{(h)}$.
        r : list, optional
            DESCRIPTION. The default is [2,5,5,5,2]. It is the ranks of the 
            tensor train decomposition.
        maxepoch : int, optional
            DESCRIPTION. The default is 5.
        batchsize : int, optional
            DESCRIPTION. The default is 100. It is the iteration number for the 
            learning algorithm.
        K : int, optional
            DESCRIPTION. The default is 1. It is the number of iterations of
            the Gibbs sampler.
        momentum : scalar (float), optional
            DESCRIPTION. The default is 0.7. It should be between 0.0 and 1.0.
        learning_rate : TYPE, optional
            DESCRIPTION. The default is 0.01.
        lambda_1 : float, optional
            DESCRIPTION. Represents $L_1$ regulazation term. The default is 0.0. 
        lambda_2 : float, optional
            DESCRIPTION. Represents $L_2$ regulazation term. The default is 0.0001.
        verbose : bool, optional        
            DESCRIPTION. The default is False. If set to True, will print 
            learning progression.

        Returns
        -------
        None.

        '''
        
        self.sigma = sigma
        self.hid_shape = hid_shape
        self.r = r
        self.maxepoch = maxepoch
        self.batchsize = batchsize
        self.K = K
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.verbose = verbose     
            
    def gradients(self, data):
        ''' This function estimates the gradients of the log-likelihood with 
        respect to model parameters.        

        Parameters
        ----------
        data : array of shape ($N,N_1^{(v)},N_2^{(v)}$).
            DESCRIPTION. It is the observed dataset.

        Returns
        -------
        None.

        '''
        
        grad_cores = {}        
        vis_bias = tf.reshape( tf.tile(self.visbiases, [self.batchsize, 1]),
                              shape = [self.batchsize] + self.vis_shape)
        hid_bias = tf.tile(self.hidbiases, [self.batchsize, 1])
        hid_bias = tf.reshape( hid_bias, shape = [self.batchsize] + self.hid_shape)
        
        sample_data = data
        for k in range(self.K):
            # Positive phase 
            probs_hid_pos, pos_state_hid =\
                helpers.probh_givenv(sample_data, hid_bias, self.cores, self.sigma)
            # First terms of gradients
            if k == 0:
                first_vis_term = tf.reduce_mean(sample_data/self.sigma**2, axis = 0)
                first_hid_term = tf.reduce_mean(probs_hid_pos, axis = 0)
                first_cores_term =\
                    helpers.gradients_energy_cores(sample_data/self.sigma**2, probs_hid_pos, self.cores)
            # Negative phase
            _, sample_data = helpers.probv_givenh(pos_state_hid, vis_bias, self.cores, self.sigma) 
            probs_hid_neg , _ = helpers.probh_givenv(sample_data, hid_bias, self.cores, self.sigma)
        self.reconst_data = sample_data
        # Second terms of gradients        
        second_vis_term = tf.reduce_mean(sample_data/self.sigma**2, axis = 0)
        second_hid_term = tf.reduce_mean(probs_hid_neg, axis = 0)
        second_cores_term =\
            helpers.gradients_energy_cores(sample_data/self.sigma**2, probs_hid_neg, self.cores)
        # Gradients
        grad_visbiases = tf.math.subtract(first_vis_term, second_vis_term)
        grad_hidbiases = tf.math.subtract(first_hid_term,second_hid_term)
        for l in range(self.l_shape):
            grad_cores['gradG' + str(l)] =\
                tf.math.subtract( first_cores_term['gradE_G'+str(l)],
                                 second_cores_term['gradE_G' + str(l)]) -\
                    self.lambda_1*self.cores['G'+str(l)] -\
                        self.lambda_2*tf.math.sign(self.cores['G'+str(l)])
        self.gradient_visbiases =\
            self.momentum*self.gradient_visbiases + self.learning_rate*grad_visbiases
        self.gradient_hidbiases =\
            self.momentum*self.gradient_hidbiases + self.learning_rate*grad_hidbiases
        for l in range(self.l_shape):
            self.gradient_cores['gradG' + str(l)] =\
                self.momentum*self.gradient_cores['gradG'+ str(l)] +\
                    (self.learning_rate)*grad_cores['gradG' + str(l)]           
        
        
    @tf.function 
    def gradient_energy_tf(energy, a, b, cores):
        ''' This function returns the gradients of the energy function with 
        respect to CF-GRBM parameters using a tensorflow function.        

        Parameters
        ----------
        energy: array of shape (N,).
            DESCRIPTION. Energy function.
        a : array of shape $(N_1^{(v)},N_2^{(v)})$.
            DESCRIPTION. bias associated with observed variables in tensorizing of GRBM.
        b : array of shape $(N_1^{(h)},N_2^{(h)})$.
            DESCRIPTION. bias associated with latent variables in tensorizing of GRBM.
        cores : dictionnary.
            DESCRIPTION. Contains the factors of the canonical decomposition.

        Returns
        -------
        energy_grad : array
            DESCRIPTION.

        '''
        energy_grad = tf.gradients(energy, [a, b], stop_gradients=[a, b])
        
        return energy_grad          
    
    def update_params_CFgrbm(self, gradient_visbiases, gradient_hidbiases, gradient_cores):
        '''This function updates CF-GRBM model parameters at the t-th iteration.        

        Parameters
        ----------        
        gradient_visbiases : array
            DESCRIPTION. Gradient of the log-likelihood with respect to the 
            visible bias.
        gradient_hidbiases : array
            DESCRIPTION. Gradient of the log-likelihood with respect to the 
            latent bias.
        gradient_cores : dictionnary
            DESCRIPTION. Contains gradient of the log-likelihood with respect
            to the factors.

        Returns
        -------
        None.

        '''
        
        self.visbiases.assign_add(gradient_visbiases)
        self.hidbiases.assign_add(gradient_hidbiases)
        for l in range(self.l_shape):
            self.cores['G'+str(l)].assign_add(gradient_cores['gradG'+ str(l)])         
    
    def optimize_CFgrbm(self, dataset):
        ''' Optimizes CF-GRBM model learning algorithm.        

        Parameters
        ----------
        dataset : array
            DESCRIPTION. Dataset used to train the model.

        Returns
        -------
        None.

        '''
        
        self.vis_shape = list(dataset.shape[1:])
        self.l_shape = len(self.vis_shape + self.hid_shape)
        numbatches = dataset.shape[0]//self.batchsize
        batchdata = tf.convert_to_tensor(helpers.makebatch(dataset, self.batchsize), 
                                         dtype = tf.dtypes.float32)
        
        self.visbiases, self.hidbiases, self.cores =\
            helpers.initialize_parameters(self.vis_shape, self.hid_shape, self.r)
        self.gradient_visbiases, self.gradient_hidbiases, self.gradient_cores = \
            helpers.initialize_gradients(self.vis_shape, self.hid_shape, self.r)
        self.reconst_error = []
        for epoch in range(self.maxepoch):
            batch_error = []
            for batch in range(numbatches):
                data = batchdata[..., batch]
                #grad_visbiases, grad_hidbiases, grad_cores, reconst_data = self.gradients(data)
                self.gradients(data)
                self.update_params_CFgrbm( self.gradient_visbiases, self.gradient_hidbiases, self.gradient_cores)
                error = helpers.reconstructor_error(data, self.reconst_data)
                batch_error.append(error)
                
                if self.verbose & (batch % 599 == 0) & (epoch > (self.maxepoch-2)):
                    print("Reconstructor error of epoch {} and bath {} is :".format(epoch, batch), error)
                    helpers.plot(self.reconst_data.numpy()) 
            epoch_error = np.mean(batch_error)
            self.reconst_error.append(epoch_error)            
            
    def run_CFgrbm(self, dataset):
        ''' Runs CF-GRBM algorithm learning.        

        Parameters
        ----------
        dataset : array
            DESCRIPTION. Dataset used to train the model.

        Returns
        -------
        None.

        '''
        
        self.vis_shape = list(dataset.shape[1:])        
        self.optimize_CFgrbm(dataset)     
                
    def get_parameter_CFgrbm(self):
        ''' Gets learned parameters after learning is done.
        

        Returns
        -------
        params: list
            DESCRIPTION. learned parameters.

        '''
        params = [self.visbiases, self.hidbiases, self.cores]
        
        return params
    
    def reconstructor_data(self):
        ''' Gets some generated data sample from the last iteration.        

        Returns
        -------
        array
            DESCRIPTION. generated data.

        '''
        
        print("recontructor data of last batch and epoch is:") 
        
        return self.reconst_data    
    
    def reconstructor_weight(self): 
        ''' reconstructs the weight matrix from the factors of the canonical 
        decomposition.        

        Returns
        -------
        array
            DESCRIPTION. weight matrix. Reconstructs the original weight.        

        '''
        
        return helpers.reconstructor_weight(self.cores)  


  
