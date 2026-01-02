# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Ting Hei Wan, Adeleke Maradesa, Baptiste Py'
__date__ = '28th June 2024'

"""
This file stores all the functions that are shared by all three DRT methods, i.e., simple, Bayesian, and Bayesian Hilbert Transform.
References: 
    [1] T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, Electrochimica Acta. 184 (2015) 483-499.
    [2] M. Saccoccio, T. H. Wan, C. Chen,F. Ciucci, Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: Ridge and lasso regression methods - A theoretical and experimental study, Electrochimica Acta. 147 (2014) 470-482.
    [3] J. Liu, T. H. Wan, F. Ciucci, A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357 (2020) 136864.
    [4] A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the regularization parameter in the distribution of relaxation times, Journal of the Electrochemical Society. 170 (2023) 030502.
"""

# Maths and data related packages
import numpy as np
import sys
from numpy import log, log10, sqrt
import pandas as pd
from math import pi
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import importlib
from cvxopt import matrix, solvers

# pyDRTtools related package  
import basics
import nearest_PD as nPD
# BHT and HMC not needed for basic DRT functionality
import time 

class EIS_object(object):
    
    # The EIS_object class stores the input data and the DRT result.
      
    def __init__(self, freq, Z_prime, Z_double_prime):
        
        """
        This is EIS_object class 
        Inputs:
            freq: frequency of the EIS measurement
            Z_prime: real part of the impedance
            Z_double_prime: imaginery part of the impedance
        """
        # define an EIS_object
        self.freq = freq
        self.Z_prime = Z_prime
        self.Z_double_prime = Z_double_prime
        self.Z_exp = Z_prime + 1j*Z_double_prime
        
        # keep a copy of the original data
        self.freq_0 = freq
        self.Z_prime_0 = Z_prime
        self.Z_double_prime_0 = Z_double_prime
        self.Z_exp_0 = Z_prime + 1j*Z_double_prime

        self.tau = 1/freq # we assume that the collocation points equal to 1/freq as default
        self.tau_fine  = np.logspace(log10(self.tau.min())-0.5,log10(self.tau.max())+0.5,10*freq.shape[0]) 
        ## select custom collocation
        # tau_fine = np.logspace(tau_min, tau_max, num = N_taus, endpoint=True)   

        self.method = 'none'
    
    @classmethod
    def from_file(cls,filename):
        
        if filename.endswith('.csv'): # import from csv file
            data = pd.read_csv(filename, header=None).to_numpy()
            freq = data[:, 0]
            Z_prime = data[:, 1]
            Z_double_prime = data[:, 2]
        
        elif filename.endswith('.txt'): # import from txt file
            data = np.loadtxt(filename)
            freq = data[:, 0]
            Z_prime = data[:, 1]
            Z_double_prime = data[:, 2]
    
        return cls(freq, Z_prime, Z_double_prime)
    
    def plot_DRT(self): # plot the DRT result
        
        basics.pretty_plot(4,4)
        plt.rc('font', family='serif', size=15)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('text', usetex=True)    
        
        if self.method == 'simple':    
            plt.plot(self.out_tau_vec, self.gamma, 'k')
            y_min = 0
            y_max = max(self.gamma)
            
        elif self.method == 'credit':
            plt.fill_between(self.out_tau_vec, self.lower_bound, self.upper_bound,  facecolor='lightgrey')
            plt.plot(self.out_tau_vec, self.gamma, color='black', label='MAP')
            plt.plot(self.out_tau_vec, self.mean, color='blue', label='mean')
            plt.plot(self.out_tau_vec, self.lower_bound, color='black', linewidth=1)
            plt.plot(self.out_tau_vec, self.upper_bound, color='black', linewidth=1)
            plt.legend(frameon=False, fontsize = 15)
            y_min = 0
            y_max = max(self.upper_bound)
            
        elif self.method == 'BHT':    
            plt.semilogx(self.out_tau_vec, self.mu_gamma_fine_re, 'b', linewidth=1)
            plt.semilogx(self.out_tau_vec, self.mu_gamma_fine_im, 'k', linewidth=1)
            y_min = min(np.concatenate((self.mu_gamma_fine_re, self.mu_gamma_fine_im)))
            y_max = max(np.concatenate((self.mu_gamma_fine_re, self.mu_gamma_fine_im)))
        
        else:
            return
        
        plt.xscale('log')
        plt.xlim(self.out_tau_vec.min(), self.out_tau_vec.max())
        plt.ylim(y_min, y_max*1.1)
        plt.xlabel(r'$f/{\rm Hz}$', fontsize=20)
        plt.ylabel(r'$\gamma(\tau)/\Omega$', fontsize=20)
    
        plt.show()

# 
def simple_run(entry, rbf_type = 'Gaussian', data_used = 'Combined Re-Im Data', induct_used = 1, der_used = '1st order', cv_type = 'GCV', reg_param = 1E-3, shape_control = 'FWHM Coefficient', coeff = 0.5):
    
    
    """
    This function enables to compute the DRT using ridge regression (also known as Tikhonov regression)
    References:
        T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, Electrochimica Acta 184 (2015) 483-499.
    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression
        reg_param: regularization parameter applied when "custom" is used for cv_type 
        shape_control: option for controlling the shape of the radial basis function (RBF) 
        coeff: magnitude of the shape control
    """
    
    # Step 1.1: define the optimization bounds
    N_freqs = entry.freq.shape[0]
    N_taus = entry.tau.shape[0]
    ###
    entry.b_re = entry.Z_exp.real
    entry.b_im = entry.Z_exp.imag

    # Step 1.2: compute epsilon
    entry.epsilon = basics.compute_epsilon(entry.freq, coeff, rbf_type, shape_control)
    
    # Step 1.3: compute A matrix
    ## assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type)
    entry.A_re_temp = basics.assemble_A_re(entry.freq, entry.tau, entry.epsilon, rbf_type)
    entry.A_im_temp = basics.assemble_A_im(entry.freq, entry.tau, entry.epsilon, rbf_type)
    
    # Step 1.4: compute M matrix  assemble_M_1(tau_vec, epsilon, rbf_type)
    if der_used == '1st order':
        entry.M_temp = basics.assemble_M_1(entry.tau, entry.epsilon, rbf_type)
    elif der_used == '2nd order':
        entry.M_temp = basics.assemble_M_2(entry.tau, entry.epsilon, rbf_type)
    
    # Step 2: conduct ridge regularization
    if data_used == 'Combined Re-Im Data': # select both parts of the impedance for the simple run
 
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            N_RL = 1 # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:,N_RL:] = entry.A_re_temp
            entry.A_re[:,0] = 1
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:,N_RL:] = entry.A_im_temp
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            # initial guess for the hyperparameter
            log_lambda_0 = log(reg_param) # initial guess for lambda
            #
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
                
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            H_combined,c_combined = basics.quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, entry.lambda_value)

            # enforce positivity constraint # N_RL
            ## bound matrix
            G = matrix(-np.identity(entry.b_re.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_re.shape[0]+N_RL))

            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_combined), matrix(c_combined),G,h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler, it will be used if needed
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im

            # only consider std of residuals in both parts
            sigma_re_im = np.std(np.concatenate([entry.res_re,entry.res_im]))
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
            Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_re.T@inv_V@entry.b_re + entry.A_im.T@inv_V@entry.b_im
           
        elif induct_used == 1: # considering the inductance
            N_RL = 2
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            entry.A_re[:,1] = 1
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:,0] = 2*pi*entry.freq
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            log_lambda_0 = log(reg_param) # initial guess for lambda
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
                
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            H_combined,c_combined = basics.quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, entry.lambda_value)

            # enforce positivity constraint # N_RL
            ## bound matrix
            G = matrix(-np.identity(entry.b_re.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_re.shape[0]+N_RL))

            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_combined), matrix(c_combined),G,h)
            x = np.array(sol['x']).flatten()
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im

            # only consider std of residuals in both parts
            sigma_re_im = np.std(np.concatenate([entry.res_re,entry.res_im]))
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
            Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_re.T@inv_V@entry.b_re + entry.A_im.T@inv_V@entry.b_im
            
    elif data_used == 'Im Data': # select the imaginary part of the impedance for the simple run
        
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            N_RL = 0 # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            log_lambda_0 = log(reg_param) # initial guess for lambda
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
                
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            H_im, c_im = basics.quad_format_separate(entry.A_im, entry.b_im, entry.M, entry.lambda_value)

            # enforce positivity constraints
            ## bound matrix
            G = matrix(-np.identity(entry.b_im.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_im.shape[0]+N_RL))

            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_im), matrix(c_im),G,h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im
            
            # only consider std of residuals in the imaginary part
            sigma_re_im = np.std(entry.res_im)
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
            
            Sigma_inv = (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_im.T@inv_V@entry.b_im
            
        elif induct_used == 1: # considering the inductance
            N_RL = 1
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:,0] = 2*pi*entry.freq
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            log_lambda_0 = log(reg_param) # initial guess for lambda
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            
            H_im, c_im = basics.quad_format_separate(entry.A_im, entry.b_im, entry.M, entry.lambda_value)
            #
            # enforce positivity constraints
            # bound matrix
            G = matrix(-np.identity(entry.b_im.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_im.shape[0]+N_RL))
            # Formulate the quadratic programming problem
            ##
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_im), matrix(c_im),G,h)
            x = np.array(sol['x']).flatten()
            # prepare for HMC sampler
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im
            
            # only consider std of residuals in the imaginary part
            sigma_re_im = np.std(entry.res_im)
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
            
            Sigma_inv = (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_im.T@inv_V@entry.b_im

    elif data_used == 'Re Data': # select the real part of the impedance for the simple run
        N_RL = 1
        entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
        entry.A_re[:, N_RL:] = entry.A_re_temp
        entry.A_re[:,0] = 1
        
        entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
        entry.A_im[:, N_RL:] = entry.A_im_temp
        entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
        entry.M[N_RL:,N_RL:] = entry.M_temp
        
        # optimally select the regularization level
        log_lambda_0 = log(reg_param) # initial guess for lambda
        if cv_type=='custom':
            entry.lambda_value = reg_param
        else:
            entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
        print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
        
        # recover the DRT using cvxopt 
        H_re,c_re = basics.quad_format_separate(entry.A_re, entry.b_re, entry.M, entry.lambda_value)
    
        # enforce positivity constraints
        # ## bound matrix
        G = matrix(-np.identity(entry.b_re.shape[0]+N_RL))
        h = matrix(np.zeros(entry.b_re.shape[0]+N_RL))
        # Formulate the quadratic programming problem
        ###
        # Solve the quadratic programming problem
        sol = solvers.qp(matrix(H_re), matrix(c_re),G,h)
        x = np.array(sol['x']).flatten()
        # prepare for HMC sampler
        entry.mu_Z_re = entry.A_re@x
        entry.mu_Z_im = entry.A_im@x       
        entry.res_re = entry.mu_Z_re-entry.b_re
        entry.res_im = entry.mu_Z_im-entry.b_im
        
        # only consider std of residuals in the real part
        sigma_re_im = np.std(entry.res_re)
        inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
        Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.lambda_value/sigma_re_im**2)*entry.M
        mu_numerator = entry.A_re.T@inv_V@entry.b_re

    entry.Sigma_inv = (Sigma_inv+Sigma_inv.T)/2
    
    # test if the covariance matrix is positive definite
    if (nPD.is_PD(entry.Sigma_inv)==False):
        entry.Sigma_inv = nPD.nearest_PD(entry.Sigma_inv) # if not, use the nearest positive definite matrix
    
    L_Sigma_inv = np.linalg.cholesky(entry.Sigma_inv)
    entry.mu = np.linalg.solve(L_Sigma_inv, mu_numerator)
    entry.mu = np.linalg.solve(L_Sigma_inv.T, entry.mu)
    # entry.mu = np.linalg.solve(entry.Sigma_inv, mu_numerator)
    
    # Step 3: obtaining the result of inductance, resistance, and gamma  
    if N_RL == 0: 
        entry.L, entry.R = 0, 0        
    elif N_RL == 1 and data_used == 'Im Data':
        entry.L, entry.R = x[0], 0    
    elif N_RL == 1 and data_used != 'Im Data':
        entry.L, entry.R = 0, x[0]
    elif N_RL == 2:
        entry.L, entry.R = x[0:2]
        
    entry.x = x[N_RL:]
    entry.out_tau_vec, entry.gamma = basics.x_to_gamma(x[N_RL:], entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.N_RL = N_RL 
    entry.method = 'simple'
    
    return entry
