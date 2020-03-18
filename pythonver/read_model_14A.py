import pystan
import rpy2
import pandas as pd
import rpy2.robjects as robjects
import re
import numpy as np
import pickle
from PyStan_control_variate.ControlVariate.control_variate import control_variate_linear, control_variate_quadratic
from PyStan_control_variate.BasicFunction.process_stan_file import getParameterNames, verifyDataType


#
data_file = 'posterior_samples/data_S_model10.R'

# read model and sampling
with open('stan_model_14A1.bin', 'rb') as fid:
    sm = pickle.load(fid)
with open('stan_fit_14A1.bin', 'rb') as fid:
    fit = pickle.load(fid)

# read data
robjects.globalenv.clear()
robjects.r['source'](data_file)
vars = list(robjects.globalenv.keys())

# prepare data
data = {}
for var in vars:
    data_ = np.array(robjects.globalenv.find(var))
    if (data_.ndim == 1) and (data_.shape[0] == 1):
        data[var] = data_[0]
    else:
        data[var] = data_
data = verifyDataType(sm, data)

# extract
parameter_names, parameter_sizes = getParameterNames(sm)
parameter_extract = fit.extract()
num_of_iter = parameter_extract['lp__'].shape[0]

unconstrain_mcmc_samples = []
constrain_mcmc_samples = []
constrain_mcmc_list = []
for i in range(num_of_iter):
    tmp_dict = {}
    tmp_array = []
    for j, parameter_name in enumerate(parameter_names):
        tmp_dict[parameter_name] = parameter_extract[parameter_name][i]
        tmp = np.array(tmp_dict[parameter_name])
        if tmp.ndim == 0:
            tmp_array.append(np.expand_dims(tmp, axis=0))
        else:
            tmp_array.append(tmp)
    unconstrain_mcmc_samples.append(fit.unconstrain_pars(tmp_dict))
    constrain_mcmc_list.append(np.concatenate(tmp_array))
unconstrain_mcmc_samples = np.asarray(unconstrain_mcmc_samples)
constrain_mcmc_array = np.asarray(constrain_mcmc_list)
mcmc_dim = constrain_mcmc_array.shape[1]
mean_constrain_mcmc = np.mean(constrain_mcmc_array, axis=0)
var_constrain_mcmc = np.var(constrain_mcmc_array, axis=0)

grad_log_prob_val = []
for i in range(num_of_iter):
    grad_log_prob_val.append(fit.grad_log_prob(unconstrain_mcmc_samples[i], adjust_transform=False))
grad_log_prob_val = np.asarray(grad_log_prob_val)

unconstrain_cv_linear_mcmc_samples = control_variate_linear(unconstrain_mcmc_samples, grad_log_prob_val)

# convert unconstrained parameters to constrained ones
# print(cv_linear_mcmc_samples[0])
constrain_cv_linear = []
for row in unconstrain_cv_linear_mcmc_samples:
    constrain_cv_linear.append(fit.constrain_pars(row))
constrain_cv_linear = np.asarray(constrain_cv_linear)
mean_constrain_cv_linear = np.mean(constrain_cv_linear, axis=0)
var_constrain_cv_linear = np.var(constrain_cv_linear, axis=0)

print('mcmc samples mean: ')
print(mean_constrain_mcmc)
print('mcmc samples variance:')
print(var_constrain_mcmc)
print('linear cv means: ')
print(mean_constrain_cv_linear[0:mcmc_dim])
print('linear cv variance: ')
print(var_constrain_cv_linear[0:mcmc_dim])
print('improvement: ')
print(np.mean(var_constrain_mcmc/var_constrain_cv_linear[0:mcmc_dim]))
