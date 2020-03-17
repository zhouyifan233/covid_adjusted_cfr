import pystan
import rpy2
import pandas as pd
import rpy2.robjects as robjects
import re
import numpy as np
import pickle
from PyStan_control_variate.ControlVariate.control_variate import control_variate_linear, control_variate_quadratic
from PyStan_control_variate.BasicFunction.process_stan_file import getParameterNames, verifyDataType


# parameters
model_file = 'models/model10.stan'
data_file = 'posterior_samples/data_S_model10.R'
# model_file = 'models/model14.stan'
# data_file = 'run_models/data_S_model14A1_2020-03-11-16-20-16.R'

# read posterior samples
# post_samples = pd.read_csv('posterior_samples/S_model10_2020-03-02-17-48-36_1.csv', comment='#')

# read R data file
#data_file = 'run_models/data_S_model13A1_2020-03-11-16-22-52.R'
robjects.globalenv.clear()
robjects.r['source'](data_file)
vars = list(robjects.globalenv.keys())

# prepare stan model
sm = pystan.StanModel(file=model_file)
with open('stan_model_10.bin', 'wb') as fid:
    pickle.dump(sm, fid)

# prepare data
data = {}
for var in vars:
    data_ = np.array(robjects.globalenv.find(var))
    if (data_.ndim == 1) and (data_.shape[0] == 1):
        data[var] = data_[0]
    else:
        data[var] = data_
data = verifyDataType(sm, data)

# sample
fit = sm.sampling(data=data, chains=1, iter=500, verbose=True,
                  control={'max_treedepth': 10, 'int_time': 72}, init=0.5)
with open('stan_fit_10.bin', 'wb') as fid:
    pickle.dump(fit, fid)

# extract
parameter_names, parameter_sizes = getParameterNames(sm)
parameter_extract = fit.extract()
num_of_iter = parameter_extract['lp__'].shape[0]

unconstrain_mcmc_samples = []
constrain_mcmc_samples = []
for i in range(num_of_iter):
    tmp_dict = {}
    for j, parameter_name in enumerate(parameter_names):
        tmp_dict[parameter_name] = parameter_extract[parameter_name][i]
    unconstrain_mcmc_samples.append(fit.unconstrain_pars(tmp_dict))
unconstrain_mcmc_samples = np.asarray(unconstrain_mcmc_samples)

grad_log_prob_val = []
for i in range(num_of_iter):
    grad_log_prob_val.append(fit.grad_log_prob(unconstrain_mcmc_samples[i], adjust_transform=False))
grad_log_prob_val = np.asarray(grad_log_prob_val)

cv_linear_mcmc_samples = control_variate_linear(unconstrain_mcmc_samples, grad_log_prob_val)

'''num_of_iter = post_samples['lp__'].shape[0]
unconstrain_mcmc_samples = []
constrain_mcmc_samples = []

for i in range(num_of_iter):
    tmp_dict = {}
    for j, parameter_name in enumerate(parameter_names):
        parameter_size = parameter_sizes[j]
        is_param = re.fullmatch(r'[A-Za-z_]+', parameter_size)
        if is_param is None:
            parameter_size = int(parameter_size)
            if parameter_size == 1:
                tmp_dict[parameter_name] = post_samples[parameter_name].iloc[i]
            elif parameter_size > 1:
                for k in range(1, parameter_size+1):
                    parameter_name_new = parameter_name + '.' + str(k)
                    tmp_dict[parameter_name_new] = post_samples[parameter_name_new].iloc[i]
            else:
                print('ERROR: parameters ' + parameter_name + ' size is invalid ' + str(parameter_size))
        else:

    unconstrain_mcmc_samples.append(fit.unconstrain_pars(tmp_dict))
unconstrain_mcmc_samples = np.asarray(unconstrain_mcmc_samples)'''


