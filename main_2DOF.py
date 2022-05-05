"""
@author: Mukesh K. Ramancha

main file to test transitional mcmc implementation
example: 2D0F system

To execute this file:
use following command on cmd promt if parallel_processing = 'multiprocessing'
    python main_2DOF.py
use following command on cmd promt if parallel_processing = 'mpi'
    mpiexec -n 8 python -m mpi4py.futures main_2DOF.py
"""

import numpy as np
import pickle
from tmcmc_mod import pdfs
from tmcmc_mod.tmcmc import run_tmcmc

# choose 'multiprocessing' for local workstation or 'mpi' for supercomputer
parallel_processing = 'multiprocessing'

# measurment data:
# eigen values of first mode
data1 = np.array([0.3860, 0.3922, 0.4157, 0.3592, 0.3615])
# eigen values of second mode
data2 = np.array([2.3614, 2.5877, 2.7070, 2.3875, 2.7272])
# eigen vector of first mode
data3 = np.array([1.68245252, 1.71103903, 1.57876073, 1.58722342, 1.61878479])

# number of particles (to approximate the posterior)
Np = 500

# prior distribution of parameters
k1 = pdfs.Uniform(lower=0.8, upper=2.2)
k2 = pdfs.Uniform(lower=0.4, upper=1.2)

# Required! a list of all parameter objects
all_pars = [k1, k2]


def log_likelihood(particle_num, s):
    """
    Required!
    log-likelihood function which is problem specific
    for the 2DOF example log-likelihood is

    Parameters
    ----------
    particle_num : int
        particle number.
    s : numpy array of size Np (number of parameters in all_pars)
        particle location in Np space

    Returns
    -------
    LL : float
        log-likelihood function value.

    """
    sig1 = 0.0191
    sig2 = 0.0809   # = 0.05*1.618
    lambda1_s = (s[0]/2+s[1]) - np.sqrt(((s[0]/2+s[1])**2 - s[0]*s[1]))
    phi12_s = (s[0]+s[1]-lambda1_s)/s[1]

    # see slide 27
    LL = (np.log((2*np.pi*sig1*sig2)**-5) +
          (-0.5*(sig1**(-2))*sum((lambda1_s-data1)**2)) +
          (-0.5*(sig2**-2)*sum((phi12_s-data3)**2)))
    return LL


# run main
if __name__ == '__main__':
    """main part to run tmcmc for the 2DOF example"""

    mytrace, comm = run_tmcmc(Np, all_pars,
                              log_likelihood, parallel_processing,
                              "status_file_2DOF.txt")

    # save results
    with open('mytrace.pickle', 'wb') as handle1:
        pickle.dump(mytrace, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    if parallel_processing == 'mpi':
        comm.Abort(0)
