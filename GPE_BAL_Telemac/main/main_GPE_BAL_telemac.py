'''
Stochastic calibration of a Telemac2d hydromorphodunamic model model using 
Surrogate-Assisted Bayesian inversion. The surrogate model is created using
Gaussian Process Regression

Methodology from:
Oladyshkin, S., Mohammadi, F., Kroeker, I., & Nowak, W. (2020). Bayesian3 Active Learning for the Gaussian Process
Emulator Using Information Theory. Entropy, 22(8), 890.

Adaptation to python and coupling with Telemac by: Eduardo Acuna and Farid Mohammadi

Contact: eduae94@gmail.com
'''

# Import libraries
import sys, os
import numpy as np
import shutil
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import init
from auxiliary_functions_BAL import *
from auxiliary_functions_telemac import *

# ---------------------------------------------------------------------------------------------------------------------
# USER INPUT  ---------------------------------------------------------------------------------------------------------

# Prior distribution of calibration parameters
N = 4  # number of calibration parameters (uncertainty parameters)
NS = 10000  # sample size I take from my prior distributions
input_distribution = np.zeros((NS, N))
input_distribution[:, 0] = np.random.uniform(0.01, 0.1, NS)
input_distribution[:, 1] = np.random.uniform(0.05, 0.4, NS)
input_distribution[:, 2] = np.random.uniform(200, 500, NS)
input_distribution[:, 3] = np.random.uniform(0.8, 1.7, NS)
parameters_name = ["CLASSES CRITICAL SHEAR STRESS FOR MUD DEPOSITION",
                   "LAYERS CRITICAL EROSION SHEAR STRESS OF THE MUD",
                   "LAYERS MUD CONCENTRATION",
                   "CLASSES SETTLING VELOCITIES"]

# Observations (measured values that are going to be used for calibration)
temp = np.loadtxt("calibration_points.txt")
n_points = temp.shape[0]
nodes = temp[:, 0].reshape(-1, 1)
observations = temp[:, 1].reshape(-1, 1)
observations_error = temp[:, 2]

# Bayesian updating
iteration_limit = 15  # number of bayesian iterations
mc_size = 10000  # mc size for parameter space
prior_distribution = np.copy(input_distribution[:mc_size, :])
d_size_AL = 1000  # number of active learning sets (sets I take from the prior to do the active learning).
mc_size_AL = 100000  # active learning sampling size
# Note: d_size_AL+ iteration_limit < mc_size
al_strategy = "RE"

# Telemac
telemac_name = "run_liquid_tel.cas"
gaia_name = "run_liquid_gaia.cas"
result_name_gaia = "'res_gaia_PC"  # PC stands for parameter combination
result_name_telemac = "'res_tel_PC"  # PC stands for parameter combination
n_processors = "12"

# Calibration parameters
initial_diameters = np.array([0.001, 0.000024, 0.0000085, 0.0000023])
calibration_variable = "BOTTOM"
auxiliary_names = ["CLASSES SEDIMENT DIAMETERS"]

# Paths
path_results = "../results"
path_simulations = "../simulations"

# END OF USER INPUT  --------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Part 1. Initialization of information  ------------------------------------------------------------------------------
BME = np.zeros((iteration_limit, 1))
RE = np.zeros((iteration_limit, 1))
al_BME = np.zeros((d_size_AL, 1))
al_RE = np.zeros((d_size_AL, 1))

# Part 2. Read initial collocation points  ----------------------------------------------------------------------------
temp = np.loadtxt(os.path.abspath(os.path.expanduser(path_results))+"/parameter_file.txt", dtype=np.str, delimiter=';')
simulation_names = temp[:, 0]
collocation_points = temp[:, 1:].astype(np.float)
n_simulation = collocation_points.shape[0]

# Part 3. Read the previously computed simulations of the numerical model in the initial collocation points -----------
temp = np.loadtxt(os.path.abspath(os.path.expanduser(path_results)) + "/" + simulation_names[0] + "_" +
                  calibration_variable + ".txt")
model_results = np.zeros((collocation_points.shape[0], temp.shape[0]))
for i, name in enumerate(simulation_names):
    model_results[i, :] = np.loadtxt(os.path.abspath(os.path.expanduser(path_results))+"/" + name + "_" +
                                     calibration_variable + ".txt")[:, 1]

# Loop for bayesian iterations
for iter in range(0, iteration_limit):
    # Part 4. Computation of surrogate model prediction in MC points using gaussian processes --------------------------
    surrogate_prediction = np.zeros((n_points, prior_distribution.shape[0]))
    surrogate_std = np.zeros((n_points, prior_distribution.shape[0]))

    for i, model in enumerate(model_results.T):
        kernel = RBF(length_scale=[0.05, 0.2, 150, 0.5], length_scale_bounds=[(0.001, 0.1), (0.001, 0.4), (5, 300), (0.02, 2)]) * np.var(model)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0002, normalize_y=True, n_restarts_optimizer=10)
        gp.fit(collocation_points, model)
        surrogate_prediction[i, :], surrogate_std[i, :] = gp.predict(prior_distribution, return_std=True)

    # Part 5. Read or compute the other errors to incorporate in the likelihood function
    loocv_error = np.loadtxt("loocv_error_variance.txt")[:, 1]
    total_error = (observations_error**2 + loocv_error)*5

    # Part 6. Computation of bayesian scores (in parameter space) -----------------------------------------------------
    BME[iter], RE[iter] = compute_bayesian_scores(surrogate_prediction.T, observations.T, total_error)
    np.savetxt("BME.txt", BME)
    np.savetxt("RE.txt", RE)


    # Part 7. Bayesian active learning (in output space) --------------------------------------------------------------
    # Index of the elements of the prior distribution that have not been used as collocation points
    aux1 = np.where((prior_distribution[:d_size_AL+iter, :] == collocation_points[:, None]).all(-1))[1]
    aux2 = np.invert(np.in1d(np.arange(prior_distribution[:d_size_AL+iter, :].shape[0]), aux1))
    al_unique_index = np.arange(prior_distribution[:d_size_AL+iter, :].shape[0])[aux2]

    for iAL in range(0, len(al_unique_index)):
        # Exploration of output subspace associated with a defined prior combination.
        al_exploration = np.random.normal(size=(mc_size_AL, n_points))*surrogate_std[:, al_unique_index[iAL]] + \
                         surrogate_prediction[:, al_unique_index[iAL]]
        # BAL scores computation
        al_BME[iAL], al_RE[iAL] = compute_bayesian_scores(al_exploration, observations.T, total_error)

    # Part 8. Selection criteria for next collocation point ------------------------------------------------------
    al_value, al_value_index = BAL_selection_criteria(al_strategy, al_BME, al_RE)

    # Part 9. Selection of new collocation point
    collocation_points = np.vstack((collocation_points, prior_distribution[al_unique_index[al_value_index], :]))

    # Part 10. Computation of the numerical model in the newly defined collocation point --------------------------
    # Update steering files
    update_steering_file(collocation_points[-1, :], parameters_name, initial_diameters, auxiliary_names, gaia_name,
                         telemac_name, result_name_gaia, result_name_telemac, n_simulation + 1 + iter)
    # Run telemac
    run_telemac(telemac_name, n_processors)

    # Extract values of interest
    updated_string = result_name_gaia[1:] + str(n_simulation+1+iter) + ".slf"
    save_name = path_results + "/PC" + str(n_simulation+1+iter) + "_" + calibration_variable + ".txt"
    results = get_variable_value(updated_string, calibration_variable, nodes, save_name)
    model_results = np.vstack((model_results, results[:, 1].T))

    # Move the created files to their respective folders
    shutil.move(result_name_gaia[1:] + str(n_simulation+1+iter) + ".slf", path_simulations)
    shutil.move(result_name_telemac[1:] + str(n_simulation+1+iter) + ".slf", path_simulations)

    # Append the parameter used to a file
    new_line = "; ".join(map('{:.3f}'.format, collocation_points[-1, :]))
    new_line = "PC" + str(n_simulation+1+iter) + "; " + new_line
    append_new_line(path_results + "/parameter_file.txt", new_line)

    # Progress report
    print("Bayesian iteration: " + str(iter+1) + "/" + str(iteration_limit))
