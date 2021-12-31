'''
Stochastic calibration of a model using Surrogate-Assisted Bayesian inversion. The surrogate model is created using
Gaussian Process Regression (BAL_GP -> Bayesian Active Learning, Gaussian Process)

Methodology from:
Oladyshkin, S., Mohammadi, F., Kroeker, I., & Nowak, W. (2020). Bayesian3 Active Learning for the Gaussian Process
Emulator Using Information Theory. Entropy, 22(8), 890.

Code logic by: Sergey Oladyshkin
(https://www.mathworks.com/matlabcentral/fileexchange/74794-bal-gpe-matlab-toolbox-bayesian-active-learning-for-gpe?s_tid=srchtitle_GPE_1)

Adaptation to python by: Eduardo Acuna and Farid Mohammadi

Contact: eduae94@gmail.com
'''

# Import libraries
import sys, os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from auxiliary_functions_BAL import *
from physical_model import analytical_function
from plots_fun import plot_likelihoods

# ---------------------------------------------------------------------------------------------------------------------
# USER INPUT  ---------------------------------------------------------------------------------------------------------

# Prior distribution of calibration parameters
N = 2  # number of calibration parameters (uncertainty parameters)
mc_size = 10000  # mc size for parameter space
prior_distribution = np.zeros((mc_size, N))
prior_distribution[:, 0] = np.random.uniform(-5, 5, mc_size)
prior_distribution[:, 1] = np.random.uniform(-5, 5, mc_size)

# Observations (in a real case this would be read directly after computing the full-complexity model)
t = np.arange(0, 1.01, 1/9)  # for my  analytical model
n_points = len(t)  # number of points (space or time) I am going to use in my calibration
synthetic_solution = [0, 0]  # assumed solution for my analytical model
observations = np.transpose(analytical_function(t, synthetic_solution).reshape(1, n_points))
measurement_error = 2

# Bayesian updating
iteration_limit = 30 # number of bayesian iterations
d_size_AL = 1000  # number of active learning sets (sets I take from the prior to do the active learning).
mc_size_AL = 10000  # sample size for output space
# Note: d_size_AL+ iteration_limit < mc_size
al_strategy = "RE"

# END OF USER INPUT  --------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Part 1. Initialization of information  ------------------------------------------------------------------------------
BME = np.zeros((iteration_limit, 1))
RE = np.zeros((iteration_limit, 1))
al_BME = np.zeros((d_size_AL, 1))
al_RE = np.zeros((d_size_AL, 1))
graph_list = []
graph_name = []

# Part 2. Initial training points for my surrogate model -------------------------------------------------------------
n_cp = 5
collocation_points = np.zeros((n_cp, N))
collocation_points[:, 0] = np.random.uniform(-5, 5, n_cp)
collocation_points[:, 1] = np.random.uniform(-5, 5, n_cp)

# Part 3. Computation of the physical model in the previously defined collocation points  -----------------------------
model_results = np.zeros((collocation_points.shape[0], n_points))
for i, c_point in enumerate(collocation_points):
    model_results[i, :] = analytical_function(t, c_point)


# Loop for Bayesian update
for iter in range(0, iteration_limit):
    # Part 4. Computation of surrogate model prediction in MC points using gaussian processes --------------------------
    surrogate_prediction = np.zeros((n_points, prior_distribution.shape[0]))
    surrogate_std = np.zeros((n_points, prior_distribution.shape[0]))

    for i, model in enumerate(model_results.T):
        kernel = RBF(length_scale=[1, 1], length_scale_bounds=[(0.01, 20), (0.01, 20)]) * np.var(model)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0002, normalize_y=True, n_restarts_optimizer=10)
        gp.fit(collocation_points, model)
        surrogate_prediction[i, :], surrogate_std[i, :] = gp.predict(prior_distribution, return_std=True)

    # Part 5. Computation of bayesian scores (in parameter space) -----------------------------------------------------
    total_error = (measurement_error**2)*np.ones(observations.shape[0])
    BME[iter], RE[iter] = compute_bayesian_scores(surrogate_prediction.T, observations.T, total_error)

    # Part 6. Bayesian active learning (in output space) --------------------------------------------------------------
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

    # Part 7. Selection criteria for next collocation point ------------------------------------------------------
    al_value, al_value_index = BAL_selection_criteria(al_strategy, al_BME, al_RE)

    # Part 8. Selection of new collocation point
    collocation_points = np.vstack((collocation_points, prior_distribution[al_unique_index[al_value_index], :]))

    # Part 9. Computation of the numerical model in the newly defined collocation point --------------------------
    model_results = np.vstack((model_results, analytical_function(t, collocation_points[-1, :])))

    # Progress report
    print("Bayesian iteration: " + str(iter+1) + "/" + str(iteration_limit))


# Part 10. Compute solution in final time step --------------------------------------------------------------------
surrogate_prediction = np.zeros((n_points, prior_distribution.shape[0]))
surrogate_std = np.zeros((n_points, prior_distribution.shape[0]))
for i, model in enumerate(model_results.T):
    kernel = RBF(length_scale=[1, 1], length_scale_bounds=[(0.01, 20), (0.01, 20)]) * np.var(model)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0002, normalize_y=True, n_restarts_optimizer=10)
    gp.fit(collocation_points, model)
    surrogate_prediction[i, :], surrogate_std[i, :] = gp.predict(prior_distribution, return_std=True)

likelihood_final = compute_fast_likelihood(surrogate_prediction.T, observations.T, total_error)

# Save final results of surrogate model to graph them later
graph_likelihood_surrogates = np.zeros((prior_distribution.shape[0], 3))
graph_likelihood_surrogates[:, :2] = prior_distribution
graph_likelihood_surrogates[:, 2] = likelihood_final
graph_list.append(np.copy(graph_likelihood_surrogates))
graph_name.append("iteration: " + str(iteration_limit))

# Part 11. Compute reference solution (only for explanatory purposes, with a real model the reference solution would
# not be available ) ---------------------------------------------------------- ---------------------------------------
likelihood_ref = compute_reference_solution(prior_distribution, t, observations.T, total_error)
graph_likelihood_reference = np.zeros((prior_distribution.shape[0], 3))
graph_likelihood_reference[:, :2] = prior_distribution
graph_likelihood_reference[:, 2] = likelihood_ref
graph_list.append(graph_likelihood_reference)
graph_name.append("reference")

# Plot comparison between surrogate model and reference solution
plot_likelihoods(graph_list, graph_name)
x=1
