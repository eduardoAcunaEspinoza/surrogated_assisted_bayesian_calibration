'''
Stochastic calibration of a model using Surrogate-Assisted Bayesian inversion. The surrogate model is created using
arbitrary polynomial chaos expansion, and the linear system of equations is solved with the Bayesian Ridge sparse
technique (BAL_aPCE -> Bayesian Active Learning, arbitrary polynomial chaos expansion)

Methodology from:
Oladyshkin, S., Mohammadi, F., Kroeker, I., & Nowak, W. (2020). Bayesian3 Active Learning for the Gaussian Process
Emulator Using Information Theory. Entropy, 22(8), 890.

Oladyshkin, S., & Nowak, W. (2012). Data-driven uncertainty quantification using the arbitrary polynomial chaos
expansion. Reliability Engineering & System Safety, 106, 179-190

Code logic by: Sergey Oladyshkin
(https://www.mathworks.com/matlabcentral/fileexchange/74794-bal-gpe-matlab-toolbox-bayesian-active-learning-for-gpe?s_tid=srchtitle_GPE_1)
(https://www.mathworks.com/matlabcentral/fileexchange/72014-apc-matlab-toolbox-data-driven-arbitrary-polynomial-chaos?s_tid=srchtitle)


Adaptation to python by: Eduardo Acuna and Farid Mohammadi

Contact: eduae94@gmail.com
'''

# Import libraries
import sys
import math
import numpy as np
from sklearn import linear_model
from aPoly_Construction import *
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

# Maximum degree of the polynomial I am going to use to generate my expansion
d = 4

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

# Number of terms in the aPCE expansion
P = int(math.factorial(N+d)/(math.factorial(N)*math.factorial(d)))

# Part 2. Computation of the orthonormal aPC polynomial basis  --------------------------------------------------------
# Construction of basis for each uncertain parameter from d=0 to d=d+1. The polynomial d+1 is used in the next step to
# compute the collocation points
orthonormal_basis = np.zeros((d+2, d+2, N))  # d+2 because it goes from 0 to d+1
for i in range(N):
    orthonormal_basis[:, :, i] = compute_orthonomal_basis(prior_distribution[:, i], d)

# Part 3. Computation of the collocation points  ----------------------------------------------------------------------
collocation_points = compute_collocation_points(prior_distribution, orthonormal_basis)

# Part 4. Computation of the polynomial degrees  ----------------------------------------------------------------------
polynomial_degree = compute_polynomial_degrees(d, N)

# Part 5. Computation of the physical model in the previously defined collocation points  -----------------------------
model_results = np.zeros((collocation_points.shape[0], n_points))
for i, c_point in enumerate(collocation_points):
    model_results[i, :] = analytical_function(t, c_point)

# Part 6. Computation of the space/time independent matrix of the aPCE  ---------------------------------------------
PSI = compute_aPCE_matrix(orthonormal_basis, collocation_points, polynomial_degree)

# Part 7. Computation of the multi-dimensional expansion coefficients for arbitrary Polynomial Chaos  ----------------
expansion_coefficients = []
for model in model_results.T:
    clf_poly = linear_model.BayesianRidge(fit_intercept=False, normalize=True, n_iter=1000, tol=1e-4)
    expansion_coefficients.append(clf_poly.fit(PSI, model))  # calculate the coefficients of the expansion

# Part 8. Pre-computation of the orthonormal basis evaluated in the MC realization (sample of prior) and the space/time
# independent matrix for the MC realizations ------------------------------------------------------------------------
if iteration_limit > 0:
    orthonormal_basis_mcpoints = precompute_basis_mc_realizations(orthonormal_basis, prior_distribution)
    PSI_surrogate_mc = compute_PSI_surrogate_mc(P, orthonormal_basis_mcpoints, polynomial_degree)

# Loop for Bayesian update
for iter in range(0, iteration_limit):
    # Part 9. Computation of surrogate model prediction in MC points using aPCE and Bayesian Ridge -------------------
    surrogate_prediction = np.zeros((n_points, prior_distribution.shape[0]))
    surrogate_std = np.zeros((n_points, prior_distribution.shape[0]))
    for i in range(n_points):
        clf_poly = expansion_coefficients[i]
        surrogate_prediction[i, :], surrogate_std[i, :] = clf_poly.predict(PSI_surrogate_mc, return_std=True)

    # Part 10. Computation of bayesian scores (in parameter space) -----------------------------------------------------
    total_error = (measurement_error**2)*np.ones(observations.shape[0])
    BME[iter], RE[iter] = compute_bayesian_scores(surrogate_prediction.T, observations.T, total_error)

    # Part 11. Bayesian active learning (in output space) --------------------------------------------------------------
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

    # Part 12. Selection criteria for next collocation point ------------------------------------------------------
    al_value, al_value_index = BAL_selection_criteria(al_strategy, al_BME, al_RE)

    # Part 13. Selection of new collocation point
    collocation_points = np.vstack((collocation_points, prior_distribution[al_unique_index[al_value_index], :]))

    # Part 14. Computation of the numerical model in the newly defined collocation point --------------------------
    model_results = np.vstack((model_results, analytical_function(t, collocation_points[-1, :])))

    # Part 15. Computation of the space/time independent matrix for aPCE (considering new collocation point) -------
    PSI = compute_aPCE_matrix(orthonormal_basis, collocation_points, polynomial_degree)

    # Part 16. Computation of the multi-dimensional expansion coefficients for aPCE (considering new collocation point)
    expansion_coefficients = []
    for model in model_results.T:
        clf_poly = linear_model.BayesianRidge(fit_intercept=False, normalize=True, n_iter=1000, tol=1e-4)
        expansion_coefficients.append(clf_poly.fit(PSI, model))  # calculate the coefficients of the expansion

    # Progress report
    print("Bayesian iteration: " + str(iter+1) + "/" + str(iteration_limit))


# Part 10. Compute solution in final time step --------------------------------------------------------------------
surrogate_prediction = np.zeros((n_points, prior_distribution.shape[0]))
surrogate_std = np.zeros((n_points, prior_distribution.shape[0]))
for i in range(n_points):
    clf_poly = expansion_coefficients[i]
    surrogate_prediction[i, :], surrogate_std[i, :] = clf_poly.predict(PSI_surrogate_mc, return_std=True)

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
