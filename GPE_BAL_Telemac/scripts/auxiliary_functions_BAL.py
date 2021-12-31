'''
Auxiliary functions for the stochastic calibration of model using Surrogate-Assisted Bayesian inversion

Contact: eduae94@gmail.com
'''

# Libraries
import numpy as np
import math

def compute_fast_likelihood(prediction, observations, error_variance):
    """
    Calculates the multivariate Gaussian likelihood between model predictions and measured/observed data, taking
    independent errors (diagonal covariance matrix)

    Input
    ----------
    prediction : array [MC, n_points]
        predicted / modelled values
    observations: array [1, n_points]
        observed / measured values
    error variance : array [n_points]]
        error of the observations

    Returns
    -------
    likelihood: array [MC]
       likelihood value

    Notes:
    * MC is the total number of model runs and n_points is the number of points considered for the comparison
    * const_mvn is the constant outside the exponent of the multivariate Gaussian likelihood. In some cases this can be
    ignored
    * Method is faster than using stats module.
    """

    cov_mat = np.diag(error_variance)
    invR = np.linalg.inv(cov_mat)

    # Calculate constants:
    #n_points = observations.shape[1]
    #det_R = np.linalg.det(cov_mat)
    #const_mvn = pow(2 * math.pi, -n_points / 2) * (1 / math.sqrt(det_R))

    # Vectorize means:
    means_vect = observations[:, np.newaxis]

    # Calculate differences and convert to 4D array (and its transpose):
    diff = means_vect - prediction  # Shape: # means
    diff_4d = diff[:, :, np.newaxis]
    transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)

    # Calculate values inside the exponent
    inside_1 = np.einsum("abcd, dd->abcd", diff_4d, invR)
    inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)
    total_inside_exponent = inside_2.transpose(2, 1, 0)
    total_inside_exponent = np.reshape(total_inside_exponent,
                                       (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))

    #likelihood = const_mvn * np.exp(-0.5 * total_inside_exponent)
    likelihood = np.exp(-0.5 * total_inside_exponent)

    if likelihood.shape[1] == 1:
        likelihood = likelihood[:, 0]

    return likelihood


def compute_bayesian_scores(prediction, observations, error_variance):
    """
    Compute the Bayesian Model Evidence (BME) and Relative entropy

    Input
    ----------
    prediction : array [MC, n_points]
        predicted / modelled values
    observations: array [1, n_points]
        observed / measured values
    error_variance : array [n_points]]
        error of the observations

    Returns
    -------
    BME: float
       bayesian model evidence
    RE: float
        relative entropy

    Notes:
    * MC is the total number of model runs and n_points is the number of points considered for the comparison
    """

    # Likelihood calculation
    likelihood = compute_fast_likelihood(prediction, observations, error_variance).reshape(1, prediction.shape[0])

    # BME calculation
    BME = np.mean(likelihood)

    # For cases in which the prediction of the surrogate is not too bad
    if BME > 0:

        # Non normalized cross entropy with rejection sampling
        #accepted = likelihood / np.amax(likelihood) >= np.random.rand(1, prediction.shape[0])
        #exp_log_pred = np.mean(np.log(likelihood[accepted]))

        # Non normalized cross entropy with bayesian weighting
        non_zero_likel = likelihood[np.where(likelihood != 0)]
        post_weigth = non_zero_likel / np.sum(non_zero_likel)
        exp_log_pred = np.sum(post_weigth * np.log(non_zero_likel))

        # Relative entropy calculation
        RE = exp_log_pred - math.log(BME)

    # For cases in which the BME is zero (point selected from the prior gives really bad results, or the surrogate still
    # is giving bad results)
    else:
        BME = 0
        RE = 0

    return BME, RE


def BAL_selection_criteria(al_strategy, al_BME, al_RE):
    """
    Gives the best value of the selected bayesian score and index of the associated parameter combination
    ----------
    al_strategy : string
        strategy for active learning, selected bayesian score
    al_BME : array [d_size_AL,1]
        bayesian model evidence of active learning sets
    al_RE: array [d_size_AL,1]
        relative entropy of active learning sets

    Returns
    -------
    al_value: float
        best value of the selected bayesian score
    al_value_index: int
        index of the associated parameter combination

    Notes:
    * d_size_AL is the number of active learning sets (sets I take from the prior to do the active learning)
    """

    if al_strategy == "BME":
        al_value = np.amax(al_BME)
        al_value_index = np.argmax(al_BME)

        if np.amax(al_BME) == 0:
            print("Warning Active Learning: all values of Bayesian model evidences equal to 0")
            print("Active Learning Action: training point have been selected randomly")

    elif al_strategy == "RE":
        al_value = np.amax(al_RE)
        al_value_index = np.argmax(al_RE)

        if np.amax(al_RE) == 0 and np.amax(al_BME) != 0:
            al_value = np.amax(al_BME)
            al_value_index = np.argmax(al_BME)
            print("Warning Active Learning: all values of Relative entropies equal to 0")
            print("Active Learning Action: training point have been selected according Bayesian model evidences")
        elif np.amax(al_RE) == 0 and np.amax(al_BME) == 0:
            al_value = np.amax(al_BME)
            al_value_index = np.argmax(al_BME)
            print("Warning Active Learning: all values of Relative entropies equal to 0")
            print("Warning Active Learning: all values of Bayesian model evidences are also equal to 0")
            print("Active Learning Action: training point have been selected randomly")

    return al_value, al_value_index

