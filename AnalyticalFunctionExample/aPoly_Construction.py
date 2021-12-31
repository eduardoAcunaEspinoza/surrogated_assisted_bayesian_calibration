'''
Auxiliary functions necessary to compute a surrogate model using the arbitrary polynomial chaos expansion technique

Methodology from:
Oladyshkin, S., & Nowak, W. (2012). Data-driven uncertainty quantification using the arbitrary polynomial chaos
expansion. Reliability Engineering & System Safety, 106, 179-190
(https://www.mathworks.com/matlabcentral/fileexchange/72014-apc-matlab-toolbox-data-driven-arbitrary-polynomial-chaos?s_tid=srchtitle)

Adaptation to python by: Eduardo Acuna and Farid Mohammadi

Contact: eduae94@gmail.com
'''

# Libraries
import numpy as np
import os
import math


def compute_orthonomal_basis(data, d):
    """
   Construction of Data-driven Orthonormal Polynomial Basis

    Parameters
    ----------
    data : array [MC]
        Raw data.
    d : int
        degree of polynomial expansion

    Returns
    -------
    Polynomial : array [d+2, d+2]
        The coefficients of the orthonormal polynomials.

     Notes:
    * MC is the total number of MonteCarlo samples
    """
    
    # Initialization
    dd = d+1  # Degree of polynomial for roots definition
    nsamples = len(data)

    # Forward linear transformation (Avoiding numerical issues)
    # MeanOfData = np.mean(Data)
    # VarOfData = np.var(Data)
    # Data = Data/MeanOfData

    # Compute raw moments for input data
    M = [np.sum(np.power(data, p)) / nsamples for p in range(2 * dd + 2)]
    
    # --------------------------------------------------------------------------
    # ----- Main Loop for Polynomial with degree up to dd
    # --------------------------------------------------------------------------
    PolyCoeff_NonNorm = np.empty((0, 1))
    Polynomial = np.zeros((dd+1, dd+1))
    
    for degree in range(dd+1):
        Mm = np.zeros((degree+1,degree+1))
        Vc = np.zeros((degree+1))

        # Define Moments Matrix Mm
        for i in range(degree+1):
            for j in range(degree+1):
                if i < degree:
                    Mm[i, j] = M[i+j]

                elif (i == degree) and (j == degree):
                    Mm[i, j] = 1

            # Numerical Optimization for Matrix Solver
            Mm[i, :] = Mm[i, :]/max(abs(Mm[i,:]))

        # Definition of Right Hand side orthogonality conditions: Vc
        for i in range(degree+1):
            Vc[i] = 1 if i == degree else 0

        # Solution: Coefficients of Non-Normal Orthogonal Polynomial: Vp Eq.(4)
        try:
            Vp = np.linalg.solve(Mm, Vc)
        except:
            inv_Mm = np.linalg.pinv(Mm)
            Vp = np.dot(inv_Mm,Vc.T)
        
        # PolyCoeff_NonNorm[degree,0:degree]=Vp #PolyCoeff_NonNorm(degree+1,1:degree+1)=Vp'
        if degree == 0:
            PolyCoeff_NonNorm = np.append(PolyCoeff_NonNorm,Vp)

        if degree !=0:
            if degree == 1:
                zero=[0]
            else:
                zero=np.zeros((degree,1))
            PolyCoeff_NonNorm = np.hstack((PolyCoeff_NonNorm , zero))

            PolyCoeff_NonNorm = np.vstack((PolyCoeff_NonNorm, Vp))

        if 100*abs(sum(abs(np.dot(Mm,Vp)) - abs(Vc))) > 0.5:
            print('\n---> Attention: Computational Error too high !')
            print('\n---> Problem: Convergence of Linear Solver')

        # Original Numerical Normalization of Coefficients with Norm and Ortho-normal Basis computation
        # Matrix Storage Note: Polynomial(i,j) correspond to coefficient number "j-1" of polynomial degree "i-1"
        P_norm = 0
        for i in range(nsamples):
            Poly = 0
            for k in range(degree+1):
                if degree == 0:
                    Poly += PolyCoeff_NonNorm[k] * (data[i] ** k)
                else:
                    Poly += PolyCoeff_NonNorm[degree, k] * (data[i] ** k)

            P_norm += Poly**2 / nsamples

        P_norm = np.sqrt(P_norm)
        
        for k in range(degree+1):
            if degree == 0:
                Polynomial[degree, k] = PolyCoeff_NonNorm[k]/P_norm
            else:
                Polynomial[degree, k] = PolyCoeff_NonNorm[degree,k]/P_norm

    # Backward linear transformation to the real data space
    # Data = Data * MeanOfData
    # for k in range(len(Polynomial)):
    #     Polynomial[:,k] = Polynomial[:,k]/(MeanOfData**(k))

    return Polynomial


def compute_collocation_points(data, orthonormal_basis):
    """
    This function computes the collocation points (parameter combination) for which the response surface is going to
    be computed. The collocation points are taken as the roots of the polynomial with degree d+1 (similar to Gauss -
    Quadrature). Then we have to choose the P most probable collocation points, being P the order of expansion

    Parameters
    ----------
    data : array [MC, N]
        Raw data.
    orthonormal_basis : array [d+2, d+2, N]
        orthonormal_basis for each random variable

    Returns
    -------
    collocation points: array [P,  N]
       The collocation points

     Notes:
    * MC is the total number of MonteCarlo samples
    * N is the number of uncertain parameters
    * d is the degree of the polynomial expansion
    * P is the number of terms in polynomial expansion

    """
    # Number of uncertain parameters
    N = data.shape[1]
    # Degree of polynomial expansion
    d = int(orthonormal_basis.shape[0]-2)
    # Number of terms in polynomial expansion
    P = math.factorial(N + d) / (math.factorial(N) * math.factorial(d))

    # Compute the roots of the polynomial of degree d+1 for each of the orthogonal basis (each uncertainty parameter)
    polynomial_roots = np.zeros((N, d + 1))
    for i in range(N):
        polynomial_coefficient = orthonormal_basis[d+1, :, i]
        polynomial_roots[i, :] = np.roots(np.flip(polynomial_coefficient))

    # Creation of all the possible combinations of the different polynomial_roots for the N uncertainty parameters
    nroot_para_mat = np.tile(np.arange(1, d + 2), (N, 1))  # Matrix with the number of roots for each parameter
    unique_combinations = np.array(np.meshgrid(*(row for row in nroot_para_mat))).T.reshape(-1, N)

    # Sort the unique_combinations based on the sum of each row. Later this is going to be the ranking to construct the
    # most probable collocation points using the most probable polynomial roots.
    sort_unique_combinations = unique_combinations[np.argsort(unique_combinations.sum(axis=1)), :].astype(float)

    # Sort the polynomial_roots based on the higher probability of occurrence. In this case, we assume that the higher
    # probability of occurrence is the mean value, and sort the values with respect con the distance from the mean
    temp = abs(np.subtract(polynomial_roots, np.transpose(np.mean(data, axis=0, keepdims=True))))
    temp_sort = np.argsort(temp, axis=1)
    sorted_polynomial_roots = polynomial_roots.copy()
    for i, row in enumerate(sorted_polynomial_roots):
        sorted_polynomial_roots[i, :] = row[temp_sort[i]]

    # Compute the most probable collocation points as the combination of the most probable polynomial roots for each
    # parameter.
    for i in range(0, sort_unique_combinations.shape[0]):
        for j in range(0, sort_unique_combinations.shape[1]):
            sort_unique_combinations[i, j] = sorted_polynomial_roots[j, int(sort_unique_combinations[i, j]) - 1]

    # Choose the P most probable collocation points (being P the order of expansion)
    collocation_points = sort_unique_combinations[0:int(P), :]
    x=1

    return collocation_points


def compute_polynomial_degrees(d, N):

    """
    This function ranks the first P polynomial combinations (combinations between the polynomial of the different
    uncertainty parameters). Similar to an approximation using a truncated Taylor series, the most important polynomials
    are the one with the lower degree, and the less important are the ones with the higher degree

    Parameters
    ----------
    d : int
       Maximum polynomial degree.
    N : int
       Number of uncertain parameters
    Returns
    -------
    polynomial_degree: array [P, N]
       Ranked combination of polynomial

     Notes:
    * P is the number of terms in polynomial expansion
    """

    # Number of terms in expansion
    P = math.factorial(N + d) / (math.factorial(N) * math.factorial(d))

    # Creation of all the possible combinations of the different polynomial degrees for the N uncertainty parameters
    degree_para_mat = np.tile(np.arange(0, d + 1), (N, 1))  # Matrix with the polynomial degrees for each parameter
    unique_degree_combinations = np.array(np.meshgrid(*(row for row in degree_para_mat), indexing='ij')).T.reshape(-1, N)
    unique_degree_combinations = np.flip(unique_degree_combinations, axis=1)

    # Sort the unique_degree_combinations based on the sum of each row
    sort_unique_degree_combinations = unique_degree_combinations[np.argsort(unique_degree_combinations.sum(axis=1),
                                                                            kind='mergesort'), :].astype(float)

    # Choose the P most important polynomial combinations
    polynomial_degree = sort_unique_degree_combinations[0:int(P), :]
    return polynomial_degree


def compute_aPCE_matrix(orthonormal_basis, collocation_points, polynomial_degree):
    """
    Setting up the space/time independent matrix of the aPCE.

    Parameters
    ----------
    orthonormal_basis : array [d+2, d+2, N]
        orthonormal_basis for each random variable
    collocation_points : array [# collocation points, N]
        parameter combination where the model is going to be evaluated
    polynomial_degree: array [P, N]
        ranking of the polynomials that are going to be used for the expansion

    Returns
    -------
    PSI: array [# collocation points, P]
       space/time independent matrix of the aPCE

     Notes:
    * d is the degree of the polynomial expansion
    * N is the number of uncertain parameters
    * P is the number of terms in polynomial expansion

    """
    # Number of uncertain parameters
    N = collocation_points.shape[1]
    # Degree of polynomial expansion
    d = int(orthonormal_basis.shape[0]-2)
    # Number of terms in expansion
    P = int(math.factorial(N + d) / (math.factorial(N) * math.factorial(d)))

    P_total = collocation_points.shape[0]
    PSI = np.zeros((P_total, P))  # as many rows as collocation points and as many columns as the order of the expansion
    for i in range(0, P):  # for each column
        for j in range(0, P_total):  # for each row of that column
            PSI[j, i] = 1
            for k in range(0, N):  # for each of my uncertain parameters
                PSI[j, i] = PSI[j, i] * np.polyval(
                    orthonormal_basis[int(polynomial_degree[i, k]), np.arange(d + 1, -1, -1),
                                      k], collocation_points[j, k])
    return PSI


def precompute_basis_mc_realizations(orthonormal_basis, prior_distribution):
    """
    Pre-computation of the orthonormal basis evaluated in the MC realization (sample of prior)

    Parameters
    ----------
    orthonormal_basis : array [d+2, d+2, N]
        orthonormal_basis for each random variable
    prior_distribution : array [MC, N]
        prior distribution of each uncertain parameter
    d: int
        degree of polynomial expansion

    Returns
    -------
    orthonormal_basis_mcpoints: array [MC, N, d+1]
       evaluation of the orthonormal basis in the samples taken form the prior distributions

     Notes:
    * d is the degree of the polynomial expansion
    * N is the number of uncertain parameters
    * MC is the total number of MonteCarlo samples
    """

    # Degree of polynomial expansion
    d = int(orthonormal_basis.shape[0]-2)
    # Number of MonteCarlo sample
    mc_size = prior_distribution.shape[0]
    # Number of uncertain parameters
    N = prior_distribution.shape[1]

    orthonormal_basis_mcpoints = np.zeros((mc_size, N, d+1))
    for i in range(0, mc_size):  # For every point of the mc sample
        for j in range(0, N):  # For every uncertain parameter
            for k in range(0, d+1):  # For every polynomial degree in the orthonormal basis
                orthonormal_basis_mcpoints[i, j, k] = np.polyval(orthonormal_basis[k, np.arange(d+1, -1, -1), j],
                                                                 prior_distribution[i, j])
    return orthonormal_basis_mcpoints


def compute_PSI_surrogate_mc(P, orthonormal_basis_mcpoints, polynomial_degree):
    """
    Compute the space/time independent matrix evaluated in the MC realization (sample of prior)

    Parameters
    ----------
    P: int
        order of the expansion
    orthonormal_basis_mcpoints : array array [MC, N, d+1]
       orthonormal basis evaluated in the MC realization
    polynomial_degree : array [P, N]
        polynomial combinations used for the chaos expansion

    Returns
    -------
    PSI_surrogate_evaluation: array [MC, P]
       Evaluation of the surrogate model in each of the MC realizations

    Notes:
    * MC is the total number of MonteCarlo samples
    * N is the number of uncertain parameters
    * d is the degree of the polynomial expansion

    """
    # Number of MonteCarlo sample
    mc_size = orthonormal_basis_mcpoints.shape[0]

    PSI_surrogate_evaluation = np.zeros((mc_size, P))
    # temp1 and temp2 help compute the rows of the PSI_surrogate_evaluation in a more efficient way (without
    # looping directly). temp2 is used later to compute all the P expansion polynomials in one single action
    temp1 = np.arange(0, orthonormal_basis_mcpoints[0, :, :].shape[1] * polynomial_degree.shape[1],
                      orthonormal_basis_mcpoints[0, :, :].shape[1]).reshape(polynomial_degree.shape[1], 1)
    temp2 = (np.transpose(polynomial_degree) + temp1).astype(int)

    for i in range(0, mc_size):  # for each mc_sample
        # Compute each row of the space/time independent matrix for the mc_sample [mc_size x P].
        PSI_surrogate_evaluation[i, :] = np.prod(np.take(orthonormal_basis_mcpoints[i, :, :], temp2), axis=0)

    return PSI_surrogate_evaluation