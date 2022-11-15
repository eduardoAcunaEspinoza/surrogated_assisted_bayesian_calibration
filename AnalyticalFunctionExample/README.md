## Introduction

The two main functions work independently of each other. Both functions make a Stochastic calibration of a model using Surrogate-Assisted Bayesian inversion. The difference between them is the technique used to create the surrogate model. 

In both cases, the model that is being calibrated is an analytical function, which is described in the file physical_model.py. The use of an analytical function has two main purposes. First, explain how the calibration method works using a fast-simple model. Second, we will be able to evaluate the calibration quality by comparing the surrogate-assisted calibration results with a Monte Carlo reference solution because we can run the analytical model thousands of times. In a  real case, this analytical function will be substituted by a full complexity model (see folder GPE_BAL_Telemac), and a reference solution cannot be computed. 

## Libraries

*Python* libraries:  *numpy*, *matplotlib*, *scikit-learn*

*Standard* libraries: *math*, *os*, *sys*

## File structure
- main_BAL_aPCE = The surrogate model is created using the arbitrary polynomial chaos expansion technique proposed by [Oladyshkin, S., & Nowak, W. (2012)](https://doi.org/10.1016/j.ress.2012.05.002). 
- main_BAL_GP = The surrogate model is created using the Gaussian Process package of [scikit-learn](https://scikit-learn.org/stable/modules/gaussian_process.html). 
- aPoly_Construction: Auxiliary functions necessary to compute a surrogate model using the arbitrary polynomial chaos expansion technique
- auxiliary_functions_BAL: Auxiliary functions for the stochastic calibration of model using Surrogate-Assisted Bayesian inversion
- physical_model: Contains a non-linear analytical function used as an example of the "physical model" that is being calibrated.
- plots_fun: Auxiliary functions to plot the results. 

## Notes
Note 1: For the main_BAL_GP, in the analytical case example, the code prints some warnings and proposes preprocessing the Gaussian Process Regression (GPR) input data. Several tests preprocessing the data were done, and the number of warnings was reduced. Nevertheless, in this version of the code, no preprocessing was made to avoid bias suggestions to other users on how the information must be preprocessed, as this procedure depends directly on the data used for the calibration. 

Note 2: The square-exponential kernel was used in the GPR, but depending on the users' data, other kernels might be better. 

Note 3. The original version of this code was implemented by [Oladyshkin et al (2020)](https://doi.org/10.3390/e2208089) and can be found in [BAL-GPE Matlab Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/74794-bal-gpe-matlab-toolbox-bayesian-active-learning-for-gpe?s_tid=FX_rc3_behav).
