# Surrogate Assisted Bayesian Calibration

## Introduction

The following repository provides a tool to do a stochastic calibration of computationally expensive models using a surrogated assisted bayesian inversion technique 
proposed by [Oladyshkin et al (2020)](https://doi.org/10.3390/e2208089). The implementation of the code is done in Python, but a MATLAB version by Oladyshkin can be 
found as [BAL-GPE Matlab Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/74794-bal-gpe-matlab-toolbox-bayesian-active-learning-for-gpe?s_tid=FX_rc3_behav).

The repository is organized as follows:

1. AnalyticalFunctionExample: In this folder an example of the metod is implemented with an analytical function
2. GPE_BAL_Telemac: In this folder the calibration methodology is coupled with TELEMAC 2D and GAIA to calibrate a 2D hydromorphodynamic model for reservoir sedimentantion
3. run_multiple_Telemac: In the folder a function to run Telemac from python multiple times is presented. This function is useful when one is creating the initial surrogate model (also known as metamodel) that will later be updated through the Bayesian Active Learning technique. 

All folders are independent of each other and a detailed explanation of the files can be found inside each folder.

# Disclaimer
No warranty is expressed or implied regarding the usefulness or completeness of the information and documentation provided. References to commercial products do not imply endorsement by the Authors. The concepts, materials, and methods used in the algorithms and described in the documentation are for informational purposes only. The Authors has made substantial effort to ensure the accuracy of the algorithms and the documentation, but the Authors shall not be held liable, nor his employer or funding sponsors, for calculations and/or decisions made on the basis of application of the scripts and documentation. The information is provided "as is" and anyone who chooses to use the information is responsible for her or his own choices as to what to do with the data. The individual is responsible for the results that follow from their decisions.

This website contains external links to other, external websites and information provided by third parties. There may be technical inaccuracies, typographical or other errors, programming bugs or computer viruses contained within the web site or its contents. Users may use the information and links at their own risk. The Authors of this web site excludes all warranties whether express, implied, statutory or otherwise, relating in any way to this web site or use of this web site; and liability (including for negligence) to users in respect of any loss or damage (including special, indirect or consequential loss or damage such as loss of revenue, unavailability of systems or loss of data) arising from or in connection with any use of the information on or access through this web site for any reason whatsoever (including negligence).

# Authors
- Eduardo Acuna
- Kilian Mouris
- Sebastian Schwindt
- Farid Mohammadi 

For question you can write an email to eduardo.espinoza@kit.edu

