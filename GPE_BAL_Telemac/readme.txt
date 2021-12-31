Methodology by:
Oladyshkin, S., Mohammadi, F., Kroeker, I., & Nowak, W. (2020). Bayesian3 Active Learning for the Gaussian Process Emulator Using Information Theory. Entropy, 22(8), 890.
----------------------
Stochastic calibration of a Telemac2d hydro-morphodynamic model using  Surrogate-Assisted Bayesian inversion. The surrogate model is created using Gaussian Process Regression. 

To run the code, run the main_GPE_BAL_telemac.py file using the main folder as a current directory from a console/terminal in which Telemac and GAIA have already been compiled. It is not recommended to run the code from PyCharm as PyCharm uses a kind of additional virtual environment when it fires up its Terminal, and because Telemac has its own environment and APIs, those might be conflicting with PyCharm. 

#main Folder: 
-main_GPE_BAL_telemac.py: Stochastic calibration of a Telemac2d hydro-morphodynamic model using  Surrogate-Assisted Bayesian inversion. The surrogate model is created using Gaussian Process Regression.
-calibration_points.txt: This file contains the index of the mesh nodes that will be used for the calibration, the respective field measurement (water elevation, bottom elevation, ...) and its associated error (measurement error). 
-loocv_error_variance.txt: Contains an additional leave-one-out cross-validation error for each calibration point that accounts for the fact that the surrogate model is an approximation of the full-complexity model.
-init.py: Reference other folders.
- Files necessary to run the hydro-morphodynamic model using Telemac2D and GAIA: 
	- bc_liquid.liq: Liquid boundary condition (flow, sediment or tracers inflow/outflow)
	- bc_steady_tel.cli: File that defines the type and location of the boundary conditions.
	- geo_banda.slf: File that defines the mesh structure for the hydro-morphodynamic model. 
	-run_liquid_gaia.cas: Numerical configuration of the sediment transport model.
	- run_liquid_tel.cas: Numerical configuration of the hydrodynamic model. 

#results Folder:
Here the files that are going to be used as initial training points for the creation of the surrogate model are stored. In this case, 15 initial training points (15 runs of the full-complexity hydro-morphodynamic model) will be used to create the surrogate model. 
- parameter_file.txt: Contains the parameter combination associated with each training point.
- PCX_XXX.txt: Contains, for each parameter combination listed in parameter_file.txt, the associated modelled results (water elevation, bottom elevation, ...). 
In each iteration, the BAL technique selects a new training point to improve the surrogate quality. Therefore, a new hydro-morphodynamic simulation will be run using the newly defined training point parameters, the parameter_file.txt will be updated, and a new .txt file with the values of a calibration variable (water elevation, bottom elevation, ...) in the nodes listed in calibration_points.txt will be generated and stored in this folder. 

#simulations Folder: 
In each iteration, the BAL technique selects a new training point to improve the surrogate quality. Therefore, a new hydro-morphodynamic simulation will be run using the newly defined training point parameters. The simulation files produced by telemac are going to be stored in this folder. 

#external_libraries Folder:
The library pputils-master by Pat Prodanovic (https://github.com/pprodano/pputils) is used to extract the results of the simulation file (.slf) into a .txt file, which is then stored in the results Folder.

#scripts Folder:
-auxiliary_fuctions_BAL: Auxiliary functions for the stochastic calibration of model using Surrogate-Assisted Bayesian inversion
- auxiliary_functions_telemac: Contains auxiliary functions used to modify the input and output of the telemac files. These functions are specific to the parameters that wanted to be changed at the time, but they can be used as a base on how to modify Telemac's input and output files
-init.py: Reference other folders.