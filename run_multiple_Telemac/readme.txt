Run multiple hydro-morphodynamic simulations of Telemac-2d, using a list of parameter combinations. The code is specific to the parameters that wanted to be changed at the time, but it can be used as the base to run other specific numerical configurations.

To run the code, run the run_multiple_telemac.py file using the main folder as a current directory from a console/terminal in which Telemac and GAIA have already been compiled. It is not recommended to run the code from PyCharm as PyCharm uses a kind of additional virtual environment when it fires up its Terminal, and because Telemac has its own environment and APIs, those might be conflicting with PyCharm. 

#main Folder: 
-run_multiple_telemac.py: runs a hydro-morphodynamic simulation using the Telemac 2D software coupled with the GAIA module for all the parameter combinations located in the file parameter_comb.txt. The parameters modified in each run are named in the variable parameters_name in the USER INPUT section of the code. These parameters should be one of the KeyWords listed in Telemac or GAIA.
-parameter_comb.txt: Have the numerical value of the parameter combinations for which the telemac software is going to be run. For this example, 4 parameters (4 columns) are going to be modified 3 times (3 lines). Therefore when the code is run, there are going to be 3 different simulations. 
-calibration_points.txt: This file contains the number of the nodes that will be used in case the values of a specific variable want to be extracted from particular nodes of the mesh.
-init.py: Reference other folders.
- Files necessary to run the hydro-morhodynamic model using Telemac2D and GAIA: 
	- bc_liquid.liq: Liquid boundary condition (flow, sediment or tracers inflow/outflow)
	- bc_steady_tel.cli: File that defines the type and location of the boundary conditions.
	- geo_banda.slf: File that defines the mesh structure for the hydro-morphodynamic model. 
	-run_liquid_gaia.cas: Numerical configuration of the sediment transport model.
	- run_liquid_tel.cas: Numerical configuretion of the hydrodynamic model. 

#simulations Folder: 
After each simulation is completed, the simulation files will be stored in this folder. 

#results Folder:
After each simulation is completed, a .txt file with the values of a specified variable (water elevation, bottom elevation, ...) in the nodes listed in calibration_points.txt will be generated and stored in this folder. 

#external_libraries Folder:
The library pputils-master by Pat Prodanovic (https://github.com/pprodano/pputils) is used to extract the results of the simulation file (.slf) into a .txt file, which is then stored in the results Folder.

#scripts Folder:
- auxiliary_functions_telemac: Contains auxiliary functions used to modify the input and output of the telemac files. These functions are specific to the parameters that wanted to be changed at the time, but they can be used as a base on how to modify Telemac's input and output files
-init.py: Reference other folders.