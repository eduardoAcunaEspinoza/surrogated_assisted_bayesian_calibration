'''
Run multiple hydro-morphodynamic simulation of Telemac-2d, using a list of parameter combinations given in the file
parameter_comb.txt. The code is specific of the parameters that wanted to be changed at the time, but it can be used as
the base to run other specific numerical configurations.

Contact: eduae94@gmail.com
'''

# Import libraries
import shutil
import numpy as np
import init
from auxiliary_functions_telemac import *

# ---------------------------------------------------------------------------------------------------------------------
# USER INPUT  -------

# Telemac
telemac_name = "run_liquid_tel.cas"
gaia_name = "run_liquid_gaia.cas"
n_processors = "12"
n_previous_simulations = 100

# Calibration parameters
calibration_variable = "BOTTOM"
initial_diameters = np.array([0.001, 0.000024, 0.0000085, 0.0000023])
prior_distribution = np.loadtxt("parameter_comb.txt")
NS = prior_distribution.shape[0]
parameters_name = ["CLASSES CRITICAL SHEAR STRESS FOR MUD DEPOSITION",
                   "LAYERS CRITICAL EROSION SHEAR STRESS OF THE MUD",
                   "LAYERS MUD CONCENTRATION",
                   "CLASSES SETTLING VELOCITIES"]
auxiliary_names = ["CLASSES SEDIMENT DIAMETERS"]

# Paths
path_calibration_points = "calibration_points.txt"
result_name_gaia = "'res_gaia_PC"  # PC stands for parameter combination
result_name_telemac = "'res_tel_PC"  # PC stands for parameter combination
path_results = "../results"
path_simulations = "../simulations"

# END OF USER INPUT  --------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Initialization of information
calibration_points = np.loadtxt("calibration_points.txt")

# Loop for the number of times I want to run my model
for i in range(NS):
    # Part 1: Update steering files---------------------------------------------------------------------------------
    n_run = i + 1 + n_previous_simulations
    update_steering_file(prior_distribution[i, :], parameters_name, initial_diameters, auxiliary_names, gaia_name,
                         telemac_name, result_name_gaia, result_name_telemac, n_run)

    # Part 2: Run model ---------------------------------------------------------------------------------------------
    run_telemac(telemac_name, n_processors)

    # Part 3: Extract values of interest ----------------------------------------------------------------------------
    updated_string = result_name_gaia[1:] + str(i + 1 + n_previous_simulations) + ".slf"
    save_name = path_results + "/PC" + str(i + 1 + n_previous_simulations) + "_" + calibration_variable + ".txt"
    modelled_results = get_variable_value(updated_string, calibration_variable, calibration_points, save_name)

    # Part 4. Move the created files to their respective folders
    shutil.move(result_name_gaia[1:] + str(i + 1 + n_previous_simulations) + ".slf", path_simulations)
    shutil.move(result_name_telemac[1:] + str(i + 1 + n_previous_simulations) + ".slf", path_simulations)

    # Part 5. Append the parameter used to a file
    new_line = "; ".join(map('{:.3f}'.format, prior_distribution[i, :]))
    new_line = "PC" + str(i + 1 + n_previous_simulations) + "; " + new_line
    append_new_line(path_results + "/parameter_file.txt", new_line)

