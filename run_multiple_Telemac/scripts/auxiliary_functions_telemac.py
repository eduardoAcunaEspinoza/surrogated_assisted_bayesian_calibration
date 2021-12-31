'''
Auxiliary functions to couple the Surrogate-Assisted Bayesian inversion technique with Telemac. These functions are
specific of the parameters that wanted to be changed at the time, but they can be used as a base on how to modify
Telemac's input and output files

Contact: eduae94@gmail.com
'''

# Import libraries
import sys, os
import subprocess
import shutil
import numpy as np
import math
from datetime import datetime
import init
from ppmodules.selafin_io_pp import *


def update_steering_file(prior_distribution, parameters_name, initial_diameters, auxiliary_names, gaia_name, telemac_name,
                         result_name_gaia, result_name_telemac, n_simulation):

    # Update deposition stress
    updated_values = np.round(np.ones(4)*prior_distribution[0], decimals=3)
    updated_string = create_string(parameters_name[0], updated_values)
    rewrite_parameter_file(parameters_name[0], updated_string, gaia_name)

    # Update erosion stress
    updated_values = np.round(np.ones(2) * prior_distribution[1], decimals=3)
    updated_string = create_string(parameters_name[1], updated_values)
    rewrite_parameter_file(parameters_name[1], updated_string, gaia_name)

    # Update density
    updated_values = np.round(np.ones(2) * prior_distribution[2], decimals=0)
    updated_string = create_string(parameters_name[2], updated_values)
    rewrite_parameter_file(parameters_name[2], updated_string, gaia_name)

    # Update settling velocity
    rho_sed, rho_water, k_visc = 2650.0, 1000.0, 0.000001004
    new_diameters = initial_diameters * prior_distribution[3]
    settling_velocity = calculate_settling_velocity(new_diameters[1:], rho_sed, rho_water, k_visc)
    updated_values = "; ".join(map('{:.3E}'.format, settling_velocity))
    updated_values = "-9; " + updated_values.replace("E-0", "E-")
    updated_string = parameters_name[3] + " = " + updated_values
    rewrite_parameter_file(parameters_name[3], updated_string, gaia_name)

    # Update other variables
    new_diameters[0] = initial_diameters[0]  # the first non-cohesive diameter stays the same
    updated_values = "; ".join(map('{:.3E}'.format, new_diameters))
    updated_values = updated_values.replace("E-0", "E-")
    updated_string = auxiliary_names[0] + " = " + updated_values
    rewrite_parameter_file(auxiliary_names[0], updated_string, gaia_name)

    # Update result file name gaia
    updated_string = "RESULTS FILE"+"=" + result_name_gaia + str(n_simulation) + ".slf'"
    rewrite_parameter_file("RESULTS FILE", updated_string, gaia_name)
    # Update result file name telemac
    updated_string = "RESULTS FILE"+"="+ result_name_telemac + str(n_simulation) + ".slf'"
    rewrite_parameter_file("RESULTS FILE", updated_string, telemac_name)


def create_string(param_name, values):
    updated_string = param_name + " = "

    for i, value in enumerate(values):
        if i == len(values)-1:
            updated_string = updated_string + str(value)
        else:
            updated_string = updated_string + str(value) + "; "

    return updated_string


def rewrite_parameter_file(param_name, updated_string, path):

    # Save the variable of interest without unwanted spaces
    variable_interest = param_name.rstrip().lstrip()

    # Open the steering file with read permission and save a temporary copy
    gaia_file = open(path, "r")
    read_steering = gaia_file.readlines()

    # If the updated_string have more 72 characters, then it divides it in two
    if len(updated_string) >= 72:
        position = updated_string.find("=") + 1
        updated_string = updated_string[:position].rstrip().lstrip() + "\n" + updated_string[position:].rstrip().lstrip()

    # Preprocess the steering file. If in a previous case, a line had more than 72 characters then it was split in 2,
    # so this loop clean all the lines that start with a number
    temp = []
    for i, line in enumerate(read_steering):
        if not isinteger(line[0]):
            temp.append(line)
        else:
            previous_line = read_steering[i-1].split("=")[0].rstrip().lstrip()
            if previous_line != variable_interest:
                temp.append(line)

    # Loop through all the lines of the temp file, until it finds the line with the parameter we are interested in,
    # and substitute it with the new formatted line
    for i, line in enumerate(temp):
        line_value = line.split("=")[0].rstrip().lstrip()
        if line_value == variable_interest:
            temp[i] = updated_string + "\n"

    # Rewrite and close the steering file
    gaia_file = open(path, "w")
    gaia_file.writelines(temp)
    gaia_file.close()


def calculate_settling_velocity(diameters, rho_sed, rho_water, k_visc):
    settling_velocity = np.zeros(diameters.shape[0])
    s = rho_sed/rho_water
    for i, d in enumerate(diameters):

        if d <= 0.0001:
            settling_velocity[i] = (s - 1) * 9.81 * d**2 / (18 * k_visc)
        elif 0.0001 < d < 0.001:
            settling_velocity[i] = 10 * k_visc / d * (math.sqrt(1 + 0.01 * (s-1) * 9.81 * d**3 / k_visc**2) - 1)
        else:
            settling_velocity[i] = 1.1 * math.sqrt((s-1) * 9.81 * d)

    return settling_velocity


def run_telemac(telemac_file_name, number_processors):
    start_time = datetime.now()
    # Run telemac
    bash_cmd = "telemac2d.py " + telemac_file_name + " --ncsize="+number_processors
    process = subprocess.Popen(bash_cmd .split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print("Telemac simulation finished")
    print("Simulation time= " + str(datetime.now() - start_time))


def run_gretel(telemac_file_name, number_processors, folder_rename):
    # Save original working directory
    original_directory = os.getcwd()

    # Access folder with results
    subfolders = [f.name for f in os.scandir(original_directory) if f.is_dir()]
    simulation_index = [i for i, s in enumerate(subfolders) if telemac_file_name in s]
    simulation_index = simulation_index[0]
    original_name = subfolders[simulation_index]
    os.rename(original_name, folder_rename)
    simulation_path = './'+ folder_rename
    os.chdir(simulation_path)

    # Run gretel code
    bash_cmd = "gretel.py --geo-file=T2DGEO --res-file=GAIRES --ncsize="+number_processors+" --bnd-file=T2DCLI"
    process = subprocess.Popen(bash_cmd .split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print("Finish Gretel for GAIA")

    bash_cmd = "gretel.py --geo-file=T2DGEO --res-file=T2DRES --ncsize="+number_processors+" --bnd-file=T2DCLI"
    process = subprocess.Popen(bash_cmd .split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print("Finish Gretel for Telemac")

    # Copy result files in original folder
    shutil.copy('GAIRES', original_directory)
    shutil.copy('T2DRES', original_directory)
    os.chdir(original_directory)


def rename_selafin(original_name, new_name):
    # When I join parallel computed meshes with my gretel subroutine, even though the files is a SELAFIN it lacks the
    # extension. Here I change the name and create the proper extention
    if os.path.exists(original_name):
        os.rename(original_name, new_name)
    else:
        print("File not found")


def get_variable_value(file_name, calibration_variable, specific_nodes=None, save_name=""):
    # Read the SELEFIN file
    slf = ppSELAFIN(file_name)
    slf.readHeader()
    slf.readTimes()

    # Get the printout times
    times = slf.getTimes()

    # Read the variables names
    variables_names = slf.getVarNames()
    # Removed unnecessary spaces from variables_names
    variables_names = [v.strip() for v in variables_names]
    # Get the position of the value of interest
    index_variable_interest = variables_names.index(calibration_variable)

    # Read the variables values in the last time step
    slf.readVariables(len(times) - 1)

    # Get the values (for each node) for the variable of interest in the last time step
    modelled_results = slf.getVarValues()[index_variable_interest, :]
    format_modelled_results = np.zeros((len(modelled_results), 2))
    format_modelled_results[:, 0] = np.arange(1, len(modelled_results) + 1, 1)
    format_modelled_results[:, 1] = modelled_results

    # Get specific values of the model results associated in certain nodes number, in case the user want to use just
    # some nodes for the comparison. This part only runs if the user specify the parameter specific_nodes. Otherwise
    # this part is ommited and all the nodes of the model mesh are returned
    if specific_nodes is not None:
        format_modelled_results = format_modelled_results[specific_nodes.astype(int) - 1, :]

    if len(save_name) != 0:
        np.savetxt(save_name, format_modelled_results, delimiter="	", fmt=['%1.0f', '%1.3f'])

    # Return the value of the variable of interest
    return format_modelled_results


def isinteger(x):

    try:
        int(x)
        return True

    except ValueError:
        return False


def append_new_line(file_name, text_to_append):
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
