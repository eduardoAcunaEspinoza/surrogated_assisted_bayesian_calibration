#!/usr/bin/env python3
#
#+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#                                                                       #
#                                 sel2plot_1d.py                        # 
#                                                                       #
#+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!
#
# Author: Pat Prodanovic, Ph.D., P.Eng. 
# 
# Date: Oct 29, 2015
#
# Updated: Oct 31, 2015 - added headers to the output *.csv file.
#
# Updated: Feb 22, 2016 - uses selafin_io_pp class (works with python 2 and 3).
#
# Purpose: Script designed to take a 2d *.slf results file and a pputils line
# (i.e., a cross section or a profile line) and drapes the results (from a
# specified variable) onto the line for all time steps in the *.slf file.
# It also writes the profile for each time step as a separate *.png file.
#
# Using: Python 2 or 3, Matplotlib, Numpy
#
# Example: 
# python sel2plot_1d.py -i res.slf -v 1 -l profile.csv -o profile_Z.csv
#
# if using two variables, then run
# python sel2plot_1d.py -i res.slf -v 0 1 -l profile.csv -o profile_Z.csv
#
# the two variables are assumed to be vectors, and will be converted to
# the magnitude (i.e., if -v 0 is uvel and -v 1 is vvel, then the output
# will be v_mag 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os,sys
import numpy as np
import matplotlib.tri as mtri
from ppmodules.selafin_io_pp import * 
from matplotlib import pyplot as plt

if (len(sys.argv) == 9):
  dummy1 = sys.argv[1]
  input_file = sys.argv[2]
  dummy2 = sys.argv[3]
  var_idx = int(sys.argv[4])
  dummy3 = sys.argv[5]
  line_file = sys.argv[6]
  dummy3 = sys.argv[7]
  line_out_file = sys.argv[8]
elif (len(sys.argv) == 10):
  dummy1 = sys.argv[1]
  input_file = sys.argv[2]
  dummy2 = sys.argv[3]
  var1_idx = int(sys.argv[4])
  var2_idx = int(sys.argv[5])
  dummy3 = sys.argv[6]
  line_file = sys.argv[7]
  dummy3 = sys.argv[8]
  line_out_file = sys.argv[9]
else:
  print('Wrong number of Arguments, stopping now...')
  print('Usage:')
  print('python sel2plot_1d.py -i res.slf -v 1 -l profile.csv -o profile_Z.csv')
  sys.exit()

# create the line_out_file 
fout = open(line_out_file, "w")

# read the line (profile or cross section) file
line_data = np.loadtxt(line_file, delimiter=',',skiprows=0,unpack=True)
shapeid = line_data[0,:]
lnx = line_data[1,:]
lny = line_data[2,:]

# create the new output variables
lnz = np.zeros(len(lnx))
sta = np.zeros(len(lnx))
tempid = np.zeros(len(lnx))
dist = np.zeros(len(lnx))

# to create the sta array
sta[0] = 0.0
tempid = shapeid
dist[0] = 0.0

for i in range(1,len(lnx)):
  if (tempid[i] - shapeid[i-1] < 0.001):
    xdist = lnx[i] - lnx[i-1]
    ydist = lny[i] - lny[i-1]
    dist[i] = np.sqrt(np.power(xdist,2.0) + np.power(ydist,2.0))
    sta[i] = sta[i-1] + dist[i]

# Read the header of the selafin result file and get geometry and
# variable names and units
res = ppSELAFIN(input_file)
res.readHeader()
res.readTimes()

# get the mesh properties from the resultfile
NELEM, NPOIN, NDP, IKLE, IPOBO, x, y = res.getMesh()

# the IKLE array starts at element 1, but matplotlib needs it to start at zero
# note that doing this invalidates the IPOBO array; but we don't need IPOBO here
IKLE[:,:] = IKLE[:,:] - 1

# variable names and units
variables = res.getVarNames()
units = res.getVarUnits()

# number of variables
numvars = len(variables)

# times list
times = res.getTimes()

# find the bottom variable (this is needed for the 1d plots of WSE)
bottom_idx = -1
for i in range(len(variables)):
  if (variables[i].find('BOTTOM') > -1):
    bottom_idx = i

if (bottom_idx == -1):
  print('Variable BOTTOM is not found in the *.slf file!')
  #print 'Exiting'
  #sys.exit()

# create a result triangulation object
triang = mtri.Triangulation(x, y, IKLE)

# store the interpolated variable as a list, for every time step
interp_var_list = list()

if (len(sys.argv) == 9):
  # if there is only one variable
  # for every time step in the results file
  for i in range(len(times)):
    res.readVariables(i)
    slf_results = res.getVarValues()
    interpolator = mtri.LinearTriInterpolator(triang, slf_results[var_idx,:])
    interp_var = interpolator(lnx, lny)
    
    # put -999.0 if the line is outside of the results file domain
    where_are_NaNs = np.isnan(interp_var)
    interp_var[where_are_NaNs] = -999.9
  
    interp_var_list.append(interp_var)
elif (len(sys.argv) == 10):
  # if there are two variables, then compute magnitude
  # for every time step in the results file
  for i in range(len(times)):
    res.readVariables(i)
    slf_results = res.getVarValues()
    interpolator1 = mtri.LinearTriInterpolator(triang, slf_results[var1_idx,:])
    interp_var1 = interpolator1(lnx, lny)
    
    interpolator2 = mtri.LinearTriInterpolator(triang, slf_results[var2_idx,:])
    interp_var2 = interpolator2(lnx, lny)
    
    # compute magnitude of the two variables
    mag = np.sqrt(np.power(interp_var1,2.0) + np.power(interp_var2,2.0))
    
    # put -999.0 if the line is outside of the results file domain
    #where_are_NaNs = np.isnan(interp_var)
    #interp_var[where_are_NaNs] = -999.9
  
    interp_var_list.append(mag)

# to interpolate the bottom variable if it exists
if (bottom_idx != -1):
  bot_interpolator = mtri.LinearTriInterpolator(triang, slf_results[bottom_idx])
  bottom_var = bot_interpolator(lnx, lny)

# convert interp_var_list to numpy array
# interp_var_array = np.zeros((len(lnx), len(times)))
# for i in range(len(lnx)):
#   for j in range(len(times)):
#     interp_var_array[i,j] = interp_var_list[j][i]

# this is more efficient
interp_var_array = np.transpose(np.asarray(interp_var_list))

# to get the var name to display
if (len(sys.argv) == 9):
  var_str = variables[var_idx]
  unit_str = units[var_idx].replace(' ', '')
else:
  var_str = variables[var1_idx]
  #unit_str = units[var1_idx]
  unit_str = units[var1_idx].replace(' ', '')
  if (var_str == 'VELOCITY U      '):
    var_str = 'VELOCITY MAG    '

# column stack it all
if (bottom_idx != -1):
  stacked = np.column_stack((shapeid, sta, bottom_var, interp_var_array))
    
  n_str = 'shapeid,station,bottom,' 
  u_str = '-,M,M,'
  for i in range(len(times)):
    n_str = n_str + var_str + str(i)+ ','
    u_str = u_str + unit_str + ','
  header_str = n_str + '\n' + u_str
  
else:
  stacked = np.column_stack((shapeid, sta, interp_var_array))
  n_str = 'shapeid,station,' 
  u_str = '-,M,'
  for i in range(len(times)):
    n_str = n_str + var_str + str(i)+ ','
    u_str = u_str + unit_str + ','
  header_str = n_str + '\n' + u_str

# write the stacked array (this has 10 char spaces, 2 after decimal)
# columns are shapeid, sta, bot, var[t=0], var[t=1], ... , var[t=n]
np.savetxt(line_out_file, stacked, fmt='%.3f', header = header_str, 
  comments = '', delimiter=',') 

# to create a different figure for every time step
fignames = list()

# if bottom is in the *.slf file
if ((bottom_idx != -1) and len(sys.argv) == 9 ):
  if (variables[var_idx].find('FREE SURFACE') > -1):
    # axis ranges
    xmin = np.amin(sta)
    xmax = np.amax(sta)
    ymin = np.amin(bottom_var)
    ymax = np.amax(interp_var_array)
  
    for i in range(len(times)):
      fignames.append(input_file.split('.',1)[0] + "{:0>5d}".format(i) + '.png')
      plt.figure(figsize=(10, 5))
      plt.grid(True)
      #plt.text(xmax-0.5*xmax, ymax-0.5*ymax, r'step: ' + str(i) + ' time: ' + str(times[i]))
      plt.xlabel('Station [m]')
      plt.ylabel(var_str + '[' + unit_str + ']')
      plt.axis([xmin, xmax, ymin, ymax])
      plt.plot(stacked[:,1], stacked[:,2] )
      plt.plot(stacked[:,1], stacked[:,3+i] )
      plt.savefig(fignames[i], c = 'k')
      plt.close()
  else:
    # axis ranges
    xmin = np.amin(sta)
    xmax = np.amax(sta)
    ymin = np.amin(interp_var_array)
    ymax = np.amax(interp_var_array)
    
    for i in range(len(times)):
      fignames.append(input_file.split('.',1)[0] + "{:0>5d}".format(i) + '.png')
      plt.figure(figsize=(10, 5))
      plt.grid(True)
      #plt.text(xmax-0.5*xmax, ymax-0.5*ymax, r'step: ' + str(i) + ' time: ' + str(times[i]))
      plt.xlabel('Station [m]')
      plt.ylabel(var_str + '[' + unit_str + ']')
      plt.axis([xmin, xmax, ymin, ymax])
      #plt.plot(stacked[:,1], stacked[:,2] )
      plt.plot(stacked[:,1], stacked[:,3+i], linewidth = 1.5 )
      plt.savefig(fignames[i], c = 'k')
      plt.close()
    
elif ( (bottom_idx == -1) and (len(sys.argv) == 9 or len(sys.argv) == 10) ):
  # axis ranges
  xmin = np.amin(sta)
  xmax = np.amax(sta)
  ymin = np.amin(interp_var_array)
  ymax = np.amax(interp_var_array)
  
  for i in range(len(times)):
    fignames.append(input_file.split('.',1)[0] + "{:0>5d}".format(i) + '.png')
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    #plt.text(xmax-0.5*xmax, ymax-0.5*ymax, r'step: ' + str(i) + ' time: ' + str(times[i]))
    plt.xlabel('Station [m]')
    #if (len(sys.argv) == 9):
    #  plt.ylabel(variables[var_idx] + '[' + units[var_idx] + ']')
    #else:
    #  plt.ylabel(variables[var1_idx] + '[' + units[var1_idx] + ']')
    plt.ylabel(var_str + '[' + unit_str + ']')
    
    plt.axis([xmin, xmax, ymin, ymax])
    #plt.plot(stacked[:,1], stacked[:,2] )
    plt.plot(stacked[:,1], stacked[:,2+i], linewidth = 1.5 )
    plt.savefig(fignames[i], c = 'k')
    plt.close()
    
elif ( (bottom_idx != -1) and (len(sys.argv) == 9 or len(sys.argv) == 10) ):
  # axis ranges
  xmin = np.amin(sta)
  xmax = np.amax(sta)
  ymin = np.amin(interp_var_array)
  ymax = np.amax(interp_var_array)
  
  for i in range(len(times)):
    fignames.append(input_file.split('.',1)[0] + "{:0>5d}".format(i) + '.png')
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    #plt.text(xmax-0.5*xmax, ymax-0.5*ymax, r'step: ' + str(i) + ' time: ' + str(times[i]))
    plt.xlabel('Station [m]')
    #if (len(sys.argv) == 9):
    #  plt.ylabel(variables[var_idx] + '[' + units[var_idx] + ']')
    #else:
    #  plt.ylabel(variables[var1_idx] + '[' + units[var1_idx] + ']')
    plt.ylabel(var_str + '[' + unit_str + ']')  
    plt.axis([xmin, xmax, ymin, ymax])
    #plt.plot(stacked[:,1], stacked[:,2] )
    plt.plot(stacked[:,1], stacked[:,3+i], linewidth = 1.5 )
    plt.savefig(fignames[i], c = 'k')
    plt.close()
