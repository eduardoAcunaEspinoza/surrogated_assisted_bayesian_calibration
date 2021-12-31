# Here is where we can compute the full-complexity model. In this case, as a test, we are just going to evaluate a non
# linear function
import numpy as np
import math


def analytical_function(time_steps, par_val):
    model_results = np.zeros((len(time_steps)))
    for i, t in enumerate(time_steps):
        model_results[i] = (par_val[0]**2 + par_val[1] - 1)**2 + par_val[0]**2 + 0.1*par_val[0] * \
                           math.exp(par_val[1]) - 2*par_val[0]*math.sqrt(0.5*t) + 1
    return model_results

