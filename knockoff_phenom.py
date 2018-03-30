import numpy as np
from jelena_instances import jelena_instance_AR

class small(jelena_instance_AR):

    signature = None
    name = 'AR(0.5)'
    n = 500
    p = 1000
    s = 30
    signal = np.sqrt(2 * np.log(p) / n)
    rho = 0.5

class medium(jelena_instance_AR):

    name = 'AR(0.5)'
    p = 1000
    n = 1000
    signal = np.sqrt(2 * np.log(p) / n)

class large(jelena_instance_AR):

    name = 'AR(0.5)'
    p = 1000
    n = 2000
    signal = np.sqrt(2 * np.log(p) / n)

class larger(jelena_instance_AR):

    name = 'AR(0.5)'
    p = 1000
    n = 2500
    signal = np.sqrt(2 * np.log(p) / n)

small.register(), medium.register(), large.register(), larger.register()

