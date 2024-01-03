'''
- This file contains ONLY kinetics functions.
- Each of the functions herein takes zero arguments. 
- Calling one of these functions returns a standard set of parameters,
  which describe (1) a "tent function" for viral kinetics, (2) time of
  symptom onset, and (3) a threshold above which individuals are assumed 
  to be infectious.
- Relevant sources for parameter values are in the accompanying manuscript
  by Middleton & Larremore. 
'''

import numpy as np

def kinetics_test():
    VL_min_infectious = 0
    VL_floor = 0
    VL_peak = 10
    
    t_latent = 0
    dt_proliferation = 5
    dt_clearance = 5
    t_peak = t_latent + dt_proliferation
    t_clearance = t_peak + dt_clearance    
    t_Sx = t_peak + np.random.uniform(-5,-0)
    
    return t_latent,t_peak,t_clearance,t_Sx,VL_floor,VL_peak,VL_min_infectious


def kinetics_flu():
    VL_min_infectious = 4 + 0.5
    VL_floor = 2.95
    VL_peak = np.random.uniform(6,8.5)
    
    t_latent = np.random.uniform(0.5,1.5)
    dt_proliferation = np.random.uniform(1,3)
    dt_clearance = np.random.uniform(2,3)
    t_peak = t_latent + dt_proliferation
    t_clearance = t_peak + dt_clearance    
    t_Sx = t_peak + np.random.uniform(-2,0)
    
    return t_latent,t_peak,t_clearance,t_Sx,VL_floor,VL_peak,VL_min_infectious

def kinetics_rsv():
    VL_min_infectious = 2.8 + 0.5
    VL_floor = 2.8
    VL_peak = np.random.uniform(4,8)
    
    t_latent = np.random.uniform(2,4)
    dt_proliferation = np.random.uniform(2,4)
    dt_clearance = np.random.uniform(3,6)
    t_peak = t_latent + dt_proliferation
    t_clearance = t_peak + dt_clearance
    t_Sx = t_peak + np.random.uniform(-1,1)
    
    return t_latent,t_peak,t_clearance,t_Sx,VL_floor,VL_peak,VL_min_infectious

def kinetics_sarscov2_founder_naive():
    VL_min_infectious = 5.5 + 0.5
    VL_floor = 3
    VL_peak = np.random.lognormal(1.99950529,0.19915982)
    while VL_peak<VL_floor:
        VL_peak = np.random.lognormal(1.99950529,0.19915982)
        
    t_latent = np.random.uniform(2.5,3.5)
    dt_proliferation = np.random.lognormal(0.87328419,0.78801765)
    while ((dt_proliferation<0.5) or (dt_proliferation>10)):
        dt_proliferation = np.random.lognormal(0.87328419,0.78801765)
    dt_clearance = np.random.lognormal(1.95305127,0.61157974)
    while ((dt_clearance<0.5) or (dt_clearance>25)):
        dt_clearance = np.random.lognormal(1.95305127,0.61157974)
    t_peak = t_latent + dt_proliferation
    t_clearance = t_peak + dt_clearance
    t_Sx = t_peak + np.random.uniform(0,3)
    
    return t_latent,t_peak,t_clearance,t_Sx,VL_floor,VL_peak,VL_min_infectious

def kinetics_sarscov2_omicron_experienced():
    VL_min_infectious = 5.5 + 0.5
    VL_floor = 3
    VL_peak = np.random.lognormal(1.87567526,0.18128526)
    while VL_peak<VL_floor:
        VL_peak = np.random.lognormal(1.87567526,0.18128526)
        
    t_latent = np.random.uniform(2.5,3.5)
    dt_proliferation = np.random.lognormal(1.05318392,0.68842803)
    while ((dt_proliferation<0.5) or (dt_proliferation>10)):
        dt_proliferation = np.random.lognormal(1.05318392,0.68842803)
    dt_clearance = np.random.lognormal(1.70427144,0.49046478)
    while ((dt_clearance<0.5) or (dt_clearance>25)):
        dt_clearance = np.random.lognormal(1.70427144,0.49046478)
    t_peak = t_latent + dt_proliferation
    t_clearance = t_peak + dt_clearance
    t_Sx = t_peak + np.random.uniform(-5,-1)
    
    return t_latent,t_peak,t_clearance,t_Sx,VL_floor,VL_peak,VL_min_infectious