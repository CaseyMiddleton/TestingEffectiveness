'''
- This file contains ONLY functions to run a set of montecarlo simulations
  for a fixed set of input parameters. To do so, it repeatedly calls
  functions found in montecarlo.py. 
- Each simulation is converted to a hash, with its montecarlo draws stored
  in [hash].csv.  If that file does not exist, the specified number of draws
  are computed. If that file does already exist, computations are skipped,
  and the simulation's summary values are logged instead.
'''

import pandas as pd
import hashlib
import types
from pathlib import Path
import numpy as np

import montecarlo as mc
import kinetics as knx

def conduct_simulation(inputs,N_draws):
    wait,supply,Q,tat,c,f,T,L,wait_exit,Q_exit,f_exit,L_exit,kinetics,testing,isolation = inputs
    opts_testing = {'test_regular':mc.test_regular,
                'test_post_exposure':mc.test_post_exposure,
                'test_post_symptoms':mc.test_post_symptoms,}
    testing_function = opts_testing[testing]

    opts_kinetics = {'kinetics_test':knx.kinetics_test,
                 'kinetics_flu':knx.kinetics_flu,
                 'kinetics_sarscov2_founder_naive':knx.kinetics_sarscov2_founder_naive,
                 'kinetics_sarscov2_omicron_experienced':knx.kinetics_sarscov2_omicron_experienced,
                 'kinetics_rsv':knx.kinetics_rsv}
    kinetics_function = opts_kinetics[kinetics]
    results = []
    for i in range(N_draws):
        outputs = mc.get_sample(wait,supply,Q,tat,c,f,T,L,
                             wait_exit,Q_exit,f_exit,L_exit,
                             kinetics_function,testing_function,isolation)
        results.append(outputs)
    return results

def save_simulation(simulation,outputs,output_columns,OUTPUT_PATH):
    df = pd.DataFrame(outputs,columns=output_columns)
    compression_opts = dict(method='zip',archive_name=simulation+'.csv')
    df.to_csv(OUTPUT_PATH / (simulation+'.zip'),
              index=False,
              compression=compression_opts)

def summarize_simulation(df):
    R_0 = np.mean(df['I0'].values)
    R_testing_perfect_isolation = np.mean(df['Itest'].values)
    R_post_exit = np.mean(df['Iexit'].values)
    R_testing = R_testing_perfect_isolation+R_post_exit
    ascertainment = compute_ascertainment(df['tDx'].values)
    n_tests_dx = np.mean(df['n_tests'].values)
    n_tests_exit = np.mean(df['n_tests_exit'].values)
    T_isolation = compute_isolation_time(df['tDx'].values,df['tExit'].values)
    TE = 1-R_testing/R_0
    outcomes = TE,ascertainment,n_tests_dx,R_0,R_testing,R_post_exit,n_tests_exit,T_isolation
    return outcomes

def compute_ascertainment(tDx):
    N = len(tDx)
    misses = np.sum(np.isinf(tDx))
    return 1-misses/N

def compute_isolation_time(tDx,tExit):
    T_isolation = tExit-tDx
    T_isolation_nonzero = T_isolation[T_isolation>0]
    if len(T_isolation_nonzero)==0:
        return 0
    else:
        return np.mean(T_isolation_nonzero)

def local_hash(inputs):
    m = hashlib.new('sha256')
    input_strings = []
    for item in inputs:
        input_strings.append(str(item))
    input_string = ','.join(input_strings)
    bytestring=bytes(input_string,encoding='utf-8')
    hashed = hashlib.sha256(bytestring).hexdigest()
    return hashed

def append_simulation_to_log(log_path,inputs,simulation,outcomes):
    simulation_check = local_hash(inputs)
    if simulation != simulation_check:
        raise Exception('hash failure')
    
    with open(log_path,'a') as myfile:
        myfile.write(str(simulation))
        for item in inputs:
            myfile.write(',')
            if isinstance(item,types.FunctionType):
                myfile.write(item.__name__)
            else:
                myfile.write(str(item))
        for item in outcomes:
            myfile.write(',')
            myfile.write(str(item))
        myfile.write('\n')
        
def initialize_log(log_path,input_columns,outcome_columns):
    if log_path.exists():
        pass
    else:
        with open(log_path,'w') as myfile:
            myfile.write('simulation')
            for col in input_columns:
                myfile.write(',')
                myfile.write(col)
            for col in outcome_columns:
                myfile.write(',')
                myfile.write(col)
            myfile.write('\n')

def is_simulation_redundant(simulation,output_folder):
    fname = output_folder / (simulation+'.zip')
    if fname.exists():
        return True
    else:
        return False