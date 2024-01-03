'''
- This file drives an experiment related to Testing Effectiveness.
- All experiment parameters should be placed in a .yaml file.
- Each experiment corresponds to one or more simulations (simulation.py)
- Each simulation corresponds to a set of montecarlo draws (montecarlo.py)
'''

import yaml
import itertools
import argparse
import pandas as pd
from pathlib import Path

import simulation as sim

OUTPUT_PATH = Path(__file__).parent.parent / 'output'
INPUT_PATH = Path(__file__).parent.parent / 'input'


parser = argparse.ArgumentParser()
parser.add_argument('experiment_file')
args = parser.parse_args()

# yaml -> dict
with open(INPUT_PATH / '{}'.format(args.experiment_file),'r') as stream:
    try:
        parsed_yaml = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

        
# dict -> variables
for name, value in parsed_yaml.items():
    if isinstance(value,list):
        globals()[name] = value
    elif name in ['N_draws','log_name']:
        globals()[name] = value
    else:
        globals()[name] = [value]


# hard-coded variables
input_columns = ['wait','supply','Q','tat','c','f','T','L','wait_exit',
    'Q_exit','f_exit','L_exit','kinetics','testing','isolation']
log_columns = ['tDx','tExit','n_tests','n_tests_exit','I0','Itest',
    'Iexit','A','P','B','tSx','m','M','first','last','infectious_threshold']
outcome_columns = ['TE','ascertainment','n_tests_dx','R_no_testing',
    'R_testing','R_post_exit','n_tests_exit','T_isolation']

# create log
log_path = OUTPUT_PATH / '{}'.format(log_name+'.csv')
sim.initialize_log(log_path,input_columns,outcome_columns)


ranges = [wait,supply,Q,tat,c,f,T,L,
           wait_exit,Q_exit,f_exit,L_exit,
           kinetics,testing,isolation]
for args in itertools.product(*ranges):
    simulation_name = sim.local_hash(args)
    if sim.is_simulation_redundant(simulation_name,OUTPUT_PATH)==True:
        print('🚀 {}'.format(simulation_name))
    else:
        outputs = sim.conduct_simulation(args,N_draws)
        sim.save_simulation(simulation_name,outputs,log_columns,OUTPUT_PATH)
        print('🧑‍💻 {}'.format(simulation_name))
    df = pd.read_csv(OUTPUT_PATH / (simulation_name+'.zip'))
    outcomes = sim.summarize_simulation(df)
    print('\tTE={:.4f}'.format(outcomes[0]))
    
    sim.append_simulation_to_log(log_path,args,simulation_name,outcomes)