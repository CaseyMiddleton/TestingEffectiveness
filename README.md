# Calculating Testing Effectiveness (TE)
The purpose of this code is to rapidly perform estimates of Testing Effectiveness, ascertainment, test consumption, and isolation days, for a variety of pathogens, diagnostics, and testing scenarios. Requirements, installation, and examples can be found below.

The code reads its inputs from a `.yaml` parameter file, and writes each ensemble of Monte Carlo simulations to a single `.csv.zip`, which is summarized in a single line `csv`. 

All code is written in pure Python. 
# Requirements
```
numpy
pandas
hashlib
pathlib
yaml
itertools
argparse
```
# Installation
Simply place the folders where you wish, and ensure that Python and the requirements are installed.

# General Usage
Call the code by
```
python [PATH_TO_SRC]/experiment.py [PARAMETER_FILE_NAME].yaml
```
The code will automatically look in the `input` folder for the parameter file.
# Example Usage
From the same directory as the README, this line runs the example_1 experiment
```
python src/experiment.py example_1.yaml
```
This will run the experiment detailed in `example_1.yaml`, investigating daily use of a rapid test for up to 3 days, under three conditions 
1. starting at symptoms
2. starting at symptoms + 1d 
3. starting at symptoms + 2d
with 10,000 draws each. For each condition, the draws will be stored in a `.csv.zip` file. The summary of all three conditions will be stored in `example_1.csv`. From these, we should learn that testing effectiveness (TE) is approximately  60%, 32%, and 6% for each of those conditions, respectively. 

Note that running the same code again will avoid re-running the Monte Carlo draws, whose outcomes remain stored in the `.zip` files. Instead, the code will recognize that the simulations have already been done, and will simply read in the results and summarize them in the summary `.csv`.

There are two other examples which are meant to showcase a few other features of the way in which `.yaml` files store parameters.
```
python src/experiment.py example_2A.yaml
python src/experiment.py example_2B.yaml
```
These examples explore the differences between founder SARS-CoV-2 in naive hosts and omicron SARS-CoV-2 in experienced hosts, and the way that fixed-duration isolation (example_2A) or test-to-exit (example_2B) may affect testing effectiveness.

# Inputs

## Note: .yaml files
All inputs should be placed in `/input` in a yaml-formatted file. This is a lightweight and intuitive markup language, and examples are provided to get started quickly. 

## Note: single values vs multiple values
You may write single values, like `Q: 7` to specify weekly testing, or multiple values in arrays, like `Q: [3.5, 7]`.  Specifying multiply values means that the code will iterate over each value in a separate set of Monte Carlo draws. For instance, the vector `Q: [3.5, 7]` would first explore twice-weekly testing, and then explore weekly testing.

## Input Parameters
-`log_name`: the name of the log file in which outputs should be written. Multiple input files may share the same log file. If a log file already exists, results will be appended to it, and will not overwrite it.

### Kinetics Parameters
- `kinetics`: the name of the pathogen kinetics function from which parameters should be drawn. Options: 
	- `kinetics_flu`
	- `kinetics_sarscov2_founder_naive`
	- `kinetics_sarscov2_omicron_experienced`
	- `kinetics_rsv`

### Testing Parameters
- `testing`: the type of testing behavior to be modeled. Options:
	- `test_regular`: testing on a regular cadence every `Q` days with uniformly random phase
	- `test_post_exposure`: testing after exposure after an additional `wait` days every `Q` days up to `supply` times
	- `test_post_symptoms`: testing after symptom onset after an additional `wait` days every `Q` days up to `supply` times
- `Q`: the number of days between tests. Positive real number.
- `wait`: the number of days to wait before testing begins. Positive real number.
	- IGNORED for `test_regular`
- `supply`: the maximum number of tests to be used for diagnosis. Positive integer.
	- IGNORED for `test_regular`

### Diagnostic Parameters
- `tat`: turnaround time from sample to answer, in days. Non-negative real number.
- `L`: analytical limit of detection, given in Log10 cp RNA/ml. Non-negative real number.
- `f`: failure rate for tests administered at pathogen loads exceeding `L`. Probability [0,1].

### Behavior Parameters
- `c`: compliance, the probability that a scheduled/intended test is actually taken. Probability [0,1].
- participation is *not* included among the behavior parameters, as it may be applied to scale outputs directly. See manuscript.

### Isolation Parameters
- `isolation`: the type of isolation to be modeled. Options:
	- `fixed`: a fixed-duration isolation for `wait_exit` days.
	- `TTE`: a test-to-exit isolation lasting until the first negative test result, where testing begins after `wait_exit` days every `Q_exit` days with an analytical limit of detection `L_exit` and failure rate `f_exit`. TTE tests are assumed to have zero turnaround time.
- `wait_exit`: the number of days to wait before leaving isolation or beginning TTE. Nonnegative real number.
- `Q_exit`: the number of days between TTE tests. Positive real number.
	- IGNORED for `fixed` isolation.
- `L_exit`: analytical limit of detection, given in Log10 cp RNA/ml.
	- IGNORED for `fixed` isolation.
- `f_exit`: failure rate for TTE tests administered at pathogen loads exceeding `L_exit`. Probability [0,1].
	- IGNORED for `fixed` isolation.
	
### Simulation Parameters
- `N_draws`: the number of Monte Carlo draws to perform.
	- Single values only; no ranges.
- `T`: the total duration of each simulation in days.

# Outputs
Outputs are saved in `.csv` files. Note that all input parameters are also stored there. In this way, the output files can be ingested by other code to display, plot, or analyze results. All outputs are estimates, computed as means over many simulations. Thus, confidence intervals may be computed as desired.

- `TE`: testing effectiveness, the proportion by which the testing program and subsequent isolation decreases the risk of transmission.
- `ascertainment`: the proportion of infections predicted to be detected.
- `n_tests_dx`: number of tests used to attempt diagnosis, whether or not successful, per INFECTED person.
- `n_tests_exit`: number of tests used to exit isolation, per ISOLATED person.
- `R_no_testing`: total infectiousness in ARBITRARY UNITS without testing.
- `R_testing`: total infectiousness in the same ARBITRARY UNITS with testing.
	- Note that `TE = 1 - R_testing / R_no_testing`.
- `R_post_exit`: the total infectiousness that occurred *after* release from isolation. This quantity is already incorporated into the outputs above.
- `T_isolation`: time spent in isolation PER ISOLATED PERSON.
