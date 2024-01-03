import numpy as np
from scipy.stats import gamma

def get_VL(t,Tlatent,Tpeak,Ypeak,Tclear,Ydetect,Yundetect,Yinf,**kwargs):
    '''
    Computes the scalar value of a hinge function with control points:
        (Tlatent, Ydetect) : time spent between exposure and log Ydetect VL
        (Tpeak, Ypeak) : time between log 3 VL and peak infection, VL at peak
        (Tclear, Yundetect) : time between peak and crossing log Ydetect VL
    Returns zero whenever (1) t<T3 or (2) hinge(t) is negative.
    '''
    if t < Tlatent:
        return 0
    if t < Tpeak:
        VL = (t-Tlatent) * (Ypeak-Ydetect)/(Tpeak-Tlatent) + Ydetect
    else:
        VL = np.max([(t-Tpeak) * (Yundetect-Ypeak)/(Tclear-Tpeak) + Ypeak,0])
    return VL

def get_inf(t,Tlatent,Tpeak,Ypeak,Tclear,Ydetect,Yundetect,Yinf,**kwargs):
    '''
    Computes infectiousness at time t, assuming infectiousness is equal to viral load above
        Yinf, the VL at which an individual becomes infectious
    Returns infectious viral load at time t.
    '''
    inf = get_VL(t,Tlatent,Tpeak,Ypeak,Tclear,Ydetect,Yundetect,Yinf) - Yinf
    return np.max([inf,0])

def is_detectable_viral(tvals,VL,sensitivity_threshold,**kwargs):
    '''
    Computes detectability by a viral test given
        VL : viral load
        sensitivity_threshold : viral test limit of detection
    Returns a vector of 0's (undetectable) and 1's (detectable)
    '''
    D = np.zeros(len(VL))
    D[np.array(VL) >= sensitivity_threshold] = 1
    return D

def prob_test_screening(tvals,frequency,**kwargs):
    '''
    Computes the probability of being tested at time t, assuming a uniform distribution of testing every frequency days.
    Returns a vector containing the probability of being tested at time t for all t in tvals
    '''
    return [1/frequency]*len(tvals)

def prob_test_symp(tvals,peak,a,b,w,n,tpd):
    '''
    peak : time of peak VL
    a,b : first and last possible times of symptom onset in relation to tpeak
    w : wait time after symptom onset before testing
    n : number of tests provided
    tpd : tests used per day
    Returns a vector of Pr(test at t) for all times in tvals
    '''
    x = min(n/tpd,b-a) # last possible test, either because you ran out or because you didn't have symptoms anymore
    slope = tpd/(b-a)
    A = np.zeros(len(tvals))
    for ii,t in enumerate(tvals):
        if t >= peak+a+w and t < peak+a+w+x:
            A[ii] = slope*(t-(peak+a+w))
        elif t >= peak+a+w+x and t < peak+b+w+n/tpd-x:
            A[ii] = slope*x
        elif t >= peak+b+w+n/tpd-x and t <= peak+b+w+n/tpd:
            A[ii] = -slope*(t-(peak+b+w+n/tpd-x)) + slope*x
    return A

def prob_test_reflex(tvals,trigger_time,wait,num_tests,tests_per_day):
    '''
    Returns the probability of being tested at time t assuming a prompt to test at time trigger_time
    where individuals test *wait* days later and then use *tests_per_day* tests daily
    until *num_tests* are exhausted
    '''
    tst = []
    for t in tvals:
        if t >= trigger_time + wait and t <= trigger_time + wait + num_tests/tests_per_day:
            tst.append(tests_per_day)
        else:
            tst.append(0)
    return tst

def create_testing_program(pathogen,**kwargs):
    ''' Get parameter values for any testing program
        pathogen : COVID, RSV, FLU
        kwargs : any test attributes that should be included in this testing program (e.g. frequency)
    '''
    param_func = pathogen + "_test_params"
    params = globals()[param_func]()
    for param_name in kwargs.keys():
        params[str(param_name)] = kwargs[param_name]
    return params

# Virus Parameter values ----------------------------------------------------------------
# source : https://github.com/skissler/CtTrajectories_B117/blob/main/code/utils.R
def convert_Ct_logGEML(Ct):
    m_conv=-3.609714286
    b_conv=40.93733333
    out = (Ct-b_conv)/m_conv * np.log10(10) + np.log10(250)
    return out

def COVID_params_stoch():
    '''
    Defines a dictionary of stochastically drawn parameter values for COVID viral load and infectiousness.
    Using data from Kissler, et al. (https://doi.org/10.7554/eLife.81849) - assuming Omicron variant with 1-2 doses
    '''
    params = {}
    params['LOD_high'] = 3;
    params['LOD_low'] = 6;
    params['Ydetect'] = params['LOD_high']; # gold standard detection
    params['Yundetect'] = params['Ydetect'];
    params['Yinf'] = 5.5;
    params['Tlatent'] = np.random.random()+2.5;

    params['Tpeak'] = params['Tlatent'] + max(0.5,np.random.lognormal(1.053,0.688))
    while params['Tpeak'] - params['Tlatent'] >= 10:
        params['Tpeak'] = params['Tlatent'] + max(0.5,np.random.lognormal(1.053,0.688))

    params['Ypeak'] = np.random.lognormal(1.876,0.181) 
    while params['Ypeak'] < params['Yundetect']:
       params['Ypeak'] = np.random.lognormal(1.876,0.181)

    params['Tclear'] = params['Tpeak'] + max(0.5,np.random.lognormal(1.704,0.490))
    while params['Tclear'] - params['Tpeak'] >= 25:
        params['Tclear'] = params['Tpeak'] + max(0.5,np.random.lognormal(1.704,0.490))

    return params

def COVID_params_det():
    params = {}
    params['LOD_high'] = 3;
    params['LOD_low'] = 6;
    params['Ydetect'] = params['LOD_high']; # gold standard detection
    params['Yundetect'] = params['Ydetect'];
    params['Yinf'] = 5.5;
    params['Tlatent'] = 3;
    params['Tpeak'] = params['Tlatent'] + 3.7;
    params['Ypeak'] = 7
    params['Tclear'] = params['Tpeak'] + 7.75
    return params

def WT_params_stoch():
    params = {}
    params['LOD_high'] = 3;
    params['LOD_low'] = 5;
    params['Ydetect'] = params['LOD_high']; # gold standard detection
    params['Yundetect'] = 3;
    params['Yinf'] = 5.5;
    params['Tlatent'] = np.random.random()+2.5;

    params['Tpeak'] = params['Tlatent'] + max(0.5,np.random.lognormal(0.873,0.788))
    while params['Tpeak'] - params['Tlatent'] >= 10:
         params['Tpeak'] = params['Tlatent'] + max(0.5,np.random.lognormal(0.873,0.788))
    
    params['Ypeak'] = np.random.lognormal(1.9995,0.199)
    while params['Ypeak'] <= params['Yundetect']:
        params['Ypeak'] = np.random.lognormal(1.9995,0.199)

    params['Tclear'] = params['Tpeak'] + max(0.5,np.random.lognormal(1.953,0.612))
    while params['Tclear'] - params['Tpeak'] >= 25:
        params['Tclear'] = params['Tpeak'] + max(0.5,np.random.lognormal(1.953,0.612))
    return params

def WT_params_det():
    params = {}
    params['LOD_high'] = 3;
    params['LOD_low'] = 5;
    params['Ydetect'] = params['LOD_high']; # gold standard detection
    params['Yundetect'] = 3;
    params['Yinf'] = 5.5;
    params['Tlatent'] = 3;
    params['Tpeak'] = params['Tlatent'] + 3.2
    params['Ypeak'] = 7.6
    params['Tclear'] = params['Tpeak'] + 8.5
    return params

def RSV_params_stoch():
    '''
    Defines a dictionary of stochastically drawn parameter values for RSV viral load and infectiousness.
    '''
    params = {}
    params['LOD_high'] = 2.8;
    params['LOD_low'] = 5;
    params['Ydetect'] = params['LOD_high'];
    params['Yundetect'] = params['Ydetect'];
    params['Yinf'] = params['Ydetect'];
    params['Tlatent'] = np.random.uniform(2,4);
    params['Tpeak'] = params['Tlatent'] + np.random.uniform(2,4)
    params['Ypeak'] = np.random.uniform(4,8)
    while params['Ypeak'] <= params['Yundetect']:
        params['Ypeak'] = np.random.uniform(4,8)
    params['Tclear'] = params['Tpeak'] + np.random.uniform(3,6)
    return params

def RSV_params_det():
    '''
    Defines a dictionary of stochastically drawn parameter values for RSV viral load and infectiousness.
    '''
    params = {}
    params['LOD_high'] = 2.8;
    params['LOD_low'] = 5;
    params['Ydetect'] = params['LOD_high'];
    params['Yundetect'] = params['Ydetect'];
    params['Yinf'] = params['Ydetect'];
    params['Tlatent'] = 3
    params['Tpeak'] = params['Tlatent'] + 3
    params['Ypeak'] = 6
    params['Tclear'] = params['Tpeak'] + 4.5
    return params

def FLU_params_stoch():
    '''
    Defines a dictionary of stochastic parameter values for FLU viral load and infectiousness.
    '''
    params = {}
    params['LOD_high'] = 2.95;
    params['LOD_low'] = 5.38;
    params['Ydetect'] = params['LOD_high']; # gold standard detection
    params['Yundetect'] = params['Ydetect'];
    params['Yinf'] = 4;
    params['Tlatent'] = np.random.uniform(0.5,1.5);
    params['Tpeak'] = params['Tlatent'] + np.random.uniform(1,3)
    params['Ypeak'] = np.random.uniform(6,8.5)
    while params['Ypeak'] <= params['Yundetect']:
        params['Ypeak'] = np.random.uniform(6,8.5)
    params['Tclear'] = params['Tpeak'] + np.random.uniform(2,5)
    return params

def FLU_params_det():
    '''
    Defines a dictionary of stochastic parameter values for FLU viral load and infectiousness.
    '''
    params = {}
    params['LOD_high'] = 2.95;
    params['LOD_low'] = 5.38;
    params['Ydetect'] = params['LOD_high']; # gold standard detection
    params['Yundetect'] = params['Ydetect'];
    params['Yinf'] = 4;
    params['Tlatent'] = 1
    params['Tpeak'] = params['Tlatent'] + 2
    params['Ypeak'] = 7.25
    params['Tclear'] = params['Tpeak'] + 3.5
    return params

# Testing parameters ---------------------------------------------------------
'''
Defines a dictionary of testing parameters
'''
def COVID_test_params():
    params = {}
    params['sensitivity_threshold'] = 3;
    params['frequency'] = 1;
    params['delay'] = 0.01;
    params['failure_rate'] = 0.05;
    params['compliance'] = 1;
    return params

def RSV_test_params():
    params = {}
    params['sensitivity_threshold'] = 2.8;
    params['frequency'] = 1;
    params['delay'] = 0.01;
    params['failure_rate'] = 0.05;
    params['compliance'] = 1;
    return params

def FLU_test_params():
    params = {}
    params['sensitivity_threshold'] = 2.95;
    params['frequency'] = 1;
    params['delay'] = 0.01;
    params['failure_rate'] = 0.05;
    params['compliance'] = 1;
    return params

# Symptom parameters ---------------------------------------------------------
def COVID_get_pct_symptomatic():
    return 0.65
def WT_get_pct_symptomatic():
    return 0.65
def RSV_get_pct_symptomatic():
    return 0.57
def FLU_get_pct_symptomatic():
    return 0.64

def COVID_get_stoch_symp_time():
    '''
    Returns the range of times that symptoms may onset
    in units of distance from tpeak
    '''
    return -5,-1

def WT_get_stoch_symp_time():
    return 0,3

def FLU_get_stoch_symp_time():
    return -2,0

def RSV_get_stoch_symp_time():
    return -1,1

## Stochastic parameter calls   -----------------------------------------------------

def get_stoch_inf_params(p):
    func_string = p + "_params_stoch"
    return globals()[func_string]()
   
def get_LOD_high(p):
    func_string = p + "_params_stoch"
    return globals()[func_string]()['LOD_high']

def get_LOD_low(p):
    func_string = p + "_params_stoch"
    return globals()[func_string]()['LOD_low']

def get_pct_symptomatic(p):
    func_string = p + "_get_pct_symptomatic"
    return globals()[func_string]()
    
def get_stoch_symp_time(p,peak):
    func_string = p + "_get_stoch_symp_time"
    return globals()[func_string]()
    
def get_test_count_params(p):
    # specificity: how many uninifected people in population think they should be testing
    if p == "COVID":
        q = 0.005       # population prevalence = 0.5%
        symp_sp = 0.05       # 5% of uninfected population is testing because they feel symptomatic
        exp_se = 0.75        # 75% of infected people are notified
        exp_sp = .5         # 50% of uninfected population gets an exposure notification but does not get infected
    elif p == "FLU":
        q = 0.005       # population prevalence = .05%
        symp_sp = 0.05
        exp_se = 0.75
        exp_sp = .5
    elif p == "RSV":
        q = 0.005       # population prevalence = .05%
        symp_sp = 0.05
        exp_se = 0.75
        exp_sp = .5
    else:
        raise Exception("Please enter a valid pathogen: COVID, FLU, or RSV")
    return q,symp_sp,exp_se,exp_sp


