'''
- This file contains ONLY functions associated with a SINGLE DRAW
  from the monte carlo simulation for testing.
- Its primary function is get_sample(...) and all other functions
  herein support that primary function.

get_sample does the following:

1. Draw a viral load
    kinetics options:
        kinetics_flu
        kinetics_sarscov2_founder_naive
        kinetics_sarscov2_omicron_experienced
        kinetics_rsv
2. Compute detectability D, using LOD
    get_D
3. Draw test administration times, using administration A.
    get_scheduled_tests options:
        test_regular
        test_post_symptoms
        test_post_exposure
4. Decide which tests are taken, using (1-c) coin for compliance.
    apply_compliance
5. Among them, decide which fail using f coin for failure.
    apply_failure
6. Decide which tests are positive, if any, by comparing to detectability D.
    get_hits
7. Get tDx by shifting first positive test forward by turnaround tat.
    get_tDx
8. Count tests consumed prior to tDx.
    count_tests
9. Compute infectiousness without testing
    area_triangle
10. Compute infectiousness with testing
    area_diagnosis
11. Compute exit from isolation, including tExit, n_tests_exit, Iexit
    compute_exit options:
        TTE
        fixed

get_sample(...) takes in:
    Administration:
        wait,supply,Q,T,testing
    Diagnostic:
        tat,L
    Compliance and Failure
        c,f,
    Virus
        thr,kinetics,

get_sample(...) returns:
    diagnosis, consumption, infectiousness:
        tDx,tExit,n_tests,n_tests_exit,I0,Itest,Iexit, 
    viral load and detectability params:
        A,P,B,tSx,m,M,first,last,thr
'''

import numpy as np

def get_sample(wait,supply,Q,tat,c,f,T,L,
    wait_exit,Q_exit,f_exit,L_exit,
    kinetics,testing,isolation):
    A,P,B,tSx,m,M,thr = kinetics()
    first,last = get_D(A,P,B,m,M,L)
    tests_scheduled = get_scheduled_tests(tSx,wait,supply,Q,T,testing)
    tests_taken = apply_compliance(tests_scheduled,c)
    valid_tests_taken = apply_failure(tests_taken,f)
    hits = get_hits(valid_tests_taken,first,last)
    tDx = get_tDx(hits,tat)
    n_tests = count_tests(tests_taken,tDx)
    I0 = area_triangle(A,P,B,m,M,thr)
    if tDx==np.inf:
        Itest = I0
        tExit = -1
        n_tests_exit = 0
        Iexit = 0
    else:
        Itest = area_diagnosis(A,P,B,m,M,thr,tDx)
        tExit,n_tests_exit,Iexit_complement = compute_exit(isolation,
            tDx,wait_exit,Q_exit,f_exit,L_exit,T,
            A,P,B,m,M,thr)
        Iexit = I0-Iexit_complement

    return tDx,tExit,n_tests,n_tests_exit,I0,Itest,Iexit, A,P,B,tSx,m,M,first,last,thr

def get_D(A,P,B,m,M,L):
    '''
    OUTPUT: 
        computes the window of detectability D, expresse as the interval [a,b]
    INPUTS: 
        Assuming a "tent function" trajectory, 
        A P B - the start, peak, and clearance times
        m M L - the start/clearance level, the peak level, and the detection limit
    '''
    # points (A,m)-(P,M)-(B,m)
    if L > M:
        a = 0
        b = 0
    elif L < m:
        a = A
        b = B
    else:
        f = (L-m)/(M-m)
        a = f*(P-A) + A
        b = f*(P-B) + B
    return a,b

def get_scheduled_tests(tSx,wait,supply,Q,T,method):
    '''
    Switch function to handle regular, exposure, and symptom testing.
    '''
    if method.__name__ == 'test_regular':
        return method(Q,T)
    elif method.__name__ == 'test_post_exposure':
        return method(wait,supply,Q)
    elif method.__name__ == 'test_post_symptoms':
        return method(tSx,wait,supply,Q)

def test_post_symptoms(tSx,wait,supply,Q):
    phase = Q*np.random.rand()
    tests_scheduled = tSx+wait+phase+np.linspace(0,Q*(supply-1),supply)
    return tests_scheduled

def test_post_exposure(wait,supply,Q):
    phase = Q*np.random.rand()
    tests_scheduled = wait+phase+np.linspace(0,Q*(supply-1),supply)
    return tests_scheduled

def test_regular(Q,T):
    phase = Q*np.random.rand()
    tests_scheduled = np.arange(phase,T,Q)
    return tests_scheduled

def apply_compliance(tests_scheduled,c):
    '''
    Among a set of scheduled tests, chucks out those that aren't taken due to compliance.
    '''
    n_tests = np.random.binomial(len(tests_scheduled),c)
    tests_taken = np.random.choice(tests_scheduled,n_tests,replace=False)
    return np.sort(tests_taken)

def apply_failure(tests_taken,f):
    '''
    Among a set of taken tests, chucks out those that fail. 
    '''
    n_tests = np.random.binomial(len(tests_taken),1-f)
    valid_tests_taken = np.random.choice(tests_taken,n_tests,replace=False)
    return np.sort(valid_tests_taken)

def get_hits(valid_tests_taken,first,last):
    return valid_tests_taken[(valid_tests_taken>=first) & (valid_tests_taken<=last)]

def get_tDx(hits,tat):
	if len(hits)==0:
		tDx = np.inf
	else:
		tDx = hits[0]+tat
	return tDx

def count_tests(tests_taken,tDx):
	if tDx==-1:
		n_tests = len(tests_taken)
	else:
		n_tests = np.sum(tests_taken<=tDx)
	return n_tests

def area_triangle(A,P,B,m,M,thr):
    '''
    Calculates the area of a triangle for infectiousness
    '''
    if M<thr:
        area = 0
    else:
        a,b = get_D(A,P,B,m,M,thr)
        height = M-thr
        base = b-a
        area = base*height/2
    return area

def area_diagnosis(A,P,B,m,M,thr,tDx):
    '''
    Calculates the area of a clipped triangle for infectiousness
    '''
    if M<thr:
        area = 0
    else:
        a,b = get_D(A,P,B,m,M,thr)
        if tDx < a:
            area = 0
        elif tDx <= P:
            base = tDx-a
            height = (tDx-a)*(M-thr)/(P-a)
            area = base*height/2
        elif tDx < b:
            base = b-tDx
            height = (b-tDx)*(M-thr)/(b-P)
            area = ((b-a)*(M-thr) - base*height)/2
        else:
            area =(b-a)*(M-thr)/2
    return area

def get_scheduled_exit_tests(tDx,wait_exit,Q_exit,T):
    phase = Q_exit*np.random.rand()
    t_start_TTE = tDx+wait_exit+phase
    exit_tests_scheduled = np.arange(t_start_TTE,T,Q_exit)
    if len(exit_tests_scheduled)==0:
        exit_tests_scheduled = np.array([tDx+wait_exit])
    return exit_tests_scheduled

def get_tExit(exit_tests_failed,exit_tests_scheduled,last):
    if len(exit_tests_failed)>0:
        # Exit by the first false negative
        tExit = exit_tests_failed[0]
    else:
        # Exit by the first true negative
        tExit = exit_tests_scheduled[exit_tests_scheduled>last][0]
    return tExit

def compute_exit(isolation,
                 tDx,wait_exit,Q_exit,f_exit,L_exit,T,
                 A,P,B,m,M,thr):
    if isolation=='fixed':
        tExit = tDx + wait_exit
        n_tests_exit = 0
        Iexit_complement = area_diagnosis(A,P,B,m,M,thr,tExit)
    elif isolation=='TTE':
        exit_tests_scheduled = get_scheduled_exit_tests(tDx,wait_exit,Q_exit,T)
        exit_tests_taken = exit_tests_scheduled
        first,last = get_D(A,P,B,m,M,L_exit)
        exit_tests_positive = get_hits(exit_tests_taken,first,last)
        positive_tests_failed = apply_failure(exit_tests_positive,1-f_exit)
        tExit = get_tExit(positive_tests_failed,exit_tests_scheduled,last)
        n_tests_exit = count_tests(exit_tests_taken,tExit)
        Iexit_complement = area_diagnosis(A,P,B,m,M,thr,tExit)
    return tExit,n_tests_exit,Iexit_complement









