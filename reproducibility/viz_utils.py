import pandas as pd
import numpy as np

def tent(x,A,P,B,m,M,thr):
    if isinstance(x,np.ndarray):
        return np.array([tent(xx,A,P,B,m,M,thr) for xx in x])
    else:
        if x < A:
            y=0
        elif x < P:
            y=(x-A)/(P-A)*(M-m)+m
        elif x < B:
            y=(B-x)/(B-P)*(M-m)+m
        elif x > B:
            y=0
        y_above_thr = np.max([0,y-thr])
        return y_above_thr

def tents(x,AA,PP,BB,mm,MM,thrthr):
    N = len(AA)
    Y = 0*x
    for idx in range(N):
        y = tent(x,AA[idx],PP[idx],BB[idx],mm[idx],MM[idx],thrthr[idx])
        Y += y
    return Y/N

def tentDx(x,A,P,B,m,M,thr,tDx):
    if isinstance(x,np.ndarray):
        return np.array([tentDx(xx,A,P,B,m,M,thr,tDx) for xx in x])
    else:
        if tDx==-1:
            y = tent(x,A,P,B,m,M,thr)
        else:
            if x<tDx:
                y = tent(x,A,P,B,m,M,thr)
            else:
                y = 0
        return y

def tentsDx(x,AA,PP,BB,mm,MM,thrthr,tDxtDx):
    N = len(AA)
    Y = 0*x
    for idx in range(N):
        y = tentDx(x,AA[idx],PP[idx],BB[idx],mm[idx],MM[idx],thrthr[idx],tDxtDx[idx])
        Y += y
    return Y/N

def smooth(x,z):
    kernel = np.ones(z)/z
    return np.convolve(x,kernel,mode='same')

def compute_curves(file_name,particip,t_max=25,max_samples=1000):
    data = pd.read_csv('../output/{}'.format(file_name))
    x = np.arange(0,t_max,0.05)
    A = data['A'].values
    P = data['P'].values
    B = data['B'].values
    m = data['m'].values
    M = data['M'].values
    if len(A)>max_samples:
        A = A[:max_samples]
        P = P[:max_samples]
        B = B[:max_samples]
        m = m[:max_samples]
        M = M[:max_samples]
    TE = 1-np.sum(data['Itest'].values+data['Iexit'].values)/np.sum(data['I0'].values)
    TE *= particip
    thr = data['infectious_threshold'].values
    tDx = data['tDx'].values
    asc = np.sum(tDx<1e6)*particip/len(tDx)
    beta_no_testing = tents(x,A,P,B,m,M,thr)
    beta_testing = tentsDx(x,A,P,B,m,M,thr,tDx)
    CCDF_FtDx = []
    for t in x:
        CCDF_FtDx.append(np.sum(tDx>t)/len(tDx))
    beta_testing_particip = particip*beta_testing + (1-particip)*beta_no_testing
    CCDF_FtDx_adjusted = 1-particip*(1-np.array(CCDF_FtDx))
    return beta_no_testing,beta_testing,CCDF_FtDx_adjusted,x,TE,asc