import importlib
from os import path
import pandas as pd
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from matplotlib.gridspec import GridSpec
# from math import ceil, floor
import numpy as np
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects # to add border to text
import random

# set smaller precision
from decimal import *
getcontext().prec = 20

# colors
from palettable.cmocean.sequential import Tempo_8 as COVID_col
COVID_colors = COVID_col.hex_colors
COVID_cmap = COVID_col.mpl_colormap
from palettable.cartocolors.sequential import Burg_7 as FLU_colors
FLU_cmap = FLU_colors.mpl_colormap
FLU_colors = FLU_colors.hex_colors
from palettable.colorbrewer.sequential import YlOrBr_7 as RSV_colors
RSV_cmap = RSV_colors.mpl_colormap
RSV_colors = RSV_colors.hex_colors
from palettable.cmocean.sequential import Gray_8 as grays
gray_cmap = grays.mpl_colormap
grays = grays.hex_colors
my_purp = "#0A014F"
light_pink = "#FF85C0"
med_pink = "#FF1F8B"
dark_pink = "#CC0063"
black = "#042A2B"
orange = "#EF682E"#"#F9A620"
mint = "#77BA99"
green = "#09AE4B"#"#4EBC5E"
gray = "#BEBEBE"
dark_gray = "#655967"
light_gray = "#949494"

from prettyplotlib import *
from stoch_params import *
from viz_utils import *

data_path = Path(__file__).parent.parent / 'output' # where to retrieve data 
sim_path = Path(__file__).parent / 'fig_output' # where to save figure data
fig_path = Path(__file__).parent / 'figs' # where to save figures


# ------------------------------------------------------------------------------------------------------------
# ---------------------------------- Figure Drivers ----------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
def main():
    # Fig2_generate_data()
    # Draw_Fig2()                 #- Is testing useful? Scenarios
    # Draw_Fig2_supp_Asc()
    # Fig2_supp_curves_generate_data()
    # Draw_Fig2_supp_curves()     #- Ensemble curves for all Fig 1 testing scenarios
    # Fig4_generate_data()
    # Draw_Fig4_symp()            #- Reflex testing heatmaps - post-symptom
    # Draw_Fig4_BandW()           #- Reflex testing Black and White supplementary heatmaps - post-symptom
    # Fig4Supp_generate_data_Asc()
    # Draw_Fig4Supp_Asc()         # Reflex testing heatmaps - ascertainment
    # Draw_Fig4_exposure()        #- Reflex testing heatmaps - post-exposure
    # Fig5_generate_data()
    # Draw_Fig5()                 #- WT vs Omicron COVID
    # Fig6_generate_data()
    Draw_Fig6()                 #- Cost-benefit of isolation policies
    # VL_samples()
    # VL_maintext()               #- Fig 3
    # sensitivity_Fig2_generate_data()
    # sensitivity_Draw_Fig2()
    # sensitivity_Fig4_generate_data()
    # sensitivity_Draw_Fig4()

# ------------------------------------------------------------------------------------------------------------
# ---------------------------------- Figure Functions ----------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

''' Create time series data '''
start = 0
stop = 40
step = 0.001
numels = int((stop - start)/step + 1)
t_vals = np.linspace(start,stop,numels)
slim_t_vals = np.linspace(start,stop,(stop-start+1))
dt = t_vals[1]-t_vals[0]

# keys to read in data from MCMC files
def get_kinetics_string(pathogen):
    if pathogen == "COVID":
        return "kinetics_sarscov2_omicron_experienced"
    elif pathogen == "RSV":
        return "kinetics_rsv"
    elif pathogen == "FLU":
        return "kinetics_flu"
    elif pathogen == "WT":
        return "kinetics_sarscov2_founder_naive"

def read_data(filename):
    # if pickle file exists, return data
    if path.exists(filename / '.pkl'):
        return pickle.load(open(filename / '.pkl','rb'))
    else:
        raise Exception("File {} does not exist".format(filename))
def save_data(data,filename):
    pickle.dump(data,open(filename / '.pkl','wb'))
def save_txt(data,filename):
    file = open(filename / ".txt","w+")
    file.writelines(str(data))
    file.close()
def create_filestring_exp1(path,comp,freq,TAT,LOD,failure_rate):
    '''
    Exp 1 - Regular screening TE
    '''
    return sim_path / 'exp_1_{}_c{}_freq{}_TAT{}_LOD{}_f{}'.format(path,float(comp),float(freq),float(TAT),float(LOD),float(failure_rate))
def create_filestring_exp23(exp_num,pathogen,num_test,tests_per_day,wait_time,LOD,f=0,TAT=0,B=1,c=1):
    '''
    Exp 2 - symptom driven testing
    Exp 3 - exposure driven testing
    '''
    return sim_path / 'exp_{}_{}_numtst{}_tpd{}_wait{}_f{}_LOD{}_TAT{}_B{}_c{}'.format(exp_num,pathogen,\
        float(num_test),float(tests_per_day),float(wait_time),float(f),float(LOD),float(round(TAT,1)),\
        float(B),float(c))

def get_expt1_params(path):
    # params for Figure 1 panel A - 2 rapid tests at symptom onset
    exp = 2; n = 2; w = 0; tpd = 1; LOD = get_LOD_low(path); TAT = 0; p = get_pct_symptomatic(path)
    return (exp, n, w, tpd, LOD, TAT, p)
def get_expt2_params(path):
    # params for Figure 1 panel B - Post-exposure PCR within w=2 + 5 days
    exp = 3; n = 1; w = 2; tpd = 1/5; LOD = get_LOD_high(path); TAT = 2;  p = 0.75
    return (exp, n, w, tpd, LOD, TAT, p)
def get_expt3_params(path):
    # params for Figure 1 panel C - Weekly RDT screening with 50% compliance
    freq = 7; LOD = get_LOD_low(path); TAT = 0; p = 1; c = 0.5;
    return (freq, LOD, TAT, p, c)
def get_fig2_tvals(path):
    dt = 0.05
    return np.arange(0,40+dt,dt)
def Fig2_generate_data():
    Fig2a = pd.read_csv(sim_path / "figure2A.csv")
    Fig2b = pd.read_csv(sim_path / "figure2B.csv")
    Fig2c = pd.read_csv(sim_path / "figure2C.csv")
    Fig2example = pd.read_csv(sim_path / "figure2_example.csv")

    pathogens = ["RSV","FLU","COVID"]

    for path in pathogens:
        TEs = []; Ascs = []
        path_string = get_kinetics_string(path)

         # expt3 : weekly rapid testing
        freq, LOD, TAT, p, c = get_expt3_params(path)
        expt3 = Fig2c.loc[(Fig2c['kinetics']==path_string) & 
                          (Fig2c['L'] == LOD) & 
                          (Fig2c['tat'] == TAT) & 
                          (Fig2c['Q'] == freq)]
        if len(expt3) > 1:
            raise Exception("You didn't query me enough :(")
        TEs.append(p*expt3['TE'].values[0])
        Ascs.append(p*expt3['ascertainment'].values[0])

        # expt1 : 2 rapid tests at symptom onset
        exp, n, w, tpd, LOD, TAT, p = get_expt1_params(path)
        expt1 = Fig2a.loc[(Fig2a['kinetics']==path_string) & 
                          (Fig2a['L'] == LOD) & 
                          (Fig2a['tat'] == TAT) & 
                          (Fig2a['wait'] == w)]
        if len(expt1) > 1:
                    raise Exception("You didn't query me enough :(")
        TEs.append(p*expt1['TE'].values[0])
        Ascs.append(p*expt1['ascertainment'].values[0])

        # expt1 : 1 PCR 2-7 days after exposure
        exp, n, w, tpd, LOD, TAT, p = get_expt2_params(path)
        expt2 = Fig2b.loc[(Fig2b['kinetics']==path_string) & 
                          (Fig2b['L'] == LOD) & 
                          (Fig2b['tat'] == TAT) & 
                          (Fig2b['wait'] == w)]
        if len(expt2) > 1:
            raise Exception("You didn't query me enough :(")
        TEs.append(p*expt2['TE'].values[0])
        Ascs.append(p*expt2['ascertainment'].values[0])


        # Create infectiousness curves ----------------------------------------------------------
        # Note this code is slow! It will take ~30 minutes to run, so keep commented out unless you need to regenerate curves
        # Plot RSV infectiousness curve - experiment 3
        # if path == "RSV":
        #     hashid = expt3['simulation'].values[0]
        #     fname = sim_path / hashid / ".zip"
        #     beta_0,beta_testing,CCDF_tDx,x,TE,asc = compute_curves(fname,p,t_max=max(get_fig2_tvals(path)))
        #     fname = data_path / path / "_fig2_exp3_curves"
        #     save_data([beta_0,beta_testing,CCDF_tDx],fname)

        #     # example panel - RSV
        #     freq = 1; LOD = get_LOD_high(path); TAT = 0; p = 0.75; c = 0.75; w = 0; n = 2;
        #     exptn = Fig2example.loc[(Fig2example['kinetics']==path_string) & 
        #                     (Fig2example['L'] == LOD) & 
        #                     (Fig2example['tat'] == TAT) & 
        #                     (Fig2example['wait'] == w) &
        #                     (Fig2example['c'] == c)&
        #                     (Fig2example['supply'] == n)]
        #     if len(exptn) > 1:
        #                 raise Exception("You didn't query me enough :(")
        #     hashid = exptn['simulation'].values[0]
        #     sim_fname = sim_path / hashid / ".zip"
        #     beta_0,beta_testing,CCDF_tDx,x,TE,asc = compute_curves(sim_fname,p,t_max=max(get_fig2_tvals(path)))
        #     fname = data_path / path / "_fig2_expn_curves"
        #     save_data([beta_0,beta_testing,CCDF_tDx],fname)
        # # Plot Flu infectiousness curve - experiment 1
        # elif path == "FLU":
        #     hashid = expt1['simulation'].values[0]
        #     fname = sim_path / hashid / ".zip"
        #     beta_0,beta_testing,CCDF_tDx,x,TE,asc = compute_curves(fname,p,t_max=max(get_fig2_tvals(path)))
        #     fname = data_path / path / "_fig2_exp1_curves"
        #     save_data([beta_0,beta_testing,CCDF_tDx],fname)
        # # Plot COVID infectiousness curve  - experiment 2
        # elif path == "COVID":
        #     hashid = expt2['simulation'].values[0]
        #     fname = sim_path / hashid / ".zip"
        #     beta_0,beta_testing,CCDF_tDx,x,TE,asc = compute_curves(fname,p,t_max=max(get_fig2_tvals(path)))
        #     fname = data_path / path / "_fig2_exp2_curves"
        #     save_data([beta_0,beta_testing,CCDF_tDx],fname)

        fname = data_path / path / "_fig2"
        save_data(TEs,fname)
        fname = data_path / path / "_fig2_Asc"
        save_data(Ascs,fname)
def Draw_Fig2():
    COVID = read_data(data_path / "COVID_fig2")
    RSV = read_data(data_path / "RSV_fig2")
    FLU = read_data(data_path / "FLU_fig2")

    exptn_curve = read_data(data_path / "RSV_fig2_expn_curves")
    expt1_curve = read_data(data_path / "FLU_fig2_exp1_curves")
    expt2_curve = read_data(data_path / "COVID_fig2_exp2_curves")
    expt3_curve = read_data(data_path / "RSV_fig2_exp3_curves")

    labs = [ "Weekly RDT\nscreening\n50% comply",\
            "2 RDTs\npost-sympt.\n0d TAT", \
            "1 PCR 2-7d\npost-expos.\n2d TAT"]; xvals = np.array([1,3,5]);
    w=0.5; dy = 0.005; f_idx = int(20/.001)
    plt.rcParams['hatch.linewidth'] = 2  # hatch linewidth
    N = 1000 # number of VL draws used in fiji simulations - used for averaging here
    c = 1; f = 0.05; # constants for experiment sims

    figure_mosaic = """
    AABC
    AABC
    AADE
    AADE
    ....
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,5))
    
    a = axes["A"].bar(xvals-w,RSV,color=RSV_colors[4],width=w,edgecolor=ALMOST_BLACK)
    b = axes["A"].bar(xvals,FLU,color=FLU_colors[3],width=w,edgecolor=ALMOST_BLACK)
    c = axes["A"].bar(xvals+w,COVID,color=COVID_colors[4],width=w,edgecolor=ALMOST_BLACK)

    axes["A"].set_xticks(xvals); axes["A"].set_xticklabels(labs);
    axes["A"].set_ylim([0,0.5]); axes["A"].set_ylabel("Testing effectiveness (TE)",size=LABEL_SIZE-2)
    axes["A"].legend([a,b,c],["RSV","Influenza A","SARS-CoV-2"],loc='upper left',frameon = False)
    axes["A"].text(xvals[0]-w,RSV[0]+dy,"C",size=LABEL_SIZE,ha="center",fontweight="bold")
    axes["A"].text(xvals[1],FLU[1]+dy,"D",size=LABEL_SIZE,ha="center",fontweight="bold")
    axes["A"].text(xvals[2]+w,COVID[2]+dy,"E",size=LABEL_SIZE,ha="center",fontweight="bold")
    axes["A"].text(-1.15,0.5,"A",size=LABEL_SIZE,ha="center",fontweight="bold")
    finalize(axes["A"],aspect=1.75)

    # Diagram of how to read A-C plots
    inf = exptn_curve[0]; resid = exptn_curve[1]; CCDF = exptn_curve[2];
    path = "RSV"; t_vals = get_fig2_tvals(path)
    axes["B"].fill_between(t_vals,inf,color="white",edgecolor=dark_gray,hatch="///")
    axes["B"].fill_between(t_vals,resid,color=light_gray)
    axes["B"].text(7.3,max(inf)-0.2,"no testing",size=LABEL_SIZE-3,c=dark_gray)
    axes["B"].text(3.2,0.075,"testing",size=LABEL_SIZE-3,c="w")
    # add F(t) curve
    ax2 = axes["B"].twinx()
    ax2.plot(t_vals,CCDF,c=ALMOST_BLACK,linewidth=2)
    ax2.set_ylim([0,1.005]); finalize_keep_frame(ax2)
    ax2.set_yticklabels([None,None,None]);
    ax2.yaxis.labelpad = 10
    ax2.tick_params(axis='y', pad=3) # move labels closer to axis
    ax2.text(17.5,CCDF[-1]+0.05,"ascertainment",rotation=90,size=LABEL_SIZE-3)
    ax2.arrow(19.5,CCDF[-1]+0.03,0,1-(CCDF[-1]+0.02)-0.02,head_width=0.4,head_length=0.03,color=ALMOST_BLACK,length_includes_head=True)
    ax2.arrow(19.5,0.98,0,-1*(1-(CCDF[-1]+0.02)-0.02),head_width=0.4,head_length=0.03,color=ALMOST_BLACK,length_includes_head=True)
    ax2.text(0.25,0.88,"B",size=LABEL_SIZE,fontweight="bold")
    ax2.text(10.1,CCDF[-1]-0.1,r"$1-p\,E[F(t)]$",size=LABEL_SIZE-4)

    # expt3: weekly rapid testing - plot RSV
    path = "RSV"; exp_num = 3
    inf = expt3_curve[0]; resid = expt3_curve[1]; CCDF = expt3_curve[2];
    t_vals = get_fig2_tvals(path)
    axes["C"].fill_between(t_vals,inf,color="white",edgecolor=RSV_colors[5],hatch="///")
    axes["C"].fill_between(t_vals,resid,color=RSV_colors[4])
    # add F(t) curve
    ax2 = axes["C"].twinx()
    ax2.plot(t_vals,CCDF,c=ALMOST_BLACK,linewidth=2)
    finalize_keep_frame(ax2,aspect=1)
    ax2.set_yticklabels([]); ax2.set_yticklabels([None,.5,1])
    ax2.tick_params(axis='y', pad=3) # move labels closer to axis
    ax2.set_ylim([0,1.005]);
    ax2.text(0.25,0.88,"C",size=LABEL_SIZE,fontweight="bold")
    
    # expt1 : 2 rapid tests at symptom onset - plot Flu
    inf = expt1_curve[0]; resid = expt1_curve[1]; CCDF = expt1_curve[2];
    t_vals = get_fig2_tvals(path)
    axes["D"].fill_between(t_vals,inf,color="white",edgecolor=FLU_colors[4],hatch="///")
    axes["D"].fill_between(t_vals,resid,color=FLU_colors[3])
    # add F(t) curve
    ax2 = axes["D"].twinx()
    ax2.plot(t_vals,CCDF,c=ALMOST_BLACK,linewidth=2)
    finalize_keep_frame(ax2,aspect=1)
    ax2.set_yticklabels([]); ax2.set_yticklabels([])
    ax2.tick_params(axis='y', pad=3) # move labels closer to axis
    ax2.set_ylim([0,1.005]); 
    ax2.text(0.25,0.88,"D",size=LABEL_SIZE,fontweight="bold")

    # expt2: 1 PCR 3 days after exposure - plot COVID
    path = "COVID"; exp_num = 2
    inf = expt2_curve[0]; resid = expt2_curve[1]; CCDF = expt2_curve[2];
    t_vals = get_fig2_tvals(path)
    axes["E"].fill_between(t_vals,inf,color="white",edgecolor=COVID_colors[5],hatch="///")
    axes["E"].fill_between(t_vals,resid,color=COVID_colors[4])
    # add F(t) curve
    ax2 = axes["E"].twinx()
    ax2.plot(t_vals,CCDF,c=ALMOST_BLACK,linewidth=2)
    ax2.set_ylim([0,1.005]); finalize_keep_frame(ax2)
    ax2.set_yticklabels([0,.5,1])
    ax2.yaxis.labelpad = 10
    ax2.tick_params(axis='y', pad=3) # move labels closer to axis
    ax2.text(0.25,0.88,"E",size=LABEL_SIZE,fontweight="bold")

    experiment_axes = [axes["B"],axes["C"],axes["D"],axes["E"]]
    ylims = [2.9,3,2.6,0.55]; xlims = [20,20,15,20]
    for ii,ax in enumerate(experiment_axes):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_yticklabels([])
        ax.set_xlim([0,xlims[ii]]); ax.set_ylim(0,ylims[ii]);
        finalize(ax,aspect=0.5); 
    axes["D"].set_xticks([0,5,10,15]); axes["E"].set_xticks([0,5,10,15,20])
    axes["B"].set_ylabel("Infectiousness"); axes["D"].set_ylabel("Infectiousness")
    plt.text(0.7,-.35,"Days since exposure",size=LABEL_SIZE,ha="center",va="center")
    plt.text(25.75,1,"Proportion of infections not yet detected",size=LABEL_SIZE,rotation=90,ha="center",va="center")

    fname = fig_path / "Fig2.png"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    fname = fig_path /  "Fig2.pdf"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    # plt.show()

def Draw_Fig2_supp_Asc():
    COVID = read_data(data_path / "COVID_fig2_Asc")
    RSV = read_data(data_path / "RSV_fig2_Asc")
    FLU = read_data(data_path / "FLU_fig2_Asc")

    exptn_curve = read_data(data_path / "RSV_fig2_expn_curves")
    expt1_curve = read_data(data_path / "FLU_fig2_exp1_curves")
    expt2_curve = read_data(data_path / "COVID_fig2_exp2_curves")
    expt3_curve = read_data(data_path / "RSV_fig2_exp3_curves")

    labs = ["Weekly RDT\nscreening\n50% comply",\
            "2 RDTs\npost-sympt.\n0d TAT", \
            "1 PCR 2-7d\npost-expos.\n2d TAT"]; xvals = np.array([1,3,5]);
    w=0.5; dy = 0.005; f_idx = int(20/.001)
    plt.rcParams['hatch.linewidth'] = 2  # hatch linewidth
    N = 1000 # number of VL draws used in fiji simulations - used for averaging here
    c = 1; f = 0.05; # constants for experiment sims

    figure_mosaic = """
    AABC
    AABC
    AADE
    AADE
    ....
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,5))
    
    a = axes["A"].bar(xvals-w,RSV,color=RSV_colors[4],width=w,edgecolor=ALMOST_BLACK)
    b = axes["A"].bar(xvals,FLU,color=FLU_colors[3],width=w,edgecolor=ALMOST_BLACK)
    c = axes["A"].bar(xvals+w,COVID,color=COVID_colors[4],width=w,edgecolor=ALMOST_BLACK)

    axes["A"].set_xticks(xvals); axes["A"].set_xticklabels(labs);
    axes["A"].set_ylim([0,0.7]); axes["A"].set_yticks([0,.1,.2,.3,.4,.5,.6,.7]); axes["A"].set_ylabel("Ascertainment",size=LABEL_SIZE-2)
    axes["A"].legend([a,b,c],["RSV","Influenza A","SARS-CoV-2"],loc='upper left',frameon = False)
    axes["A"].text(xvals[0]-w,RSV[0]+dy,"C",size=LABEL_SIZE,ha="center",fontweight="bold")
    axes["A"].text(xvals[1],FLU[1]+dy,"D",size=LABEL_SIZE,ha="center",fontweight="bold")
    axes["A"].text(xvals[2]+w,COVID[2]+dy,"E",size=LABEL_SIZE,ha="center",fontweight="bold")
    axes["A"].text(-1.15,0.7,"A",size=LABEL_SIZE,ha="center",fontweight="bold")
    finalize(axes["A"],aspect=1.75)

    # Diagram of how to read A-C plots
    inf = exptn_curve[0]; resid = exptn_curve[1]; CCDF = exptn_curve[2];
    path = "RSV"; t_vals = get_fig2_tvals(path)
    axes["B"].fill_between(t_vals,inf,color="white",edgecolor=dark_gray,hatch="///")
    axes["B"].fill_between(t_vals,resid,color=light_gray)
    axes["B"].text(7.3,max(inf)-0.2,"no testing",size=LABEL_SIZE-3,c=dark_gray)
    axes["B"].text(3.2,0.075,"testing",size=LABEL_SIZE-3,c="w")
    # add F(t) curve
    ax2 = axes["B"].twinx()
    ax2.plot(t_vals,CCDF,c=ALMOST_BLACK,linewidth=2)
    ax2.set_ylim([0,1.005]); finalize_keep_frame(ax2)
    ax2.set_yticklabels([None,None,None]);
    ax2.yaxis.labelpad = 10
    ax2.tick_params(axis='y', pad=3) # move labels closer to axis
    ax2.text(17.5,CCDF[-1]+0.05,"ascertainment",rotation=90,size=LABEL_SIZE-3)
    ax2.arrow(19.5,CCDF[-1]+0.03,0,1-(CCDF[-1]+0.02)-0.02,head_width=0.4,head_length=0.03,color=ALMOST_BLACK,length_includes_head=True)
    ax2.arrow(19.5,0.98,0,-1*(1-(CCDF[-1]+0.02)-0.02),head_width=0.4,head_length=0.03,color=ALMOST_BLACK,length_includes_head=True)
    ax2.text(0.25,0.88,"B",size=LABEL_SIZE,fontweight="bold")
    ax2.text(10.1,CCDF[-1]-0.1,r"$1-p\,E[F(t)]$",size=LABEL_SIZE-4)

    # expt3: weekly rapid testing - plot RSV
    path = "RSV"; exp_num = 3
    inf = expt3_curve[0]; resid = expt3_curve[1]; CCDF = expt3_curve[2];
    t_vals = get_fig2_tvals(path)
    axes["C"].fill_between(t_vals,inf,color="white",edgecolor=RSV_colors[5],hatch="///")
    axes["C"].fill_between(t_vals,resid,color=RSV_colors[4])
    # add F(t) curve
    ax2 = axes["C"].twinx()
    ax2.plot(t_vals,CCDF,c=ALMOST_BLACK,linewidth=2)
    finalize_keep_frame(ax2,aspect=1)
    ax2.set_yticklabels([]); ax2.set_yticklabels([None,.5,1])
    ax2.tick_params(axis='y', pad=3) # move labels closer to axis
    ax2.set_ylim([0,1.005]);
    ax2.text(0.25,0.88,"C",size=LABEL_SIZE,fontweight="bold")
    
    # expt1 : 2 rapid tests at symptom onset - plot Flu
    inf = expt1_curve[0]; resid = expt1_curve[1]; CCDF = expt1_curve[2];
    t_vals = get_fig2_tvals(path)
    axes["D"].fill_between(t_vals,inf,color="white",edgecolor=FLU_colors[4],hatch="///")
    axes["D"].fill_between(t_vals,resid,color=FLU_colors[3])
    # add F(t) curve
    ax2 = axes["D"].twinx()
    ax2.plot(t_vals,CCDF,c=ALMOST_BLACK,linewidth=2)
    finalize_keep_frame(ax2,aspect=1)
    ax2.set_yticklabels([]); ax2.set_yticklabels([])
    ax2.tick_params(axis='y', pad=3) # move labels closer to axis
    ax2.set_ylim([0,1.005]); 
    ax2.text(0.25,0.88,"D",size=LABEL_SIZE,fontweight="bold")

    # expt2: 1 PCR 3 days after exposure - plot COVID
    path = "COVID"; exp_num = 2
    inf = expt2_curve[0]; resid = expt2_curve[1]; CCDF = expt2_curve[2];
    t_vals = get_fig2_tvals(path)
    axes["E"].fill_between(t_vals,inf,color="white",edgecolor=COVID_colors[5],hatch="///")
    axes["E"].fill_between(t_vals,resid,color=COVID_colors[4])
    # add F(t) curve
    ax2 = axes["E"].twinx()
    ax2.plot(t_vals,CCDF,c=ALMOST_BLACK,linewidth=2)
    ax2.set_ylim([0,1.005]); finalize_keep_frame(ax2)
    ax2.set_yticklabels([0,.5,1])
    ax2.yaxis.labelpad = 10
    ax2.tick_params(axis='y', pad=3) # move labels closer to axis
    ax2.text(0.25,0.88,"E",size=LABEL_SIZE,fontweight="bold")

    experiment_axes = [axes["B"],axes["C"],axes["D"],axes["E"]]
    ylims = [2.9,3,2.6,0.55]; xlims = [20,20,15,20]
    for ii,ax in enumerate(experiment_axes):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_yticklabels([])
        ax.set_xlim([0,xlims[ii]]); ax.set_ylim(0,ylims[ii]);
        finalize(ax,aspect=0.5); 
    axes["D"].set_xticks([0,5,10,15]); axes["E"].set_xticks([0,5,10,15,20])
    axes["B"].set_ylabel("Infectiousness"); axes["D"].set_ylabel("Infectiousness")
    plt.text(0.7,-.35,"Days since exposure",size=LABEL_SIZE,ha="center",va="center")
    plt.text(25.75,1,"Proportion of infections not yet detected",size=LABEL_SIZE,rotation=90,ha="center",va="center")

    fname = fig_path / "Fig2Supp_Asc.png"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    fname = fig_path /  "Fig2Supp_Asc.pdf"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    plt.show()
def Fig2_supp_curves_generate_data():
    Fig2a = pd.read_csv(sim_path / "figure2A.csv")
    Fig2b = pd.read_csv(sim_path / "figure2B.csv")
    Fig2c = pd.read_csv(sim_path / "figure2C.csv")

    pathogens = ["FLU","RSV","COVID"]

    for pathogen in pathogens:
        # expt1 : 2 rapid tests at symptom onset ---------------------------------------------
        exp, n, w, tpd, LOD, TAT, p = get_expt1_params(pathogen)
        # Check to see if data exists
        fname = data_path / pathogen / "_fig2_exp1_curves"
        if path.exists(fname+'.pkl'):
            print(fname," already exists. Skipping simulation.")
        # Otherwise, compute curves
        else:
            print("Computing ensemble curves for ",fname)
            path_string = get_kinetics_string(pathogen)

            expt1 = Fig2a.loc[(Fig2a['kinetics']==path_string) & 
                                (Fig2a['L'] == LOD) & 
                                (Fig2a['tat'] == TAT) & 
                                (Fig2a['wait'] == w)]
            if len(expt1) > 1:
                raise Exception("You didn't query me enough :(")
            
            hashid = expt1['simulation'].values[0]
            sim_fname = sim_path / hashid / ".zip"
            beta_0,beta_testing,CCDF_tDx,x,TE,asc = compute_curves(sim_fname,p,t_max=max(get_fig2_tvals(pathogen)))
            save_data([beta_0,beta_testing,CCDF_tDx],fname)

        # expt2 : 1 PCR 2-7 days after exposure --------------------------------------------------
        exp, n, w, tpd, LOD, TAT, p = get_expt2_params(pathogen)
        # Check to see if data exists
        fname = data_path / pathogen / "_fig2_exp2_curves"
        if path.exists(fname+'.pkl'):
            print(fname," already exists. Skipping simulation.")
        # Otherwise, compute curves
        else:
            print("Computing ensemble curves for ",fname)
            path_string = get_kinetics_string(pathogen)

            expt2 = Fig2b.loc[(Fig2b['kinetics']==path_string) & 
                                (Fig2b['L'] == LOD) & 
                                (Fig2b['tat'] == TAT) & 
                                (Fig2b['wait'] == w)]
            if len(expt2) > 1:
                raise Exception("You didn't query me enough :(")
            
            hashid = expt2['simulation'].values[0]
            sim_fname = sim_path / hashid / ".zip"
            beta_0,beta_testing,CCDF_tDx,x,TE,asc = compute_curves(sim_fname,p,t_max=max(get_fig2_tvals(pathogen)))
            save_data([beta_0,beta_testing,CCDF_tDx],fname)

        # expt3 : weekly rapid testing ----------------------------------------------
        freq, LOD, TAT, p, c = get_expt3_params(pathogen)
        # Check to see if data exists
        fname = data_path / pathogen / "_fig2_exp3_curves"
        if path.exists(fname+'.pkl'):
            print(fname," already exists. Skipping simulation.")
        # Otherwise, compute curves
        else:
            print("Computing ensemble curves for ",fname)
            path_string = get_kinetics_string(pathogen)

            expt3 = Fig2c.loc[(Fig2c['kinetics']==path_string) & 
                                (Fig2c['L'] == LOD) & 
                                (Fig2c['tat'] == TAT) & 
                                (Fig2c['Q'] == freq)]
            if len(expt3) > 1:
                raise Exception("You didn't query me enough :(")
            
            hashid = expt3['simulation'].values[0]
            sim_fname = sim_path / hashid / ".zip"
            beta_0,beta_testing,CCDF_tDx,x,TE,asc = compute_curves(sim_fname,p,t_max=max(get_fig2_tvals(pathogen)))
            save_data([beta_0,beta_testing,CCDF_tDx],fname)
def Draw_Fig2_supp_curves():
    ''' 3x3 grid of ensemble curves shown in Fig 1 B-E'''

    plt.rcParams['hatch.linewidth'] = 2  # hatch linewidth
    N = 1000 # number of VL draws used in fiji simulations - used for averaging here
    c = 1; f = 0.05; # constants for experiment sims

    # Pathogens and experiments will be plotted as follows
    '''
    Expt1: FLU, RSV, COVID
    Expt2: FLU, RSV, COVID
    Expt3: FLU, RSV, COVID
    '''
    pathogens = ["FLU","RSV","COVID"]
    exp_nums = ["1","2","3"]
    color_vector = [FLU_colors,RSV_colors,COVID_colors]
    exp_to_subplot = [["A","B","C"],["D","E","F"],["G","H","I"]]

    figure_mosaic = """
    ABC
    DEF
    GHI
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(8,6))

    left_axes = ["A","D","G"]; right_axes = ["C","F","I"]; bottom_axes = ["G","H","I"]; top_axes = ["A","B","C"]
    path_labels = ["Influenza A","RSV","SARS-CoV-2"]

    for col,pathogen in enumerate(pathogens):
        for row,exp in enumerate(exp_nums):
            # Get axis nuber and color to plot with
            axnum = exp_to_subplot[row][col]
            ax = axes[axnum]
            colors = color_vector[col]
            # Read data
            curve = read_data(data_path / pathogen / "_fig2_exp" / exp / "_curves")
            inf = curve[0]; resid = curve[1]; F = curve[2]
            t_vals = get_fig2_tvals(pathogen)

            # plot curves
            ax.fill_between(t_vals,inf,color="white",edgecolor=colors[5],hatch="///")
            ax.fill_between(t_vals,resid,color=colors[4])
            ymax = max(inf)
            ax.set_xlim([0,15]); ax.set_ylim([0,ymax+ymax*0.05])
            ax.set_yticks([])
            # add F(t) curve
            ax2 = ax.twinx()
            ax2.plot(t_vals,F,c=ALMOST_BLACK,linewidth=2)
            #ax2.set_yticklabels([]); ax2.set_yticklabels([None,.5,1])
            ax2.tick_params(axis='y', pad=3) # move labels closer to axis
            ax2.set_ylim([0,1.005]); 
    
            # Axes labels and ticks along edges
            if axnum in left_axes:
                ax.set_ylabel("Infectiousness")

            if axnum in bottom_axes:
                ax.set_xlabel("Days since exposure")
            else:
                ax.set_xticklabels([])
            
            if axnum in right_axes:
                ax2.set_yticks([0,0.5,1])
                if axnum == "F":
                    ax2.set_ylabel("Proportion of infections not yet detected")
            else:
                ax2.set_yticklabels([])
            
            if axnum in top_axes:
                ax.set_title(path_labels[col])

            finalize(ax); finalize_keep_frame(ax2)

    fname = fig_path / "Fig2supp_curves_nolabs.png"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    fname = fig_path /  "Fig2supp_curves_nolabs.pdf"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    plt.show()

def Fig4_generate_data():
    ''' TE heatmap for symptom based and exposure based reflex testing '''
    Fig4a = pd.read_csv(sim_path / "figure4A.csv")
    Fig4b = pd.read_csv(sim_path / "figure4B.csv")
    Fig4c = pd.read_csv(sim_path / "figure4C.csv")

    Fig4asupp = pd.read_csv(sim_path / "figure4A_exposure.csv")
    Fig4bsupp = pd.read_csv(sim_path / "figure4B_exposure.csv")
    Fig4csupp = pd.read_csv(sim_path / "figure4C_exposure.csv")

    wait_times_exp = np.arange(0,11,1)
    wait_times_symp = np.arange(0,6,1)
    num_tests = np.arange(6,0,-1)

    pathogens = ["COVID","FLU","RSV"]
    for path in pathogens:
        if path == "RSV":
            Fig = Fig4a
        elif path == "FLU":
            Fig = Fig4b
        else:
            Fig = Fig4c
        path_string = get_kinetics_string(path)

        # Post-symptom testing
        p = get_pct_symptomatic(path)
        LOD = get_LOD_low(path); TAT = 0;
        df = pd.DataFrame()
        for w in wait_times_symp:
            scenario_results = []
            for n in num_tests:
                data = Fig.loc[(Fig['kinetics']==path_string) & 
                               (Fig['L']==LOD) & 
                               (Fig['tat']==TAT) & 
                               (Fig['wait']==w) & 
                               (Fig['supply']==n)]
                if len(data) > 1:
                    raise Exception("You didn't query me enough :(")
                scenario_results.append(p*data['TE'].values[0])
            df_string = "wait" + str(w)
            df[df_string] = scenario_results
        # Save data
        fname = data_path / path / "_fig4_symp"
        save_data(df,fname)

        # PCR post-symptom
        LOD = get_LOD_high(path); TAT = 2;
        scenario_results = []
        for w in wait_times_symp:
            data = Fig.loc[(Fig['kinetics']==path_string) & 
                           (Fig['L']==LOD) & 
                           (Fig['tat']==TAT) & 
                           (Fig['wait']==w) & 
                           (Fig['supply']==n)]
            if len(data) > 1:
                raise Exception("You didn't query me enough :(")
            scenario_results.append(p*data['TE'].values[0])
        df = np.expand_dims(np.array(scenario_results), axis=0)
        # Save data
        fname = data_path / path / "_fig4_symp_PCR"
        save_data(df,fname)

    for path in pathogens:
        # Post-exposure testing
        if path == "RSV":
            Fig = Fig4asupp
        elif path == "FLU":
            Fig = Fig4bsupp
        else:
            Fig = Fig4csupp
        path_string = get_kinetics_string(path)

        df = pd.DataFrame()
        p = 0.75
        LOD = get_LOD_low(path); TAT = 0;
        for w in wait_times_exp:
            scenario_results = []
            for n in num_tests:
                data = Fig.loc[(Fig['kinetics']==path_string) & 
                               (Fig['L']==LOD) & 
                               (Fig['tat']==TAT) & 
                               (Fig['wait']==w) & 
                               (Fig['supply']==n)]
                if len(data) > 1:
                    raise Exception("You didn't query me enough :(")
                scenario_results.append(p*data['TE'].values[0])
            df_string = "wait" + str(w)
            df[df_string] = scenario_results
        # Save data
        fname = data_path / path / "_fig4_exposure"
        save_data(df,fname)

        # PCR post-exposure
        LOD = get_LOD_high(path); TAT = 2;
        scenario_results = []
        for w in wait_times_exp:
            data = Fig.loc[(Fig['kinetics']==path_string) & 
                           (Fig['L']==LOD) & 
                           (Fig['tat']==TAT) & 
                           (Fig['wait']==w) & 
                           (Fig['supply']==n)]
            if len(data) > 1:
                raise Exception("You didn't query me enough :(")
            scenario_results.append(p*data['TE'].values[0])
        df = np.expand_dims(np.array(scenario_results), axis=0)
        # Save data
        fname = data_path / path / "_fig4_exposure_PCR"
        save_data(df,fname)
def Draw_Fig4_symp():
    ''' Prompted testing with different wait times and number of daily tests '''

    num_tests = np.arange(1,7)

    # for symptom-based
    COVID = read_data(data_path / "COVID_fig4_symp")
    FLU = read_data(data_path / "FLU_fig4_symp")
    RSV = read_data(data_path / "RSV_fig4_symp")

    wait_times = np.arange(0,6,1)
    centers = np.min(wait_times), np.max(wait_times), np.min(num_tests), np.max(num_tests)
    extent = create_extent(COVID,centers)
    PCR_centers = np.min(wait_times), np.max(wait_times), 0.5, 1.5
    PCR_extent = create_extent(COVID,PCR_centers)

    COVID_PCR = read_data(data_path / "COVID_fig4_symp_PCR")
    RSV_PCR = read_data(data_path / "RSV_fig4_symp_PCR")
    FLU_PCR = read_data(data_path / "FLU_fig4_symp_PCR")

    max_reduction_COVID = 0.5
    min_reduction_COVID = 0
    max_reduction_FLU = 0.5
    min_reduction_FLU = 0
    max_reduction_RSV = 0.5
    min_reduction_RSV = 0

    figure_mosaic = """
    ABC
    DEF
    DEF
    DEF
    DEF
    DEF
    GHI
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,5))

    ## Plotting data   -------------------------------------------------------------------------
    im_RSV = axes["D"].imshow(RSV,extent=extent,cmap=RSV_cmap,vmin=min_reduction_RSV,vmax=max_reduction_RSV)
    axes["G"].imshow(RSV_PCR,extent=PCR_extent,cmap=RSV_cmap,vmin=min_reduction_RSV,vmax=max_reduction_RSV)
    cbar_RSV = plt.colorbar(im_RSV,ax=axes["A"],fraction=0.7,orientation="horizontal")
    cbar_RSV.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
    axes["A"].text(0.5,0.7,'RSV testing effectiveness',ha='center',size=LABEL_SIZE-2)
    # plot stars over max TE
    for n in range(max(num_tests)):
        TEs = []
        for ii,row in enumerate(RSV):
            TEs.append(RSV[row][n])
        max_TE = TEs.index(max(TEs))
        axes["D"].scatter(max_TE,len(num_tests)-n,c="white",marker="*")
    max_TE = np.argmax(RSV_PCR)
    axes["G"].scatter(max_TE,1,c="white",marker="*")

    im_FLU = axes["E"].imshow(FLU,extent=extent,cmap=FLU_cmap,vmin=min_reduction_FLU,vmax=max_reduction_FLU)
    axes["H"].imshow(FLU_PCR,extent=PCR_extent,cmap=FLU_cmap,vmin=min_reduction_FLU,vmax=max_reduction_FLU)
    cbar_FLU = plt.colorbar(im_FLU,ax=axes["B"],fraction=0.7,orientation="horizontal")
    cbar_FLU.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
    axes["B"].text(0.5,0.7,'Influenza testing effectiveness',ha='center',size=LABEL_SIZE-2)
    # plot stars over max TE
    for n in range(max(num_tests)):
        TEs = []
        for ii,row in enumerate(FLU):
            TEs.append(FLU[row][n])
        max_TE = TEs.index(max(TEs))
        axes["E"].scatter(max_TE,len(num_tests)-n,c="white",marker="*")
    max_TE = np.argmax(FLU_PCR)
    axes["H"].scatter(max_TE,1,c="white",marker="*")

    im_COVID = axes["F"].imshow(COVID,extent=extent,cmap=COVID_cmap,vmin=min_reduction_COVID,vmax=max_reduction_COVID)
    axes["I"].imshow(COVID_PCR,extent=PCR_extent,cmap=COVID_cmap,vmin=min_reduction_COVID,vmax=max_reduction_COVID)
    cbar_COVID = plt.colorbar(im_COVID,ax=axes["C"],fraction=0.7,orientation="horizontal")
    cbar_COVID.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
    axes["C"].text(0.5,0.7,'SARS-CoV-2 testing effectiveness',ha='center',size=LABEL_SIZE-2)
    # plot stars over max TE
    for n in range(max(num_tests)):
        TEs = []
        for ii,row in enumerate(COVID):
            TEs.append(COVID[row][n])
        max_TE = TEs.index(max(TEs))
        axes["F"].scatter(max_TE,len(num_tests)-n,c="white",marker="*")
    max_TE = np.argmax(COVID_PCR)
    axes["I"].scatter(max_TE,1,c="white",marker="*")

    x_axes = [axes["D"],axes["E"],axes["F"],axes["G"],axes["H"],axes["I"]]
    for a in x_axes:
        a.set_xticks([0,1,2,3,4,5])
    
    axes["G"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)
    axes["H"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)
    axes["I"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)

    axes["D"].set_yticks([1,2,3,4,5,6])
    axes["D"].set_ylabel("Rapid tests available")
    axes["G"].set_ylabel("PCR")

    for a in [axes["E"],axes["F"],axes["H"],axes["I"]]:
        a.set_yticks([])

    for a in [axes["D"],axes["E"],axes["F"]]:
        finalize_keep_frame(a)
        force_aspect(a,1)
        a.set_xticklabels([])
        a.set_yticks([1,2,3,4,5,6])
    
    for a in [axes["A"],axes["B"],axes["C"]]:
        remove_ax(a)

    for a in [axes["G"],axes["H"],axes["I"]]:
        finalize_keep_frame(a,aspect=4.75)
        a.set_yticks([1])

    # axes["D"].text(5.25,6.25,"RDT tests\n0 day TAT",size=18,va="top",ha="right")
    # axes["G"].text(5.25,1.25,"PCR, 2 day",size=18,va="top",ha="right")

    xlabpad = -0.15; ylabpad = 2.4;
    label_subplots(axes,x_pads=[xlabpad,xlabpad,xlabpad],y_pad=ylabpad,labels=["A","B","C"],fontsize=LABEL_SIZE)
    plt.subplots_adjust(hspace=-0.4) # space between rows
    plt.tight_layout

    fname = fig_path / "Fig4.png"
    plt.savefig(fname,dpi=300,bbox_inches="tight") # bbox_inches prevents x-label from being cutoff
    fname = fig_path / "Fig4.pdf"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    # plt.show()
def Draw_Fig4_exposure():
    ''' Prompted testing with different wait times and number of daily tests '''

    num_tests = np.arange(1,7)

    # for symptom-based
    COVID = read_data(data_path / "COVID_fig4_exposure")
    FLU = read_data(data_path / "FLU_fig4_exposure")
    RSV = read_data(data_path / "RSV_fig4_exposure")

    wait_times = np.arange(0,11,1)
    centers = np.min(wait_times), np.max(wait_times), np.min(num_tests), np.max(num_tests)
    extent = create_extent(COVID,centers)
    PCR_centers = np.min(wait_times), np.max(wait_times), 0.5, 1.5
    PCR_extent = create_extent(COVID,PCR_centers)

    COVID_PCR = read_data(data_path / "COVID_fig4_exposure_PCR")
    RSV_PCR = read_data(data_path / "RSV_fig4_exposure_PCR")
    FLU_PCR = read_data(data_path / "FLU_fig4_exposure_PCR")

    max_reduction_COVID = 0.75
    min_reduction_COVID = 0
    max_reduction_FLU = 0.75
    min_reduction_FLU = 0
    max_reduction_RSV = 0.75
    min_reduction_RSV = 0

    figure_mosaic = """
    ABC
    DEF
    DEF
    DEF
    DEF
    DEF
    GHI
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,5))

    ## Plotting data   -------------------------------------------------------------------------
    im_RSV = axes["D"].imshow(RSV,extent=extent,cmap=RSV_cmap,vmin=min_reduction_RSV,vmax=max_reduction_RSV)
    axes["G"].imshow(RSV_PCR,extent=PCR_extent,cmap=RSV_cmap,vmin=min_reduction_RSV,vmax=max_reduction_RSV)
    cbar_RSV = plt.colorbar(im_RSV,ax=axes["A"],fraction=0.7,orientation="horizontal")
    cbar_RSV.set_ticks([0,0.25,0.5,0.75])
    axes["A"].text(0.5,0.7,'RSV testing effectiveness',ha='center',size=LABEL_SIZE-2)
    # plot stars over max TE
    for n in range(max(num_tests)):
        TEs = []
        for ii,row in enumerate(RSV):
            TEs.append(RSV[row][n])
        max_TE = TEs.index(max(TEs))
        axes["D"].scatter(max_TE,len(num_tests)-n,c="white",marker="*")
    max_TE = np.argmax(RSV_PCR)
    axes["G"].scatter(max_TE,1,c="white",marker="*")

    im_FLU = axes["E"].imshow(FLU,extent=extent,cmap=FLU_cmap,vmin=min_reduction_FLU,vmax=max_reduction_FLU)
    axes["H"].imshow(FLU_PCR,extent=PCR_extent,cmap=FLU_cmap,vmin=min_reduction_FLU,vmax=max_reduction_FLU)
    cbar_FLU = plt.colorbar(im_FLU,ax=axes["B"],fraction=0.7,orientation="horizontal")
    cbar_FLU.set_ticks([0,0.25,0.5,0.75])
    axes["B"].text(0.5,0.7,'Influenza testing effectiveness',ha='center',size=LABEL_SIZE-2)
    # plot stars over max TE
    for n in range(max(num_tests)):
        TEs = []
        for ii,row in enumerate(FLU):
            TEs.append(FLU[row][n])
        max_TE = TEs.index(max(TEs))
        axes["E"].scatter(max_TE,len(num_tests)-n,c="white",marker="*")
    max_TE = np.argmax(FLU_PCR)
    axes["H"].scatter(max_TE,1,c="white",marker="*")

    im_COVID = axes["F"].imshow(COVID,extent=extent,cmap=COVID_cmap,vmin=min_reduction_COVID,vmax=max_reduction_COVID)
    axes["I"].imshow(COVID_PCR,extent=PCR_extent,cmap=COVID_cmap,vmin=min_reduction_COVID,vmax=max_reduction_COVID)
    cbar_COVID = plt.colorbar(im_COVID,ax=axes["C"],fraction=0.7,orientation="horizontal")
    cbar_COVID.set_ticks([0,0.25,0.5,0.75])
    axes["C"].text(0.5,0.7,'SARS-CoV-2 testing effectiveness',ha='center',size=LABEL_SIZE-2)
    # plot stars over max TE
    for n in range(max(num_tests)):
        TEs = []
        for ii,row in enumerate(COVID):
            TEs.append(COVID[row][n])
        max_TE = TEs.index(max(TEs))
        axes["F"].scatter(max_TE,len(num_tests)-n,c="white",marker="*")
    max_TE = np.argmax(COVID_PCR)
    axes["I"].scatter(max_TE,1,c="white",marker="*")

    x_axes = [axes["D"],axes["E"],axes["F"],axes["G"],axes["H"],axes["I"]]
    for a in x_axes:
        a.set_xticks([0,2,4,6,8,10])
    
    axes["G"].set_xlabel("Delay before testing\n(days post-exposure)",size=LABEL_SIZE)
    axes["H"].set_xlabel("Delay before testing\n(days post-exposure)",size=LABEL_SIZE)
    axes["I"].set_xlabel("Delay before testing\n(days post-exposure)",size=LABEL_SIZE)

    axes["D"].set_yticks([1,2,3,4,5,6])
    axes["D"].set_ylabel("Rapid tests available")
    axes["G"].set_ylabel("PCR")

    for a in [axes["E"],axes["F"],axes["H"],axes["I"]]:
        a.set_yticks([])

    for a in [axes["D"],axes["E"],axes["F"]]:
        finalize_keep_frame(a)
        force_aspect(a,1)
        a.set_xticklabels([])
        a.set_yticks([1,2,3,4,5,6])
    
    for a in [axes["A"],axes["B"],axes["C"]]:
        remove_ax(a)

    for a in [axes["G"],axes["H"],axes["I"]]:
        finalize_keep_frame(a,aspect=4.75)
        a.set_yticks([1])

    # axes["D"].text(5.25,6.25,"RDT tests\n0 day TAT",size=18,va="top",ha="right")
    # axes["G"].text(5.25,1.25,"PCR, 2 day",size=18,va="top",ha="right")

    xlabpad = -0.15; ylabpad = 2.4;
    label_subplots(axes,x_pads=[xlabpad,xlabpad,xlabpad],y_pad=ylabpad,labels=["A","B","C"],fontsize=LABEL_SIZE)
    plt.subplots_adjust(hspace=-0.4) # space between rows
    plt.tight_layout

    fname = fig_path / "Fig4Supp_exposure.png"
    plt.savefig(fname,dpi=300,bbox_inches="tight") # bbox_inches prevents x-label from being cutoff
    fname = fig_path / "Fig4Supp_exposure.pdf"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    # plt.show()
def Draw_Fig4_BandW():
    ''' Prompted testing with different wait times and number of daily tests '''
    ''' Heatmaps in black and white with values displayed'''

    num_tests = np.arange(1,7)

    # for symptom-based
    COVID = read_data(data_path / "COVID_fig4_symp")
    FLU = read_data(data_path / "FLU_fig4_symp")
    RSV = read_data(data_path / "RSV_fig4_symp")

    wait_times = np.arange(0,6,1)
    centers = np.min(wait_times), np.max(wait_times), np.min(num_tests), np.max(num_tests)
    extent = create_extent(COVID,centers)
    PCR_centers = np.min(wait_times), np.max(wait_times), 0.5, 1.5
    PCR_extent = create_extent(COVID,PCR_centers)

    COVID_PCR = read_data(data_path / "COVID_fig4_symp_PCR")
    RSV_PCR = read_data(data_path / "RSV_fig4_symp_PCR")
    FLU_PCR = read_data(data_path / "FLU_fig4_symp_PCR")

    max_reduction_COVID = 0.5
    min_reduction_COVID = 0
    max_reduction_FLU = 0.5
    min_reduction_FLU = 0
    max_reduction_RSV = 0.5
    min_reduction_RSV = 0

    figure_mosaic = """
    ABC
    DEF
    DEF
    DEF
    DEF
    DEF
    GHI
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,5))

    ## Plotting data   -------------------------------------------------------------------------
    im_RSV = axes["D"].imshow(RSV,extent=extent,cmap="Greys",vmin=min_reduction_RSV,vmax=max_reduction_RSV)
    axes["G"].imshow(RSV_PCR,extent=PCR_extent,cmap="Greys",vmin=min_reduction_RSV,vmax=max_reduction_RSV)
    cbar_RSV = plt.colorbar(im_RSV,ax=axes["A"],fraction=0.7,orientation="horizontal")
    cbar_RSV.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
    axes["A"].text(0.5,0.7,'RSV testing effectiveness',ha='center',size=LABEL_SIZE-2)
    # Add TE values to each square
    for n in range(max(num_tests)):
        for ii,row in enumerate(RSV):
            txt=axes["D"].text(ii,6-n,str(round(RSV[row][n],2)),c="white",ha="center",va="center")
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
    for ii, val in enumerate(RSV_PCR[0]):
        txt=axes["G"].text(ii,1,str(round(val,2)),c="white",ha="center",va="center")
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

    im_FLU = axes["E"].imshow(FLU,extent=extent,cmap="Greys",vmin=min_reduction_FLU,vmax=max_reduction_FLU)
    axes["H"].imshow(FLU_PCR,extent=PCR_extent,cmap="Greys",vmin=min_reduction_FLU,vmax=max_reduction_FLU)
    cbar_FLU = plt.colorbar(im_FLU,ax=axes["B"],fraction=0.7,orientation="horizontal")
    cbar_FLU.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
    axes["B"].text(0.5,0.7,'Influenza testing effectiveness',ha='center',size=LABEL_SIZE-2)
    # Add TE values to each square
    for n in range(max(num_tests)):
        for ii,row in enumerate(FLU):
            txt=axes["E"].text(ii,6-n,str(round(FLU[row][n],2)),c="w",ha="center",va="center")
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
    for ii, val in enumerate(FLU_PCR[0]):
        txt=axes["H"].text(ii,1,str(round(val,2)),c="w",ha="center",va="center")
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

    im_COVID = axes["F"].imshow(COVID,extent=extent,cmap="Greys",vmin=min_reduction_COVID,vmax=max_reduction_COVID)
    axes["I"].imshow(COVID_PCR,extent=PCR_extent,cmap="Greys",vmin=min_reduction_COVID,vmax=max_reduction_COVID)
    cbar_COVID = plt.colorbar(im_COVID,ax=axes["C"],fraction=0.7,orientation="horizontal")
    cbar_COVID.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
    axes["C"].text(0.5,0.7,'SARS-CoV-2 testing effectiveness',ha='center',size=LABEL_SIZE-2)
    # Add TE values to each square
    for n in range(max(num_tests)):
        for ii,row in enumerate(COVID):
            txt=axes["F"].text(ii,6-n,str(round(COVID[row][n],2)),c="w",ha="center",va="center")
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
    for ii, val in enumerate(COVID_PCR[0]):
        txt=axes["I"].text(ii,1,str(round(val,2)),c="w",ha="center",va="center")
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

    x_axes = [axes["D"],axes["E"],axes["F"],axes["G"],axes["H"],axes["I"]]
    for a in x_axes:
        a.set_xticks([0,1,2,3,4,5])
    
    axes["G"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)
    axes["H"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)
    axes["I"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)

    axes["D"].set_yticks([1,2,3,4,5,6])
    axes["D"].set_ylabel("Rapid tests available")
    axes["G"].set_ylabel("PCR")

    for a in [axes["E"],axes["F"],axes["H"],axes["I"]]:
        a.set_yticks([])

    for a in [axes["D"],axes["E"],axes["F"]]:
        finalize_keep_frame(a)
        force_aspect(a,1)
        a.set_xticklabels([])
        a.set_yticks([1,2,3,4,5,6])
    
    for a in [axes["A"],axes["B"],axes["C"]]:
        remove_ax(a)

    for a in [axes["G"],axes["H"],axes["I"]]:
        finalize_keep_frame(a,aspect=4.75)
        a.set_yticks([1])

    # axes["D"].text(5.25,6.25,"RDT tests\n0 day TAT",size=18,va="top",ha="right")
    # axes["G"].text(5.25,1.25,"PCR, 2 day",size=18,va="top",ha="right")

    xlabpad = -0.15; ylabpad = 2.4;
    label_subplots(axes,x_pads=[xlabpad,xlabpad,xlabpad],y_pad=ylabpad,labels=["A","B","C"],fontsize=LABEL_SIZE)
    plt.subplots_adjust(hspace=-0.4) # space between rows

    fname = fig_path / "Fig4Supp_BandW.png"
    plt.savefig(fname,dpi=300,bbox_inches="tight") # bbox_inches prevents x-label from being cutoff
    fname = fig_path / "Fig4Supp_BandW.pdf"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    # plt.show()

def Fig4Supp_generate_data_Asc():
    ''' TE heatmap for symptom based and exposure based reflex testing '''
    Fig4a = pd.read_csv(sim_path / "figure4A.csv")
    Fig4b = pd.read_csv(sim_path / "figure4B.csv")
    Fig4c = pd.read_csv(sim_path / "figure4C.csv")

    Fig4asupp = pd.read_csv(sim_path / "figure4A_exposure.csv")
    Fig4bsupp = pd.read_csv(sim_path / "figure4B_exposure.csv")
    Fig4csupp = pd.read_csv(sim_path / "figure4C_exposure.csv")

    wait_times_exp = np.arange(0,11,1)
    wait_times_symp = np.arange(0,6,1)
    num_tests = np.arange(6,0,-1)

    pathogens = ["COVID","FLU","RSV"]
    for path in pathogens:
        if path == "RSV":
            Fig = Fig4a
        elif path == "FLU":
            Fig = Fig4b
        else:
            Fig = Fig4c
        path_string = get_kinetics_string(path)

        # Post-symptom testing
        p = get_pct_symptomatic(path)
        LOD = get_LOD_low(path); TAT = 0;
        df = pd.DataFrame()
        for w in wait_times_symp:
            scenario_results = []
            for n in num_tests:
                data = Fig.loc[(Fig['kinetics']==path_string) & 
                               (Fig['L']==LOD) & 
                               (Fig['tat']==TAT) & 
                               (Fig['wait']==w) & 
                               (Fig['supply']==n)]
                if len(data) > 1:
                    raise Exception("You didn't query me enough :(")
                scenario_results.append(p*data['ascertainment'].values[0])
            df_string = "wait" + str(w)
            df[df_string] = scenario_results
        # Save data
        fname = data_path / path / "_fig4_symp_Asc"
        save_data(df,fname)

        # PCR post-symptom
        LOD = get_LOD_high(path); TAT = 2;
        scenario_results = []
        for w in wait_times_symp:
            data = Fig.loc[(Fig['kinetics']==path_string) & 
                           (Fig['L']==LOD) & 
                           (Fig['tat']==TAT) & 
                           (Fig['wait']==w) & 
                           (Fig['supply']==n)]
            if len(data) > 1:
                raise Exception("You didn't query me enough :(")
            scenario_results.append(p*data['ascertainment'].values[0])
        df = np.expand_dims(np.array(scenario_results), axis=0)
        # Save data
        fname = data_path / path / "_fig4_symp_PCR_Asc"
        save_data(df,fname)
def Draw_Fig4Supp_Asc():
    ''' Prompted testing with different wait times and number of daily tests '''
    ''' Heatmaps in black and white with values displayed'''

    num_tests = np.arange(1,7)

    # for symptom-based
    COVID = read_data(data_path / "COVID_fig4_symp_Asc")
    FLU = read_data(data_path / "FLU_fig4_symp_Asc")
    RSV = read_data(data_path / "RSV_fig4_symp_Asc")

    wait_times = np.arange(0,6,1)
    centers = np.min(wait_times), np.max(wait_times), np.min(num_tests), np.max(num_tests)
    extent = create_extent(COVID,centers)
    PCR_centers = np.min(wait_times), np.max(wait_times), 0.5, 1.5
    PCR_extent = create_extent(COVID,PCR_centers)

    COVID_PCR = read_data(data_path / "COVID_fig4_symp_PCR_Asc")
    RSV_PCR = read_data(data_path / "RSV_fig4_symp_PCR_Asc")
    FLU_PCR = read_data(data_path / "FLU_fig4_symp_PCR_Asc")

    max_reduction_COVID = 0.7
    min_reduction_COVID = 0
    max_reduction_FLU = 0.7
    min_reduction_FLU = 0
    max_reduction_RSV = 0.7
    min_reduction_RSV = 0

    figure_mosaic = """
    ABC
    DEF
    DEF
    DEF
    DEF
    DEF
    GHI
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,5))

    ## Plotting data   -------------------------------------------------------------------------
    im_RSV = axes["D"].imshow(RSV,extent=extent,cmap="Greys",vmin=min_reduction_RSV,vmax=max_reduction_RSV)
    axes["G"].imshow(RSV_PCR,extent=PCR_extent,cmap="Greys",vmin=min_reduction_RSV,vmax=max_reduction_RSV)
    cbar_RSV = plt.colorbar(im_RSV,ax=axes["A"],fraction=0.7,orientation="horizontal")
    cbar_RSV.set_ticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
    axes["A"].text(0.5,0.7,'RSV Ascertainment',ha='center',size=LABEL_SIZE-2)
    # Add TE values to each square
    for n in range(max(num_tests)):
        for ii,row in enumerate(RSV):
            txt=axes["D"].text(ii,6-n,str(round(RSV[row][n],2)),c="white",ha="center",va="center")
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
    for ii, val in enumerate(RSV_PCR[0]):
        txt=axes["G"].text(ii,1,str(round(val,2)),c="white",ha="center",va="center")
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

    im_FLU = axes["E"].imshow(FLU,extent=extent,cmap="Greys",vmin=min_reduction_FLU,vmax=max_reduction_FLU)
    axes["H"].imshow(FLU_PCR,extent=PCR_extent,cmap="Greys",vmin=min_reduction_FLU,vmax=max_reduction_FLU)
    cbar_FLU = plt.colorbar(im_FLU,ax=axes["B"],fraction=0.7,orientation="horizontal")
    cbar_FLU.set_ticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
    axes["B"].text(0.5,0.7,'Influenza Ascertainment',ha='center',size=LABEL_SIZE-2)
    # Add TE values to each square
    for n in range(max(num_tests)):
        for ii,row in enumerate(FLU):
            txt=axes["E"].text(ii,6-n,str(round(FLU[row][n],2)),c="w",ha="center",va="center")
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
    for ii, val in enumerate(FLU_PCR[0]):
        txt=axes["H"].text(ii,1,str(round(val,2)),c="w",ha="center",va="center")
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

    im_COVID = axes["F"].imshow(COVID,extent=extent,cmap="Greys",vmin=min_reduction_COVID,vmax=max_reduction_COVID)
    axes["I"].imshow(COVID_PCR,extent=PCR_extent,cmap="Greys",vmin=min_reduction_COVID,vmax=max_reduction_COVID)
    cbar_COVID = plt.colorbar(im_COVID,ax=axes["C"],fraction=0.7,orientation="horizontal")
    cbar_COVID.set_ticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
    axes["C"].text(0.5,0.7,'SARS-CoV-2 Ascertainment',ha='center',size=LABEL_SIZE-2)
    # Add TE values to each square
    for n in range(max(num_tests)):
        for ii,row in enumerate(COVID):
            txt=axes["F"].text(ii,6-n,str(round(COVID[row][n],2)),c="w",ha="center",va="center")
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
    for ii, val in enumerate(COVID_PCR[0]):
        txt=axes["I"].text(ii,1,str(round(val,2)),c="w",ha="center",va="center")
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

    x_axes = [axes["D"],axes["E"],axes["F"],axes["G"],axes["H"],axes["I"]]
    for a in x_axes:
        a.set_xticks([0,1,2,3,4,5])
    
    axes["G"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)
    axes["H"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)
    axes["I"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)

    axes["D"].set_yticks([1,2,3,4,5,6])
    axes["D"].set_ylabel("Rapid tests available")
    axes["G"].set_ylabel("PCR")

    for a in [axes["E"],axes["F"],axes["H"],axes["I"]]:
        a.set_yticks([])

    for a in [axes["D"],axes["E"],axes["F"]]:
        finalize_keep_frame(a)
        force_aspect(a,1)
        a.set_xticklabels([])
        a.set_yticks([1,2,3,4,5,6])
    
    for a in [axes["A"],axes["B"],axes["C"]]:
        remove_ax(a)

    for a in [axes["G"],axes["H"],axes["I"]]:
        finalize_keep_frame(a,aspect=4.75)
        a.set_yticks([1])

    # axes["D"].text(5.25,6.25,"RDT tests\n0 day TAT",size=18,va="top",ha="right")
    # axes["G"].text(5.25,1.25,"PCR, 2 day",size=18,va="top",ha="right")

    xlabpad = -0.15; ylabpad = 2.4;
    label_subplots(axes,x_pads=[xlabpad,xlabpad,xlabpad],y_pad=ylabpad,labels=["A","B","C"],fontsize=LABEL_SIZE)
    plt.subplots_adjust(hspace=-0.4) # space between rows

    fname = fig_path / "Fig4Supp_Asc.png"
    plt.savefig(fname,dpi=300,bbox_inches="tight") # bbox_inches prevents x-label from being cutoff
    fname = fig_path / "Fig4Supp_Asc.pdf"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    # plt.show()

def Fig5_generate_data():
    # Comparing [2x weekly, weekly, symp-based] screening with [low sens 0 day TAT, high sense 2 day TAT]
    c = 1; p = 1; f = 0.05; exp_num = 2;
    wt_LODs = [3,5]; om_LODs = [3,6]; TATs = [2,0]; freqs = [3.5,7]


    Fig5 = pd.read_csv(sim_path / "figure5.csv")

    # Omicron Variant - COVID
    path = "COVID"
    path_string = get_kinetics_string(path)
    for ii,LOD in enumerate(om_LODs):
        TE = []
        # regular screening
        p = 1
        for freq in freqs:
            data = Fig5.loc[(Fig5['kinetics']==path_string) &
                            (Fig5['testing']=="test_regular") & 
                            (Fig5['L']==LOD) & 
                            (Fig5['tat']==TATs[ii]) & 
                            (Fig5['Q']==freq)]
            if len(data) > 1:
                raise Exception("You didn't query me enough :(")
            TE.append(p*data['TE'].values[0])
        # symptom-based testing
        w = 0; tpd = 1; n = 1; # immediate testing with 1 test
        p = get_pct_symptomatic(path)
        data = Fig5.loc[(Fig5['kinetics']==path_string) &
                        (Fig5['testing']=="test_post_symptoms") & 
                        (Fig5['L']==LOD) & 
                        (Fig5['tat']==TATs[ii]) & 
                        (Fig5['wait']==w) & 
                        (Fig5['supply']==n) &
                        (Fig5['Q']==1)]
        if len(data) > 1:
            raise Exception("You didn't query me enough :(")
        TE.append(p*data['TE'].values[0])
        # save data
        fname = data_path / path / "_fig5_LOD" / str(LOD)
        save_data(TE,fname)

    # wildtype
    path = "WT"
    path_string = get_kinetics_string(path)
    for ii,LOD in enumerate(wt_LODs):
        TE = []
        # regular screening
        p = 1
        for freq in freqs:
            data = Fig5.loc[(Fig5['kinetics']==path_string) & 
                            (Fig5['testing']=="test_regular") & 
                            (Fig5['L']==LOD) & 
                            (Fig5['tat']==TATs[ii]) & 
                            (Fig5['Q']==freq)]
            if len(data) > 1:
                raise Exception("You didn't query me enough :(")
            TE.append(p*data['TE'].values[0])
        # symptom-based testing
        w = 0; tpd = 1; n = 1; # immediate testing with 1 test
        p = get_pct_symptomatic(path)
        data = Fig5.loc[(Fig5['kinetics']==path_string) &
                        (Fig5['testing']=="test_post_symptoms") & 
                        (Fig5['L']==LOD) & 
                        (Fig5['tat']==TATs[ii]) & 
                        (Fig5['wait']==w) & 
                        (Fig5['supply']==n) &
                        (Fig5['Q']==1)]
        if len(data) > 1:
            raise Exception("You didn't query me enough :(")
        TE.append(p*data['TE'].values[0])
        # Save data
        fname = data_path / path / "_fig5_LOD" / str(LOD)
        save_data(TE,fname)
def Draw_Fig5():
    om_data_highsens = read_data(data_path / "COVID_fig5_LOD3")
    om_data_lowsens = read_data(data_path / "COVID_fig5_LOD6")
    wt_data_highsens = read_data(data_path / "WT_fig5_LOD3")
    wt_data_lowsens = read_data(data_path / "WT_fig5_LOD5")

    labs = ["2x weekly","Weekly","Post-Sx"]; xvals = np.array([1,3,5]);
    N = 15; w=0.75; np.random.seed(44)

    figure_mosaic = """
    AAAAAAAA.CCCCC
    .........DDDDD
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,6))

    # Compare trajectories (top - panel axes['A']) ---------------------------------------------------------------
    xmax = 15

    # Wildtype COVID
    inf_params = WT_params_det()
    peak = inf_params['Tpeak']
    VL = [get_VL(t,**inf_params) for t in t_vals]
    LOD = get_LOD_low("WT")
    # Calculate detectability by PCR and RDT
    min_time_detectable_PCR = np.where(np.array(VL)>0)[0][0]
    max_time_detectable_PCR = np.where(np.array(VL)>get_LOD_high("COVID"))[0][-1]
    min_time_detectable_RDT = np.where(np.array(VL)>=LOD)[0][0]
    gap = (min_time_detectable_RDT - min_time_detectable_PCR)*dt
    print("Wild-type detectability gap: ",gap)
    # symptom onset times
    symp_times = WT_get_stoch_symp_time()
    symp_low = inf_params['Tpeak'] + symp_times[1]; symp_high = inf_params['Tpeak'] + symp_times[0]; 
    print("wild-type symptom onset times: ", symp_low, " - ", symp_high )
    # plot
    wt, = axes['A'].plot(t_vals,VL,c=COVID_colors[3],linewidth=3)
    axes['A'].plot([0,35],[LOD,LOD],c=COVID_colors[3],linestyle="dashed")
    axes['A'].text(xmax,LOD,"founder RDT",ha="right",va="center",c=COVID_colors[3],backgroundcolor="white")

    # Omicron variant
    inf_params = COVID_params_det()
    peak = inf_params['Tpeak']
    VL = [get_VL(t,**inf_params) for t in t_vals]
    LOD = get_LOD_low("COVID")
    # Calculate detectability by PCR and RDT
    min_time_detectable_PCR = np.where(np.array(VL)>0)[0][0]
    max_time_detectable_PCR = np.where(np.array(VL)>get_LOD_high("COVID"))[0][-1]
    min_time_detectable_RDT = np.where(np.array(VL)>=LOD)[0][0]
    gap = (min_time_detectable_RDT - min_time_detectable_PCR)*dt
    print("Omicron detectability gap: ",gap )
    # symptom onset times
    symp_times = COVID_get_stoch_symp_time()
    symp_low = inf_params['Tpeak'] + symp_times[1]; symp_high = inf_params['Tpeak'] + symp_times[0]; 
    print("omicron symptom onset times: ", symp_low, " - ", symp_high )
    # plot
    om, = axes['A'].plot(t_vals,VL,c=COVID_colors[5],linewidth=3)
    axes['A'].plot([0,35],[LOD,LOD],c=COVID_colors[5],linestyle="dashed")
    axes['A'].text(xmax,LOD,"omicron RDT LOD",ha="right",va="center",c=COVID_colors[6],backgroundcolor="white")

    # axes adjustments
    axes['A'].set_ylabel("$Log_{10}$ viral load",size=LABEL_SIZE-2); #($\log_{10}$ cp RNA/mL)
    #axes['A'].legend([wt,om],["typical founder","typical omicron"],loc='upper right')
    axes['A'].set_ylim([3,8]); axes['A'].set_xlim([0,xmax]); axes['A'].set_xticks([0,5,10,15])
    axes['A'].set_xlabel("Days since exposure",size=LABEL_SIZE-2,labelpad=1);
    finalize(axes['A'])

    # WT results - top right (axes['C'])  ---------------------------------------------------------------
    a = axes['C'].bar(xvals-w/2,wt_data_highsens,color=light_gray,width=w,edgecolor=ALMOST_BLACK)
    b = axes['C'].bar(xvals+w/2,wt_data_lowsens,color=COVID_colors[3],width=w,edgecolor=ALMOST_BLACK)
    axes['C'].set_xticks(xvals); axes['C'].set_xticklabels(labs,size=LABEL_SIZE-4);
    axes['C'].set_ylim([0,1]); axes['C'].set_ylabel("Testing effectiveness (TE)",size=LABEL_SIZE-2)
    axes['C'].legend([a,b],["PCR","RDT"],loc='upper right')
    axes['C'].set_title("Founder/Naive SARS-CoV-2",{'fontsize':LABEL_SIZE,'fontweight':'bold'},pad=10)
    axes['C'].set_ylim([0,0.75]); axes['C'].set_yticks([0,0.25,0.5,0.75]); axes['C'].set_yticklabels([0,0.25,0.5,0.75])
    finalize(axes['C'],aspect=1.75)

    # omicron results - bottom right (axes['D'])  ---------------------------------------------------------------
    a = axes['D'].bar(xvals-w/2,om_data_highsens,color=light_gray,width=w,edgecolor=ALMOST_BLACK)
    b = axes['D'].bar(xvals+w/2,om_data_lowsens,color=COVID_colors[5],width=w,edgecolor=ALMOST_BLACK)
    axes['D'].set_xticks(xvals); axes['D'].set_xticklabels(labs,size=LABEL_SIZE-4); 
    axes['D'].set_ylim([0,1]); axes['D'].set_ylabel("Testing effectiveness (TE)")
    axes['D'].legend([a,b],["PCR","RDT"],loc='upper right')
    axes['D'].set_title("Omicron/Experienced SARS-CoV-2",{'fontsize':LABEL_SIZE,'fontweight':'bold'},pad=15)
    axes['D'].set_ylim([0,0.75]); axes['D'].set_yticks([0,0.25,0.5,0.75]); axes['D'].set_yticklabels([0,0.25,0.5,0.75])
    finalize(axes['D'],aspect=1.75)

    plt.subplots_adjust(wspace=10, # space between cols
                    hspace=0.4 ) # space between rows
    
    fname = fig_path / "Fig5_nolabs.png"
    plt.savefig(fname,dpi=300)
    fname = fig_path / "Fig5_nolabs.pdf"
    plt.savefig(fname,dpi=300)
    plt.show()

def create_filestring_exp4(path,testing,isolation,w,n):
    '''
    Exp 4 - isolation behaviors
    '''
    test_string = testing + str(w) + str(n)
    return sim_path / 'exp_4_{}_{}_detection_{}_isolation_LOD6_c1'.format(path,test_string,isolation)
def Fig6_generate_data():
    ''' Comparing isolation policy cost-benefit '''
    # symptomatic testing with 1 or two tests to diagnost and isolation of 5 days or TTE
    Fig6 = pd.read_csv(sim_path / "figure6.csv")
    pathogen = "COVID"

    # Three bar charts: TE, test consumption, and days isolated using 1 or two tests
    freq = "symp" # symptomatic testing
    p = get_pct_symptomatic(pathogen)
    w = 1 # wait one day after symptom onset
    num_tests = [1,2]
    # initialize dictionaries
    five_day_iso = {}; TTE_iso = {};
    for iso in [five_day_iso,TTE_iso]:
        iso['TE'] = []; iso['test_consumption'] = []; iso['days_isolated'] = []
    # Five day isolation
    for n in num_tests:
        data = Fig6.loc[(Fig6['isolation']=="fixed") 
                        & (Fig6['supply']==n)]
        if len(data) > 1:
            raise Exception("You didn't query me enough :(")
        five_day_iso['TE'].append(p*data['TE'].values[0])
        five_day_iso['test_consumption'].append(data['n_tests_dx'].values[0] + data['n_tests_exit'].values[0])
        five_day_iso['days_isolated'].append(data['T_isolation'].values[0])
    # TTE isolation
    for n in num_tests:
        data = Fig6.loc[(Fig6['isolation']=="TTE") & (Fig6['supply']==n)]
        if len(data) > 1:
            raise Exception("You didn't query me enough :(")
        TTE_iso['TE'].append(p*data['TE'].values[0])
        TTE_iso['test_consumption'].append(data['n_tests_dx'].values[0] + data['n_tests_exit'].values[0])
        TTE_iso['days_isolated'].append(data['T_isolation'].values[0])
    # save data
    fname = data_path / "Fig6_barchart"
    save_data([five_day_iso,TTE_iso],fname)

    # Histogram - read from montecarlo simulation data
    hash = "2850778d294e8334637990ab7ee1340735b5d56b3cc50a9ff1590293a4501432"
    data = pd.read_csv(sim_path / hash / ".zip")
    data = data.loc[data['tDx'] < 40] # disregard infinite time entries - those that are never detected
    durations = data['tExit'] - data['tDx']
    fname = data_path / "Fig6_histogram"
    save_data(durations,fname)
def Draw_Fig6():
    data = read_data(data_path / "Fig6_barchart")
    hist_data = read_data(data_path / "Fig6_histogram")
    five_day_iso = data[0]; TTE_iso = data[1];

    # set colors
    isolations = [five_day_iso, TTE_iso]
    cols = [COVID_colors[6],COVID_colors[3]]
    lstyles = ["solid","solid"]

    figure_mosaic = """
    AABBCCDDD
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,3))
    width = 0.325; dx = width/2; xvals = np.array([1,2]); # horizontal locations for barchart

    # Plot TE (axes['A'])
    axes['A'].bar(xvals-dx,five_day_iso['TE'],width,color=cols[0],edgecolor=ALMOST_BLACK)
    axes['A'].bar(xvals+dx,TTE_iso['TE'],width,color=cols[1],edgecolor=ALMOST_BLACK)
    axes['A'].set_ylabel("Testing effectiveness (TE)",size=LABEL_SIZE-2);
    axes['A'].set_xticks([1,2]); axes['A'].set_xticklabels([])
    axes['A'].set_ylim([0,0.41])
    axes['A'].set_xticklabels(["1","2"],size=LABEL_SIZE-2)
    finalize(axes['A'])

    # Plot test consumption (axes['B'])
    axes['B'].bar(xvals-dx,five_day_iso['test_consumption'],width,color=cols[0],edgecolor=ALMOST_BLACK)
    axes['B'].bar(xvals+dx,TTE_iso['test_consumption'],width,color=cols[1],edgecolor=ALMOST_BLACK)
    axes['B'].set_ylabel("Avg. total tests used",size=LABEL_SIZE-2);
    axes['B'].set_xticks([1,2]); axes['B'].set_xticklabels([])
    axes['B'].set_ylim([0,3.2])
    axes['B'].legend(["5 day isolation","test-to-exit"],loc='upper left',frameon=False)
    axes['B'].set_xticklabels(["1","2"],size=LABEL_SIZE-2)
    axes['B'].set_yticks([0,1,2,3]); axes['B'].set_yticklabels([0,1,2,3])
    finalize(axes['B'])

    # Plot days isolated (axes['C'])
    axes['C'].bar(xvals-dx,five_day_iso['days_isolated'],width,color=cols[0],edgecolor=ALMOST_BLACK)
    axes['C'].bar(xvals+dx,TTE_iso['days_isolated'],width,color=cols[1],edgecolor=ALMOST_BLACK)
    axes['C'].set_ylabel("Avg. days isolated",size=LABEL_SIZE-2);
    axes['C'].set_xticks([1,2])
    axes['C'].set_xticklabels(["1","2"],size=LABEL_SIZE-2)
    axes['C'].set_ylim([0,5.3]);
    finalize(axes['C'])

    # Histogram of isolation days
    axes['D'].hist(np.clip(hist_data,0,9),bins=np.array([1,2,3,4,5,6,7,8,9]),density=True,color=cols[1],edgecolor=ALMOST_BLACK)
    axes['D'].plot([TTE_iso['days_isolated'][0],TTE_iso['days_isolated'][0]],[0,0.67],color=ALMOST_BLACK,linestyle="dashed")
    axes['D'].text(TTE_iso['days_isolated'][0],0.67,"mean",c=ALMOST_BLACK,ha="center",size=10)
    axes['D'].text(6.5,0.55,"Distribution of days\nspent in isolation\n(TTE, 2 tests to Dx)",size=LABEL_SIZE,ha="center",va="top")
    axes['D'].set_xticks([1,2,3,4,5,6,7,8]); axes['D'].set_xticklabels(["1","2","3","4","5","6","7","8+"])
    axes['D'].text(4.55,-0.19,"Days spent in isolation",size=LABEL_SIZE,fontweight="bold",ha="center")
    finalize(axes['D'])

    #axes['B'].text(1.5,-0.8,"Tests to diagnose",size=LABEL_SIZE,fontweight="bold",ha="center")
    axes['A'].set_xlabel("Tests to diagnose",size=LABEL_SIZE,fontweight="bold")
    axes['B'].set_xlabel("Tests to diagnose",size=LABEL_SIZE,fontweight="bold")
    axes['C'].set_xlabel("Tests to diagnose",size=LABEL_SIZE,fontweight="bold")

    label_subplots(axes,x_pads=[0.05,0.05,0.05,0.025],y_pad=1.05,labels=["A","B","C","D"],fontsize=LABEL_SIZE)

    plt.tight_layout()
    fname = fig_path / "Fig6.png"
    plt.savefig(fname,dpi=300)
    fname = fig_path / "Fig6.pdf"
    plt.savefig(fname,dpi=300)
    # plt.show()

def VL_samples():
    ''' Plot viral trajectories of COVID, RSV, and FLU with LODs '''
    figure_mosaic = """
    ABC
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,4))
    bbox_dict = dict(fc='w',ec='w',pad=0.1,alpha=1)
    # plot stochastic draws of each viral trajectory in gray
    n = 20 # number of sample trajectories to show
    np.random.seed(44)

    for ii in range(n):
        # COVID VL
        COVID_params = COVID_params_stoch()
        COVID_VL = np.array([get_VL(t,**COVID_params) for t in t_vals])
        first_det = np.where(COVID_VL>=get_LOD_high("COVID"))[0][0]
        last_det = np.where(COVID_VL>=get_LOD_high("COVID"))[0][-1]
        COVID_plot = COVID_VL[first_det:last_det+1]
        COVID_t_vals = t_vals[first_det:last_det+1]
        
        # Flu VL
        FLU_params = FLU_params_stoch()
        FLU_VL = np.array([get_VL(t,**FLU_params) for t in t_vals])
        first_det = np.where(FLU_VL>=get_LOD_high("FLU"))[0][0]
        last_det = np.where(FLU_VL>=get_LOD_high("FLU"))[0][-1]
        FLU_plot = FLU_VL[first_det:last_det+1]
        FLU_t_vals = t_vals[first_det:last_det+1]
        
        # RSV VL
        RSV_params = RSV_params_stoch()
        RSV_VL = np.array([get_VL(t,**RSV_params) for t in t_vals])
        first_det = np.where(RSV_VL>=get_LOD_high("RSV"))[0][0]
        last_det = np.where(RSV_VL>=get_LOD_high("RSV"))[0][-1]
        RSV_plot = RSV_VL[first_det:last_det+1]
        RSV_t_vals = t_vals[first_det:last_det+1]
       
        # Plot
        axes["C"].plot(COVID_t_vals,COVID_plot,color=dark_gray,zorder=1,linewidth=1,alpha=0.3)
        axes["B"].plot(FLU_t_vals,FLU_plot,color=dark_gray,zorder=1,linewidth=1,alpha=0.3)
        axes["A"].plot(RSV_t_vals,RSV_plot,color=dark_gray,zorder=1,linewidth=1,alpha=0.3)

    # Highlight one deterministic trajectory
    # COVID
    COVID_params = COVID_params_det()
    COVID_VL = np.array([get_VL(t,**COVID_params) for t in t_vals])
    first_det = np.where(COVID_VL>=get_LOD_high("COVID"))[0][0]
    last_det = np.where(COVID_VL>=get_LOD_high("COVID"))[0][-1]
    COVID_plot = COVID_VL[first_det:last_det+1]
    COVID_t_vals = t_vals[first_det:last_det+1]
    axes["C"].plot(COVID_t_vals,COVID_plot,color=COVID_colors[5],zorder=1,linewidth=2,alpha=1)
    # Flu 
    FLU_params = FLU_params_det()
    FLU_VL = np.array([get_VL(t,**FLU_params) for t in t_vals])
    first_det = np.where(FLU_VL>=get_LOD_high("FLU"))[0][0]
    last_det = np.where(FLU_VL>=get_LOD_high("FLU"))[0][-1]
    FLU_plot = FLU_VL[first_det:last_det+1]
    FLU_t_vals = t_vals[first_det:last_det+1]
    axes["B"].plot(FLU_t_vals,FLU_plot,color=FLU_colors[5],zorder=1,linewidth=2,alpha=1)
    # RSV 
    RSV_params = RSV_params_det()
    RSV_VL = np.array([get_VL(t,**RSV_params) for t in t_vals])
    first_det = np.where(RSV_VL>=get_LOD_high("RSV"))[0][0]
    last_det = np.where(RSV_VL>=get_LOD_high("RSV"))[0][-1]
    RSV_plot = RSV_VL[first_det:last_det+1]
    RSV_t_vals = t_vals[first_det:last_det+1]
    axes["A"].plot(RSV_t_vals,RSV_plot,color=RSV_colors[5],zorder=1,linewidth=2,alpha=1)

    # Plot limits of detection
    COVID_low = get_LOD_low("COVID")*np.ones(len(t_vals))
    COVID_high = get_LOD_high("COVID")*np.ones(len(t_vals))
    FLU_low = get_LOD_low("FLU")*np.ones(len(t_vals))
    FLU_high = get_LOD_high("FLU")*np.ones(len(t_vals))
    RSV_low = get_LOD_low("RSV")*np.ones(len(t_vals))
    RSV_high = get_LOD_high("RSV")*np.ones(len(t_vals))
    axes["C"].plot(t_vals,COVID_low,color=dark_gray,linewidth=2,linestyle="dashed")
    axes["C"].plot(t_vals,COVID_high,color=dark_gray,linewidth=2)#,linestyle="dashed")
    axes["B"].plot(t_vals[0:12500],FLU_low[0:12500],color=dark_gray,linewidth=2,linestyle="dashed")
    axes["B"].plot(t_vals,FLU_high,color=dark_gray,linewidth=2)#,linestyle="dashed")
    axes["A"].plot(t_vals,RSV_low,color=dark_gray,linewidth=2,linestyle="dashed")
    axes["A"].plot(t_vals,RSV_high,color=dark_gray,linewidth=2)#,linestyle="dashed")

    # plot infectiousness thresholds
    axes["C"].fill_between(t_vals,COVID_params['Yinf'],100,color=light_gray,alpha=0.15)
    axes["B"].fill_between(t_vals,FLU_params['Yinf'],100,color=light_gray,alpha=0.15)
    axes["A"].fill_between(t_vals,RSV_params['Yinf'],100,color=light_gray,alpha=0.15)

    axes["C"].arrow(16,COVID_params['Yinf'],0,1,head_width=0.4,head_length=0.15,color=dark_gray)
    axes["C"].arrow(24,COVID_params['Yinf'],0,1,head_width=0.4,head_length=0.15,color=dark_gray)
    axes["C"].text(20,COVID_params['Yinf']+1.4,"Infectious\nvirus",size=LABEL_SIZE-2,color=dark_gray,ha="center",va="top")

    axes["B"].text(14.8,FLU_low[0],"RDT",color=dark_gray,va="center",ha="right",\
        size=LABEL_SIZE,zorder=3)
    axes["B"].text(15,FLU_high[0],"RT-qPCR",color=dark_gray,va="center",ha="right",\
        size=LABEL_SIZE,bbox=bbox_dict,zorder=3)

    ymax = 9.5; ymin = 2.5
    axes["C"].set_ylim([ymin,ymax]); # COVID ylim
    axes["B"].set_ylim([ymin,ymax]); # FLU ylim
    axes["A"].set_ylim([ymin,ymax]); # RSV ylim

    axes["C"].set_xlabel("Days since exposure"); axes["B"].set_xlabel("Days since exposure"); axes["A"].set_xlabel("Days since exposure");
    axes["A"].set_ylabel("Viral load (log$_{10}$ cp RNA/mL)")
    axes["C"].set_xlim([0,25]); axes["B"].set_xlim([0,15]); axes["A"].set_xlim([0,15]);
    axes["C"].set_xticks([0,5,10,15,20,25]); axes["C"].set_xticklabels([0,5,10,15,20,25])
    finalize(axes["A"],aspect=1)
    finalize(axes["B"],aspect=1)
    finalize(axes["C"],aspect=1)

    # Add subplot labels
    axes["A"].text(-1.2,9.5,"A",size=LABEL_SIZE,ha="center",fontweight="bold")
    axes["B"].text(-1.2,9.5,"B",size=LABEL_SIZE,ha="center",fontweight="bold")
    axes["C"].text(-1.2,9.5,"C",size=LABEL_SIZE,ha="center",fontweight="bold")

    plt.tight_layout()
    
    fname = fig_path / "viral_trajectories.png"
    plt.savefig(fname,dpi=300)
    fname = fig_path / "viral_trajectories.pdf"
    plt.savefig(fname,dpi=300)
    plt.show()
def VL_maintext():
    ''' Plot key attributes of of COVID, RSV, and FLU viral and symptom kinetics '''
    figure_mosaic = """
    ABC
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,4))
    bbox_dict = dict(fc='w',ec='w',pad=0.1,alpha=0.9)
    # plot stochastic draws of each viral trajectory in gray
    n = 15 # number of sample trajectories to show
    np.random.seed(40)

    # Panel A: RSV ------------------------------------------------------------------------
    ii=0;
    while ii < n:
        RSV_params = RSV_params_stoch()
        # infectious timing
        RSV_VL = np.array([get_VL(t,**RSV_params) for t in t_vals])
        inf_indexes = np.where(RSV_VL >= RSV_params['Yinf'])
        try:
            first_inf = t_vals[inf_indexes[0][0]]; 
            last_inf = t_vals[inf_indexes[0][-1]];
        except:
            print("Skipping non-infectious draw for RSV")
            continue
        # RDT detectability
        try:
            first_det = t_vals[np.where(RSV_VL>=get_LOD_low("RSV"))[0][0]]
            last_det = t_vals[np.where(RSV_VL>=get_LOD_low("RSV"))[0][-1]]
        except:
            first_det = None; last_det = None;
        # symptom timing
        if ii/n <= 1 - get_pct_symptomatic("RSV"):
            RSV_symp = None
        else:
            RSV_symp_times = RSV_get_stoch_symp_time()
            RSV_symp = RSV_params['Tpeak'] + np.random.uniform(RSV_symp_times[0],RSV_symp_times[1])
            while RSV_symp < 1:
                RSV_symp = RSV_params['Tpeak'] + np.random.uniform(RSV_symp_times[0],RSV_symp_times[1])
        # Plot individual data
        axes["A"].scatter(RSV_symp,ii,marker="o",c="w",edgecolors="k",zorder=3) # symptom time
        axes["A"].plot([first_inf,last_inf],[ii,ii],c=RSV_colors[3],linewidth=4,alpha=0.5) # infectious time
        axes["A"].plot([first_det,last_det],[ii,ii],c=RSV_colors[5]) # detectable time
        ii += 1

    # Panel B: FLU ------------------------------------------------------------------------
    ii=0;
    while ii < n:
        FLU_params = FLU_params_stoch()
        # infectious timing
        FLU_VL = np.array([get_VL(t,**FLU_params) for t in t_vals])
        inf_indexes = np.where(FLU_VL >= FLU_params['Yinf'])
        try:
            first_inf = t_vals[inf_indexes[0][0]]; 
            last_inf = t_vals[inf_indexes[0][-1]];
        except:
            print("Skipping non-infectious draw for flu")
            continue
        # RDT detectability
        try:
            first_det = t_vals[np.where(FLU_VL>=get_LOD_low("FLU"))[0][0]]
            last_det = t_vals[np.where(FLU_VL>=get_LOD_low("FLU"))[0][-1]]
        except:
            first_det = None; last_det = None;
        # symptom timing
        if ii/n <= 1 - get_pct_symptomatic("FLU"):
            FLU_symp = None
        else:
            FLU_symp_times = FLU_get_stoch_symp_time()
            FLU_symp = FLU_params['Tpeak'] + np.random.uniform(FLU_symp_times[0],FLU_symp_times[1])
            while FLU_symp < 1:
                FLU_symp = FLU_params['Tpeak'] + np.random.uniform(FLU_symp_times[0],FLU_symp_times[1])
        # Plot individual data
        axes["B"].scatter(FLU_symp,ii,marker="o",c="w",edgecolors="k",zorder=3) # symptom time
        axes["B"].plot([first_inf,last_inf],[ii,ii],c=FLU_colors[3],linewidth=4,alpha=0.5) # infectious time
        axes["B"].plot([first_det,last_det],[ii,ii],c=FLU_colors[5]) # detectable time
        ii += 1

    # Panel C: COVID ------------------------------------------------------------------------
    ii=0;
    while ii < n:
        COVID_params = COVID_params_stoch()
        # infectious timing
        COVID_VL = np.array([get_VL(t,**COVID_params) for t in t_vals])
        inf_indexes = np.where(COVID_VL >= COVID_params['Yinf'])
        try:
            first_inf = t_vals[inf_indexes[0][0]]; 
            last_inf = t_vals[inf_indexes[0][-1]];
        except:
            print("Skipping non-infectious draw for SARS-CoV-2")
            continue
        if last_inf - first_inf <= 1:
            print("Skipping non-infectious draw for SARS-CoV-2")
            continue
        # RDT detectability
        try:
            first_det = t_vals[np.where(COVID_VL>=get_LOD_low("COVID"))[0][0]]
            last_det = t_vals[np.where(COVID_VL>=get_LOD_low("COVID"))[0][-1]]
        except:
            first_det = None; last_det = None;
        # symptom timing
        if ii/n <= 1 - get_pct_symptomatic("COVID"):
            COVID_symp = None
        else:
            COVID_symp_times = COVID_get_stoch_symp_time()
            COVID_symp = COVID_params['Tpeak'] + np.random.uniform(COVID_symp_times[0],COVID_symp_times[1])
            while COVID_symp < 1:
                COVID_symp = COVID_params['Tpeak'] + np.random.uniform(COVID_symp_times[0],COVID_symp_times[1])
        # Plot individual data
        axes["C"].scatter(COVID_symp,ii,marker="o",c="w",edgecolors="k",zorder=3) # symptom time
        axes["C"].plot([first_inf,last_inf],[ii,ii],c=COVID_colors[3],linewidth=4,alpha=0.5) # infectious time
        axes["C"].plot([first_det,last_det],[ii,ii],c=COVID_colors[5]) # detectable time
        ii += 1


    # Changes to all axes
    xmax = 15; xmin = 0
    axlabs = ["A","B","C"]
    for axlab in axlabs:
        # x-axis limits
        axes[axlab].set_xlim([xmin,xmax])
        # x-axis grid lines 
        axes[axlab].grid(axis="x")
        # subplot labels
        axes[axlab].text(-1,n,axlab,size=LABEL_SIZE,ha="center",fontweight="bold")
        # X-axis label
        axes[axlab].set_xlabel("Days since exposure")
        # No y ticks
        axes[axlab].set_yticks([])

    lpad = 12#25
    axes["A"].set_title("RSV",size=LABEL_SIZE,fontweight="bold",pad=lpad)
    axes["B"].set_title("Influenza A",size=LABEL_SIZE,fontweight="bold",pad=lpad)
    axes["C"].set_title("SARS-CoV-2",size=LABEL_SIZE,fontweight="bold",pad=lpad)
    if lpad == 25:
        axes["C"].text(7.5,n+0.65,"omicron/experienced",size=LABEL_SIZE-3,ha="center")
    else:
        axes["C"].text(7.5,n-0.25,"omicron/experienced",size=LABEL_SIZE-3,ha="center")

    axes["A"].set_ylabel("Individual"); 
    asp = 0.85
    for axlab in axlabs:
        axes[axlab].set_aspect(asp)
        finalize(axes[axlab])

    plt.tight_layout()
    
    fname = fig_path / "viral_kinetics_maintext.png"
    plt.savefig(fname,dpi=300)
    fname = fig_path / "viral_kinetics_maintext.pdf"
    plt.savefig(fname,dpi=300)
    plt.show()

def jitter_plots():
    # Panel B - jitter plots
    omicron_6 = sim_path / '18b20679510954da24bcd0e58f73cf33ce101900b8d3bb0e9bc45738edba5a86.zip'
    founder_5 = sim_path / '9fac1f6ad22a116191d6a8ea4b8cb4332bbde2512b5df3a30d2b6b5afbf839df.zip'
    files = [founder_5,omicron_6]
    N = 1000
    ax = axes['L']
    ymax = 16

    np.random.seed(42)
    w = 0.7
    hw = 0.7/2
    jitter = (np.random.rand(int(1e5))-0.5)*w
    labels = ['Founder\nnaive','Omicron\nexp.']
    colors = [COVID_colors[3],COVID_colors[5]] # founder, omicron
    for idx,file in enumerate(files):
        df = pd.read_csv(file)
        data = df['last'].values - df['first'].values
        data = data[data<=ymax]
        ax.scatter(idx+jitter[:N],data[:N],20,
                color=colors[idx],
                alpha=0.25,
                clip_on=False,
                zorder=3)
        ax.plot([idx-hw,idx+hw],[np.median(data)]*2,
                lw=2,color='k',zorder=4)
        [m,M] = np.quantile(data,[0.25,0.75])
        rect = Rectangle((idx-hw,m),w,M-m,linewidth=1,edgecolor='k',facecolor='none',zorder=4,clip_on=False)
        ax.add_patch(rect)
        IQR = M-m
        high = np.min([M+1.5*IQR,np.max(data)])
        low = np.max([m-1.5*IQR,np.min(data)])
        ax.plot([idx]*2,[M,high],'k',zorder=4,clip_on=False)
        ax.plot([idx-hw/2,idx+hw/2],[high,high],'k',zorder=4,clip_on=False)
        ax.plot([idx]*2,[m,low],'k',zorder=4,clip_on=False)
        ax.plot([idx-hw/2,idx+hw/2],[low,low],'k',zorder=4,clip_on=False)
    ax.set_ylabel('Days detectable by RDT')
    ax.set_ylim([0,ymax])
    ax.set_xlim([0-w,1+w])
    ax.set_yticks(np.arange(0,ymax+0.1,2))
    ax.set_xticks([0,1])
    ax.set_xticklabels(labels)

    ax = axes['M']
    ymax = 10
    for idx,file in enumerate(files):
        df = pd.read_csv(file)
        data = df['first'].values-df['A'].values
        data = data[data>0]
        data = data[data<=ymax]
        ax.scatter(idx+jitter[:N],data[:N],20,
                color=colors[idx],
                alpha=0.25,
                clip_on=False,zorder=3)
        ax.plot([idx-hw,idx+hw],[np.median(data)]*2,
                lw=2,color='k',zorder=4)
        [m,M] = np.quantile(data,[0.25,0.75])
        rect = Rectangle((idx-hw,m),w,M-m,linewidth=1,edgecolor='k',facecolor='none',zorder=4,clip_on=False)
        ax.add_patch(rect)
        IQR = M-m
        high = np.min([M+1.5*IQR,np.max(data)])
        low = np.max([m-1.5*IQR,np.min(data)])
        ax.plot([idx]*2,[M,high],'k',zorder=4,clip_on=False)
        ax.plot([idx-hw/2,idx+hw/2],[high,high],'k',zorder=4,clip_on=False)
        ax.plot([idx]*2,[m,low],'k',zorder=4,clip_on=False)
        ax.plot([idx-hw/2,idx+hw/2],[low,low],'k',zorder=4,clip_on=False)
    ax.set_ylabel(r'$\Delta$LOD/m (days)',labelpad=-7)
    ax.set_ylim([0,ymax])
    ax.set_yticks(np.arange(0,ymax+0.1,2))
    ax.set_xlim([0-w,1+w])
    ax.set_xticks([0,1])
    xx = ax.get_xlim()
    yy = ax.get_ylim()
    rect = Rectangle((xx[0],yy[0]),xx[1]-xx[0],np.abs(yy[0])+2,linewidth=0,facecolor='k',alpha=0.1,zorder=1)
    ax.add_patch(rect)
    ax.plot(xx,[2,2],c=[0.3,0.3,0.3],ls='--',zorder=2)
    ax.set_xticklabels(labels)
    ax.text(xx[1]-0.03,1,r'$\Delta$TAT',rotation=90,ha='right',va='center',c=[0.3,0.3,0.3])
    ax.annotate("", xy=(xx[1], 2), xytext=(xx[1], 0),
                arrowprops=dict(arrowstyle="<->",color=[0.3,0.3,0.3]))
    
    finalize(axes['L']); finalize(axes['M'])

# Sensitivity analysis supplementary figures
def get_sensitivity_scenario_LOD(pathogen,LOD_scenario):
    # Returns the appropriate LOD for the given scenario
    func_string = pathogen / "_params_stoch"
    if LOD_scenario == "Low":
        return globals()[func_string]()['LOD_low'] - 0.5
    else:
        return globals()[func_string]()['LOD_low'] + 0.5
def sensitivity_Fig2_generate_data():
    # creating a 2x2 grid combining increased/decreased infectiousness thresholds and increased/decreased RDT LOD
    inf_scenarios = ["Low","High"]; # lower and higher infectious thresholds
    LOD_scenarios = ["Low","High"] # lower sensitivit [higher LOD] test and higher sensitivity [lower LOD] test
    pathogens = ["RSV","FLU","COVID"]

    for inf_scenario in inf_scenarios:
        
        # read in data for each of the three figure 1 scenarios
        Fig2a = pd.read_csv("{}sens_{}Inf/Figure2A_{}Inf.csv".format(sim_path,inf_scenario,inf_scenario))
        Fig2b = pd.read_csv("{}sens_{}Inf/Figure2B_{}Inf.csv".format(sim_path,inf_scenario,inf_scenario))
        Fig2c = pd.read_csv("{}sens_{}Inf/Figure2C_{}Inf.csv".format(sim_path,inf_scenario,inf_scenario))

        for LOD_scenario in LOD_scenarios:

            for path in pathogens:
                TEs = []; Ascs = []
                path_string = get_kinetics_string(path)

                # expt3 : weekly rapid testing
                freq, LOD, TAT, p, c = get_expt3_params(path)
                LOD = get_sensitivity_scenario_LOD(path,LOD_scenario)
                expt3 = Fig2c.loc[(Fig2c['kinetics']==path_string) & 
                                (Fig2c['L'] == LOD) & 
                                (Fig2c['tat'] == TAT) & 
                                (Fig2c['Q'] == freq)]
                # make sure we got the experiment we are looking for
                if len(expt3) > 1:
                    # if there are two entries, check if they are just repeats of the same experiment
                    if len(expt3) == 2 and expt3['simulation'].values[0] == expt3['simulation'].values[1]:
                        pass
                    else:
                        raise Exception("You didn't query me enough :(")
                TEs.append(p*expt3['TE'].values[0])
                Ascs.append(p*expt3['ascertainment'].values[0])
                
                # expt1 : 2 rapid tests at symptom onset
                exp, n, w, tpd, LOD, TAT, p = get_expt1_params(path)
                LOD = get_sensitivity_scenario_LOD(path,LOD_scenario)
                expt1 = Fig2a.loc[(Fig2a['kinetics']==path_string) & 
                                (Fig2a['L'] == LOD) & 
                                (Fig2a['tat'] == TAT) & 
                                (Fig2a['wait'] == w)]
                # make sure we got the experiment we are looking for
                if len(expt1) > 1:
                    # if there are two entries, check if they are just repeats of the same experiment
                    if len(expt1) == 2 and expt1['simulation'].values[0] == expt1['simulation'].values[1]:
                        pass
                    else:
                        raise Exception("You didn't query me enough :(")
                TEs.append(p*expt1['TE'].values[0])
                Ascs.append(p*expt1['ascertainment'].values[0])

                # expt2 : 1 PCR 2-7 days after exposure
                exp, n, w, tpd, LOD, TAT, p = get_expt2_params(path)
                expt2 = Fig2b.loc[(Fig2b['kinetics']==path_string) & 
                                (Fig2b['L'] == LOD) & 
                                (Fig2b['tat'] == TAT) & 
                                (Fig2b['wait'] == w)]
                # make sure we got the experiment we are looking for
                if len(expt2) > 1:
                    # if there are two entries, check if they are just repeats of the same experiment
                    if len(expt2) == 2 and expt2['simulation'].values[0] == expt2['simulation'].values[1]:
                        pass
                    else:
                        raise Exception("You didn't query me enough :(")
                TEs.append(p*expt2['TE'].values[0])
                Ascs.append(p*expt2['ascertainment'].values[0])

                fname = "{}_fig2_{}inf_{}LOD".format(data_path / path, inf_scenario, LOD_scenario)
                save_data(TEs,fname)
def sensitivity_Draw_Fig2():
    inf_scenarios = ["Low","High"]; # lower and higher infectious thresholds
    LOD_scenarios = ["Low","High"] # lower RDT LOD and higher RDT LOD

    # figure mosaic coder
    # Low sens/low inf thresh       low sens/ high inf thresh
    # High sens/low inf thresh       high sens/ high inf thresh
    axis_finder = [["A","B"],["C","D"]]

    figure_mosaic = """
    AB
    CD
    """

    fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,5))

    for col,inf_scenario in enumerate(inf_scenarios):
        for row,LOD_scenario in enumerate(LOD_scenarios):
            # get axis to plot on
            axnum = str(axis_finder[row][col])
            ax = axes[axnum]

            # read in data for each pathogen
            COVID = read_data(data_path / "COVID_fig2_" /  inf_scenario / "inf_" / LOD_scenario / "LOD")
            RSV = read_data(data_path / "RSV_fig2_" /  inf_scenario / "inf_" / LOD_scenario / "LOD")
            FLU = read_data(data_path / "FLU_fig2_" /  inf_scenario / "inf_" / LOD_scenario / "LOD")

            labs = [ "Weekly RDT\nscreening\n50% comply",\
                    "2 RDTs\npost-sympt.\n0d TAT", \
                    "1 PCR 2-7d\npost-expos.\n2d TAT"]; xvals = np.array([1,3,5]);
            w=0.5; dy = 0.005; f_idx = int(20/.001)
            plt.rcParams['hatch.linewidth'] = 2  # hatch linewidth
            N = 1000 # number of VL draws used in fiji simulations - used for averaging here
            c = 1; f = 0.05; # constants for experiment sims
            
            a = ax.bar(xvals-w,RSV,color=RSV_colors[4],width=w,edgecolor=ALMOST_BLACK)
            b = ax.bar(xvals,FLU,color=FLU_colors[3],width=w,edgecolor=ALMOST_BLACK)
            c = ax.bar(xvals+w,COVID,color=COVID_colors[4],width=w,edgecolor=ALMOST_BLACK)
            
            # bottom axes
            if axnum in ["C","D"]:  
                ax.set_xticklabels(labs);
            else:
                ax.set_xticklabels([])  
            
            # left axes
            if axnum in ["A","C"]:
                ax.set_ylabel("Testing effectiveness",size=LABEL_SIZE-4)
            else:
                ax.set_yticklabels([])
            
            # sanity check to make sure scenarios are correct
            # ax.text(0,0.4,"{}Inf,{}Sens".format(inf_scenario,LOD_scenario))

            ax.set_xticks(xvals); ax.set_ylim([0,0.5]); 
            # y-axis grid lines 
            ax.grid(axis="y",zorder=3)
            ax.set_axisbelow(True)
            # finalize
            finalize(ax,aspect=1.75)

    axes["A"].legend([a,b,c],["RSV","Influenza A","SARS-CoV-2"],loc='upper left',frameon = True)
    plt.tight_layout()

    fname = fig_path / "Fig2_sensitivity_nolabs.png"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    fname = fig_path /  "Fig2_sensitivity_nolabs.pdf"
    plt.savefig(fname,dpi=300,bbox_inches="tight")
    plt.show()

def sensitivity_Fig4_generate_data():
    ''' TE heatmap for symptom based and exposure based reflex testing '''
    # creating a 2x2 grid combining increased/decreased infectiousness thresholds and increased/decreased RDT LOD
    inf_scenarios = ["Low","High"]; # lower and higher infectious thresholds
    LOD_scenarios = ["Low","High"] # lower sensitivit [higher LOD] test and higher sensitivity [lower LOD] test
    pathogens = ["RSV","FLU","COVID"]

    wait_times_symp = np.arange(0,6,1)
    num_tests = np.arange(6,0,-1)

    for inf_scenario in inf_scenarios:
        
        # read in data for each of the three figure 1 scenarios
        Fig4 = pd.read_csv("{}sens_{}Inf/Figure4A_{}Inf.csv".format(sim_path,inf_scenario,inf_scenario))

        for LOD_scenario in LOD_scenarios:
            for path in pathogens:
                path_string = get_kinetics_string(path)
                p = get_pct_symptomatic(path)
                LOD = get_sensitivity_scenario_LOD(path,LOD_scenario)
                df = pd.DataFrame()
                for w in wait_times_symp:
                    scenario_results = []
                    for n in num_tests:
                        data = Fig4.loc[(Fig4['kinetics']==path_string) & 
                                    (Fig4['L']==LOD) & 
                                    (Fig4['wait']== w) & 
                                    (Fig4['supply']==n)]
                        if len(data) > 1:
                            # if there are two entries, check if they are just repeats of the same experiment
                            if len(data) == 2 and data['simulation'].values[0] == data['simulation'].values[1]:
                                pass
                            else:
                                raise Exception("You didn't query me enough :(")
                        scenario_results.append(p*data['TE'].values[0])
                    df_string = "wait" + str(w)
                    df[df_string] = scenario_results
                # Save data
                fname = "{}_fig4_{}inf_{}LOD".format(data_path / path, inf_scenario, LOD_scenario)
                save_data(df,fname)
def sensitivity_Draw_Fig4():
    ''' Prompted testing with different wait times and number of daily tests '''

    num_tests = np.arange(1,7)

    inf_scenarios = ["Low","High"]; # lower and higher infectious thresholds
    LOD_scenarios = ["Low","High"] # lower RDT LOD and higher RDT LOD

    # figure mosaic coder
    # Low sens/low inf thresh       low sens/ high inf thresh
    # High sens/low inf thresh       high sens/ high inf thresh
    axis_finder = [[["A","B","C"],["D","E","F"]],[["H","I","J"],["K","L","M"]]]

    for col,inf_scenario in enumerate(inf_scenarios):
        for row,LOD_scenario in enumerate(LOD_scenarios):
            figure_mosaic = """
            ABC
            DEF
            DEF
            DEF
            DEF
            DEF
            """

            fig,axes = plt.subplot_mosaic(mosaic=figure_mosaic,figsize=(10,4))

            # read in data for each pathogen
            COVID = read_data(data_path / "COVID_fig4_" /  inf_scenario / "inf_" / LOD_scenario / "LOD")
            RSV = read_data(data_path / "RSV_fig4_" /  inf_scenario / "inf_" / LOD_scenario / "LOD")
            FLU = read_data(data_path / "FLU_fig4_" /  inf_scenario / "inf_" / LOD_scenario / "LOD")

            wait_times = np.arange(0,6,1)
            centers = np.min(wait_times), np.max(wait_times), np.min(num_tests), np.max(num_tests)
            extent = create_extent(COVID,centers)

            max_reduction_COVID = 0.5
            min_reduction_COVID = 0
            max_reduction_FLU = 0.5
            min_reduction_FLU = 0
            max_reduction_RSV = 0.5
            min_reduction_RSV = 0

            ## Plotting data   -------------------------------------------------------------------------
            im_RSV = axes["D"].imshow(RSV,extent=extent,cmap=RSV_cmap,vmin=min_reduction_RSV,vmax=max_reduction_RSV)
            cbar_RSV = plt.colorbar(im_RSV,ax=axes["A"],fraction=0.7,orientation="horizontal")
            cbar_RSV.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
            axes["A"].text(0.5,0.7,'RSV Testing effectiveness',ha='center',size=LABEL_SIZE-2)
            # plot stars over max TE
            for n in range(max(num_tests)):
                TEs = []
                for ii,row in enumerate(RSV):
                    TEs.append(RSV[row][n])
                max_TE = TEs.index(max(TEs))
                axes["D"].scatter(max_TE,len(num_tests)-n,c="white",marker="*")

            im_FLU = axes["E"].imshow(FLU,extent=extent,cmap=FLU_cmap,vmin=min_reduction_FLU,vmax=max_reduction_FLU)
            cbar_FLU = plt.colorbar(im_FLU,ax=axes["B"],fraction=0.7,orientation="horizontal")
            cbar_FLU.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
            axes["B"].text(0.5,0.7,'Influenza Testing effectiveness',ha='center',size=LABEL_SIZE-2)
            # plot stars over max TE
            for n in range(max(num_tests)):
                TEs = []
                for ii,row in enumerate(FLU):
                    TEs.append(FLU[row][n])
                max_TE = TEs.index(max(TEs))
                axes["E"].scatter(max_TE,len(num_tests)-n,c="white",marker="*")

            im_COVID = axes["F"].imshow(COVID,extent=extent,cmap=COVID_cmap,vmin=min_reduction_COVID,vmax=max_reduction_COVID)
            cbar_COVID = plt.colorbar(im_COVID,ax=axes["C"],fraction=0.7,orientation="horizontal")
            cbar_COVID.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
            axes["C"].text(0.5,0.7,'SARS-CoV-2 Testing effectiveness',ha='center',size=LABEL_SIZE-2)
            # plot stars over max TE
            for n in range(max(num_tests)):
                TEs = []
                for ii,row in enumerate(COVID):
                    TEs.append(COVID[row][n])
                max_TE = TEs.index(max(TEs))
                axes["F"].scatter(max_TE,len(num_tests)-n,c="white",marker="*")

            x_axes = [axes["D"],axes["E"],axes["F"]]
            for a in x_axes:
                a.set_xticks([0,1,2,3,4,5])
            
            axes["D"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)
            axes["E"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)
            axes["F"].set_xlabel("Delay before testing\n(days post-Sx)",size=LABEL_SIZE)

            axes["D"].set_yticks([1,2,3,4,5,6])
            axes["D"].set_ylabel("Rapid tests available")

            for a in [axes["E"],axes["F"]]:
                a.set_yticks([])

            for a in [axes["D"],axes["E"],axes["F"]]:
                finalize_keep_frame(a)
                force_aspect(a,1)
            
            for a in [axes["A"],axes["B"],axes["C"]]:
                remove_ax(a)

            xlabpad = -0.15; ylabpad = 2.4;
            #label_subplots(axes,x_pads=[xlabpad,xlabpad,xlabpad],y_pad=ylabpad,labels=["A","B","C"],fontsize=LABEL_SIZE)
            #plt.subplots_adjust(hspace=-0.4) # space between rows

            fname = "{}Fig4_sensitivity_{}Inf_{}LOD.png".format(fig_path,inf_scenario,LOD_scenario)
            plt.savefig(fname,dpi=300,bbox_inches="tight") # bbox_inches prevents x-label from being cutoff
            fname = "{}Fig4_sensitivity_{}Inf_{}LOD.pdf".format(fig_path,inf_scenario,LOD_scenario)
            plt.savefig(fname,dpi=300,bbox_inches="tight")
            plt.show()



if __name__ == "__main__":
    main()
