# -*- coding: utf-8 -*-

# %% imports
from aux_EcoPG import N, Nd, f, c, mi, mj, qc, qr
from aux_EcoPG import fc_reward_subs, q_subs
from con_Emergence import obtain_conEmergence, m, y, Xg

from agents.detAC import detAC
from aux_QuiverPlot import plot_quiver, plot_trajectory
from aux_EcoPG import R, T

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import copy


# %% parameters
gammin = 0.953  # gamma limit minimum 
gammax = 1.0 # gamma limit maximum
reward_subs = fc_reward_subs  # specific reward subsitution scheme

N_v, Nd_v, f_v, c_v, qc_v, qr_v = 2, 0, 1.2, 5, 0.02, 0.0001  # param. values
paramsubsis = {N: N_v, Nd: Nd_v, f: f_v, c: c_v, qc: qc_v, qr: qr_v} 

# labels 
ylab = u"$X^2_{s\mathsf{c}}$"; ylabP = u"$X^2_{\mathsf{pc}}$"
xlabP = u"$X^1_{\mathsf{pc}}$"; xlabD = u"$X^1_{\mathsf{gc}}$"


#%% analytical calcuation with sympy
    
# behavior space: separatic condition for Emergence of cooperation
conEm = obtain_conEmergence(qc_vs=[qc, qc], m_vs=[m, m], qr_vs=[qr, qr],
                            Xsubs={}, reward_subs=reward_subs)
# set cooperation in degraded state
X_1o0 = {Xg[0, 0, 0]: 1.0, Xg[1, 0, 0]: 1.0}
# solve strategy separatix for discount factor gamma 
gam_of = sp.solve(conEm, y)
# lambdify solution 
lam_gam = sp.lambdify((Xg[0, 1, 0], Xg[1, 1, 0], qc, qr, m),
                      gam_of[1].subs(X_1o0))
# solve strategy separatix for strategy X
X_of = sp.solve(conEm, Xg[1,1,0])
# lambdify solution
lam_X0 = sp.lambdify((Xg[0,1,0], y, qc, qr, m),
                     X_of[0].subs(X_1o0).simplify())


# %% quiver plot
def obt_behavior(X000, X100, X010, X110):
    Xinit = np.zeros((2, 2, 2))
    Xinit[0, 0, 0] = X000
    Xinit[1, 0, 0] = X100
    Xinit[0, 1, 0] = X010
    Xinit[1, 1, 0] = X110

    Xinit[0, 0, 1] = 1.0 - Xinit[0, 0, 0]
    Xinit[1, 0, 1] = 1.0 - Xinit[1, 0, 0]
    Xinit[0, 1, 1] = 1.0 - Xinit[0, 1, 0]
    Xinit[1, 1, 1] = 1.0 - Xinit[1, 1, 0]

    return Xinit

def _plot_lamX(ax, y_v, m_v, style):
    xlim = ax.get_xlim(); ylim = ax.get_ylim()

    X0sC = np.linspace(0, 1, 101)
    ax.plot(X0sC, lam_X0(X0sC, y_v, qc_v, qr_v, m_v), **style)

    ax.set_xlim(xlim); ax.set_ylim(ylim)

# quiver plots
def qplot(fig, xc, yc, size, m_v, y_v, Label="", plotTrajectories=False,
          test=False):
    
    figfrac = fig.get_figwidth() / fig.get_figheight()
    gs = gridspec.GridSpec(1, 2)
    wsp = 0.07  # space parameter
    gs.update(wspace=wsp, left=xc, bottom=yc,
              right=xc+2*(size+0.5*wsp)/figfrac, top=yc+size)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])

    # agent parameters
    alpha = 0.2; beta = 100.0; gammas = np.array([y_v, y_v])
    # init agents
    agents = detAC(np.array(T.subs(q_subs).subs(paramsubsis).tolist())
                   .astype(float),
                   np.array(R.subs(reward_subs).subs(paramsubsis)
                             .subs({mi: m_v, mj: m_v}).tolist())
                   .astype(float),
                   alpha, beta, gammas)
    
    # plot quiver
    if test:
        pAs = [0.0, 0.3, 0.7, 1.0]  # for testing
    else:
        pAs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax1, ax2 = plot_quiver(agents, axes=[ax1, ax2], pAs=pAs, sf=0.22)

    # separatirx
    _plot_lamX(ax2, y_v, m_v, style={"color": "red", "ls": "--"})

    # Trajectories
    if plotTrajectories:
        XisaS = [0.25, 0.5, 0.75]
        for X0s0 in XisaS:
            for X1s0 in XisaS:
                X1 = obt_behavior(X000=X0s0, X100=X1s0, X010=X0s0, X110=X1s0)
                rt, fpr = plot_trajectory(agents, X1, axes=[ax1, ax2],
                                          Tmax=7500, color="k", alpha=0.75,
                                          lw=0.5, ms=0.5)
    # decorations
    for ax in [ax1, ax2]:
        ax.set_title("")
        ax.set_yticks([0, 1]); ax.set_xticks([0, 1])
        ax.set_ylim(0, 1); ax.set_xlim(0, 1)
        ax.yaxis.labelpad = -10; ax.xaxis.labelpad = -10
    ax1.set_xlabel(xlabD); ax1.set_ylabel(ylab)
    ax2.set_xlabel(xlabP); ax2.set_yticklabels(())

    ax1.annotate(r"de$\mathsf{g}$raded", (0.0, 1.0),
                 textcoords="axes fraction", ha="left", va="bottom")
    ax2.annotate(r"$\mathsf{p}$rosperous", (1.0, 1.0),
                 textcoords="axes fraction", ha="right", va="bottom")
    
    bbox_props = dict(boxstyle="round", fc="gray", alpha=0.9, lw=1)
    ax2.annotate(Label, (-wsp/2, 1.0+wsp), xycoords="axes fraction",
                 color="w", ha="center", va="bottom", bbox=bbox_props)

    return ax1, ax2


# %% critial gamma plot
def yplot(fig, xc, yc, m_v, Label="", Lcol="k", lenY=0.16):

    figfrac = fig.get_figwidth() / fig.get_figheight()
    ax = fig.add_axes([xc, yc, 2*lenY/figfrac, 2*lenY])

    # plotting data
    xs = np.linspace(0, 1, 1001); ys = np.linspace(0, 1, 1001)
    X, Y = np.meshgrid(xs, ys)
    Z = lam_gam(X, Y, qc_v, qr_v, m_v)

    # colormap
    palette = copy(plt.cm.Reds)
    palette.set_over('k', 1.0);
    palette.set_under('w', 1.0)
    palette.set_bad('b', 1.0)
    GamProps = {"cmap": palette, "vmin": gammin, "vmax": gammax}

    # plot
    cb = ax.pcolormesh(X, Y, Z, **GamProps)

    # decorations
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel(xlabP); ax.set_ylabel(ylabP)
    ax.yaxis.labelpad = -10; ax.xaxis.labelpad = -10
    ax.annotate(r"$\mathsf{p}$rosperous", (1.0, 1.0),
                textcoords="axes fraction", ha="right", va="bottom")
    ax.annotate(Label, xy=(0.025, 0.975), fontsize="large", fontweight=750,
                xycoords="axes fraction", ha="center", va="top", color=Lcol)

    return cb, ax


# %% the plot
test = False  # test plot - use reduced computation for faster execution

fsf = 0.56
fig = plt.figure(num=1, figsize=(fsf*10, fsf*12))

# parameters
mH, mM, mL = -6,  -3, -1
yval = 0.99
pT = True  # plot trajectories
hspace = 0.33

# Plot mild collapse impact
qaxes = qplot(fig, 0.05, 0.05+2*hspace, 0.16, mL, yval, Label="Tragedy",
              plotTrajectories=pT, test=test)

cb, gaxL = yplot(fig, 0.585, 0.05+2*hspace, mL, Lcol="white", lenY=0.1)

_plot_lamX(gaxL, yval, mL, style={"color": "w", "ls": "--"})

gaxL.annotate("A", xy=(0.035, 0.28+2*hspace),
             fontsize="large", fontweight=750, 
             xycoords="figure fraction", ha="left", va="center")
gaxL.annotate("Mild collapse impact", xy=(0.07, 0.28+2*hspace),
             fontsize="large", fontweight=500, 
             xycoords="figure fraction", ha="left", va="center")
gaxL.annotate(f"(m={mL})", xy=(0.39, 0.28+2*hspace),
             fontsize="medium", fontweight=300, 
             xycoords="figure fraction", ha="left", va="center")

# Plot medium collapse impacts
qaxes = qplot(fig, 0.05, 0.05+hspace, 0.16, mM, yval, Label="Coordination",
              plotTrajectories=pT, test=test)

cb, gaxM = yplot(fig, 0.585, 0.05+hspace, mM, Lcol="white", lenY=0.1)

_plot_lamX(gaxM, yval, mM, style={"color": "w", "ls": "--"})

gaxM.annotate("B", xy=(0.035, 0.28+1*hspace),
             fontsize="large", fontweight=750, 
             xycoords="figure fraction", ha="left", va="center")
gaxM.annotate("Medium collapse impact", xy=(0.07, 0.28+1*hspace),
             fontsize="large", fontweight=500, 
             xycoords="figure fraction", ha="left", va="center")
gaxM.annotate(f"(m={mM})", xy=(0.445, 0.28+1*hspace),
             fontsize="medium", fontweight=300, 
             xycoords="figure fraction", ha="left", va="center")

# Plot severe collapse impacts
qaxes = qplot(fig, 0.05, 0.05, 0.16, mH, yval, Label="Comedy",
              plotTrajectories=pT, test=test)

cb, gaxH = yplot(fig, 0.585, 0.05, mH, Lcol="white", lenY=0.1)

_plot_lamX(gaxH, yval, mH, style={"color": "w", "ls": "--"})

gaxM.annotate("C", xy=(0.035, 0.28+0*hspace),
             fontsize="large", fontweight=750, 
             xycoords="figure fraction", ha="left", va="center")
gaxM.annotate("Severe collapse impact", xy=(0.07, 0.28+0*hspace),
             fontsize="large", fontweight=500, 
             xycoords="figure fraction", ha="left", va="center")
gaxM.annotate(f"(m={mH})", xy=(0.43, 0.28+0*hspace),
             fontsize="medium", fontweight=300, 
             xycoords="figure fraction", ha="left", va="center")
    
# colorbar axes
cbax = fig.add_axes([0.86, 0.1, 0.02, 0.8])
cbar = fig.colorbar(cb, extend="max", extendrect=True,
                    extendfrac=0.05, orientation="vertical", cax=cbax,
                    label=r"Critical discount factor $\gamma_{crit}$")

plt.savefig("figs/fig05_LD.png", dpi=300)
