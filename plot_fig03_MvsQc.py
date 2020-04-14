# -*- coding: utf-8 -*-

# %% Imports
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from aux_EcoPG import R, T
from aux_EcoPG import rti, rpi, rsi, rri
from aux_EcoPG import N, Nd, f, c, mi, qc, qr
from aux_EcoPG import fc_reward_subs, q_subs
from con_Game import obtain_con_sols_Game, yi, obtain_con_sols_NFGame
from con_Game import con0_style, con1_style, con2_style

# %% Parameters
ysym = mi  # the symbol for the y-axis 
xsym = qc  # the symbol for the x-axis
reward_subs = fc_reward_subs  # specific reward subsitution sceme
paramsubsis = {N: 2, Nd: 0, f: 1.2, c: 5}

# obtain solutions to ciritical curves
sol0, sol1, sol2 = obtain_con_sols_Game(R, T.subs(q_subs), ysym=ysym)


# %% Plot
def lamprep(sol, paramsubsis):
    return sp.lambdify((xsym), sol.subs(reward_subs).subs(paramsubsis))

#  plot all three ciritical curves for specific parameter values
def plot_triple(sol0=None, sol1=None, sol2=None,
                yi_v=-1, qr_v=-1, style_add=[{}, {}, {}], style_feature={},
                paramsubsis=paramsubsis):
    qc_vs = np.linspace(0.00000000001, 1.0, 10001)

    def prep(sol):
        return lamprep(sol, {yi: yi_v, qr: qr_v, **paramsubsis})(qc_vs)\
            * np.ones_like(qc_vs)

    if sol0 is not None:
        plt.plot(qc_vs, prep(sol0), **{**con0_style, **style_add[0],
                                       **style_feature})
    if sol1 is not None:
        plt.plot(qc_vs, prep(sol1), **{**con1_style, **style_add[1],
                                       **style_feature})
    if sol2 is not None:
        plt.plot(qc_vs, prep(sol2), **{**con2_style, **style_add[2],
                                       **style_feature})

fsf = 0.65  # figure scale factor
plt.figure(figsize=(fsf*9, fsf*6))
# grey out impact values above zero
plt.fill_between([0, 1], [4, 4], color="k", alpha=0.2)

yH = 0.99  # plot for high discount factor value
styYH = [{"color": "k"}, {"color": "darkblue"}, {"color": "darkgreen"}]
plot_triple(sol0[0], sol1[0], sol2[0], yi_v=yH, qr_v=0.0001,
            style_add=styYH, style_feature={"ls": "-"})
plot_triple(sol0[0], sol1[0], sol2[0], yi_v=yH, qr_v=0.01,
            style_add=styYH, style_feature={"ls": "--", "lw": 1.5})
plot_triple(sol0[0], sol1[0], sol2[0], yi_v=yH, qr_v=0.1,
            style_add=styYH, style_feature={"ls": ":", "lw": 2.5})

yL = 0  # plot for low discount factor value
styYL = [{"color": "lightgray"}, {"color": "lightblue"},
         {"color": "lightgreen"}]
plot_triple(sol0[0], sol1[0], sol2[0], yi_v=yL, qr_v=0.0001,
            style_add=styYL, style_feature={"ls": "-"})
plot_triple(sol0[0], sol1[0], sol2[0], yi_v=yL, qr_v=0.01,
            style_add=styYL, style_feature={"ls": "--", "lw": 1.5})
plot_triple(sol0[0], sol1[0], sol2[0], yi_v=yL, qr_v=0.1,
            style_add=styYL, style_feature={"ls": ":", "lw": 2.5})

# decorations
plt.ylim(-7, 2)
plt.xlim(0, 1)
plt.xlabel(r"Collapse leverage $q_c$")
plt.ylabel(r"Collapse impact $m$")

# LEGEND
legend_elements = [
                   Line2D([0], [0], marker='o', color='w',
                          label=r'Dilemma at $\gamma=0.99$',
                          markerfacecolor='k', markersize=10),
                   Line2D([0], [0], marker='o', color='w',
                          label=r'Greed at $\gamma=0.99$',
                          markerfacecolor='darkblue', markersize=10),
                   Line2D([0], [0], marker='o', color='w',
                          label='Fear at $\gamma=0.99$',
                          markerfacecolor='darkgreen', markersize=10),
                   Line2D([0], [0], marker='o', color='w',
                          label=r'Dilemma at $\gamma=0$',
                          markerfacecolor='lightgray', markersize=10),
                   Line2D([0], [0], marker='o', color='w',
                          label=r'Greed at $\gamma=0$',
                          markerfacecolor='lightblue', markersize=10),
                   Line2D([0], [0], marker='o', color='w',
                          label='Fear at $\gamma=0$',
                          markerfacecolor='lightgreen', markersize=10)]

legend_elements2 = [Line2D([0], [0], color='k', lw=1,
                           label=r'$q_r = 0.0001$'),
                    Line2D([0], [0], color='k', ls="--", lw=1.5,
                           label=r'$q_r = 0.01$'),
                    Line2D([0], [0], color='k', ls=":", lw=2.5,
                           label=r'$q_r = 0.1$')]

legend1 = plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1.0),
                     loc="upper left", ncol=1,
                     borderaxespad=0., frameon=True)
legend2 = plt.legend(handles=legend_elements2, bbox_to_anchor=(0.0, 1.02),
                     loc="lower left", borderaxespad=0, ncol=3, frameon=True)

plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

plt.subplots_adjust(top=0.89, bottom=0.13, left=0.11, right=0.64, hspace=0.2,
                    wspace=0.2)

plt.savefig("figs/fig03_MvsQc.png", dpi=300)



# %% Normal Form Game Plot for SI

# solutions of critical curves for the normal form game versions
def NFsols(inclusive, threshold, ysym=None):
    """
    reward combi: I(nclusive) or E(xclusive)
    coll:  T(threshold) or M(arginal)
    """
    E = not inclusive
    T = threshold

    colp = qc if T else qc/N
    COLp = qc if T else (N-1)/N * qc

    reward = rri
    temptation = (1 if not E else (1-colp)) * rti + colp*mi
    sucker = (1 if not E else (1-COLp)) * rsi + COLp*mi
    punishment = (1 if not E else (1-qc)) * rpi + qc*mi

    sols = obtain_con_sols_NFGame(*map(lambda expr: expr.subs(reward_subs),
                                  [reward, temptation, sucker, punishment]),
                                  ysym=ysym)
    return sols


plt.figure(figsize=(6, 4))

# plot inclusive reward with marginal collapse risk
IMs0, IMs1, IMs2 = NFsols(inclusive=True, threshold=False, ysym=ysym)
plot_triple(sol0=IMs0[0], sol1=IMs1[0], sol2=IMs2[0], style_add=styYL,
            style_feature={"ls": "-", "color":"lightseagreen", "alpha": 1.0,
                           "label": "IM"})

# plot inclusive reward with thresholded collapse risk
ITs0, ITs1, ITs2 = NFsols(inclusive=True, threshold=True, ysym=ysym)
plot_triple(sol0=ITs0[0], sol1=ITs1[0],
            style_feature={"color": "mediumpurple", "label": "IT"})

# plot exclusive reward with thresholded collapse risk
ETs0, ETs1, ETs2 = NFsols(inclusive=False, threshold=True, ysym=ysym)
plot_triple(sol0=ETs0[0], sol1=ETs1[0],
            style_feature={"color": "deeppink", "label": "ET"})

# plot exclusive reward with marginal collapse risk
EMs0, EMs1, EMs2 = NFsols(inclusive=False, threshold=False, ysym=ysym)
plot_triple(EMs0[0], EMs1[0], EMs2[0],
            style_add=[{"color": "k"}, {"color": "lightblue"},
                       {"color": "lightgreen"}],
            style_feature={"ls": "-", "lw": "1.5", "alpha": 1.0,
                           "label": "EM"})

# LEGEND
legend_elements = [Line2D([0], [0], color='k',
                          label=r'Dilemma for all games'),
                   Line2D([0], [0], color='lightblue',
                          label=r'Greed (EM)'),
                   Line2D([0], [0], color='lightgreen',
                          label='Fear (EM)'),
                   Line2D([0], [0], color='lightseagreen',
                          label='Greed & Fear (IM)'),
                   Line2D([0], [0], color='mediumpurple',
                          label='Greed (IT)'),
                   Line2D([0], [0], color='deeppink',
                          label='Greed (ET)')]
plt.legend(handles=legend_elements, frameon=False)

# decorations
plt.ylim(-7, 2)
plt.xlim(0, 1)
plt.xlabel(r"Collapse risk leverage $q_c$")
plt.ylabel(r"Collapse impact $m$")
plt.tight_layout()

plt.savefig("figs/SIfig_MvsQc.png", dpi=300)
