# -*- coding: utf-8 -*-

# %% Imports
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from aux_EcoPG import R, T
from aux_EcoPG import N, Nd, b, f, c, mi, qc, qr
from aux_EcoPG import bc_reward_subs, fc_reward_subs, q_subs
from con_Game import obtain_con_sols_Game, yi 
from con_Game import con0_style, con1_style, con2_style

# %% Parameters
ysym = mi  # the symbol for the y-axis 
xsym = N   # the symbol for the x-axis
paramsubsis = {Nd: 0, b: 3, f: 1.2, c: 5}

# obtain solutions to ciritical curves
sol0, sol1, sol2 = obtain_con_sols_Game(R, T.subs(q_subs), ysym=ysym)


# %% prep
def lamprep(sol, reward_subs, paramsubsis):
    return sp.lambdify((xsym), sol.subs(reward_subs).subs(paramsubsis))

#  plot all three ciritical curves for specific parameter values
def plot_triple(x_vs, sol0=None, sol1=None, sol2=None,
                qc_v=-1, yi_v=-1, qr_v=-1, reward_subs=bc_reward_subs,
                style_add=[{}, {}, {}], style_feature={},
                paramsubsis=paramsubsis, ax=plt.gca()):

    def prep(sol):
        return lamprep(sol, reward_subs,
                       {yi: yi_v, qc: qc_v, qr: qr_v, **paramsubsis})(x_vs)\
            * np.ones_like(x_vs)

    if sol0 is not None:
        ax.plot(x_vs, prep(sol0), **{**con0_style, **style_add[0],
                                     **style_feature})
    if sol1 is not None:
        ax.plot(x_vs, prep(sol1), **{**con1_style, **style_add[1],
                                     **style_feature})
    if sol2 is not None:
        ax.plot(x_vs, prep(sol2), **{**con2_style, **style_add[2],
                                     **style_feature})

# %% MAIN PLOT

def plot_for_y(yi_v, ax1, ax2):
    N_vs = np.arange(2, 250)
    mul = 1/N

    ax1.plot([0, max(N_vs)], [0, 0], color="red", alpha=0.5)
    ax2.plot([0, max(N_vs)], [0, 0], color="red", alpha=0.5)

    styYH = [{"color": "k"}, {"color": "blue"}, {"color": "green"}]
    linestlyes = [":", "-", "--"]

    rsubs = fc_reward_subs  # reward subsitution sceme

    for i, f_v in enumerate([0.0, 1.2, 2.4]):
        paramsubsis[f] = f_v
        ls = linestlyes[i]

        plot_triple(N_vs, mul*sol0[0], mul*sol1[0], mul*sol2[0],
                    yi_v=yi_v, qc_v=0.02, qr_v=0.0001, reward_subs=rsubs,
                    style_add=styYH,
                    style_feature={"ls": ls, "marker": ".", "alpha": 0.75},
                    ax=ax1)

        plot_triple(N_vs, mul*sol0[0], mul*sol1[0], mul*sol2[0],
                    yi_v=yi_v, qc_v=0.02, qr_v=0.0001, reward_subs=rsubs,
                    style_add=styYH, style_feature={"ls": ls, "alpha": 0.75},
                    ax=ax2)

fsf = 1.0  # figure scale factor
fig = plt.figure(figsize=(fsf*6, fsf*4))
# plot axes paramters
le = 0.12; ri = 0.96; to = 0.7; bo = 0.12; hmi = 0.58; vs = 0.1; hs = 0.02
ly = 1.1; dlx = 0.
# axes
ax11 = fig.add_axes([le, bo, hmi-le-0.5*hs, to-bo])
ax12 = fig.add_axes([hmi+0.5*hs, bo, ri-hmi-0.5*hs, to-bo ])

plot_for_y(yi_v=0.99, ax1=ax11, ax2=ax12)  # the plot

# decorations
ax12.spines["right"].set_visible(False)
ax12.spines["top"].set_visible(False)
ax11.spines["top"].set_visible(False)
ax11.spines["right"].set_visible(False)
ax11.set_xlim(1.6, 5.5); ax12.set_xlim(5.5, 250)
ylim1 = (-7.8, 5.6); ax11.set_ylim(*ylim1); ax12.set_ylim(*ylim1)
ax12.set_yticklabels([])
ax11.set_ylabel(r"Collapse impact per actor $m/N$")
# grey out impact values above zero
ax11.fill_between([0, 300], [6, 6], color="k", alpha=0.2)
ax12.fill_between([0, 300], [6, 6], color="k", alpha=0.2)

ax11.annotate(r"Number of actors $N$", xy=(le+(ri-le)/2, bo-0.07),
              xycoords="figure fraction", va="top", ha="center")


# Legend
legend_elements1 = [Line2D([0], [0], marker='o', color='w',
                           label=r'Dilemma',
                           markerfacecolor='k', markersize=8),
                    Line2D([0], [0], marker='o', color='w',
                           label=r'Greed',
                           markerfacecolor='blue', markersize=8),
                    Line2D([0], [0], marker='o', color='w',
                           label='Fear',
                           markerfacecolor='green', markersize=8)]

legend_elements2 = [Line2D([0], [0], marker='.', color='darkgray', ls=":",
                           label=r'$f = 0$'),
                    Line2D([0], [0], marker='.', color='darkgray', ls="-",
                           label=r'$f = 1.2$'),
                    Line2D([0], [0], marker='.', color='darkgray', ls="--",
                           label=r'$f = 2.4$')]

legend1 = ax11.legend(handles=legend_elements1, bbox_to_anchor=(0.5-dlx, ly),
                      loc='lower left',
                      borderaxespad=0., frameon=False)
legend2 = ax12.legend(handles=legend_elements2, bbox_to_anchor=(0.5+dlx, ly),
                      loc='lower right',
                      borderaxespad=0., frameon=False)

plt.savefig("../figs/fig04_MvsN.png", dpi=300)




#%% fc bc reward schemes comparison for SI

def plot_for_y(yi_v, ax1, ax2):
    styYH = [{"color": "k"}, {"color": "blue"}, {"color": "green"}]
    N_vs = np.arange(2, 250)
    mul = 1 / N

    ax1.plot([0, max(N_vs)], [0, 0], color="red", alpha=0.5)
    ax2.plot([0, max(N_vs)], [0, 0], color="red", alpha=0.5)

    rsubs = fc_reward_subs
    plot_triple(N_vs, mul*sol0[0], mul*sol1[0], mul*sol2[0],
                yi_v=yi_v, qc_v=0.02, qr_v=0.0001, reward_subs=rsubs,
                style_add=styYH, style_feature={"ls": "-", "marker": "."},
                ax=ax1)
    plot_triple(N_vs, mul*sol0[0], mul*sol1[0], mul*sol2[0],
                yi_v=yi_v, qc_v=0.02, qr_v=0.0001, reward_subs=rsubs,
                style_add=styYH, style_feature={"ls": "-"},
                ax=ax2)

    rsubs = bc_reward_subs
    plot_triple(N_vs, mul*sol0[0], mul*sol1[0], mul*sol2[0],
                yi_v=yi_v, qc_v=0.02, qr_v=0.0001, reward_subs=rsubs,
                style_add=styYH, style_feature={"ls": "--", "marker": "."},
                ax=ax1)
    plot_triple(N_vs, mul*sol0[0], mul*sol1[0], mul*sol2[0],
                yi_v=yi_v, qc_v=0.02, qr_v=0.0001, reward_subs=rsubs, ax=ax2,
                style_add=styYH, style_feature={"ls": "--"})


fsf = 0.9 # figure scale factor
fig = plt.figure(figsize=(fsf*6, fsf*6))
# plot axes paramters 
le = 0.13; ri = 0.96; to = 0.8; bo = 0.11; hmi = 0.58; vmi = bo + (to-bo)/2
vs = 0.1; hs = 0.02; ly = 1.2; dlx = 0.2
# axes
ax21 = fig.add_axes([le, bo, hmi-le-0.5*hs, vmi-bo-0.5*hs])
ax22 = fig.add_axes([hmi+0.5*hs, bo, ri-hmi-0.5*hs, vmi-bo-0.5*hs])
ax11 = fig.add_axes([le, vmi+0.5*vs, hmi-le-0.5*hs, to-vmi-0.5*vs])
ax12 = fig.add_axes([hmi+0.5*hs, vmi+0.5*vs, ri-hmi-0.5*hs, to-vmi-0.5*vs])


yi1 = 0.99  # do the plot
plot_for_y(yi1, ax11, ax12)
ax12.annotate(f"$\gamma = {yi1}$", xy=(-0.15, 0.25), xycoords="axes fraction",
              bbox=dict(boxstyle='square', fc='white'))

yi2 = 0.95  # do the plot
plot_for_y(yi2, ax21, ax22)
ax22.annotate(f"$\gamma = {yi2}$", xy=(-0.15, 0.75), xycoords="axes fraction",
              bbox=dict(boxstyle='square', fc='white'))

# decorations
ax12.spines["right"].set_visible(False)
ax12.spines["top"].set_visible(False)
ax11.spines["top"].set_visible(False)
ax11.spines["right"].set_visible(False)
ax22.spines["right"].set_visible(False)
ax22.spines["top"].set_visible(False)
ax21.spines["top"].set_visible(False)
ax21.spines["right"].set_visible(False)
xlim1 = (1.6, 8.5); ax11.set_xlim(*xlim1); ax21.set_xlim(*xlim1)
xlim2 = (8.5, 250); ax12.set_xlim(*xlim2); ax22.set_xlim(*xlim2)
ylim1 = (-7.8, 4.6); ax11.set_ylim(*ylim1); ax12.set_ylim(*ylim1)
ylim2 = (-17.5, 10.5); ax21.set_ylim(*ylim2); ax22.set_ylim(*ylim2)
ax11.set_xticklabels([]); ax12.set_xticklabels([])
ax12.set_yticklabels([]); ax22.set_yticklabels([])

ax11.annotate(r"Collapse impact per actor $m/N$", xy=(le-0.09, bo+(to-bo)/2),
              xycoords="figure fraction", ha="right", va="center",
              rotation=90)
ax21.annotate(r"Number of actors $N$", xy=(le+(ri-le)/2, bo-0.07),
              xycoords="figure fraction", va="top", ha="center")


# Legend
legend_elements1 = [Line2D([0], [0], marker='.', color='k', ls="--",
                           label=r'Dilemma (bc)'),
                    Line2D([0], [0], marker='.', color='blue', ls="--",
                           label=r'Greed (bc)'),
                    Line2D([0], [0], marker='.', color='green', ls="--",
                           label=r'Fear (bc)')]

legend_elements2 = [Line2D([0], [0], marker='.', color='k', ls="-",
                           label=r'Dilemma (fc)'),
                    Line2D([0], [0], marker='.', color='blue', ls="-",
                           label=r'Greed (fc)'),
                    Line2D([0], [0], marker='.', color='green', ls="-",
                           label=r'Fear (fc)')]

legend1 = ax11.legend(handles=legend_elements2, bbox_to_anchor=(0.5-dlx, ly),
                      loc='lower left',
                      borderaxespad=0., frameon=False)
legend2 = ax12.legend(handles=legend_elements1, bbox_to_anchor=(0.5+dlx, ly),
                      loc='lower right',
                      borderaxespad=0., frameon=False)

plt.savefig("../figs/SIfig_MvsN_fcbc.png", dpi=300)

