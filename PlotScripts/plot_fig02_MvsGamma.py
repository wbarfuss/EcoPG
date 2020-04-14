# -*- coding: utf-8 -*-

# %% imports
import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import copy

from aux_EcoPG import R, T
from aux_EcoPG import N, Nd, f, c, mi, qc, qr
from aux_EcoPG import fc_reward_subs, q_subs
from con_Game import obtain_con_sols_Game, yi


# %% parameters

ysym = mi  # the symbol for the y-axis
xsym = yi  # the symbol for the x-axis
xvs = np.linspace(0.94, 0.9999999, 101)  # numeric values for x-axis
gammin = 0.953  # xlimit minimum 
gammax = 1.0 # xlimit maximum
reward_subs = fc_reward_subs  # specific reward subsitution sceme
state = 1  # the envriomental state (1=prosperous)

N_v, Nd_v, f_v, c_v, qc_v, qr_v = 2, 0, 1.2, 5, 0.02, 0.0001  # param. values
paramsubsis = {N: N_v, Nd: Nd_v, f: f_v, c: c_v, qc: qc_v, qr: qr_v} 

# obtain solutions to ciritical curves
sol0, sol1, sol2 = obtain_con_sols_Game(R, T.subs(q_subs), ysym=ysym)

# %% plot preparations
def prep(sol):
    subsis = dict(zip([N, Nd, f, c, qc, qr],
                      [N_v, Nd_v, f_v, c_v, qc_v, qr_v]))
    lam = sp.lambdify((xsym), sol.subs(reward_subs).subs(subsis))
    return lam


# %% THE PLOT
fsf = 0.6  # figure scale factor
fig = plt.figure(num=1, figsize=(fsf*8, fsf*8))
figfrac = fig.get_figwidth() / fig.get_figheight()
                  
# place main axis in the center
xc=0.5; yc=0.5; lenCen=0.39; cbextendfrac=0.05;
ax = fig.add_axes([xc-lenCen/figfrac + 0.04, yc-lenCen + 0.04,
                   2*lenCen/figfrac, 2*lenCen])
# colorbar axes
cbax = fig.add_axes([xc-lenCen/figfrac + 0.04, yc-lenCen-0.01,
                     2*lenCen/figfrac*(1 + cbextendfrac), 0.03])
# grey out impact values above zero
ax.fill_between([0, 1], [4, 4], color="k", alpha=0.2)

# plot critical curves
ax.plot(xvs, prep(sol0[0])(xvs), c="k", label="Dilemma")
ax.plot(xvs, prep(sol1[0])(xvs), c="blue", label="Greed")
ax.plot(xvs, prep(sol2[0])(xvs), c="green", label="Fear")
# place legend
ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.5))

# decorations    
ax.set_ylim(-7, 3)
ax.set_xlim(gammin, gammax)
ax.set_ylabel("Collape imapct m")

ax.annotate("Collapse avoidance\nsuboptimal", xy=(0.98, 0.92),
            xycoords="axes fraction", ha="right", va="center")
ax.annotate("Tragedy\nRegime", xy=(0.45, 0.6),
            xycoords="axes fraction", ha="center", va="center")
ax.annotate("Coordination\nRegime", xy=(0.7, 0.3),
            xycoords="axes fraction", ha="center", va="center")
ax.annotate("Comedy\nRegime", xy=(0.84, 0.1),
            xycoords="axes fraction", ha="left", va="top")

# colorbar
palette = copy(plt.cm.Reds)
palette.set_over('k', 1.0)
palette.set_under('w', 1.0)
palette.set_bad('b', 1.0)

cmap = plt.get_cmap(palette)
norm = mpl.colors.Normalize(vmin=gammin, vmax=gammax)
cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb.set_array([])
    
fig.colorbar(cb, extend="max", extendrect=True, extendfrac=cbextendfrac,
             orientation="horizontal", cax=cbax,
             label=r"Discount factor $\gamma$")

plt.savefig("../figs/fig02_MvsGamma.png", dpi=300)
