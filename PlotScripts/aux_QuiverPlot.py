"""Plot Quiver for DetRL"""
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

def prob_grid():
    """Returns the action probability grid for one axis as a list."""
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 


def gridded_policies(pAs, Z):
    """Transforms action probabilites (pAs) into policy grid.

    Z : number of states
    """
    Xs = []  # one X has signature Xisa
    for ps in it.product(pAs, repeat=2*Z):
        X = np.zeros((2, Z, 2))
        for i in range(2):
            for s in range(Z):
                X[i, s, 0] = ps[i*Z+s]
                X[i, s, 1] = 1 - ps[i*Z+s]
        Xs.append(X)

    return Xs


def TDerror_difference(Xs, agents):
    """Compute TDerros for Xs and ltype ("A", "Q", "S")."""
    return [agents.TDerror(X) for X in Xs]


def DeltaX_difference(Xs, agents):
    """Compute X(t-1)-X(t) for Xs and ltype ("A", "Q", "S")."""
    return [agents.TDstep(X) - X for X in Xs]


def quiverformat_difference(diffs, Xs, pAs, Z):
    """Translate differences (diffs) into quiverformat for
    Xs: behaviors
    pAs : action probs (grid of quiver plot)
    Z: nr of states
    """
    n = len(pAs)

    Xagentsdata = np.zeros((Z, n, n, n**(2*(Z-1))))
    Yagentsdata = np.zeros((Z, n, n, n**(2*(Z-1))))
    # for each state(Z): n in X, n in Y direction + values in remaining states

    for i, pX in enumerate(pAs):
        for j, pY in enumerate(pAs):
            k = np.zeros(Z, dtype=int)
            for b, Beh in enumerate(Xs):
                for s in range(Z):
                    if Beh[0, s, 0] == pX and Beh[1, s, 0] == pY:
                        Xagentsdata[s, j, i, k[s]] = diffs[b][0, s, 0]
                        Yagentsdata[s, j, i, k[s]] = diffs[b][1, s, 0]
                        k[s] += 1

    return Xagentsdata, Yagentsdata


def plot_quiver(agents,
                difftype="TDe",
                pAs=prob_grid(),
                axes=None, **kwargs):
    assert agents.R.shape[0] == 2, "Number of agents needs to be 2"
    assert agents.R.shape[2] == 2, "Nummber of actions needs to be 2"
    Z = agents.R.shape[1]  # Number of states

    Xs = gridded_policies(pAs, Z)

    diffs = TDerror_difference(Xs, agents) if difftype == "TDe"\
        else DeltaX_difference(Xs, agents)

    dX, dY = quiverformat_difference(diffs, Xs, pAs, Z)
    X, Y = np.meshgrid(pAs, pAs)

    axes = _do_the_plot(dX, dY, X, Y, pAs, Z, axes, **kwargs)
    return axes

def scale(x, a):
    return np.sign(x) * a * np.abs(x)

def scale2(x, y, a):
    l = (x**2 + y**2)**(1/2)
    k = l**a
    return k/l * x, k/l * y

def _do_the_plot(dX, dY, X, Y, pAs, Z, axes, sf=1, **kwargs):
    # figure and axes
    if axes==None:
        fig, axes = plt.subplots(1, Z, figsize=(4*Z, 4))
    else:
        assert len(axes) == Z, "Number of axes must equal number of states Z"

    # quiver keywords
    qkwargs = {"units":"xy", "angles":"xy", "scale":None, "scale_units":"xy",
               "headwidth":4.5, "pivot":"tail", **kwargs}

    # individuals
    Nr = len(pAs)**(2*(Z-1))
    for i in range(Nr):
        for s in range(Z):
            DX = dX[s, :, :, i]
            DY = dY[s, :, :, i]
            LEN = (DX**2 + DY**2)**0.5
            axes[s].quiver(X, Y, *scale2(DX, DY, sf), LEN,
                           alpha=1/Nr, **qkwargs)           

    # averages
    for s in range(Z):
        DX = dX[s].mean(axis=-1)
        DY = dY[s].mean(axis=-1)
        LEN = (DX**2 + DY**2)**0.5
        axes[s].quiver(X, Y, *scale2(DX, DY, sf), LEN,
                       cmap="viridis", **qkwargs)

    for i, ax in enumerate(axes):
        ax.set_title("State " + str(i))
        ax.set_xlabel(u"$X^0_{s0}$")
    axes[0].set_ylabel(u"$X^1_{s0}$")

    return axes


def plot_trajectory(agents,
                    Xinit, axes,
                    Tmax=10000, fpepsilon=0.000001, **kwargs):
    Z = agents.R.shape[1]  # Number of states
    (AXps, AYps), rt, fpr = trajectory(agents, Xinit,
                                       Tmax=Tmax,
                                       fpepsilon=fpepsilon)

    for s in range(Z):
        axes[s].plot(AXps[:, s], AYps[:, s], **kwargs)
        axes[s].plot([AXps[0, s]], [AYps[0, s]], 'x', **kwargs)
        if fpr: # fixpointreached
            axes[s].plot([AXps[-1, s]], [AYps[-1, s]], 'o', **kwargs)

    return rt, fpr


def trajectory(agents, Xinit,
               Tmax=10000, fpepsilon=0.000001):
    X = Xinit
    agentXprobs, agentYprobs = [], []
    fixpreached = False
    t = 0
    rewardtraj = []
    while not fixpreached and t < Tmax:
        agentXprobs.append(X[0, :, 0])
        agentYprobs.append(X[1, :, 0])

        Ris = agents.obtain_Ris(X)
        δ = agents.obtain_statdist(X)   
        rewardtraj.append(np.sum(δ.T * Ris, axis=1))

        Xnew = agents.TDstep(X)
        fixpreached = np.linalg.norm(Xnew - X) < fpepsilon
        X = Xnew
        t += 1

    return (np.array(agentXprobs), np.array(agentYprobs)), rewardtraj,\
        fixpreached
