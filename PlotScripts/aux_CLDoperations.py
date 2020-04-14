"""
Collective Learning Dynamics in sympy.
    -   only for N=2 agents, M=2 actions, Z=2 states (for now)
"""

import sympy as sp
import numpy as np


def tupeltoindexstring(tup):
    string = ""
    for element in tup:
        string += str(element)
    return string


def generalT(N, M, Z):
    """
    general transition tensor
    """
    dim = np.concatenate(([Z],
                          [M for _ in range(N)],
                          [Z]))
    Tsas = np.zeros(dim, dtype=object)

    for index, _ in np.ndenumerate(Tsas):
        sas = tupeltoindexstring(index)
        Tsas[index] = sp.symbols(f"T_{sas}")

    return sp.Array(Tsas)


def generalR(N, M, Z):
    """
    general reward tensor
    """
    dim = np.concatenate(([N],
                          [Z],
                          [M for _ in range(N)],
                          [Z]))
    Risas = np.zeros(dim, dtype=object)

    for index, _ in np.ndenumerate(Risas):
        sas = tupeltoindexstring(index[1:])
        Risas[index] = sp.symbols(f"R^{index[0]}_{sas}")

    return sp.Array(Risas)


def generalBehavior(N, M, Z):
    """
    general behavioral strategy
    """
    dim = np.concatenate(([N],
                          [Z],
                          [M]))
    X = np.zeros(dim, dtype=object)

    for index, _ in np.ndenumerate(X):
        sa = tupeltoindexstring(index[1:])
        X[index] = sp.symbols(f"X^{index[0]}_{sa}")

    return sp.Array(X)


X = generalBehavior(2, 2, 2)

Xg = X.subs(X[0, 0, 1], 1-X[0, 0, 0]).subs(X[1, 0, 1], 1-X[1, 0, 0]).\
    subs(X[0, 1, 1], 1-X[0, 1, 0]).subs(X[1, 1, 1], 1-X[1, 1, 0])


#
#   Derivations
#
def obtain_Tss(T, X):
    N = X.shape[0]
    Z = T.shape[0]
    Tss = np.zeros((Z, Z), dtype=object)
    for index, _ in np.ndenumerate(np.zeros(T.shape)):
        Tsas = T[index]
        s = index[0]
        jA = index[1:-1]
        sprim = index[-1]

        allX = 1
        for n in range(N):
            allX *= X[n, s, jA[n]]

        Tss[s, sprim] += Tsas*allX

    return sp.Matrix(Tss)


def obtain_Ris(R, T, X):
    Z = T.shape[0]
    N = R.shape[0]
    Ris = np.zeros((N, Z), dtype=object)
    for index, _ in np.ndenumerate(np.zeros(R.shape)):
        i = index[0]
        s = index[1]
        jA = index[2:-1]
        # sprim = index[-1]

        Risas = R[index]
        Tsas = T[index[1:]]

        allX = 1
        for n in range(N):
            allX *= X[n, s, jA[n]]

        Ris[i, s] += Risas*Tsas*allX

    return sp.Matrix(Ris)


def obtain_VIs(R, T, X, i, ys):
    y = ys[i]

    Tss = obtain_Tss(T, X)
    Tss.simplify()
    M = (sp.eye(2) - y * Tss)
    M.simplify()
    invM = M.inv()
    invM.simplify()
    Ris = obtain_Ris(R, T, X)[i, :]
    Ris.simplify()

    Vis = ((1 - y) * invM) * Ris.T
    Vis.simplify()
    return Vis


def obtain_NextV0sa(V0s, T, X):
    Z = T.shape[0]

    NextV0sa = sp.Matrix(np.zeros((Z, 2), dtype=object))
    j = 1

    for s in range(2):
        for a in range(2):
            for sprim in range(2):
                for b in range(2):
                    NextV0sa[s, a] += X[j, s, b] *\
                        T[s, a, b, sprim] * V0s[sprim]

    return sp.Matrix(NextV0sa)


def obtain_NextV1sa(V1s, T, X):
    Z = T.shape[0]

    NextV1sa = sp.Matrix(np.zeros((Z, 2), dtype=object))
    j = 0

    for s in range(2):
        for a in range(2):
            for sprim in range(2):
                for b in range(2):
                    NextV1sa[s, a] += X[j, s, b] *\
                        T[s, b, a, sprim] * V1s[sprim]

    return sp.Matrix(NextV1sa)


def obtain_NextVIsa(VIs, T, X, i):
    Z = T.shape[0]

    NextVIsa = sp.Matrix(np.zeros((Z, 2), dtype=object))
    j = (i + 1) % 2

    for s in range(Z):
        for a in range(2):
            for sprim in range(Z):
                for b in range(2):
                    NextVIsa[s, a] += X[j, s, b] *\
                        T[s, b, a, sprim] * VIs[sprim]

    return sp.Matrix(NextVIsa)


def obtain_Risa(R, T, X):
    Z = T.shape[0]
    N = R.shape[0]
    assert N == 2

    Risa = np.zeros((N, Z, 2), dtype=object)
    for index, _ in np.ndenumerate(np.zeros(R.shape)):
        i = index[0]
        j = 1-i

        s = index[1]
        jA = index[2:-1]

        Risas = R[index]
        Tsas = T[index[1:]]

        Risa[i, s, jA[i]] += Risas*Tsas*X[j, s, jA[j]]

    return sp.Array(Risa)


