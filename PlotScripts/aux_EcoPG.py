# -*- coding: utf-8 -*-
"""
The EcoPG for sympy

The Ecological Public Goods model to use for symbolic calculations using sympy
"""
import numpy as np
import sympy as sp

import aux_CLDoperations as cld


# %% Reward Tensor
R = cld.generalR(N=2, M=2, Z=2)

# fill reward tensor with specific reward variables
rti, rpi, rsi, rri = sp.symbols("r_t^i r_p^i r_s^i r_r^i")
rtj, rpj, rsj, rrj = sp.symbols("r_t^j r_p^j r_s^j r_r^j")
mi, mj = sp.symbols("m^i m^j")

R = np.array(R.tolist())

R[0, 1, 0, 0, 1] = rri
R[0, 1, 1, 0, 1] = rti
R[0, 1, 0, 1, 1] = rsi
R[0, 1, 1, 1, 1] = rpi

R[1, 1, 0, 0, 1] = rrj
R[1, 1, 0, 1, 1] = rtj
R[1, 1, 1, 0, 1] = rsj
R[1, 1, 1, 1, 1] = rpj

R[0, 0, :, :, :] = mi
R[0, :, :, :, 0] = mi
R[1, 0, :, :, :] = mj
R[1, :, :, :, 0] = mj

R = sp.Array(R)


# %% Benefit and Cost substitiutions
N = sp.symbols("N")
Nd = sp.symbols("N_D")
b, c = sp.symbols("b c")
f = sp.symbols("f")

bc_reward_subs = {rri: (N-Nd)*b-c, rti: (N-Nd-1) * b, rsi: b-c, rpi: 0,
                  rrj: (N-Nd)*b-c, rsj: (N-Nd-1)*b - c, rtj: b, rpj: 0}

fc_reward_subs = {rri: (N-Nd)/N*f*c - c, rti: (N-Nd-1)/N*f*c,
                  rsi: 1/N*f*c - c, rpi: 0,
                  rrj: (N-Nd)/N*f*c - c, rsj: (N-Nd-1)/N*f*c - c,
                  rtj: 1/N*f*c, rpj: 0}

# %% Transition Tensor
T = cld.generalT(N=2, M=2, Z=2)

# fill transition tensor with specific transition probabilities variables
qci, qri, qcj, qrj, qck = sp.symbols("q_c^i q_r^i q_c^j q_r^j q_c^k")

T = np.array(T.tolist())

T[1, 0, 0, 0] = Nd/N*qck
T[1, 0, 0, 1] = 1 - Nd/N*qck

T[1, 1, 0, 0] = qci/N + Nd/N*qck
T[1, 1, 0, 1] = 1 - qci/N - Nd/N*qck

T[1, 0, 1, 0] = ((N-Nd-1)*qcj + Nd*qck)/N
T[1, 0, 1, 1] = 1 - ((N-Nd-1)*qcj + Nd*qck)/N

T[1, 1, 1, 0] = (qci + (N-Nd-1)*qcj + Nd*qck)/N
T[1, 1, 1, 1] = 1 - (qci + (N-Nd-1)*qcj + Nd*qck)/N


T[0, 1, 1, 1] = 0
T[0, 1, 1, 0] = 1

T[0, 0, 1, 1] = qri/N
T[0, 0, 1, 0] = 1 - qri/N

T[0, 1, 0, 1] = (N-Nd-1)*qrj/N
T[0, 1, 0, 0] = 1 - (N-Nd-1)*qrj/N

T[0, 0, 0, 1] = (qri + (N-Nd-1)*qrj)/N
T[0, 0, 0, 0] = 1 - (qri + (N-Nd-1)*qrj)/N

T = sp.Array(T)

# %% Transition probs substitution
qc, qr = sp.symbols("q_c q_r")
q_subs = {qci: qc, qcj: qc, qri: qr, qrj: qr, qck: qc}
