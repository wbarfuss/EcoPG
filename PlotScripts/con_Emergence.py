# -*- coding: utf-8 -*-
"""
condition for the emergemence of cooperation out of CLD
"""

# %% imports
from aux_CLDoperations import X, Xg
from aux_EcoPG import N, Nd, qci, qcj, qri, qrj, b, c, f, mi, mj
from aux_EcoPG import R, T
from aux_EcoPG import fc_reward_subs

import aux_CLDoperations as cld
import sympy as sp

# %% parameters
y, m, qc, qr = sp.symbols("\gamma m q_c q_r")
X_0o5 = {X[0, 0, 0]: 0.5, X[1, 0, 0]: 0.5}


# %% 
def obtain_conEmergence(qc_vs, qr_vs, m_vs, Xsubs, y_vs=[y, y],
                        reward_subs=fc_reward_subs, b_v=3, c_v=5, f_v=1.2,
                        R=R, T=T):

    # %% R and T
    R = R.subs(reward_subs)
    T = T.subs(Nd, 0).subs(N, 2)
    R = R.subs(Nd, 0).subs(N, 2)

    R = R.subs({b: b_v, c: c_v, f: f_v, mi: m_vs[0], mj: m_vs[1]})
    T = T.subs({qci: qc_vs[0], qcj: qc_vs[1], qri: qr_vs[0], qrj: qr_vs[1]})

    # %%
    V0s = cld.obtain_VIs(R=R, T=T, X=Xg.subs(Xsubs), i=0, ys=y_vs)
    V1s = cld.obtain_VIs(R=R, T=T, X=Xg.subs(Xsubs), i=1, ys=y_vs)

    # %% NextValue of Agent _sa
    V00, V01 = sp.symbols("V^0_0 V^0_1")
    V0sGen = sp.Matrix([[V00, V01]])
    NextV0sa = cld.obtain_NextV0sa(V0sGen, T, Xg.subs(Xsubs))
    NextV0sa.simplify()

    V10, V11 = sp.symbols("V^1_0 V^1_1")
    V1sGen = sp.Matrix([[V10, V11]])
    NextV1sa = cld.obtain_NextV1sa(V1sGen, T, Xg.subs(Xsubs))
    NextV1sa.simplify()

    # %% Risa
    Risa = cld.obtain_Risa(R, T, Xg.subs(Xsubs))
    R0sa = sp.Matrix(Risa[0, :, :]).reshape(2, 2)
    R0sa.simplify()
    R1sa = sp.Matrix(Risa[1, :, :]).reshape(2, 2)
    R1sa.simplify()

    # %% Temporal difference error
    td0sa = (1-y_vs[0])*R0sa + y_vs[0]*NextV0sa
    td0sa.simplify()
    td1sa = (1-y_vs[1])*R1sa + y_vs[1]*NextV1sa
    td1sa.simplify()

    # %% Ansatz
    cond = sp.Eq(td0sa[1, 0]-td0sa[1, 1], td1sa[1, 1]-td1sa[1, 0])
    condition = cond.subs(V00, V0s[0]).subs(V01, V0s[1]).\
        subs(V10, V1s[0]).subs(V11, V1s[1])
    condition = condition.simplify()

    return condition
