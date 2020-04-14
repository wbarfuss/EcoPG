# -*- coding: utf-8 -*-

# %% imports
from aux_CLDoperations import X, Xg
import aux_CLDoperations as cld

import sympy as sp

# %% Definitions
yi, yj = sp.symbols("\gamma^i \gamma^j")

con0_style = {"color": "k", "ls": "--"}
con1_style = {"color": "blue"}
con2_style = {"color": "green"}

# %%
def obtain_con_sols_Game(R, T, ysym=None,
                         y_vs=[yi, yj], state=1, deg0o5=False):

    # %% Behavior
    if not deg0o5:
        Xcc = {X[0, 1, 0]: 1, X[0, 0, 0]: 1, X[1, 1, 0]: 1, X[1, 0, 0]: 1}
        Xdd = {X[0, 1, 0]: 0, X[0, 0, 0]: 1, X[1, 1, 0]: 0, X[1, 0, 0]: 1}
        Xcd = {X[0, 1, 0]: 1, X[0, 0, 0]: 1, X[1, 1, 0]: 0, X[1, 0, 0]: 1}
        Xdc = {X[0, 1, 0]: 0, X[0, 0, 0]: 1, X[1, 1, 0]: 1, X[1, 0, 0]: 1}
    else:
        Xcc = {X[0, 1, 0]: 1, X[0, 0, 0]: 0.5, X[1, 1, 0]: 1, X[1, 0, 0]: 0.5}
        Xdd = {X[0, 1, 0]: 0, X[0, 0, 0]: 0.5, X[1, 1, 0]: 0, X[1, 0, 0]: 0.5}
        Xcd = {X[0, 1, 0]: 1, X[0, 0, 0]: 0.5, X[1, 1, 0]: 0, X[1, 0, 0]: 0.5}
        Xdc = {X[0, 1, 0]: 0, X[0, 0, 0]: 0.5, X[1, 1, 0]: 1, X[1, 0, 0]: 0.5}

    # %% compute Value Functions

    Vcc = cld.obtain_VIs(R, T, Xg.subs(Xcc), i=0, ys=y_vs)
    Vdd = cld.obtain_VIs(R, T, Xg.subs(Xdd), i=0, ys=y_vs)
    Vcd = cld.obtain_VIs(R, T, Xg.subs(Xcd), i=0, ys=y_vs)
    Vdc = cld.obtain_VIs(R, T, Xg.subs(Xdc), i=0, ys=y_vs)

    # %% values
    reward, punishment, temptation, sucker =\
        map(lambda v: v[state],
            [Vcc, Vdd, Vdc, Vcd])

    # %% Conditions
    # Dilemma: P==R
    con0 = sp.simplify(sp.Eq(punishment, reward))

    # Greed: T == R
    con1 = sp.simplify(sp.Eq(temptation, reward))

    # Fear: P == S
    con2 = sp.simplify(sp.Eq(punishment, sucker))

    # %% solutions
    if ysym is not None:
        sol0 = sp.solve(con0, ysym)
        print(f"con0 Dilemma has {len(sol0)} solution(s)")

        sol1 = sp.solve(con1, ysym)
        print(f"con1 Greed has {len(sol1)} solution(s)")

        sol2 = sp.solve(con2, ysym)
        print(f"con2 Fear has {len(sol2)} solution(s)")

        return sol0, sol1, sol2
    else:
        return con0, con1, con2


def obtain_con_sols_NFGame(reward, temptation, sucker, punishment, ysym=None):
    # %% Conditions
    # Dilemma: P==R
    con0 = sp.simplify(sp.Eq(punishment, reward))

    # Greed: T == R
    con1 = sp.simplify(sp.Eq(temptation, reward))

    # Fear: P == S
    con2 = sp.simplify(sp.Eq(punishment, sucker))

    # %% solutions
    if ysym is not None:
        sol0 = sp.solve(con0, ysym)
        print(f"con0 Dilemma has {len(sol0)} solution(s)")

        sol1 = sp.solve(con1, ysym)
        print(f"con1 Greed has {len(sol1)} solution(s)")

        sol2 = sp.solve(con2, ysym)
        print(f"con2 Fear has {len(sol2)} solution(s)")

        return sol0, sol1, sol2
    else:
        return con0, con1, con2
