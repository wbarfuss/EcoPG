"""
This is detQ.py, the determinsitic limit of Q learning.
"""
import itertools as it
import numpy as np
from .detMAE import detMAE

class detAC(detMAE):

    def __init__(self,
                 TranstionTensor,
                 RewardTensor,
                 alpha,
                 beta,
                 gamma,
                 roundingprec=9):

        detMAE.__init__(self,
                        TranstionTensor,
                        RewardTensor,
                        alpha,
                        beta,
                        gamma,
                        roundingprec)

    # =========================================================================
    #   Temporal difference error
    # =========================================================================

    def TDerror(self, X, norm=True):
        Risa = self.obtain_Risa(X)
        NextVisa = self.obtain_NextVisa(X)

        n = np.newaxis
        TDe = (1-self.gamma[:,n,n])*Risa + self.gamma[:,n,n]*NextVisa
        if norm:
            TDe = TDe - TDe.mean(axis=2, keepdims=True)
        # TDe = TDe.filled(0)
        return TDe

    # =========================================================================
    #   Behavior profile averages
    # =========================================================================

    def obtain_NextVisa(self, X):
        """
        For ac learning
        """
        Vis = self.obtain_Vis(X)
        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        j2k = list(range(4, 4+self.N-1))                      # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))    # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[j2k[o], s, e2f[o]] for o in range(self.N-1)]  # sum ids
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                Vis, [i, sprim],
                self.T, [s]+b2d+[sprim]] + otherX + [[i, s, a]]

        return np.einsum(*args)
