# updated for freud3 by Jin Wang. 2024/04/16
from freud import *
from freud import order as shop
import numpy as np


# Interface for an order parameter class, implementations must extend this.
class OrderParameter:
    def __init__(self):
        raise NotImplementedError

    def getOrder(self, snap):
        raise NotImplementedError


# new version by Jin Wang 2024/04/16
# calculate the local Q_l OP (with 2nd shell average) using of nearest neighbors
class OrderParameterAveQlNear(OrderParameter):
    def __init__(self, l, rcut, kn):
        self.sl = shop.Steinhardt(l, average=True)
        self.kn = kn
        self.rcut = rcut

    def getOrder(self, snap):
        return np.mean(self.sl.compute(system=snap, neighbors={"num_neighbors": self.kn, "r_max": self.rcut}).particle_order)


# calculate the local Q_l OP (with 2nd shell average) using of nearest neighbors
class OrderParameterQlNear(OrderParameter):
    def __init__(self, l, rcut, kn):
        self.sl = shop.Steinhardt(l)
        self.kn = kn
        self.rcut = rcut

    def getOrder(self, snap):
        return np.mean(self.sl.compute(system=snap, neighbors={"num_neighbors": self.kn, "r_max": self.rcut}).particle_order)


class OrderParameter2DPsi6_minusPsi4(OrderParameter):
    def __init__(self):
        self.p6 = shop.Hexatic(k=6)
        self.p4 = shop.Hexatic(k=4)

    def getOrder(self, snap):
        psi6 = np.abs(np.mean(self.p6.compute(system=snap, neighbors={"num_neighbors": 6}).particle_order))
        psi4 = np.abs(np.mean(self.p4.compute(system=snap, neighbors={"num_neighbors": 4}).particle_order))
        return psi6 - psi4


class OrderParameter2DPsi_l(OrderParameter):
    def __init__(self, l):
        self.l = l
        self.pl = shop.Hexatic(k=l)

    def getOrder(self, snap):
        psi = np.abs(np.mean(self.pl.compute(system=snap, neighbors={"num_neighbors": self.l}).particle_order))
        return psi

# calculate the Delta_dq2
class OrderParameterDelta_dq2(OrderParameter):
    def __init__(self, symmetries):
        self.sym = symmetries

    def compute_dq2(self, q):
        return min(np.dot((q - q2), (q - q2)) for q2 in self.symmetries)

    def getOrder(self, snap):
        qs = snap.particles.orientation
        dq2 = []
        for q in qs:
            dq2.append(self.compute_dq2(q))
        return np.std(dq2,ddof=1)
