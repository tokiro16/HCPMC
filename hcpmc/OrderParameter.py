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
        return np.mean(
            self.sl.compute(
                system=snap, neighbors={"num_neighbors": self.kn, "r_max": self.rcut}
            ).particle_order
        )


# calculate the local Q_l OP (with 2nd shell average) using of nearest neighbors
class OrderParameterQlNear(OrderParameter):
    def __init__(self, l, rcut, kn):
        self.sl = shop.Steinhardt(l)
        self.kn = kn
        self.rcut = rcut

    def getOrder(self, snap):
        return np.mean(
            self.sl.compute(
                system=snap, neighbors={"num_neighbors": self.kn, "r_max": self.rcut}
            ).particle_order
        )


class OrderParameter2DPsi6_minusPsi4(OrderParameter):
    def __init__(self):
        self.p6 = shop.Hexatic(k=6)
        self.p4 = shop.Hexatic(k=4)

    def getOrder(self, snap):
        psi6 = np.abs(
            np.mean(
                self.p6.compute(
                    system=snap, neighbors={"num_neighbors": 6}
                ).particle_order
            )
        )
        psi4 = np.abs(
            np.mean(
                self.p4.compute(
                    system=snap, neighbors={"num_neighbors": 4}
                ).particle_order
            )
        )
        return psi6 - psi4

