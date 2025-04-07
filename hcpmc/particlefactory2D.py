# Author: Jin Wang
"""A 'particlefactory2D' can generate hoomd integrator for given 2d shape.

Now we support the following families:
RoundedSquare, TruncatedSquare(not imple), RoundedTriangle, TruncatedTriangle(not imple)


.. invisible-code-block: python

    factory2d = hcpmc.particlefactory2D.RoundedSquare(zeta=0)
    mc = factory.get_integrator()
    symmetries = factory.get_symmetries()

"""

import hoomd
import coxeter
import numpy as np
import math



class RoundedSquare:
    """Create an instance of RoundedSquare particle with specified zeta = sigma/(L+sigma) ."""

    def __init__(self, zeta: float = 0):
        if zeta > 1 or zeta < 0:
            raise ValueError("zeta must be between 0 and 1")
        if zeta == 1:
            self.shape = coxeter.shapes.ConvexSpheropolygon([[-1e-6, -1e-6], [1e-6, -1e-6], [1e-6, 1e-6], [-1e-6, 1e-6]], radius=1)
            self.shape.area = 1
        else:
            self.shape = coxeter.shapes.ConvexSpheropolygon([[-1, -1], [1, -1], [1, 1], [-1, 1]], radius=zeta / (1 - zeta))
            self.shape.area = 1

    def get_integrator(self):
        """Get the hoomd integrator object."""
        mc = hoomd.hpmc.integrate.ConvexSpheropolygon(default_d=0.1, default_a=0.1)
        mc.shape["S0"] = dict(vertices=self.shape.vertices[:, :2].tolist(), sweep_radius=self.shape.radius)
        return mc

    def get_symmetries(self):
        """Get the symmetry operations that are applied to this particle."""
        symmetries = [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
            [-np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
            [np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2],
        ]
        return symmetries

    def get_gsd_shape(self):
        """Get the gsd shape object."""
        return {"type": "Polygon", "rounding_radius": self.shape.radius, "vertices": self.shape.vertices[:, :2].tolist()}

class RoundedTriangle:
    """Create an instance of RoundedTriangle particle with specified zeta = sigma/(L+sigma) ."""

    def __init__(self, zeta: float = 0):
        if zeta > 1 or zeta < 0:
            raise ValueError("zeta must be between 0 and 1")
        if zeta == 1:
            self.shape = coxeter.shapes.ConvexSpheropolygon([[-1e-6, -np.sqrt(3) / 3 * 1e-6], [1e-6, -np.sqrt(3) / 3 * 1e-6], [0, 2 * np.sqrt(3) / 3 * 1e-6]], radius=1)
            self.shape.area = 1
        else:
            self.shape = coxeter.shapes.ConvexSpheropolygon([[-1, -np.sqrt(3) / 3], [1, -np.sqrt(3) / 3], [0, 2 * np.sqrt(3) / 3]], radius=zeta / (1 - zeta))
            self.shape.area = 1

    def get_integrator(self):
        """Get the hoomd integrator object."""
        mc = hoomd.hpmc.integrate.ConvexSpheropolygon(default_d=0.1, default_a=0.1)
        mc.shape["S0"] = dict(vertices=self.shape.vertices[:, :2], sweep_radius=self.shape.radius)
        return mc

    def get_symmetries(self):
        """Get the symmetry operations that are applied to this particle."""
        symmetries = [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [1 / 2, 0, 0, np.sqrt(3) / 2],
            [-1 / 2, 0, 0, -np.sqrt(3) / 2],
            [-1 / 2, 0, 0, np.sqrt(3) / 2],
            [1 / 2, 0, 0, -np.sqrt(3) / 2],
        ]
        return symmetries

    def get_gsd_shape(self):
        """Get the gsd shape object."""
        return {"type": "Polygon", "vertices": self.shape.vertices[:, :2].tolist(), "rounding_radius": self.shape.radius}


class ConvexPolygon:
    """Create an instance of any shape particle with the given vertices."""

    def __init__(self, vertices):
        self.shape = coxeter.shapes.ConvexPolygon(vertices)
        self.shape.area = 1

    def get_integrator(self):
        """Get the hoomd integrator object."""
        mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=0.1, default_a=0.1)
        mc.shape["S0"] = dict(vertices=np.array(self.shape.vertices)[:, :2])
        return mc

    def get_gsd_shape(self):
        """Get the gsd shape object."""
        return {"type": "Polygon", "vertices": self.shape.vertices[:, :2].tolist(), "rounding_radius": 0}
