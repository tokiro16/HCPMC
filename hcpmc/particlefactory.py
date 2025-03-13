# Author: Jin Wang
"""A 'particlefactory' can generate hoomd integrator for given shape.

Now we support the following families:
Family423, Family323+, TruncatedTetrahedron, SlantedCube


.. invisible-code-block: python

    factory = hcpmc.particlefactory.Family423(a=1,c=2)
    mc = factory.get_integrator()
    symmetries = factory.get_symmetries()

"""

import hoomd
import coxeter
import numpy as np
import math


class Family423:
    """Create an instance of family 423 particle with specified a,b,c."""

    def __init__(self, a, c, b=2):
        if a < 1 or a > 2:
            raise ValueError("a must be between 1 and 2")
        if c < 2 or c > 3:
            raise ValueError("c must be between 2 and 3")
        if b != 2:
            raise ValueError("b must be 2")
        self.shape = coxeter.families.Family423.get_shape(a, c)
        self.shape.volume = 1

    def get_integrator(self):
        """Get the hoomd integrator object."""
        mc = hoomd.hpmc.integrate.ConvexPolyhedron(default_d=0.1, default_a=0.1)
        mc.shape["S0"] = dict(vertices=self.shape.vertices)
        return mc

    def get_symmetries(self):
        """Get the symmetry operations that are applied to this particle."""
        symmetries = [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0.7071, 0.7071, 0, 0],
            [-0.7071, 0.7071, 0, 0],
            [0, 1, 0, 0],
            [0.7071, 0, 0.7071, 0],
            [-0.7071, 0, 0.7071, 0],
            [0, 0, 1, 0],
            [0.7071, 0, 0, 0.7071],
            [-0.7071, 0, 0, 0.7071],
            [0, 0, 0, 1],
            [-0.7071, -0.7071, 0, 0],
            [0.7071, -0.7071, 0, 0],
            [0, -1, 0, 0],
            [-0.7071, 0, -0.7071, 0],
            [0.7071, 0, -0.7071, 0],
            [0, 0, -1, 0],
            [-0.7071, 0, 0, -0.7071],
            [0.7071, 0, 0, -0.7071],
            [0, 0, 0, -1],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0, 0, 0.7071, 0.7071],
            [0, 0, -0.7071, 0.7071],
            [0, 0.7071, 0, 0.7071],
            [0, -0.7071, 0, 0.7071],
            [0, 0.7071, 0.7071, 0],
            [0, -0.7071, 0.7071, 0],
            [0, 0, -0.7071, -0.7071],
            [0, 0, 0.7071, -0.7071],
            [0, -0.7071, 0, -0.7071],
            [0, 0.7071, 0, -0.7071],
            [0, -0.7071, -0.7071, 0],
            [0, 0.7071, -0.7071, 0],
        ]
        return symmetries

    def get_gsd_shape(self):
        """Get the gsd shape object."""
        return self.shape.gsd_shape_spec


class Family323Plus:
    """Create an instance of family 323+ particle with specified a,b,c."""

    def __init__(self, a, c, b=1):
        if a < 1 or a > 3:
            raise ValueError("a must be between 1 and 3")
        if c < 1 or c > 3:
            raise ValueError("c must be between 1 and 3")
        if b != 1:
            raise ValueError("b must be 1")
        self.shape = coxeter.families.Family323Plus.get_shape(a, c)
        self.shape.volume = 1

    def get_integrator(self):
        """Get the hoomd integrator object."""
        mc = hoomd.hpmc.integrate.ConvexPolyhedron(default_d=0.1, default_a=0.1)
        mc.shape["S0"] = dict(vertices=self.shape.vertices)
        return mc

    def get_symmetries(self):
        """Get the symmetry operations that are applied to this particle."""
        symmetries = [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5, 0.5],
        ]
        return symmetries

    def get_gsd_shape(self):
        """Get the gsd shape object."""
        return self.shape.gsd_shape_spec


class TruncatedTetrahedron:
    """Create an instance of family Truncated Tetrahedron particle with truncation."""

    def __init__(self, truncation):
        if truncation < 0 or truncation > 1:
            raise ValueError("truncation must be between 0 and 1")
        self.shape = coxeter.families.TruncatedTetrahedronFamily.get_shape(truncation)
        self.shape.volume = 1

    def get_integrator(self):
        """Get the hoomd integrator object."""
        mc = hoomd.hpmc.integrate.ConvexPolyhedron(default_d=0.01, default_a=0.01)
        mc.shape["S0"] = dict(vertices=self.shape.vertices)

    def get_symmetries(self):
        """Get the symmetry operations that are applied to this particle."""
        symmetries = [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5, 0.5],
        ]
        return symmetries

    def get_gsd_shape(self):
        """Get the gsd shape object."""
        return self.shape.gsd_shape_spec


class SlantedCube:
    """Create an instance of Slanted Cube particle with theta."""

    def vertices(self, theta):
        """Generate the vertices of the slanted cube"""
        theta = math.radians(theta)
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([np.cos(theta), 0, np.sin(theta)])
        origin = np.array([-0.5 * (1 + np.cos(theta)), -0.5, -0.5 * np.sin(theta)])
        vert = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    temp = origin + v1 * i + v2 * j + v3 * k
                    vert.append(temp.tolist())
        return vert

    def __init__(self, theta):
        if theta > 90 or theta < 0:
            raise ValueError("theta must be between 0 and 90")
        self.shape = coxeter.shapes.ConvexPolyhedron(self.vertices(theta))
        self.shape.volume = 1

    def normalize_vector(self, v):
        """Normalize a given vector"""
        v = np.array(v)
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Cannot normalize the zero vector")
        return v / norm

    def symmetry_c2(self, axis):
        """Return the symmetry operation corresponding to the given axis"""
        axis = self.normalize_vector(axis)
        axis = np.insert(axis, 0, 0)
        return axis

    def get_integrator(self):
        """Get the hoomd integrator object."""
        mc = hoomd.hpmc.integrate.ConvexPolyhedron(default_d=0.1, default_a=0.1)
        mc.shape["S0"] = dict(vertices=self.shape.vertices)
        return mc

    def get_symmetries(self):
        """Get the symmetry operations that are applied to this particle."""
        theta = math.radians(self.theta)
        symmetries = [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            self.symmetry_c2([1 + math.cos(theta), 0, math.sin(theta)]),
            -self.symmetry_c2([1 + math.cos(theta), 0, math.sin(theta)]),
            self.symmetry_c2([1 - math.cos(theta), 0, -math.sin(theta)]),
            -self.symmetry_c2([1 - math.cos(theta), 0, -math.sin(theta)]),
        ]
        return symmetries

    def get_gsd_shape(self):
        """Get the gsd shape object."""
        return self.shape.gsd_shape_spec

class Anyshape:
    """Create an instance of any shape particle with the given vertices."""

    def __init__(self, vertices):
        self.shape = coxeter.shapes.ConvexPolyhedron(vertices)
        self.shape.volume = 1

    def get_integrator(self):
        """Get the hoomd integrator object."""
        mc = hoomd.hpmc.integrate.ConvexPolyhedron(default_d=0.1, default_a=0.1)
        mc.shape["S0"] = dict(vertices=self.shape.vertices)
        return mc
