# updated for hoomd4 by Jin Wang. 2024/04/16

import numpy as np
import random
import hoomd
import math
from .. import OrderParameter
import sys


"""
Management class for an umbrella window a single (spring constant and order parameter target pair).
"""


class UmbrellaWindow:
    """An umbrella Window"""

    def __init__(self, k, ordertarget, trajectorylength, logname, simulation, orderparameter):
        """Constructor for Umbrella Window. 给定要进行Umbrella的模拟以及弹簧常数
        trajectorylength指的是进行metropolis判断的间隔步数
        logname是一串字符，会打开一个logname的文件用来记录模拟的结果

        parameters
        --------------
        k : float
            Umbrella spring constant with convention E = k*(O-Ot)^2 (no 1/2).
        ordertarget : float
            Target value of order parameter Ot
        trajectorylength : int
            Length of umbrella sampling trial trajectory before energy evaluation
        logname : str
            name of output file to which umbrella measurements are written
        simulation : HOOMDsimulation
            Hoomd simulation object for this class to manage
        orderparameter : OrderParameter.OrderParameter
            The orderparameter used for the harmonic umbrella bias.
        """

        self.k = k
        self.ordertarget = ordertarget
        self.trajectorylength = trajectorylength
        self.simulation = simulation

        if not isinstance(orderparameter, OrderParameter.OrderParameter):
            raise TypeError("given orderparameter object is not of class OrderParameter")
        self.orderparameter = orderparameter

        self.log = open(logname + ".dat", "w")
        self.loggingenabled = False  # Toggle writing to log file on/off

        self.acceptcount = 0
        self.rejectcount = 0

        # Umbrella prior state
        self.snapshotlast = None
        self.energylast = None
        self.orderlast = None

        # Umbrella current state
        self.snapshot = None
        self.order = None
        self.energy = None

        # Fill snapshot, positions with initial state
        self.getState()
        self.copyStateToPriorState()

        # Report some output of internal variables after construction to console
        self.printUmbrellaVariables()

    def getAcceptanceStatistics(self):
        """Get the acceptance and rejection counts"""
        return [self.acceptcount, self.rejectcount]

    def resetAcceptanceStatistics(self):
        """Reset statistics to zero"""
        self.acceptcount = 0
        self.rejectcount = 0

    def getAcceptanceRatio(self):
        """Return acceptance ratio. Violates naming convention"""
        ratio = self.acceptcount / (self.acceptcount + self.rejectcount)
        return ratio

    def printAcceptanceStatistics(self):
        """Prints acceptance statistics to console"""
        total = self.acceptcount + self.rejectcount
        print(" Accepted " + str(self.acceptcount) + "/" + str(total) + ", (" + str(self.acceptcount / float(total) * 100) + "%)")

    def enableLogging(self):
        """Enable logging to file"""
        self.loggingenabled = True

    def disableLogging(self):
        """Disable logging to file"""
        self.loggingenabled = False

    def writeLogValue(self, value):
        self.log.write(str(self.simulation.timestep) + " " + str(value) + "\n")
        if (self.acceptcount + self.rejectcount) % 10 == 0:
            self.log.flush()

    def calcEnergy(self, ordervalue):
        return self.k / 2 * (ordervalue - self.ordertarget) * (ordervalue - self.ordertarget)

    def resetState(self):
        """Resets HOOMD to prior snapshot"""
        self.simulation.state.set_snapshot(self.snapshotlast)

    def getState(self):
        """Get all state data from HOOMD, and calculate order and energy"""

        self.snapshot = self.simulation.state.get_snapshot()
        self.order = self.orderparameter.getOrder(self.snapshot)
        self.energy = self.calcEnergy(self.order)

    def copyStateToPriorState(self):
        """Move current state over prior state"""
        self.snapshotlast = self.snapshot
        self.energylast = self.energy
        self.orderlast = self.order

    def copyPriorStateToState(self):
        """Move prior state to current state"""
        self.snapshot = self.snapshotlast
        self.energy = self.energylast
        self.order = self.orderlast

    def Metropolis(self):
        """Perform an Umbrella Evaluation to accept/reject current simulation state"""
        # Load new state from simulator
        self.getState()
        # Umbrella MC reject or accept
        if self.energylast > self.energy:  # Accepted and don't risk overflow error
            self.acceptcount += 1
            if self.loggingenabled:
                self.writeLogValue(self.order)  # log current order
            self.copyStateToPriorState()  # prepare for next step

        elif math.exp(self.energylast - self.energy) < random.random():  # Reject
            self.resetState()  # Reload prior snapshot into simulation
            self.rejectcount += 1
            if self.loggingenabled:
                self.writeLogValue(self.orderlast)  # log current order

        else:  # Accepted
            self.acceptcount += 1
            if self.loggingenabled:
                self.writeLogValue(self.order)  # log current order
            self.copyStateToPriorState()  # prepare for next step

        if (self.acceptcount + self.rejectcount) % 100 == 0:
            self.printAcceptanceStatistics()

    def runUmbrellaTrials(self, numtrials):
        """Using Metropolis to implement biased simulation"""
        for i in range(numtrials):
            self.simulation.run(self.trajectorylength)
            self.Metropolis()

        # If last move was rejected, we need to copyPriorStateToState to state incase
        # so use of getOrder() isn't wrong (or getEnergy etc).  If the last move
        # was accepted, then priorstate == state, so there is no problem
        # This is done here instead of the overlap loop for efficiency.
        self.copyPriorStateToState()

    def forceUpdate(self):
        """Updates this class's internal state from the HOOMD system.  Use if system state has been altered or run outside of the UmbrellaWindow class."""
        self.getState()
        self.copyStateToPriorState()

    def setHarmonicConstant(self, newk):
        """Reset k.  Remember to make new logfile if changing"""
        self.k = newk

    def getHarmonicConstant(self):
        """Get current value of k"""
        return self.k

    def setOrderTarget(self, newordertarget):
        """Reset Order.  Remember to make new logfile if changing"""
        self.ordertarget = newordertarget

    def getOrderTarget(self):
        """Get current order target value from this window class"""
        return self.ordertarget

    def getOrder(self):
        """Get the last value of order calculated"""
        return self.order

    def getOrderLast(self):
        """Get the value of order stored as the current prior reference state.  This is equal to current order if the last move with accepted."""
        return self.orderlast

    def getEnergy(self):
        """Return the most recently calculated value of energy"""
        return self.energy

    def getEnergyLast(self):
        """Return the most recently calculated value of the order parameter"""
        return self.energylast

    # Create a new log to write samples to
    def setLog(self, logname, extension="txt", mode="w"):
        self.log.close()
        self.log = open(logname + ".txt", mode)

    def setTrajectoryLength(self, newtrajectorylength):
        self.trajectorylength = newtrajectorylength

    def getTrajectoryLength(self):
        return self.trajectorylength

    def getCurrentSnapshot(self):
        return self.snapshot()

    def printUmbrellaVariables(self):
        print("\nUmbrella Variables:\n")
        print("k is set to " + str(self.k) + "\n")
        print("ordertarget is set to " + str(self.ordertarget) + "\n")
        print("trajectorylength is set to " + str(self.trajectorylength) + "\n")
        print("order is set to " + str(self.order) + "\n")
        print("energy is set to " + str(self.energy) + "\n")
