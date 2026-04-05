from Localization import *
import numpy as np
from Pose import *

class DR_3DOFDifferentialDrive(Localization):
    """
    Dead Reckoning Localization for a Differential Drive Mobile Robot.
    """
    def __init__(self, index, kSteps, robot, x0, *args):
        """
        Constructor of the :class:`prlab.DR_3DOFDifferentialDrive` class.

        :param args: Rest of arguments to be passed to the parent constructor
        """

        super().__init__(index, kSteps, robot, x0, *args)  # call parent constructor

        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.t_1 = 0.0  # t_1 is the previous time at which we iterated the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn
        self.nu_k_1 = np.zeros((3, 1))  # previous velocity vector
        
    def Localize(self, xk_1, uk):  # motion model
        """
        Motion model for the 3DOF (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`) Differential Drive Mobile robot using as input the readings of the wheel encoders (:math:`u_k=[n_L~n_R]^T`).

        :parameter xk_1: previous robot pose estimate (:math:`x_{k-1}=[x_{k-1}~y_{k-1}~\psi_{k-1}]^T`)
        :parameter uk: input vector (:math:`u_k=[u_{k}~v_{k}~w_{k}~r_{k}]^T`)
        :return xk: current robot pose estimate (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`)
        """

        # Extract previous pose
        eta_k_1 = Pose3D(xk_1[0:3])  # previous position
        # Create velocity vector
        nu_k = np.zeros((3, 1))  # current velocity vector

        # Store previous state and input for Logging purposes
        self.etak_1 = xk_1  # store previous state
        if uk is not None:
            self.uk = uk  # store input
            # Dead Reckoning Localization
            # Calculate current velocities based on the wheel encoder readings
            n_L = uk[0, 0]  # left wheel encoder reading
            n_R = uk[1, 0]  # right wheel encoder reading
            v_L = ((n_L * 2 * np.pi / self.robot.pulse_x_wheelTurns) * self.robot.encoder_reading_frequency) * self.wheelRadius  # left wheel linear velocity
            v_R = ((n_R * 2 * np.pi / self.robot.pulse_x_wheelTurns) * self.robot.encoder_reading_frequency) * self.wheelRadius  # right wheel linear velocity
            v = (v_R + v_L) / 2  # linear velocity
            w = (v_R - v_L) / self.wheelBase  # angular velocity
            nu_k[0, 0] = v  # linear velocity
            nu_k[1, 0] = 0  # lateral velocity
            self.nu_k_1 = nu_k  # store previous velocity
            

        else:
            # no new encoder readings using previous velocities
            nu_k[0, 0] = self.nu_k_1[0, 0]  # previous linear velocity
            nu_k[2, 0] = self.nu_k_1[2, 0]  # previous angular velocity

        # update new pose
        eta_k = Pose3D.oplus(eta_k_1, Pose3D(nu_k * self.dt))  # new robot pose with noise
        
        # construct the new state vector
        xk = np.vstack((eta_k.reshape(3,1), nu_k.reshape(3,1)))  # new robot state
        
        return xk

        # Mapping 

        pass

    def GetInput(self):
        """
        Get the input for the motion model. In this case, the input is the readings from both wheel encoders.

        :return: uk:  input vector (:math:`u_k=[n_L~n_R]^T`)
        """

        uk, Rk = self.robot.ReadEncoders()  # get the wheel encoder readings
        
        return uk
        

