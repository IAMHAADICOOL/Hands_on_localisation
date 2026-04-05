from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *
from MapFeature import *


class EKF_3DOFDifferentialDriveInputDisplacement(GFLocalization, DR_3DOFDifferentialDrive, EKF):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor. Creates the list of  :class:`IndexStruct.IndexStruct` instances which is required for the automated plotting of the results.
        Then it defines the inital stawe vecto mean and covariance matrix and initializes the ancestor classes.

        :param kSteps: number of iterations of the localization loop
        :param robot: simulated robot object
        :param args: arguments to be passed to the base class constructor
        """

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((3, 1))  # initial state x0=[x y psi]^T
        P0 = np.zeros((3, 3))  # initial covariance

        
        # this is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]

        # previous yaw reading
        self.zk_1 = None

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1 
        
        super().__init__(index, kSteps, robot, x0, P0, *args)

    def f(self, xk_1, uk):
        # Extract previous pose
        eta_k_1 = xk_1 # previous position

        # update new pose
        xk_bar = Pose3D.oplus(Pose3D(xk_1), Pose3D(uk))
        
        return xk_bar
    
# there is a problem in the computation of the Jacobians
    
 
    def Jfx(self, xk_1, uk=None):  
        psi = xk_1[2, 0]
        
        
        if uk is not None:
            dx = uk[0, 0]
            dy = uk[1, 0]
        else:
            dx = dy = 0.0
        
        J = np.array([
            [1, 0, -dx * np.sin(psi) - dy * np.cos(psi)],
            [0, 1,  dx * np.cos(psi) - dy * np.sin(psi)],
            [0, 0,  1]
        ])
        return J

    def Jfw(self, xk_1, uk=None):  
        psi = xk_1[2, 0]
        
        J = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [0,            0,            1]
        ])
        return J

    def h(self, xk):  #:hm(self, xk):
        h = xk[2,0]
        return h  # return the expected observations

    def GetInput(self):
        """
        zk = self.zk_1
        :return: uk,Qk
        """
        uk, Qk = self.robot.ReadEncoders()  # get the wheel encoder readings
        nu_k = np.zeros((3,1)) # current velocity vector
        
        # Store previous state and input for Logging purposes
        if uk is not None:
            self.uk = uk  # store input
            # Calculate current velocities based on the wheel encoder readings
            n_L = uk[0, 0]  # left wheel encoder reading
            n_R = uk[1, 0]  # right wheel encoder reading
            v_L = ((n_L * 2 * np.pi / self.robot.pulse_x_wheelTurns) * self.robot.encoder_reading_frequency) * self.wheelRadius  # left wheel linear velocity
            v_R = ((n_R * 2 * np.pi / self.robot.pulse_x_wheelTurns) * self.robot.encoder_reading_frequency) * self.wheelRadius  # right wheel linear velocity
            
            v = (v_R + v_L) / 2  # linear velocity
            w = (v_R - v_L) / self.wheelBase  # angular velocity
            nu_k[0, 0] = v  # linear velocity
            nu_k[1, 0] = 0  # lateral velocity
            nu_k[2, 0] = w  # angular velocity
            self.nu_k_1 = nu_k  # store previous velocity


        else:
            # no new encoder readings using previous velocities
            # print(f"This is the shape of nu_k {nu_k.shape} inside else")
            nu_k[0, 0] = self.nu_k_1[0, 0]  # previous linear velocity
            nu_k[1, 0] = self.nu_k_1[1, 0]  # lateral velocity
            nu_k[2, 0] = self.nu_k_1[2, 0]  # previous angular velocity
        B_1 = ((2*np.pi/self.robot.pulse_x_wheelTurns)*self.robot.encoder_reading_frequency) * self.wheelRadius
        # The following is the mathematics involved in computing the covariances through the magic table:
        # [vr,vL]^T = [nr, nl]^T * [[B, 0],[B,0]]
        # Taking [[B,0],[B,0]] = C
        # assuming [vr, vl]^T = v
        # and [nr, nl]^T = n
        # therefore
        # v' = n * C
        # The covariance of the error in of n is Qk
        # therefore, using the magic table, the covariance of error of v' will be
        # C*Qk*C^T=Q_v'
        # where 
        C = np.array([[B_1, 0],
                        [0, B_1]])
        Q_v_rl = C@Qk@C.T
        # now, even though there is no velocity reading in y, to stay consistent
        # with matrix multiplication, we will have to add one column for velocity along y
        # and appropiately extend the Q_v_rl
        Q_vrl_modified = np.asarray([[Q_v_rl[0,0], 0, 0],
                            [0,0,0],
                            [0,0,Q_v_rl[1,1]]])
        # the velocity in x, y and angular velocity can be written as
        # matrix multiplication
        # [v_x, v_y, v_theta]^T = [[1/2, 0, 1/2],
        #                          [0, 0, 0],
        #                          [1/L, 0, 1/L]]*[v_R, 0, v_L]^
        # L is the self.wheelBase
        # the following equations implement the same, just in scalar form
        # Assuming [[1/2, 0, 1/2],
        #           [0, 0, 0],
#                       [1/L, 0, 1/L]]=D, we have
        # v = D*v'
        # now the covariance of error of v' is Q_vrl_modified
        # therefore, using the magic table, the covariance of error of v will be:
        # Q_v = D*Q_vrl_modified*D^T
        L = self.wheelBase
        D = np.asarray([[0.5, 0, 0.5],
                        [0,0,0],
                        [1/L, 0, -1/L]])
        Q_v = D @ Q_vrl_modified @ D.T

        # now finally, its time to compute displacement in x,y, theta wrt to robot frame
        # [delta_x, delta_y, delta_theta]^T = [v_x, v_y, v_theta]^T*[[dt, 0, 0],
        #                                                            [0, dt, dt],
        #                                                            [0, 0, dt]]
        # Assuming E = [[dt, 0, 0],
                    # [0, dt, dt],
                    # [0, 0, dt]]
        # now, the covariance of error in v_x, v_y, v_theta is Q_v, therefore,
        # using the magic table, the covariance of error in displacement would be 
        # Q_delta = E*Q_v*E^T
        E = np.asarray([[self.dt, 0, 0],
                        [0, self.dt, 0],
                        [0, 0, self.dt]])
        Q_displacement = E @ Q_v @ E.T

        uk = nu_k * self.dt
        return uk, Q_displacement

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        zk, Rk = self.robot.ReadCompass()
        
        if zk is not None:
            zk = np.array([[zk]]) if np.isscalar(zk) else np.atleast_2d(zk).reshape(-1, 1)
            self.zk_1 = zk
        else:
            if self.zk_1 is None:
                # First measurement, use initial state estimate
                self.zk_1 = np.array([[self.xk_1[2, 0]]])  # Initial yaw
            # zk = self.zk_1

        if np.isscalar(Rk):
            Rk = np.array([[Rk]])
        # not sure if the following jacobians are correct
        Hk = np.array([[0, 0, 1]])
        Vk = np.array([[1]])
        
        return zk, Rk, Hk, Vk


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = Pose3D(np.zeros((3, 1)))
    P0 = np.zeros((3, 3))
    dd_robot = EKF_3DOFDifferentialDriveInputDisplacement(kSteps,robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5,0, 0.03]]).T)

    exit(0)