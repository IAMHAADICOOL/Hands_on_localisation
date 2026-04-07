from FEKFSLAM import *
from FEKFMBL import *
from EKF_3DOFDifferentialDriveInputDisplacement import *
from Pose import *
from blockarray import *
from MapFeature import *
import numpy as np
from FEKFSLAMFeature import *
import gtsam

class FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM(FEKFSLAM2DCartesianFeature, FEKFSLAM, EKF_3DOFDifferentialDriveInputDisplacement):
    def __init__(self, *args):

        self.Feature = globals()["CartesianFeature"]
        self.Pose = globals()["Pose3D"]
        self.isam = gtsam.ISAM2()
        super().__init__(*args)


    # def GetFeatures(self):
    # Get features is inherited from EKF_3DOFDifferentialDriveInputDisplacement


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6, 1))
    kSteps = 2000
    alpha = 0.99

    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    x0 = Pose3D(np.zeros((3, 1)))
    dr_robot = DR_3DOFDifferentialDrive(index, kSteps, robot, x0)
    robot.SetMap(M)


    n_features = len(M)
    total_dim = 3 + n_features * 2

    x0_robot = np.zeros((3, 1)) # Initial robot pose (0,0,0)
    x0_slam = np.zeros((total_dim, 1))
    x0_slam[0:3] = x0_robot

    for i, feature in enumerate(M):
        idx = 3 + i * 2
        x0_slam[idx : idx + 2] = feature
        
    # Set landmark uncertainty to be small (e.g., 0.01) as per instructions
    P0_slam = np.eye(total_dim) * 0.01 
    # P0 = np.zeros((3, 3))
    P0_robot = np.diag([0.1, 0.1, 0.01])
    P0_slam[0:3, 0:3] = P0_robot


    auv = FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM(M, alpha, kSteps, robot)

    
    usk=np.array([[0.5, 0, 0.03]]).T
    auv.LocalizationLoop(x0_slam, P0_slam, usk)

    exit(0)
