from GaussianFilter import *
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X
from Pose import *

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
class EKF(GaussianFilter):
    """
    Extended Kalman Filter class. Implements the :class:`GaussianFilter` interface for the particular case of the Extended Kalman Filter.
    """
    def __init__(self, x0, P0, *args):
        """
        Constructor of the EKF class.

        :param x0: initial mean state vector
        :param P0: initial covariance matrix
        :param args: arguments to be passed to the parent class
        """
        self.minK = 150  # minimum number of range measurements to process initially
        self.incK = 25  # minimum number of new range measurements to process for one ISAM update
        self.isam = gtsam.ISAM2()
        self.initial = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        # Noise models will be initialized lazily in Update()
        self.PRIOR_NOISE = None
        self.ODOMETRY_NOISE = None
        self.MEASUREMENT_NOISE = None
        super().__init__(x0, P0, *args)  # call parent constructor

    def f(self, xk_1, uk): # motion model
        """
        Motion model of the EKF **to be overwritten by the child class**.

        :param xk_1: previous mean state vector
        :param uk: input vector
        :return xk_bar, Pk_bar: predicted mean state vector and its covariance matrix
        """
        pass

    def Jfx(self, xk_1):
        """
        Jacobian of the motion model with respect to the state vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        pass

    def Jfw(self, xk_1):
        """
        Jacobian of the motion model with respect to the noise vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        pass

    def h(self, xk):  # observation model
        """
        The observation model of the EKF is given by:

        .. math::
            z_k=h(x_k,v_k)
            :label: eq-EKF-observation-model

        This method computes the mean of this direct observation model. Therefore it does not depend on v_k since it is
        a zero mean Gaussian noise.

        :param xk: mean of the predicted state vector. By default it is taken from the class attribute.
        :return: expected observation vector
        """
        pass

    def Prediction(self, uk, Qk, xk_1=None, Pk_1=None):
        """
        Prediction step of the EKF. It calls the motion model and its Jacobians to predict the state vector and its covariance matrix.

        :param uk: input vector
        :param Qk: covariance matrix of the noise vector
        :param xk_1: previous mean state vector. By default it is taken from the class attribute. Otherwise it updates the class attribute.
        :param Pk_1: covariance matrix of the previous state vector. By default it is taken from the class attribute. Otherwise it updates the class attribute.
        :return xk_bar, Pk_bar: predicted mean state vector and its covariance matrix. Also updated in the class attributes.
        """
        # logging for plotting
        self.Pk_1 = Pk_1 if Pk_1 is not None else self.Pk_1
        self.xk_1 = xk_1 if xk_1 is not None else self.xk_1
        

        self.uk = uk
        self.Qk = Qk  # store the input and noise covariance for logging

        self.xk_bar = self.f(self.xk_1,self.uk)
        
        self.Pk_bar = (self.Jfx(self.xk_1, self.uk) @ self.Pk_1 @ 
                    self.Jfx(self.xk_1, self.uk).T + 
                    self.Jfw(self.xk_1, self.uk) @ self.Qk @ 
                    self.Jfw(self.xk_1, self.uk).T)
        return self.xk_bar, self.Pk_bar

    # def Update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk, k, xk_1, uk):
    #     """
    #     Update step of the EKF. It calls the observation model and its Jacobians to update the state vector and its covariance matrix.

    #     :param zk: observation vector
    #     :param Rk: covariance matrix of the noise vector
    #     :param xk_bar: predicted mean state vector.
    #     :param Pk_bar: covariance matrix of the predicted state vector.
    #     :param Hk: Jacobian of the observation model with respect to the state vector.
    #     :param Vk: Jacobian of the observation model with respect to the noise vector.
    #     :return xk,Pk: updated mean state vector and its covariance matrix. Also updated in the class attributes.
    #     """
    #     # logging for plotting
    #     # print("This is Pk_bar inside update call in EKF.py",Pk_bar)
        
    #     # Lazy initialization of noise models (handle MRO edge cases)
    #     # if self.PRIOR_NOISE is None:
    #     self.PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    #     # if self.ODOMETRY_NOISE is None:
    #     #     self.ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    #     # MEASUREMENT_NOISE will be created from Rk when available
        
   





    #     self.xk_bar = xk_bar
    #     self.Pk_bar = Pk_bar
    #     self.zk = zk
    #     # in the case when the observation is just one number
    #     if type(zk) is int:
    #         self.nz = 0
    #     else:
    #         self.nz = zk.shape[0];  # store dimensionality of the observation
    #     self.Rk = Rk  # store the observation and noise covariance for logging
    #     # print("This is shape of Hk:",Hk.shape)
    #     # print("This is shape of Vk:",Vk.shape)
    #     # print("This is shape of Rk:",Rk.shape)
    #     # print("This is shape of Pk_bar:",Pk_bar.shape)
    #     # Calculate Kalman gain
    #     K = self.Pk_bar @ Hk.T @ np.linalg.inv(Hk @ self.Pk_bar @ Hk.T + Vk @ self.Rk @ Vk.T)  # Kalman gain
    #     # print("This is shape of K:",K.shape)
    #     # Update state and covariance matrix
    #     # print("This is shape of zk inside EKF update:",self.zk.shape)
    #     # Update state and covariance matrix
    #     # FIX: Calculate innovation and selectively wrap the angular component.
    #     # print("This is zk inside EKF update:",self.zk)
    #     # print("This is h(xk_bar) inside EKF update:",self.h(self.xk_bar))
    #     innovation = self.zk - self.h(self.xk_bar)
    #     # print(f"This is she shape of innovation {innovation.shape}")
    #     # Assuming the compass (yaw) measurement is the FIRST element in the stacked vector (index 0)
    #     # The following attribute is defined in the class FEKFMBL
    #     if getattr(self, 'zm_observed', True): # self.zm_observed is defined in the class FEKFSLAM and FEKFMBL. It is set to True when a compass measurement is observed and False otherwise. This is to avoid wrapping when there is no compass measurement.
    #         # print("Wrapping angle inside Update in EKF")
    #         innovation[0, 0] = wrap_angle(innovation[0, 0])
        
    #     self.xk = self.xk_bar + K@innovation




        
    #     # Lazy initialization of ISAM objects (handle MRO edge cases)
    #     if not hasattr(self, 'graph') or self.graph is None:
    #         self.graph = gtsam.NonlinearFactorGraph()
    #     if not hasattr(self, 'initial') or self.initial is None:
    #         self.initial = gtsam.Values()
    #     if not hasattr(self, 'isam') or self.isam is None:
    #         self.isam = gtsam.ISAM2()

    #     if k == 0:
    #     # We must add both X0 and X1 to the system the first time
    #         self.initial.insert(X(0), gtsam.Pose2(xk_1[0, 0], xk_1[1, 0], xk_1[2, 0]))
    #         self.graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(xk_1[0, 0], xk_1[1, 0], xk_1[2, 0]), self.PRIOR_NOISE))

    #     # Insert poses into the initial estimate
    #     # self.initial.insert(X(k), gtsam.Pose2(self.xk_1[0, 0], self.xk_1[1, 0], self.xk_1[2, 0]))
    #     self.initial.insert(X(k+1), gtsam.Pose2(self.xk[0, 0], self.xk[1, 0], self.xk[2, 0]))
        
    #     # Create odometry noise from EKF predicted covariance Pk_bar
    #     # Extract diagonal of Pk_bar to capture the motion uncertainty
    #     # if Pk_bar is not None and Pk_bar.size > 0:
    #     diag_Pk_bar = np.diag(Pk_bar)
    #     # Clamp diagonal elements to avoid near-zero values, then take sqrt
    #     diag_Pk_bar_clamped = np.maximum(diag_Pk_bar, 1e-6)
    #     odometry_sigmas = np.sqrt(diag_Pk_bar_clamped)
    #     odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(odometry_sigmas)
    #     # else:
    #     #     # Fallback to default if Pk_bar is unavailable
    #     #     if self.ODOMETRY_NOISE is None:
    #     #         self.ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    #     #     odometry_noise = self.ODOMETRY_NOISE
        
    #     # Add odometry factor with covariance from EKF prediction
    #     odometry = gtsam.Pose2(uk[0,0], uk[1,0], uk[2,0])
    #     self.graph.add(gtsam.BetweenFactorPose2(X(k), X(k+1), odometry, odometry_noise))
    #     # Create MEASUREMENT_NOISE from actual Rk if not already set
    #     # if Rk is not None:
    #     #     if np.ndim(Rk) == 2:
    #     #         sigmas = np.sqrt(np.diag(Rk))
    #     #     elif np.ndim(Rk) == 1:
    #     #         sigmas = np.sqrt(Rk)
    #     #     else:
    #     #         sigmas = np.sqrt(np.atleast_1d(Rk))
    #     #     self.MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        
    #     # Add compass factor as a rotation prior on X(k) when a compass measurement is available.
    #     if getattr(self, 'zm_observed', True) and isinstance(self.zk, np.ndarray) and self.zk.size > 0:
    #         compass_yaw = wrap_angle(float(self.zk[0, 0]))
    #         compass_var = float(self.Rk[0, 0]) if np.ndim(self.Rk) == 2 else float(self.Rk)
    #         compass_sigma = np.sqrt(max(compass_var, 1e-12))
    #         compass_noise = gtsam.noiseModel.Isotropic.Sigma(1, compass_sigma)
    #         self.graph.add(
    #             gtsam.PoseRotationPrior2D(X(k+1), gtsam.Rot2(compass_yaw), compass_noise)
    #         )

    #     # iSAM2 incremental update
    #     self.isam.update(self.graph, self.initial)
    #     result = self.isam.calculateEstimate()
    #     marginals = gtsam.Marginals(self.graph, result)

    #     # Extract covariance and pose from iSAM2 result
    #     self.Pk = marginals.marginalCovariance(X(k+1))
    #     pose_result = result.atPose2(X(k+1))
    #     self.xk = Pose3D(np.array([[pose_result.x()], [pose_result.y()], [pose_result.theta()]]))
    #     self.graph = gtsam.NonlinearFactorGraph()
    #     self.initial = gtsam.Values()
    #     return self.xk, self.Pk
    
    def Update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk, k, xk_1, uk, Qk):
        """
        Update step of the EKF + ISAM2.
        - Calculates standard EKF update first.
        - Uses ISAM2 to perform global smoothing of the trajectory.
        """
        # --- 1. Standard EKF Update Logic ---
        # Calculate Kalman gain
        S = Hk @ Pk_bar @ Hk.T + Vk @ Rk @ Vk.T
        K = Pk_bar @ Hk.T @ np.linalg.inv(S)
        
        # Calculate innovation and wrap the yaw angle
        innovation = zk - self.h(xk_bar)
        if getattr(self, 'zm_observed', True):
            innovation[0, 0] = wrap_angle(innovation[0, 0])
        
        # Corrected EKF state
        self.xk = xk_bar + K @ innovation

        # --- JOSEPH FORM IMPLEMENTATION ---
        # This replaces the simple P = (I - KH)P to maintain positive-definiteness
        I = np.eye(len(self.xk))
        # Note: We use Pk_bar here as it is the 'previous' covariance for this update step
        # We also include Vk to stay consistent with your Kalman Gain calculation
        term = I - K @ Hk
        self.Pk = term @ Pk_bar @ term.T + K @ (Vk @ Rk @ Vk.T) @ K.T

        # --- 2. ISAM2 Incremental Logic ---
        # These containers only hold the NEW data for this specific step 'k'
        self.PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))

        new_factors = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()

        # STEP 0: Anchor the graph (Run only once at the very beginning)
        # print(f"DEBUG: Step k={k}, connecting {X(k)} to {X(k+1)}")
        # print(f"\n--- DEBUG Step k={k} ---")
        # print(f"State vector xk_bar shape: {xk_bar.shape}")
        # print(f"Covariance Pk_bar shape: {Pk_bar.shape}")
        if k == 0:
            # We must provide X0 and its Prior for the system to have an origin [cite: 18, 48]
            x0_pose = gtsam.Pose2(xk_1[0, 0], xk_1[1, 0], xk_1[2, 0])
            new_values.insert(X(0), x0_pose)
            new_factors.add(gtsam.PriorFactorPose2(X(0), x0_pose, self.PRIOR_NOISE))

        
        diag_Pk_bar = np.diag(Qk)
        # Clamp diagonal elements to avoid near-zero values, then take sqrt
        diag_Pk_bar_clamped = np.maximum(diag_Pk_bar, 1e-6)
        odometry_sigmas = np.sqrt(diag_Pk_bar_clamped)
        odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(odometry_sigmas)
        # ODOMETRY_SIGMAS = np.array([0.1, 0.1, 0.05]) # Adjust based on your lab settings
        # odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(ODOMETRY_SIGMAS)
        # STEP k: Add the NEW Pose guess and the NEW BetweenFactor
        # X(k+1) is the variable we just estimated using the EKF math above
        new_pose_guess = gtsam.Pose2(self.xk[0, 0], self.xk[1, 0], self.xk[2, 0])
        new_values.insert(X(k + 1), new_pose_guess)
        
        # The BetweenFactor connects the previous pose X(k) to the new pose X(k+1) [cite: 19, 50]
        # We use a fixed ODOMETRY_NOISE as it represents sensor uncertainty, not state uncertainty
        odometry_measurement = gtsam.Pose2(uk[0, 0], uk[1, 0], uk[2, 0])
        new_factors.add(gtsam.BetweenFactorPose2(X(k), X(k + 1), odometry_measurement, odometry_noise))

        # STEP k: Add Observations (e.g., Compass/Rotation Prior)
        if getattr(self, 'zm_observed', True) and isinstance(zk, np.ndarray):
            compass_yaw = wrap_angle(float(zk[0, 0]))
            # Calculate sigma from the measurement noise Rk
            compass_sigma = np.sqrt(max(float(Rk[0, 0]), 1e-12))
            compass_noise = gtsam.noiseModel.Isotropic.Sigma(1, compass_sigma)
            # Add a prior constraint on the rotation of the LATEST pose
            new_factors.add(gtsam.PoseRotationPrior2D(X(k + 1), gtsam.Rot2(compass_yaw), compass_noise))

        try:
            self.isam.update(new_factors, new_values)
            result = self.isam.calculateEstimate()
            
            # Extract globally smoothed results
            # marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), result)
            self.Pk = self.isam.marginalCovariance(X(k+1))
            pose_res = result.atPose2(X(k + 1))
            
            # Update state with optimized values
            self.xk = np.array([[pose_res.x()], [pose_res.y()], [pose_res.theta()]])
            
        except RuntimeError as e:
            print(f"ISAM2 Error at step {k}: {e}")
            # If ISAM2 fails, we fall back to the EKF result so the loop doesn't die
            self.Pk = Pk_bar
        return self.xk, self.Pk