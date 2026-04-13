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
        # self.minK = 150  # minimum number of range measurements to process initially
        # self.incK = 25  # minimum number of new range measurements to process for one ISAM update
        # self.isam = gtsam.ISAM2()
        # self.initial = gtsam.Values()
        # self.graph = gtsam.NonlinearFactorGraph()
        # Noise models will be initialized lazily in Update()
        # self.PRIOR_NOISE = None
        # self.ODOMETRY_NOISE = None
        # self.MEASUREMENT_NOISE = None
        # Accumulators for batch updates
        # self.new_factors = gtsam.NonlinearFactorGraph()
        # self.new_values = gtsam.Values()
        # self.step_counter = 0  # Counter to track steps for batch updates
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

    
    def Update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk, k, xk_1, uk, Qk, zf, Rf, association, pose_index=None, last_pose_step=None, accumulated_odom=None):
        """
        Update step of the EKF + ISAM2 with batch accumulation.
        - Calculates standard EKF update first.
        - Accumulates factors and values over k steps.
        - Performs ISAM2 update every self.incK steps.
        """
        # --- 1. Standard EKF Update Logic ---
        # Calculate Kalman gain
                # Use provided values or fall back to instance variables
        pose_idx = pose_index if pose_index is not None else self.pose_index
        accum_odom = accumulated_odom if accumulated_odom is not None else self.accumulated_odom
        # if zk is not None:
        #     S = Hk @ Pk_bar @ Hk.T + Vk @ Rk @ Vk.T
        #     K = Pk_bar @ Hk.T @ np.linalg.inv(S)
        #     innovation = zk - self.h(xk_bar)
        #     if getattr(self, 'zm_observed', True):
        #         innovation[0, 0] = wrap_angle(innovation[0, 0])
            
        #     # Correct state and covariance (Joseph Form for stability)
        #     self.xk = xk_bar + K @ innovation
        #     I = np.eye(len(self.xk))
        #     term = I - K @ Hk
        #     self.Pk = term @ Pk_bar @ term.T + K @ (Vk @ Rk @ Vk.T) @ K.T
        # else:
        #     self.xk, self.Pk = xk_bar, Pk_bar
        #     print("No observation at this step, skipping EKF update.")
        self.xk, self.Pk = xk_bar, Pk_bar
        # --- 2. ISAM2 Incremental Logic with Accumulation ---
        # Initialize noise models once
        self.PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))

        # STEP 0: Anchor the graph (Run only once at the very beginning)
        if pose_idx == 0:
            # We must provide X0 and its Prior for the system to have an origin
            x0_pose = gtsam.Pose2(xk_1[0, 0], xk_1[1, 0], xk_1[2, 0])
            self.new_values.insert(X(pose_idx), x0_pose)
            self.new_factors.add(gtsam.PriorFactorPose2(X(pose_idx), x0_pose, self.PRIOR_NOISE))

            # PriorFactors for every landmark in the state


            landmark_prior_noise = gtsam.noiseModel.Isotropic.Sigma(2, 0.1)
            for j in range(len(self.M)):
                l_key = L(j)
                l_pos = gtsam.Point2(np.asarray([self.M[j][0], self.M[j][1]]).reshape(2,))
                self.new_values.insert(l_key, l_pos)
                self.new_factors.add(gtsam.PriorFactorPoint2(l_key, l_pos, landmark_prior_noise))

        # --- ADD ODOMETRY FACTOR ---
        # diag_Pk_bar = np.diag(Qk)
        # diag_Pk_bar_clamped = np.maximum(diag_Pk_bar, 1e-6)
        # odometry_sigmas = np.sqrt(diag_Pk_bar_clamped)
        # odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(odometry_sigmas)
        
        # # Add the NEW Pose guess and the NEW BetweenFactor to accumulators
        # new_pose_guess = gtsam.Pose2(self.xk[0, 0], self.xk[1, 0], self.xk[2, 0])
        # self.new_values.insert(X(k + 1), new_pose_guess)
        
        # odometry_measurement = gtsam.Pose2(uk[0, 0], uk[1, 0], uk[2, 0])
        # self.new_factors.add(gtsam.BetweenFactorPose2(X(k), X(k + 1), odometry_measurement, odometry_noise))

        odom_to_use = gtsam.Pose2(self.rel_disp[0,0], self.rel_disp[1,0], self.rel_disp[2,0])
        
        # Determine noise model based on accumulation:
        # - No accumulation: use motion/odometry noise Qk (single-step, direct odometry)
        # - With accumulation: use state covariance Pk_bar (accumulated steps with intermediate poses)
        if accum_odom is not None:
            # Accumulation occurred: use state covariance for odometry factor noise
            odometry_sigmas = np.sqrt(Pk_bar.diagonal()[:3])
        else:
            # No accumulation: use motion/odometry noise Qk for single-step odometry
            odometry_sigmas = np.sqrt(Qk.diagonal()[:3]) if Qk is not None else np.sqrt(Pk_bar.diagonal()[:3])
        
        # odometry_sigmas = np.maximum(odometry_sigmas, 1e-6)  # Ensure numerically valid
        odometry_noise = gtsam.noiseModel.Gaussian.Covariance(self.rel_cov + np.eye(self.xB_dim)*1e-6)
        
        # Add the NEW Pose guess and the NEW BetweenFactor to accumulators
        new_pose_guess = gtsam.Pose2(self.xk[0, 0], self.xk[1, 0], self.xk[2, 0])
        self.new_values.insert(X(pose_idx + 1), new_pose_guess)
        
        # odometry_measurement = gtsam.Pose2(odom_to_use[0, 0], odom_to_use[1, 0], odom_to_use[2, 0])
        self.new_factors.add(gtsam.BetweenFactorPose2(X(pose_idx), X(pose_idx + 1), odom_to_use, odometry_noise))

        # --- ADD LANDMARK MEASUREMENT FACTORS ---
        if getattr(self, 'zf_observed', True):
            sig_x = np.sqrt(Rf[0, 0])
            sig_y = np.sqrt(Rf[1, 1])
            bearing_sigma = sig_x
            range_sigma = sig_x
            # br_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([bearing_sigma, range_sigma]))

            for i, landmark_idx in enumerate(association):
                if landmark_idx is not None:
                    idx = i * self.zfi_dim
                    x_meas = zf[idx]
                    y_meas = zf[idx + 1]
                    
                    range_val = np.sqrt(x_meas**2 + y_meas**2)
                    bearing_val = np.arctan2(y_meas, x_meas)

                    J = np.array([
                        [-y_meas / (range_val**2), x_meas / (range_val**2)],
                        [x_meas / range_val,    y_meas / range_val]
                    ]).reshape(2, 2)
                    Rf_i = Rf[i*self.zfi_dim:(i+1)*self.zfi_dim, i*self.zfi_dim:(i+1)*self.zfi_dim]
                    R_polar = J @ Rf_i @ J.T
                    br_noise = gtsam.noiseModel.Gaussian.Covariance(R_polar)

                    self.new_factors.add(gtsam.BearingRangeFactor2D(
                        X(pose_idx + 1), L(int(landmark_idx)), 
                        gtsam.Rot2(bearing_val), range_val, br_noise))

        # --- ADD COMPASS/ROTATION MEASUREMENT ---
        if getattr(self, 'zm_observed', True) and isinstance(zk, np.ndarray):
            compass_yaw = wrap_angle(float(zk[0, 0]))
            compass_sigma = np.sqrt(max(float(Rk[0, 0]), 1e-12))
            compass_noise = gtsam.noiseModel.Isotropic.Sigma(1, compass_sigma)
            self.new_factors.add(gtsam.PoseRotationPrior2D(X(pose_idx + 1), gtsam.Rot2(compass_yaw), compass_noise))

        # Increment step counter
        # self.step_counter += 1
        
        # --- 3. PERFORM BATCH UPDATE EVERY incK STEPS ---
        print(f"\n--- Step k={k}, Counter={self.step_counter}, Accumulated factors={self.new_factors.size()} ---")
        
        # if self.step_counter >= self.incK:
        print(f"ISAM2 BATCH UPDATE: Processing {self.step_counter} accumulated steps !!!")
        try:
            self.isam.update(self.new_factors, self.new_values)
            result = self.isam.calculateEstimate()
            pose_res = result.atPose2(X(pose_idx + 1))
            
            if abs(pose_res.x()) > 1e5 or abs(pose_res.y()) > 1e5:
                print(f"Divergence detected at step {k}. Skipping ISAM re-injection.")
                # Reset accumulators for next batch
                self.new_factors = gtsam.NonlinearFactorGraph()
                self.new_values = gtsam.Values()
                self.step_counter = 0
                return self.xk, self.Pk

            # Check for missing keys
            if k > 0 and not result.exists(X(pose_idx + 1)):
                print(f"!!! CRITICAL: Key {X(pose_idx + 1)} (x{pose_idx + 1}) is missing from ISAM2!")

            for i in range(self.new_factors.size()):
                factor = self.new_factors.at(i)
                for key in factor.keys():
                    if not result.exists(key) and not self.new_values.exists(key):
                        print(f"FACTOR ERROR: Factor {i} refers to missing Key {gtsam.DefaultKeyFormatter(key)}")

            print(f"ISAM2 batch update successful at step {k}. Extracting results...")
            
            # --- Extract optimized state ---
            pose_res = result.atPose2(X(pose_idx + 1))
            optimized_xk = np.array([[pose_res.x()], [pose_res.y()], [pose_res.theta()]])
            
            keys = gtsam.KeyVector()
            keys.append(X(pose_idx + 1))

            for j in range(self.nf):
                if result.exists(L(j)):
                    l_res = result.atPoint2(L(j))
                    optimized_xk = np.vstack((optimized_xk, l_res.reshape(2, 1)))
                    keys.append(L(j))
                else:
                    print(f"!!! WARNING: L({j}) missing from ISAM result at step {k} !!!")
                    l_old = self.xk[self.xBpose_dim + j*self.zfi_dim : self.xBpose_dim + (j+1)*self.zfi_dim]
                    optimized_xk = np.vstack((optimized_xk, l_old))
            
            self.xk = optimized_xk

            # # --- Extract covariance (OLD VERSION - only pose marginal) ---
            # new_Pk = self.Pk.copy()
            # print(f"STEP {k}: Extracting pose marginal covariance from ISAM2...")
            # marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), result)
            # # Extract only the pose marginal covariance to keep global uncertainty bounded
            # # (not full joint with landmarks, which can inflate uncertainty)
            # pose_cov = marginals.marginalCovariance(X(pose_idx + 1))
            # if not np.any(np.isnan(pose_cov)):
            #     # Update only the pose part (first 3x3), keep landmark structure
            #     new_Pk[0:3, 0:3] = pose_cov
            # 
            # self.Pk = new_Pk

            # --- Extract covariance (NEW VERSION - pose + landmark marginals) ---
            new_Pk = self.Pk.copy()
            print(f"STEP {k}: Extracting pose and landmark marginal covariances from ISAM2...")
            marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), result)
            
            # Extract pose marginal to keep global uncertainty bounded
            pose_cov = marginals.marginalCovariance(X(pose_idx + 1))
            if not np.any(np.isnan(pose_cov)):
                new_Pk[0:3, 0:3] = pose_cov
            
            # Also extract and update landmark marginals
            for j in range(self.nf):
                if result.exists(L(j)):
                    try:
                        landmark_cov = marginals.marginalCovariance(L(j))
                        if not np.any(np.isnan(landmark_cov)):
                            new_Pk[self.xBpose_dim + j*self.zfi_dim : self.xBpose_dim + (j+1)*self.zfi_dim,
                                   self.xBpose_dim + j*self.zfi_dim : self.xBpose_dim + (j+1)*self.zfi_dim] = landmark_cov
                    except Exception as e:
                        print(f"Warning: Could not extract landmark {j} marginal: {e}")
            
            self.Pk = new_Pk
            
        except RuntimeError as e:
            print(f"ISAM2 Error at step {k}: {e}")
            self.Pk = Pk_bar
        
        # --- RESET ACCUMULATORS FOR NEXT BATCH ---
        print(f"Resetting factor/value accumulators for next batch...")
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
            # self.step_counter = 0

        return self.xk, self.Pk