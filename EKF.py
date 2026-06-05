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
        Update step of the EKF + ISAM2.
        - Calculates standard EKF update first.
        - Uses ISAM2 to perform global smoothing of the trajectory.
        """
        # --- 1. Standard EKF Update Logic ---
        # Calculate Kalman gain

        pose_idx = pose_index if pose_index is not None else self.pose_index

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
        # --- 2. ISAM2 Incremental Logic ---
        # These containers only hold the NEW data for this specific step 'k'
        self.xk, self.Pk = xk_bar, Pk_bar
        self.PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))

        # new_factors = gtsam.NonlinearFactorGraph()
        # new_values = gtsam.Values()

        # STEP 0: Anchor the graph (Run only once at the very beginning)
        print(f"DEBUG: Step k={k}, connecting {X(k)} to {X(k+1)}")
        # print(f"\n--- DEBUG Step k={k} ---")
        # print(f"State vector xk_bar shape: {xk_bar.shape}")
        # print(f"Covariance Pk_bar shape: {Pk_bar.shape}")
        if pose_idx == 0:
            # We must provide X0 and its Prior for the system to have an origin [cite: 18, 48]
            x0_pose = gtsam.Pose2(xk_1[0, 0], xk_1[1, 0], xk_1[2, 0])
            self.new_values.insert(X(pose_idx), x0_pose)
            self.new_factors.add(gtsam.PriorFactorPose2(X(pose_idx), x0_pose, self.PRIOR_NOISE))

            # PriorFactors for every landmark in the state
            # landmark_prior_noise = gtsam.noiseModel.Isotropic.Sigma(2, 0.1) # Small uncertainty
            # for j in range(len(self.M)):
            #     l_key = L(j)
            #     # landmark = np.array
            #     l_pos = gtsam.Point2(np.asarray([self.M[j][0], self.M[j][1]]).reshape(2,))
            #     new_values.insert(l_key, l_pos)
            #     new_factors.add(gtsam.PriorFactorPoint2(l_key, l_pos, landmark_prior_noise))

        if getattr(self, 'zf_observed', True):
            # Measurement noise: sig_x and sig_y from Rf
            sig_x = np.sqrt(Rf[0,0])
            sig_y = np.sqrt(Rf[1,1])
            # For isotropic noise in GTSAM, we can take the average or use a Diagonal model
            bearing_sigma = 0.1 # Adjust based on sensor quality
            range_sigma = sig_x # Simplified
            
            # br_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([bearing_sigma, range_sigma]))

            for i, landmark_idx in enumerate(association):
                if landmark_idx is not None:
                    # 1. Correctly extract the Cartesian pair for this landmark
                    idx = i * self.zfi_dim
                    x_meas = zf[idx]
                    y_meas = zf[idx+1]
                    
                    # 2. Convert Cartesian to Bearing/Range
                    range_val = np.sqrt(x_meas**2 + y_meas**2)
                    bearing_val = np.arctan2(y_meas, x_meas)
                    J = np.array([
                        [-y_meas / (range_val**2), x_meas / (range_val**2)],
                        [x_meas / range_val,    y_meas / range_val]
                    ]).reshape(2, 2)
                    Rf_i = Rf[i*self.zfi_dim:(i+1)*self.zfi_dim, i*self.zfi_dim:(i+1)*self.zfi_dim]
                    R_polar = J @ Rf_i @ J.T
                    br_noise = gtsam.noiseModel.Gaussian.Covariance(R_polar)
                    # 3. Add a BearingRange factor (Constrains the landmark in 2D)
                    self.new_factors.add(gtsam.BearingRangeFactor2D(
                        X(pose_idx + 1), L(int(landmark_idx)), 
                        gtsam.Rot2(bearing_val), range_val, br_noise))
        # eps = 1e-6
        # diag_Pk_bar = np.diag(Qk)
        # # Clamp diagonal elements to avoid near-zero values, then take sqrt
        # diag_Pk_bar_clamped = np.maximum(diag_Pk_bar, 1e-6)
        # odometry_sigmas = np.sqrt(diag_Pk_bar_clamped)
        odometry_noise = gtsam.noiseModel.Gaussian.Covariance(self.rel_cov + np.eye(self.xB_dim)*1e-6)
        # ODOMETRY_SIGMAS = np.array([0.1, 0.1, 0.05]) # Adjust based on your lab settings
        # odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(ODOMETRY_SIGMAS)
        # STEP k: Add the NEW Pose guess and the NEW BetweenFactor
        # X(k+1) is the variable we just estimated using the EKF math above
        new_pose_guess = gtsam.Pose2(self.xk[0, 0], self.xk[1, 0], self.xk[2, 0])
        self.new_values.insert(X(pose_idx + 1), new_pose_guess)
        
        # The BetweenFactor connects the previous pose X(k) to the new pose X(k+1) [cite: 19, 50]
        # We use a fixed ODOMETRY_NOISE as it represents sensor uncertainty, not state uncertainty
        odom_to_use = gtsam.Pose2(self.rel_disp[0,0], self.rel_disp[1,0], self.rel_disp[2,0])

        self.new_factors.add(gtsam.BetweenFactorPose2(X(pose_idx), X(pose_idx + 1), odom_to_use, odometry_noise))

        # STEP k: Add Observations (e.g., Compass/Rotation Prior)
        if getattr(self, 'zm_observed', True) and isinstance(zk, np.ndarray):
            compass_yaw = wrap_angle(float(zk[0, 0]))
            # Calculate sigma from the measurement noise Rk
            compass_sigma = np.sqrt(max(float(Rk[0, 0]), 1e-12))
            compass_noise = gtsam.noiseModel.Isotropic.Sigma(1, compass_sigma)
            # Add a prior constraint on the rotation of the LATEST pose
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
            
            # --- 1. Extraction with Key Safety ---
            pose_res = result.atPose2(X(pose_idx + 1))
            optimized_xk = np.array([[pose_res.x()], [pose_res.y()], [pose_res.theta()]])
            
            keys = gtsam.KeyVector()
            keys.append(X(pose_idx+1))

            for j in range(self.nf):
                # Check if the landmark actually made it into the graph
                if result.exists(L(j)):
                    l_res = result.atPoint2(L(j))
                    optimized_xk = np.vstack((optimized_xk, l_res.reshape(2, 1)))
                    keys.append(L(j))
                else:
                    # If it's missing, we must keep the EKF estimate to avoid shape errors
                    print(f"!!! WARNING: L({j}) missing from ISAM result at step {k} !!!")
                    l_old = self.xk[self.xBpose_dim + j*self.zfi_dim : self.xBpose_dim + (j+1)*self.zfi_dim]
                    optimized_xk = np.vstack((optimized_xk, l_old))
            
            self.xk = optimized_xk

            # # --- 2. Covariance Extraction with Individual Checks ---
            marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), result)
            
            # # # Request the joint matrix
            full_joint_matrix = marginals.jointMarginalCovariance(keys).fullMatrix()
            
            if np.any(np.isnan(full_joint_matrix)):
                print(f"\n[DEBUG] NaN found in Joint Matrix at Step {k}. Checking marginals:")
                for key in keys:
                    m = marginals.marginalCovariance(key)
                    if np.any(np.isnan(m)):
                        # Symbol.string() helps see if it's x3, l2, etc.
                        print(f"  -> Key {gtsam.DefaultKeyFormatter(key)} contains NaNs!")
                
                # Fallback: If joint is broken, try to use Pose marginal + identity for landmarks
                # This prevents the SVD converge crash
                self.Pk = Pk_bar 
            else:
                self.Pk = full_joint_matrix
            # # Define the size of the full state vector
            # dim = self.xBpose_dim + self.nf * self.zfi_dim
            # new_Pk = np.zeros((dim, dim))

            # # 1. Get Robot Pose Marginal (The 3x3 top-left block)
            # new_Pk[0:3, 0:3] = marginals.marginalCovariance(X(pose_idx + 1))

            # # 2. Get individual Landmark Marginals (The 2x2 diagonal blocks)
            # for j in range(self.nf):
            #     start = self.xBpose_dim + j * self.zfi_dim
            #     if result.exists(L(j)):
            #         try:
            #             # Extract only the 2x2 diagonal block for this landmark
            #             new_Pk[start:start+2, start:start+2] = marginals.marginalCovariance(L(j))
            #         except:
            #             # If a specific landmark fails, give it a tiny identity covariance 
            #             # so the SVD plotter doesn't crash
            #             # new_Pk[start:start+2, start:start+2] = np.eye(2) * 0.1
            #             pass
            #     else:
            #         # Fallback for landmarks not in the graph
            #         # new_Pk[start:start+2, start:start+2] = np.eye(2) * 0.1
            #         pass

            # self.Pk = new_Pk
            # # joint = marginals.jointMarginalCovariance(keys).fullMatrix()

            # # The joint matrix is ordered by keys: [pose(3x3), L0(2x2), L1(2x2), ...]
            # # Build your full Pk from it
            # # self.Pk = joint  # This IS the correct full covariance
            # print(f"ISAM2 batch update successful at step {k}. Extracting results...")

        except RuntimeError as e:
            print(f"ISAM2 Error at step {k}: {e}")
            # If ISAM2 fails, we fall back to the EKF result so the loop doesn't die
            pass

        print(f"Resetting factor/value accumulators for next batch...")
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
            # self.step_counter = 0
        return self.xk, self.Pk