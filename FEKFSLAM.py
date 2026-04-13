from MapFeature import *
from FEKFMBL import *

class FEKFSLAM(FEKFMBL):
    """
    Class implementing the Feature-based Extended Kalman Filter for Simultaneous Localization and Mapping (FEKFSLAM).
    It inherits from the FEKFMBL class, which implements the Feature Map based Localization (MBL) algorithm using an EKF.
    :class:`FEKFSLAM` extends :class:`FEKFMBL` by adding the capability to map features previously unknown to the robot.
    """
 
    def __init__(self,  *args):

        super().__init__(*args)

        # self.xk_1 # state vector mean at time step k-1 inherited from FEKFMBL

        self.nzm = 0  # number of measurements observed
        self.nzf = 0  # number of features observed

        self.H = None  # Data Association Hypothesis
        self.nf = 0  # number of features in the state vector

        self.plt_MappedFeaturesEllipses = []

        return

 
    def AddNewFeatures(self, xk, Pk, znp, Rnp):
        """
        This method adds new features to the map. Given:

        * The SLAM state vector mean and covariance:

        .. math::
            {{}^Nx_{k}} & \\approx {\\mathcal{N}({}^N\\hat x_{k},{}^NP_{k})}\\\\
            {{}^N\\hat x_{k}} &=   \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k}}&=
            \\begin{bmatrix}
            {{}^NP_{B}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance

        * And the vector of non-paired feature observations (feature which have not been associated with any feature in the map), and their covariance matrix:

            .. math::
                {z_{np}} &=   \\left[ {}^Bz_{F_1} ~  \\cdots ~ {}^Bz_{F_{n_{zf}}}  \\right]^T \\\\
                {R_{np}}&= \\begin{bmatrix}
                {}^BR_{F_1} &  \\cdots & 0  \\\\
                \\vdots &  \\ddots & \\vdots \\\\
                0 & \\cdots & {}^BR_{F_{n_{zf}}}
                \\end{bmatrix}
                :label: FEKFSLAM-non-paire-feature-observations

        this method creates a grown state vector ([xk_plus, Pk_plus]) by adding the new features to the state vector.
        Therefore, for each new feature :math:`{}^Bz_{F_i}`, included in the vector :math:`z_{np}`, and its corresponding feature observation noise :math:`{}^B R_{F_i}`, the state vector mean and covariance are updated as follows:

            .. math::
                {{}^Nx_{k}^+} & \\approx {\\mathcal{N}({}^N\\hat x_{k}^+,{}^NP_{k}^+)}\\\\
                {{}^N x_{k}^+} &=
                \\left[ {{}^N x_{B_k}^T} ~ {{}^N x_{F_1}^T} ~ \\cdots ~{{}^N x_{F_n}^T}~ |~\\left({{}^N x_{B_k} \\boxplus ({}^Bz_{F_i} }+v_k)\\right)^T \\right]^T \\\\
                {{}^N\\hat x_{k}^+} &=
                \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~ \\cdots ~{{}^N\\hat x_{F_n}^T}~ |~{{}^N\\hat x_{B_k} \\boxplus {}^Bz_{F_i}^T } \\right]^T \\\\
                {P_{k}^+}&= \\begin{bmatrix}
                {{}^NP_{B_k}}  &  {{}^NP_{B_kF_1}}   &  \\cdots   &  {{}^NP_{B_kF_n}} & | & {{}^NP_{B_k} J_{1 \\boxplus}^T}\\\\
                {{}^NP_{F_1B_k}}  &  {{}^NP_{F_1}}   &  \\cdots   &  {{}^NP_{F_1F_n}} & | & {{}^NP_{F_1B_k} J_{1 \\boxplus}^T}\\\\
                \\vdots  & \\vdots & \\ddots  & \\vdots & | &  \\vdots \\\\
                {{}^NP_{F_nB_k}}  &  {{}^NP_{F_nF_1}}   &  \\cdots   &  {{}^NP_{F_n}}  & | & {{}^NP_{F_nB_k} J_{1 \\boxplus}^T}\\\\
                \\hline
                {J_{1 \\boxplus} {}^NP_{B_k}}  &  {J_{1 \\boxplus} {}^NP_{B_kF_1}}   &  \\cdots   &  {J_{1 \\boxplus} {}^NP_{B_kF_n}}  & | &  {J_{1 \\boxplus} {}^NP_R J_{1 \\boxplus} ^T} + {J_{2\\boxplus}} {{}^BR_{F_i}} {J_{2\\boxplus}^T}\\\\
                \\end{bmatrix}
                :label: FEKFSLAM-add-a-new-feature

        :param xk: state vector mean
        :param Pk: state vector covariance
        :param znp: vector of non-paired feature observations (they have not been associated with any feature in the map)
        :param Rnp: Matrix of non-paired feature observation covariances
        :return: [xk_plus, Pk_plus] state vector mean and covariance after adding the new features
        """

        if znp is None or len(znp) == 0:
            return xk, Pk
        
        xk_plus = xk
        Pk_plus = Pk

        number_of_new_features = znp.size // self.zfi_dim
        ## To be completed by the student
        self.nf += number_of_new_features
        for i in range(number_of_new_features):
            
            idx_start = i * self.zfi_dim
            idx_end = idx_start + self.zfi_dim
            
            zi = znp[idx_start:idx_end]
            Ri = Rnp[idx_start:idx_end, idx_start:idx_end]
            
            
            x_B = self.GetRobotPose(xk_plus) 
            
            x_new_feature = self.g(x_B, self.Feature(zi))
            
            # self.M.append(x_new_feature)
            
            xk_plus = np.vstack((xk_plus, x_new_feature))
            
            J1 = self.Jgx(x_B, self.Feature(zi)) 
            J2 = self.Jgv(x_B, self.Feature(zi))

            P_B = self.GetRobotPoseCovariance(Pk_plus)
            
            
            P_new_new = J1 @ P_B @ J1.T + J2 @ Ri @ J2.T
            
            # Calculate Cross-Correlations (Bottom-Left block)
            # need the correlation between the Robot and EVERYTHING else in the map
            # P_robot_all is the rows of Pk corresponding to the robot pose
            P_robot_all = Pk_plus[0:self.xBpose_dim, :]
            
            # Formula: J1 * P_robot_all
            P_new_old = J1 @ P_robot_all
            
            # Construct the new Pk matrix using block logic
            # [ P_old       P_new_old.T ]
            # [ P_new_old   P_new_new   ]
            
            Pk_plus = np.block([
                [Pk_plus,     P_new_old.T],
                [P_new_old,   P_new_new]
            ])

        return xk_plus, Pk_plus

    def Prediction(self, uk, Qk, xk_1, Pk_1):
        """
        This method implements the prediction step of the FEKFSLAM algorithm. It predicts the state vector mean and
        covariance at the next time step. Given state vector mean and covariance at time step k-1:

        .. math::
            {}^Nx_{k-1} & \\approx {\\mathcal{N}({}^N\\hat x_{k-1},{}^NP_{k-1})}\\\\
            {{}^N\\hat x_{k-1}} &=   \\left[ {{}^N\\hat x_{B_{k-1}}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k-1}}&=
            \\begin{bmatrix}
            {{}^NP_{B_{k-1}}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance-k-1

        the control input and its covariance :math:`u_k` and :math:`Q_k`, the method computes the state vector mean and covariance at time step k:

        .. math::
            {{}^N\\hat{\\bar x}_{k}} &=   \\left[ {f} \\left( {{}^N\\hat{x}_{B_{k-1}}}, {u_{k}}  \\right)  ~  { {}^N\\hat x_{F_1}^T} \\cdots { {}^N\\hat x_{F_n}^T}\\right]^T\\\\
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}
            :label: FEKFSLAM-prediction-step

        where

        .. math::
            {F_{1_k}} &= \\left.\\frac{\\partial {f_S({}^Nx_{k-1},u_k,w_k)}}{\\partial {{}^Nx_{k-1}}}\\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}} \\\\
             &=
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{B_{k-1}}}} &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{Fn}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}} \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_x}} & {0} & \\cdots & {0} \\\\
            {0}   & {I} & \\cdots & {0} \\\\
            \\vdots& \\vdots  & \\ddots & \\vdots  \\\\
            {0}   & {0} & \\cdots & {I} \\\\
            \\end{bmatrix}
            \\\\{F_{2_k}} &= \\left. \\frac{\\partial {f({}^Nx_{k-1},u_k,w_k)}}{\\partial {w_{k}}} \\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}}
            =
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {w_{k}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {w_{k}}}\\\\
            \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {w_{k}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_w}}\\\\
            {0}\\\\
            \\vdots\\\\
            {0}\\\\
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-Jacobian

        obtaining the following covariance matrix:
        
        .. math::
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}{{}^N\\bar P_{k}}
             &=
            \\begin{bmatrix}
            {J_{f_x}P_{B_{k-1}} J_{f_x}^T} + {J_{f_w}Q J_{f_w}^T}  & |  &  {J_{f_x}P_{B_kF_1}} & \\cdots & {J_{f_x}P_{B_kF_n}}\\\\
            \\hline
            {{}^NP_{F_1B_k} J_{f_x}^T} & |  &  {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_n}}\\\\
            \\vdots & | & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_nB_k} J_{f_x}^T} & | &  {{}^NP_{F_nF_1}} & \\cdots & {{}^NP_{F_n}}
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-covariance

        The method returns the predicted state vector mean (:math:`{}^N\\hat{\\bar x}_k`) and covariance (:math:`{{}^N\\bar P_{k}}`).

        :param uk: Control input
        :param Qk: Covariance of the Motion Model noise
        :param xk_1: State vector mean at time step k-1
        :param Pk_1: Covariance of the state vector at time step k-1
        :return: [xk_bar, Pk_bar] predicted state vector mean and covariance at time step k
        """

        r_dim = self.xBpose_dim 
        
        
        xB_1 = xk_1[0:r_dim]      
        xF_1 = xk_1[r_dim:]       
        
        
        xB_bar = self.f(xB_1, uk)
        
        
        xk_bar = np.vstack((xB_bar, xF_1))
        
        
        Jfx = self.Jfx(xB_1, uk)
        Jfw = self.Jfw(xB_1, uk)
        
        
        
        
        P_RR_1 = Pk_1[0:r_dim, 0:r_dim]  # Robot-Robot
        P_RM_1 = Pk_1[0:r_dim, r_dim:]   # Robot-Map correlations
        # P_MR_1 = Pk_1[r_dim:, 0:r_dim] # Map-Robot (not strictly needed as we transpose P_RM)
        P_MM_1 = Pk_1[r_dim:, r_dim:]    # Map-Map correlations
        
        # print(f"Value of Jfx: {Jfx}")
        # print(f"Value of Jfw: {Jfw}")
        # print(f"Value of P_RR_1: {P_RR_1}")
        # print(f"Value of P_RM_1: {P_RM_1}")
        # print(f"Value of Qk: {Qk}")
        # Top-Left: Robot Covariance (Standard EKF prediction)
        P_RR_bar = Jfx @ P_RR_1 @ Jfx.T + Jfw @ Qk @ Jfw.T
        
        # Top-Right: Robot-Map Cross Covariance
        # The correlation is updated by the robot's motion Jacobian
        P_RM_bar = Jfx @ P_RM_1
        
        # Bottom-Left: Map-Robot Cross Covariance (Transpose of Top-Right)
        P_MR_bar = P_RM_bar.T
        
        # Bottom-Right: Map-Map Covariance (Remains Unchanged)
        P_MM_bar = P_MM_1
        
        # Reassemble Pk_bar
        Pk_bar = np.block([
            [P_RR_bar, P_RM_bar],   
            [P_MR_bar, P_MM_bar]
        ])

        self.xk_bar = xk_bar
        self.Pk_bar = Pk_bar

        rel_pose = Pose3D(self.rel_disp)
        self.rel_disp = rel_pose.oplus(Pose3D(uk))
        J1_rel = rel_pose.J_1oplus(Pose3D(uk))
        J2_rel = rel_pose.J_2oplus()
        self.rel_cov = J1_rel @ self.rel_cov @ J1_rel.T + J2_rel @ Qk @ J2_rel.T

        return xk_bar, Pk_bar

    #TODO Following code has been copied from GFLocalisation.py file
    # because we have to override the LocalizationLoop function too for 
    # FEKFSLAM
    def LocalizationLoop(self, x0, P0, usk):
        """
        Localization loop. During *self.kSteps* it calls the :meth:`Localize` method for each time step.

        :param x0: initial state vector
        :param P0: initial covariance matrix
        """

        xk_1 = x0
        # temp = xk_1
        Pk_1 = P0

        zf, Rf = self.GetFeatures(xk_1)
        xk_1, Pk_1 = self.AddNewFeatures(xk_1, Pk_1, zf, Rf)
        xsk_1 = self.robot.xsk_1
        
        # Initialize sparse pose tracking
        self.pose_index = 0
        self.last_pose_step = -1
        self.accumulated_odom = None
        
        i = 1
        for self.k in range(self.kSteps):
            # print("This is the shape of xk_1 at the start",temp.shape)
            xsk = self.robot.fs(xsk_1, usk)  # Simulate the robot motion
            xk, Pk, xk_bar, zk = self.Localize(xk_1, Pk_1, self.k)  # Localize the robot
            # print(f"This is the shape of my xk after localise {self.xk.shape}")
            # print(f"At {i} iteration")
            # if i==12:
                # break
            # print(f"Estimate pose is {xk}")
            xsk_1 = xsk  # current state becomes previous state for next iteration
            xk_1 = xk
            Pk_1 = Pk
            # print(f"This is the shape of xk {xk.shape}")
            # print(f"This is the shape xk_bar {xk_bar.shape}")
            # print()
            # xk[2,0]=wrap_angle(xk[2,0])
            # xk_bar[2,0]=wrap_angle(xk_bar[2,0])
            # self.Log(xsk, xk, Pk, xk_bar, zk)
            # self.PlotUncertainty(xk, Pk)
            # self.PlotTrajectory()
            # i+=1
        # print(self.nf)
        self.PlotXY(estimation=True)
        self.PlotState()
        # self.PlotState()  # plot the state estimation results
        plt.show()

    def Localize(self, xk_1, Pk_1, k):
        """
        This method implements the FEKFSLAM algorithm. It localizes the robot and maps the features in the environment.
        It implements a single interation of the SLAM algorithm, given the current state vector mean and covariance.
        The unique difference it has with respect to its ancestor :meth:`FEKFMBL.Localize` is that it calls the method
        :meth:`AddNewFeatures` to add new non-paired features to the map.

        :param xk_1: state vector mean at time step k-1
        :param Pk_1: covariance of the state vector at time step k-1
        :param k: time step
        :return: [xk, Pk] state vector mean and covariance at time step k
        """

        ## To be completed by the student
        uk, Qk = self.GetInput()
        xk_bar, Pk_bar = self.Prediction(uk, Qk, xk_1, Pk_1)
        zm, Rm, Hm, Vm = self.GetMeasurements()
        self.zm_observed = (zm is not None)
        zf, Rf = self.GetFeatures(xk_bar)
        if zf is None:
            self.zf_observed=False
        elif (len(zf)==0):
            self.zf_observed=False
        else:
            self.zf_observed=True
        self.n_zf = 0 if zf is None else len(zf) // self.zfi_dim
        self.H = self.DataAssociation(xk_bar, Pk_bar, zf, Rf)
        zk, Rk, Hk, Vk, znp, Rnp = self.StackMeasurementsAndFeatures(zm, Rm, Hm, Vm, zf, Rf, self.H)
        
        # --- SPARSE POSE LOGIC: Only add pose when compass reading exists ---
        if zk is not None:
            # Compress odometry: use accumulated if available, else single step
            print(f"\n=== COMPASS READING at step {k}: Adding pose ===")
            xk, Pk = self.Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk, k, xk_1, uk, Qk,
                                pose_index=self.pose_index)
            self.xk, self.Pk = xk, Pk
            
            # Update pose tracking
            self.pose_index += 1
            self.last_pose_step = k
            self.rel_disp = np.zeros((self.xB_dim, 1))
            self.rel_cov = np.zeros((self.xB_dim, self.xB_dim))
            self.accumulated_odom = None      # Reset accumulation
            print(f"Pose index now: {self.pose_index}")
        else:
            # No compass reading: accumulate odometry for next pose
            print(f"\nStep {k}: No compass reading - accumulating odometry")
            xk, Pk = xk_bar, Pk_bar
            self.xk, self.Pk = xk_bar, Pk_bar
            
            # Accumulate odometry (as 3x1 matrix)
            # if self.accumulated_odom is None:
                # self.accumulated_odom = uk.copy()
            # else:
                # self.accumulated_odom = self.accumulated_odom + uk
            
            # print(f"Accumulated odometry shape: {self.accumulated_odom.shape}, sum: {np.sum(self.accumulated_odom)}")
        
        # Use the variable names zm, zf, Rf, znp, Rnp so that the plotting functions work
        xk, Pk = self.AddNewFeatures(xk, Pk, znp, Rnp)
        self.xk, self.Pk = xk, Pk
        
        self.Log(self.robot.xsk, self.GetRobotPose(self.xk), self.GetRobotPoseCovariance(self.Pk),
                 self.GetRobotPose(self.xk_bar), zm)  # log the results for plotting

        self.PlotUncertainty(zf, Rf, znp, Rnp)
        return self.xk, self.Pk, xk_bar, zk

    def PlotMappedFeaturesUncertainty(self):
        """
        This method plots the uncertainty of the mapped features. It plots the uncertainty ellipses of the mapped
        features in the environment. It is called at each Localization iteration.
        """
        # remove previous ellipses
        for i in range(len(self.plt_MappedFeaturesEllipses)):
            self.plt_MappedFeaturesEllipses[i].remove()
        self.plt_MappedFeaturesEllipses = []

        self.xk=BlockArray(self.xk,self.xF_dim, self.xB_dim)
        self.Pk=BlockArray(self.Pk,self.xF_dim, self.xB_dim)

        # draw new ellipses
        for Fj in range(self.nf):
            feature_ellipse = GetEllipse(self.xk[[Fj]],
                                         self.Pk[[Fj,Fj]])  # get the ellipse of the feature (x_Fj,P_Fj)
            plt_ellipse, = plt.plot(feature_ellipse[0], feature_ellipse[1], 'r')  # plot it
            self.plt_MappedFeaturesEllipses.append(plt_ellipse)  # and add it to the list

    def PlotUncertainty(self, zf, Rf, znp, Rnp):
        """
        This method plots the uncertainty of the robot (blue), the mapped features (red), the expected feature observations (black) and the feature observations.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of feature observations
        :param znp: vector of non-paired feature observations
        :param Rnp: covariance matrix of non-paired feature observations
        :return:
        """
        if self.k % self.robot.visualizationInterval == 0:
            self.PlotRobotUncertainty()
            self.PlotFeatureObservationUncertainty(znp, Rnp,'b')
            self.PlotFeatureObservationUncertainty(zf, Rf,'g')
            self.PlotExpectedFeaturesObservationsUncertainty()
            self.PlotMappedFeaturesUncertainty()
