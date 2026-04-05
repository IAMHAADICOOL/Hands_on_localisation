import numpy as np
from math import atan2, cos, sin

class Pose3D(np.ndarray):
    """
    Definition of a robot pose in 3 DOF (x, y, yaw). The class inherits from a ndarray.
    This class extends the ndarray with the $oplus$ and $ominus$ operators and the corresponding Jacobians.
    """
    def __new__(cls, input_array):
        """
        Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarry numpy class.

        :param input_array: array used to initialize the class
        :returns: the instance of a Pose3D class object
        """
        assert input_array.shape == (3, 1), "mean must be a 3x1 vector"

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def oplus(AxB, BxC):
        """
        Given a Pose3D object *AxB* (the self object) and a Pose3D object *BxC*, it returns the Pose3D object *AxC*.

        .. math::
            \\mathbf{{^A}x_B} &= \\begin{bmatrix} ^Ax_B & ^Ay_B & ^A\\psi_B \\end{bmatrix}^T \\\\
            \\mathbf{{^B}x_C} &= \\begin{bmatrix} ^Bx_C & ^By_C & & ^B\\psi_C \\end{bmatrix}^T \\\\

        The operation is defined as:

        .. math::
            \\mathbf{{^A}x_C} &= \\mathbf{{^A}x_B} \\oplus \\mathbf{{^B}x_C} =
            \\begin{bmatrix}
                ^Ax_B + ^Bx_C  \\cos(^A\\psi_B) - ^By_C  \\sin(^A\\psi_B) \\\\
                ^Ay_B + ^Bx_C  \\sin(^A\\psi_B) + ^By_C  \\cos(^A\\psi_B) \\\\
                ^A\\psi_B + ^B\\psi_C
            \\end{bmatrix}
            :label: eq-oplus3dof

        :param BxC: C-Frame pose expressed in B-Frame coordinates
        :returns: C-Frame pose expressed in A-Frame coordinates
        """
        
        # Make sure that AxB and BxC are Pose3D objects
        if(isinstance(AxB, Pose3D) == False):
            raise TypeError("AxB must be a Pose3D object")
        if(isinstance(BxC, Pose3D) == False):
            raise TypeError("BxC must be a Pose3D object")
        
        A_psi_B = AxB[2,0]
        cos_psi_B = cos(A_psi_B)
        sin_psi_B = sin(A_psi_B)

        
        Ax_C = np.zeros((3,1))
        Ax_C[0,0] = AxB[0,0] + BxC[0,0] * cos_psi_B - BxC[1,0] * sin_psi_B
        Ax_C[1,0] = AxB[1,0] + BxC[0,0] * sin_psi_B + BxC[1,0] * cos_psi_B
        Ax_C[2,0] = AxB[2,0] + BxC[2,0]

        # Wrap angle to [-pi, pi]
        while Ax_C[2,0] > np.pi:
            Ax_C[2,0] -= 2 * np.pi
        while Ax_C[2,0] < -np.pi:
            Ax_C[2,0] += 2 * np.pi
        
        return Pose3D(Ax_C)

    def ominus(AxB):
        """
        Inverse pose compounding of the *AxB* pose (the self objetc):

        .. math::
            ^Bx_A = \\ominus ^Ax_B =
            \\begin{bmatrix}
                -^Ax_B \\cos(^A\\psi_B) - ^Ay_B \\sin(^A\\psi_B) \\\\
                ^Ax_B \\sin(^A\\psi_B) - ^Ay_B \\cos(^A\\psi_B) \\\\
                -^A\\psi_B
            \\end{bmatrix}
            :label: eq-ominus3dof

        :returns: A-Frame pose expressed in B-Frame coordinates (eq. :eq:`eq-ominus3dof`)
        """

        if (isinstance(AxB, Pose3D) == False):
            raise TypeError("AxB must be a Pose3D object")
        A_psi_B = AxB[2,0]
        cos_psi_B = cos(A_psi_B)
        sin_psi_B = sin(A_psi_B)

        Bx_A = np.zeros((3,1))
        Bx_A[0,0] = -AxB[0,0] * cos_psi_B - AxB[1,0] * sin_psi_B
        Bx_A[1,0] = AxB[0,0] * sin_psi_B - AxB[1,0] * cos_psi_B
        Bx_A[2,0] = -AxB[2,0]

        return Pose3D(Bx_A)
