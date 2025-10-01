# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). 
            It will contain four points: two for each parallel line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # BEGIN YOUR CODE HERE
    x1,y1,x2,y2=points[0][0],points[0][1],points[1][0],points[1][1]
    x3,y3,x4,y4=points[2][0],points[2][1],points[3][0],points[3][1]
    a1=(y2-y1)/(x2-x1)
    a2=(y3-y4)/(x3-x4)
    b1=y1-a1*x1
    b2=y3-a2*x3
    # a1x+b1=a2x+b2
    x=(b2-b1)/(a1-a2)
    y=a1*x+b1
    return np.array([x,y])
    # END YOUR CODE HERE

'''
COMPUTE_K_FROM_VANISHING_POINTS
Makes sure to make it so the bottom right element of K is 1 at the end.
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    # BEGIN YOUR CODE HERE
    A=[]
    x1,y1=vanishing_points[0]
    x2,y2=vanishing_points[1]
    x3,y3=vanishing_points[2]
    A.append([x1*x2+y1*y2,x1+x2,y1+y2,1])
    A.append([x1*x3+y1*y3,x1+x3,y1+y3,1])
    A.append([x3*x2+y3*y2,x3+x2,y3+y2,1])
    u,sigma,v=np.linalg.svd(A,full_matrices=True)
    w_params=v[-1,:]
    w=np.array([
        [w_params[0],0,w_params[1]],
        [0,w_params[0],w_params[2]],
        [w_params[1],w_params[2],w_params[3]]
    ],dtype=np.float64)
    # w=(KK^T)^{-1}=K^T^{-1}K^{-1}=K^{-1}^TK^{-1}
    k=np.linalg.cholesky(w)
    k=np.linalg.inv(k).T
    k=k/k[2][2]
    return k
    # END YOUR CODE HERE

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # BEGIN YOUR CODE HERE
    v1=np.array([vanishing_pair1[0][0],vanishing_pair1[0][1],1])
    v2=np.array([vanishing_pair1[1][0],vanishing_pair1[1][1],1])
    v3=np.array([vanishing_pair2[0][0],vanishing_pair2[0][1],1])
    v4=np.array([vanishing_pair2[1][0],vanishing_pair2[1][1],1])
    K_inv=np.linalg.inv(K)
    d1=K_inv.dot(v1)
    d1=d1/np.linalg.norm(d1)
    d2=K_inv.dot(v2)
    d2=d2/np.linalg.norm(d2)
    d3=K_inv.dot(v3)
    d3=d3/np.linalg.norm(d3)
    d4=K_inv.dot(v4)
    d4=d4/np.linalg.norm(d4)
    n1=np.cross(d1,d2)
    n1=n1/np.linalg.norm(n1)
    n2=np.cross(d3,d4)
    n2=n2/np.linalg.norm(n2)
    cos_theta=np.dot(n1,n2)
    angle=np.arccos(cos_theta)*180/math.pi
    return angle
    # END YOUR CODE HERE

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # BEGIN YOUR CODE HERE
    K_inv=np.linalg.inv(K)
    points=vanishing_points1.shape[0]
    ones=np.ones((points,1))
    vp1=np.hstack((vanishing_points1,ones))
    vp2=np.hstack((vanishing_points2,ones))
    dd1=K_inv @ vp1.T
    dd2=K_inv @ vp2.T
    dd1=dd1/np.linalg.norm(dd1,axis=0)
    dd2=dd2/np.linalg.norm(dd2,axis=0)
    R=np.linalg.lstsq(dd1.T,dd2.T,rcond=None)[0].T
    return R
    # END YOUR CODE HERE

'''
TEST_P3
Test function. Do not modify.
'''
def test_p3():
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[1080, 598],[1840, 478],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[4, 878],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))


if __name__ == '__main__':
    test_p3()
