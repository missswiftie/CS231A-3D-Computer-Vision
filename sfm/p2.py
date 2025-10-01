import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    _,_,v=np.linalg.svd(F.T)
    e=v[-1]
    return e/e[-1]
    #raise Exception('Not Implemented Error')

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    height,weight=im.shape
    t=np.array([
        [1,0,-0.5*weight],
        [0,1,-0.5*height],
        [0,0,1]
    ])
    transition_e=np.matmul(t,e)
    denom=np.sqrt(transition_e[0]**2+transition_e[1]**2)
    alpha=1 if transition_e[0]>=0 else -1
    r=np.array([
        [alpha*transition_e[0]/denom,alpha*transition_e[1]/denom,0],
        [-alpha*transition_e[1]/denom,alpha*transition_e[0]/denom,0],
        [0,0,1]
    ])
    transition_rotate_e=np.matmul(r,transition_e)
    g=np.array([
        [1,0,0],
        [0,1,0],
        [-1/transition_rotate_e[0],0,1]
    ])
    return np.matmul(np.matmul(np.linalg.inv(t),g),np.matmul(r,t))
    #raise Exception('Not Implemented Error')

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    h2=compute_H(e2,im2)
    e2x=np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    v=np.array([1,1,1])
    m=np.matmul(e2x,F)+np.matmul(e2.reshape(3,1),v.reshape(1,3))
    p1_hat=np.matmul(np.matmul(h2,m),points1.T).T
    p2_hat=np.matmul(h2,points2.T).T
    w = p1_hat / p1_hat[:, 2].reshape(-1, 1)
    b = p2_hat / p2_hat[:, 2].reshape(-1, 1)
    b=b[:,0]
    a=np.linalg.lstsq(w, b, rcond=None)[0]
    ha=np.array([
        [a[0],a[1],a[2]],
        [0,1,0],[0,0,1]
    ])
    h1=np.matmul(np.matmul(ha,h2),m)
    return h1,h2
    #raise Exception('Not Implemented Error')

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
