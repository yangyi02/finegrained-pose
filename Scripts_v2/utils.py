from math import *
import numpy as np
import scipy.io
import scipy.misc


def compute_iou(segment1, segment2):
    assert(len(segment1.shape) == 2 and len(segment2.shape) == 2)
    if not segment1.shape[0] == segment2.shape[0] or not segment1.shape[1] == segment2.shape[1]:
        segment2 = segment1.astype(np.float)
        segment2 = scipy.misc.imresize(segment2, segment1.shape, 'nearest')
    if np.max(segment1) == 255:
        segment1 = segment1 / 255
    segment1 = segment1.astype(np.bool)
    if np.max(segment2) == 255:
        segment2 = segment2 / 255
    segment2 = segment2.astype(np.bool)
    inter = np.logical_and(segment1, segment2)
    union = np.logical_or(segment1, segment2)
    iou = np.sum(inter) * 1.0 / np.sum(union)
    return iou


def angle_to_rotmat(azimuth, elevation, theta):
    """
    Convert angles to rotation matrix
    """
    Ry = np.array([
        [cos(azimuth), 0., -sin(azimuth)],
        [0., 1., 0.],
        [sin(azimuth), 0.,  cos(azimuth)]])
    Rx = np.array([
        [1.,             0.,              0.],
        [0., cos(elevation), -sin(elevation)],
        [0., sin(elevation),  cos(elevation)]])
    Rz = np.array([
        [cos(theta), -sin(theta), 0.],
        [sin(theta),  cos(theta), 0.],
        [0.,           0.,        1.]])
    Rtmp = np.dot(Rx, Ry)
    R = np.dot(Rz, Rtmp)
    return R


def proj(x3d, R, d, uv, f):
    """
    Perspective projection of 3D points onto image plane
    Parameters:
        x3d: 3D coordinates of shape N x 3, where N is the number of points
        R: Rotation matrix of shape 3 x 3
        d: parameter 'd' in the paper
        uv: principal point (u, v) in the paper
        f: focal length 'f' in the paper

    Return:
        x2d: 2D coordinates of shape N x 2, where N is the number of points
    """
    u = uv[0]
    v = uv[1]
    # construct projection matrix
    intrinsic = np.array([
        [f,   0.,   u],
        [0.,   f,   v],
        [0.,     0.,  1.]])
    T = np.array([[0.], [0.], [d]])
    extrinsic = np.concatenate((R, -T), axis=1)
    P = np.dot(intrinsic, extrinsic)
    # project
    x3d_homo = np.concatenate((x3d, np.ones((x3d.shape[0], 1))), axis=1).T
    x = np.dot(P, x3d_homo)
    x[0, :] = np.divide(x[0, :], x[2, :])
    x[1, :] = np.divide(x[1, :], x[2, :])
    x2d = x[0:2, :].T

    return x2d


def read_obj(obj_file_name, target_file_name='NOT_SAVE', verbose=False, opt_save=False):
    """
    Read .obj CAD model file
    """
    with open(obj_file_name, 'r') as f:
        lines = f.readlines()

    vertices = []
    faces = []
    for i in range(len(lines)):
        line = lines[i].strip()
        # format of each line:
        # a vertex:
        # v x_coord y_coord z_coord
        # a face:
        # f v1/vn1/vt1 v2/vn2/vt2 v3/vn3/vt3
        # currently only take v1, v2 and v3, ignore normal and texture info
        line_segs = line.split(' ')
        if line_segs[0] == 'v':  # process a vertex
            x = float(line_segs[1])
            y = float(line_segs[2])
            z = float(line_segs[3])
            vertices.append([x, y, z])
            if verbose and len(vertices) > 0:
                print("vertex: ", vertices[-1])
        elif line_segs[0] == 'f':  # process a face
            face_tmp = []
            for j in range(1, 4):
                face_tmp.append(int(line_segs[j].split('/')[0]) - 1)
            faces.append(face_tmp)
            if verbose and len(faces) > 0:
                print("face: ", faces[-1])

    # put read info into numpy arrays
    num_vertex = len(vertices)
    num_face = len(faces)
    if verbose:
        print("Finish parsing file, num_vertex=%d, num_face=%d" % (num_vertex, num_face))
    vertices_tmp = np.zeros((num_vertex, 3), dtype=float)
    faces_tmp = np.zeros((num_face, 3), dtype=int)
    for i in range(num_vertex):
        vertices_tmp[i, :] = np.array(vertices[i], dtype=float)
    for i in range(num_face):
        faces_tmp[i, :] = np.array(faces[i], dtype=float)
    # return a dictionary storing the vertex/face info
    obj_data = dict()
    obj_data['vertices'] = vertices_tmp
    obj_data['faces'] = faces_tmp
    if verbose:
        print(vertices_tmp.shape, faces_tmp.shape)
    if opt_save:
        scipy.io.savemat(target_file_name, obj_data)
    return obj_data
