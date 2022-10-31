#!/usr/bin/env python
"""
Copyright 2018, Zixin Luo, HKUST.
Geometry computations.
"""

from __future__ import print_function

import numpy as np
import cv2
import math


def interpolate_depth(pos, depth):
    ids = np.array(range(0, pos.shape[0]))

    h, w = depth.shape

    i = pos[:, 0]
    j = pos[:, 1]

    i_top_left = np.floor(i).astype(np.int32)
    j_top_left = np.floor(j).astype(np.int32)
    valid_top_left = np.logical_and(i_top_left >= 0, j_top_left >= 0)

    i_top_right = np.floor(i).astype(np.int32)
    j_top_right = np.ceil(j).astype(np.int32)
    valid_top_right = np.logical_and(i_top_right >= 0, j_top_right < w)

    i_bottom_left = np.ceil(i).astype(np.int32)
    j_bottom_left = np.floor(j).astype(np.int32)
    valid_bottom_left = np.logical_and(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = np.ceil(i).astype(np.int32)
    j_bottom_right = np.ceil(j).astype(np.int32)
    valid_bottom_right = np.logical_and(i_bottom_right < h, j_bottom_right < w)

    # Valid corner
    valid_corner = np.logical_and(
        np.logical_and(valid_top_left, valid_top_right),
        np.logical_and(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corner]
    j_top_left = j_top_left[valid_corner]

    i_top_right = i_top_right[valid_corner]
    j_top_right = j_top_right[valid_corner]

    i_bottom_left = i_bottom_left[valid_corner]
    j_bottom_left = j_bottom_left[valid_corner]

    i_bottom_right = i_bottom_right[valid_corner]
    j_bottom_right = j_bottom_right[valid_corner]

    ids = ids[valid_corner]

    # Valid depth
    valid_depth = np.logical_and(
        np.logical_and(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        np.logical_and(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.astype(np.float32)
    dist_j_top_left = j - j_top_left.astype(np.float32)
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = np.stack([i, j], axis=1)
    return [interpolated_depth, pos, ids]


"""
     用两个位置生成索引

"""
def interpolate_depth_new(pos, depth, pos1, depth1):
    ids = np.array(range(0, pos.shape[0]))

    h, w = depth.shape
    """
           pos的判断
    """
    i = pos[:, 0]
    j = pos[:, 1]

    i_top_left = np.floor(i).astype(np.int32)
    j_top_left = np.floor(j).astype(np.int32)
    valid_top_left = np.logical_and(i_top_left >= 0, j_top_left >= 0)

    i_top_right = np.floor(i).astype(np.int32)
    j_top_right = np.ceil(j).astype(np.int32)
    valid_top_right = np.logical_and(i_top_right >= 0, j_top_right < w)

    i_bottom_left = np.ceil(i).astype(np.int32)
    j_bottom_left = np.floor(j).astype(np.int32)
    valid_bottom_left = np.logical_and(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = np.ceil(i).astype(np.int32)
    j_bottom_right = np.ceil(j).astype(np.int32)
    valid_bottom_right = np.logical_and(i_bottom_right < h, j_bottom_right < w)

    """
           pos1的判断
    """
    i1 = pos1[:, 0]
    j1 = pos1[:, 1]

    i_top_left1 = np.floor(i1).astype(np.int32)
    j_top_left1 = np.floor(j1).astype(np.int32)
    valid_top_left1 = np.logical_and(i_top_left1 >= 0, j_top_left1 >= 0)

    i_top_right1 = np.floor(i1).astype(np.int32)
    j_top_right1 = np.ceil(j1).astype(np.int32)
    valid_top_right1 = np.logical_and(i_top_right1 >= 0, j_top_right1 < w)

    i_bottom_left1 = np.ceil(i1).astype(np.int32)
    j_bottom_left1 = np.floor(j1).astype(np.int32)
    valid_bottom_left1 = np.logical_and(i_bottom_left1 < h, j_bottom_left1 >= 0)

    i_bottom_right1 = np.ceil(i1).astype(np.int32)
    j_bottom_right1 = np.ceil(j1).astype(np.int32)
    valid_bottom_right1 = np.logical_and(i_bottom_right1 < h, j_bottom_right1 < w)


    # Valid corner
    valid_corner = np.logical_and(
        np.logical_and(
            np.logical_and(valid_top_left, valid_top_right),
            np.logical_and(valid_bottom_left, valid_bottom_right)
        ),
        np.logical_and(
            np.logical_and(valid_top_left1, valid_top_right1),
            np.logical_and(valid_bottom_left1, valid_bottom_right1)
        )
    )
    """
           pos
    """
    i_top_left = i_top_left[valid_corner]
    j_top_left = j_top_left[valid_corner]

    i_top_right = i_top_right[valid_corner]
    j_top_right = j_top_right[valid_corner]

    i_bottom_left = i_bottom_left[valid_corner]
    j_bottom_left = j_bottom_left[valid_corner]

    i_bottom_right = i_bottom_right[valid_corner]
    j_bottom_right = j_bottom_right[valid_corner]

    """
           pos1
    """
    i_top_left1 = i_top_left1[valid_corner]
    j_top_left1 = j_top_left1[valid_corner]

    i_top_right1 = i_top_right1[valid_corner]
    j_top_right1 = j_top_right1[valid_corner]

    i_bottom_left1 = i_bottom_left1[valid_corner]
    j_bottom_left1 = j_bottom_left1[valid_corner]

    i_bottom_right1 = i_bottom_right1[valid_corner]
    j_bottom_right1 = j_bottom_right1[valid_corner]

    ids = ids[valid_corner]

    # Valid depth
    print(i_bottom_right1.min(), j_bottom_right1.min())
    valid_depth = np.logical_and(
        np.logical_and(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        np.logical_and(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        ),
        np.logical_and(
            depth1[i_top_left1, j_top_left1] > 0,
            depth1[i_top_right1, j_top_right1] > 0
        ),
        np.logical_and(
            depth1[i_bottom_left1, j_bottom_left1] > 0,
            depth1[i_bottom_right1, j_bottom_right1] > 0
        )
    )
    """
         pos
    """
    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    """
           pos1
    """
    i_top_left1 = i_top_left1[valid_depth]
    j_top_left1 = j_top_left1[valid_depth]

    i_top_right1 = i_top_right1[valid_depth]
    j_top_right1 = j_top_right1[valid_depth]

    i_bottom_left1 = i_bottom_left1[valid_depth]
    j_bottom_left1 = j_bottom_left1[valid_depth]

    i_bottom_right1 = i_bottom_right1[valid_depth]
    j_bottom_right1 = j_bottom_right1[valid_depth]

    ids = ids[valid_depth]

    # Interpolation
    """
         pos
    """
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.astype(np.float32)
    dist_j_top_left = j - j_top_left.astype(np.float32)
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = np.stack([i, j], axis=1)

    """
         pos1
    """
    i1 = i1[ids]
    j1 = j1[ids]
    dist_i_top_left1 = i1 - i_top_left1.astype(np.float32)
    dist_j_top_left1 = j1 - j_top_left1.astype(np.float32)
    w_top_left1 = (1 - dist_i_top_left1) * (1 - dist_j_top_left1)
    w_top_right1 = (1 - dist_i_top_left1) * dist_j_top_left1
    w_bottom_left1 = dist_i_top_left1 * (1 - dist_j_top_left1)
    w_bottom_right1 = dist_i_top_left1 * dist_j_top_left1

    interpolated_depth1 = (
        w_top_left1 * depth[i_top_left1, j_top_left1] +
        w_top_right1 * depth[i_top_right1, j_top_right1] +
        w_bottom_left1 * depth[i_bottom_left1, j_bottom_left1] +
        w_bottom_right1 * depth[i_bottom_right1, j_bottom_right1]
    )

    pos1 = np.stack([i1, j1], axis=1)

    return [interpolated_depth, pos, interpolated_depth1, pos1, ids]





import tensorflow as tf
def validate_and_interpolate(pos, inputs, validate_corner=True, validate_val=None, nd=False):
    if nd:
        h, w, c = inputs.tolist()
    else:
        print(7777777777777, inputs.shape)
        # h, w = inputs.shape()
        h = inputs.shape[0]
        w = inputs.shape[1]

    ids = tf.range(0, tf.shape(pos)[0])

    i = pos[:, 0]
    j = pos[:, 1]

    i_top_left = tf.cast(tf.math.floor(i), tf.int32)
    j_top_left = tf.cast(tf.math.floor(j), tf.int32)

    i_top_right = tf.cast(tf.math.floor(i), tf.int32)
    j_top_right = tf.cast(tf.math.ceil(j), tf.int32)

    i_bottom_left = tf.cast(tf.math.ceil(i), tf.int32)
    j_bottom_left = tf.cast(tf.math.floor(j), tf.int32)

    i_bottom_right = tf.cast(tf.math.ceil(i), tf.int32)
    j_bottom_right = tf.cast(tf.math.ceil(j), tf.int32)

    if validate_corner:
        # Valid corner
        valid_top_left = tf.logical_and(i_top_left >= 0, j_top_left >= 0)
        valid_top_right = tf.logical_and(i_top_right >= 0, j_top_right < w)
        valid_bottom_left = tf.logical_and(i_bottom_left < h, j_bottom_left >= 0)
        valid_bottom_right = tf.logical_and(i_bottom_right < h, j_bottom_right < w)

        valid_corner = tf.logical_and(
            tf.logical_and(valid_top_left, valid_top_right),
            tf.logical_and(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corner]
        j_top_left = j_top_left[valid_corner]

        i_top_right = i_top_right[valid_corner]
        j_top_right = j_top_right[valid_corner]

        i_bottom_left = i_bottom_left[valid_corner]
        j_bottom_left = j_bottom_left[valid_corner]

        i_bottom_right = i_bottom_right[valid_corner]
        j_bottom_right = j_bottom_right[valid_corner]

        ids = ids[valid_corner]

    if validate_val is not None:
        # Valid depth
        valid_depth = tf.logical_and(
            tf.logical_and(
                tf.gather_nd(inputs, tf.stack([i_top_left, j_top_left], axis=-1)) > 0,
                tf.gather_nd(inputs, tf.stack([i_top_right, j_top_right], axis=-1)) > 0
            ),
            tf.logical_and(
                tf.gather_nd(inputs, tf.stack([i_bottom_left, j_bottom_left], axis=-1)) > 0,
                tf.gather_nd(inputs, tf.stack([i_bottom_right, j_bottom_right], axis=-1)) > 0
            )
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]

        ids = ids[valid_depth]

    # Interpolation
    i = tf.gather(i, ids)
    j = tf.gather(j, ids)
    dist_i_top_left = i - tf.cast(i_top_left, tf.float32)
    dist_j_top_left = j - tf.cast(j_top_left, tf.float32)
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    if nd:
        w_top_left = w_top_left[..., None]
        w_top_right = w_top_right[..., None]
        w_bottom_left = w_bottom_left[..., None]
        w_bottom_right = w_bottom_right[..., None]

    interpolated_val = (
        w_top_left * tf.gather_nd(inputs, tf.stack([i_top_left, j_top_left], axis=-1)) +
        w_top_right * tf.gather_nd(inputs, tf.stack([i_top_right, j_top_right], axis=-1)) +
        w_bottom_left * tf.gather_nd(inputs, tf.stack([i_bottom_left, j_bottom_left], axis=-1)) +
        w_bottom_right * tf.gather_nd(inputs, tf.stack([i_bottom_right, j_bottom_right], axis=-1))
    )

    pos = tf.stack([i, j], axis=1)
    return [interpolated_val, pos, ids]



def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = (pos - 0.5) / 2
    return pos


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos


def grid_positions(h, w):
    x_rng = range(0, w)
    y_rng = range(0, h)
    xv, yv = np.meshgrid(x_rng, y_rng)
    return np.reshape(np.stack((yv, xv), axis=-1), (-1, 2))


def relative_pose(pose0, pose1):
    """Compute relative pose.
    Args:
        pose: [R, t]
    Returns:
        rel_pose: [rel_R, rel_t]
    """
    rel_R = np.matmul(pose1[0], pose0[0].T)
    center0 = -np.matmul(pose0[1].T, pose0[0]).T
    center1 = -np.matmul(pose1[1].T, pose1[0]).T
    rel_t = np.matmul(pose1[0], center0 - center1)
    return [rel_R, rel_t]


def warp(pos0, rel_pose, depth0, K0, depth1, K1):
    def swap_axis(data):
        return np.stack([data[:, 1], data[:, 0]], axis=-1)

    # print(6845906849, len(pos0))
    z0, pos0, ids = interpolate_depth(pos0, depth0)
    # print(444444444444444, len(pos0), len(z0), ids, len(ids))
    # z0, pos0, ids = validate_and_interpolate(pos0, depth0, validate_val=0)
    # print("555555555", z0.shape, z0)

    uv0_homo = np.concatenate([swap_axis(pos0), np.ones((pos0.shape[0], 1))], axis=-1)
    xy0_homo = np.matmul(np.linalg.inv(K0), uv0_homo.T)
    xyz0_homo = np.concatenate([np.expand_dims(z0, 0) * xy0_homo,
                                np.ones((1, pos0.shape[0]))], axis=0)

    xyz1 = np.matmul(rel_pose, xyz0_homo)
    # print(123, rel_pose, xyz0_homo.shape, xyz1.shape, xyz0_homo)
    xy1_homo = xyz1 / np.expand_dims(xyz1[-1, :], axis=0)
    uv1 = np.matmul(K1, xy1_homo).T[:, 0:2]

    pos1 = swap_axis(uv1)
    # annotated_depth, pos1, new_ids = validate_and_interpolate(pos1, depth1)
    annotated_depth, pos1, new_ids = interpolate_depth(pos1, depth1)
    # print("AAAAAAAAAAAAA", annotated_depth)

    ids = ids[new_ids]
    pos0 = pos0[new_ids]
    z0 = z0[new_ids]
    estimated_depth = xyz1.T[new_ids, -1]
    # print("estimated_depth", estimated_depth, len(estimated_depth))

    inlier_mask = np.abs(estimated_depth - annotated_depth) < 0.05

    ids = ids[inlier_mask]
    pos0 = pos0[inlier_mask]
    pos1 = pos1[inlier_mask]
    dep0 = z0[inlier_mask]
    dep1 = annotated_depth[inlier_mask]

    return pos0, pos1, ids, dep0, dep1


def undist_points(pts, K, dist, img_size=None):
    n = pts.shape[0]
    new_pts = pts
    if img_size is not None:
        hs = img_size / 2
        new_pts = np.stack([pts[:, 2] * hs[0] + hs[0], pts[:, 5] * hs[1] + hs[1]], axis=1)

    new_dist = np.zeros((5), dtype=np.float32)
    new_dist[0] = dist[0]
    new_dist[1] = dist[1]
    new_dist[4] = dist[2]

    upts = cv2.undistortPoints(np.expand_dims(new_pts, axis=1), K, new_dist)
    upts = np.squeeze(cv2.convertPointsToHomogeneous(upts), axis=1)
    upts = np.matmul(K, upts.T).T[:, 0:2]

    if img_size is not None:
        new_upts = pts.copy()
        new_upts[:, 2] = (upts[:, 0] - hs[0]) / hs[0]
        new_upts[:, 5] = (upts[:, 1] - hs[1]) / hs[1]
        return new_upts
    else:
        return upts


def skew_symmetric_mat(v):
    v = v.flatten()
    M = np.stack([
        (0, -v[2], v[1]),
        (v[2], 0, -v[0]),
        (-v[1], v[0], 0),
    ], axis=0)
    return M


def get_essential_mat(t0, t1, R0, R1):
    """
    Args:
        t: 3x1 mat.
        R: 3x3 mat.
    Returns:
        e_mat: 3x3 essential matrix.
    """
    dR = np.matmul(R1, R0.T)  # dR = R_1 * R_0^T
    dt = t1 - np.matmul(dR, t0)  # dt = t_1 - dR * t_0

    dt = dt.reshape(1, 3)
    dt_ssm = skew_symmetric_mat(dt)

    e_mat = np.matmul(dt_ssm, dR)  # E = dt_ssm * dR
    e_mat = e_mat / np.linalg.norm(e_mat)
    return e_mat


def get_epipolar_dist(kpt_coord0, kpt_coord1, K0, K1, ori_img_size0, ori_img_size1, e_mat, eps=1e-6):
    """
    Compute (symmetric) epipolar distances.
    Args:
        kpt_coord: Nx2 keypoint coordinates, normalized to [-1, +1].
        K: 3x3 intrinsic matrix.
        ori_img_size: original image size (width, height)
        e_mat: Precomputed essential matrix.
        get_epi_dist_mat: Whether to get epipolar distance in matrix form or vector form.
        eps: Epsilon.
    Returns:
        epi_dist: N-d epipolar distance.
    """
    def _get_homo_coord(coord):
        homo_coord = np.concatenate([coord, np.ones_like(coord[:, 0, None])], axis=-1)
        return homo_coord

    # print(111111, kpt_coord0.shape)
    uv0_homo = _get_homo_coord(kpt_coord0 * ori_img_size0 / 2 + ori_img_size0 / 2)
    uv1_homo = _get_homo_coord(kpt_coord1 * ori_img_size1 / 2 + ori_img_size1 / 2)
    # normalize keypoint coordinates with camera intrinsics.
    xy0_homo = np.matmul(np.linalg.inv(K0), uv0_homo.T)
    xy1_homo = np.matmul(np.linalg.inv(K1), uv1_homo.T)
    # epipolar lines in the first image.
    Ex0 = np.matmul(e_mat, xy0_homo)  # Bx3xN
    # epipolar lines in the second image.
    Etx1 = np.matmul(e_mat.T, xy1_homo) # Bx3xN
    # get normal vectors.
    line_norm0 = Ex0[0, :] ** 2 + Ex0[1, :] ** 2
    line_norm1 = Etx1[0, :] ** 2 + Etx1[1, :] ** 2
    x1Ex0 = np.sum(xy1_homo * Ex0, axis=0)
    epi_dist = (x1Ex0 ** 2) / (line_norm0 + line_norm1 + eps)
    epi_dist = np.sqrt(epi_dist)
    return epi_dist


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                      [m01+m10,     m11-m00-m22, 0.0,         0.0],
                      [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                      [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def evaluate_R_t(R_gt, t_gt, R, t, eps=1e-15):
    t = t.flatten()
    t_gt = t_gt.flatten()

    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    return err_q, err_t



