import torch
from torch import sin, cos, tensor, eye

def rotation_matrix_rpy(roll, pitch, yaw):
    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    R = tensor([[r00, r01, r02],
                [r10, r11, r12],
                [r20, r21, r22]])
    return R


def make_transform_matrix(xyz, rpy):
    T = eye(4)
    R = rotation_matrix_rpy(rpy[0], rpy[1], rpy[2])
    T[:3, :3] = R
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    return T


def prismatic_transform(xyz, rpy, axis, q):
    tf_origin = make_transform_matrix(xyz, rpy).to(device=q.device)

    n_sample, n_horizen = q.size()

    if torch.linalg.norm(axis) < 1e-12:
        axis = tensor([1.0, 0.0, 0.0], device=q.device)
    else:
        axis = axis / torch.linalg.norm(axis)

    tf_slide = torch.eye(4, device=q.device).expand(n_sample, n_horizen, 4, 4).clone()
    tf_slide[..., :3, 3] = axis.view(1, 1, 3) * q.unsqueeze(-1)

    if tf_origin.ndim == 2:
        tf_origin = tf_origin.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
        tf_origin = tf_origin.expand(n_sample, n_horizen, 4, 4)

    return tf_origin @ tf_slide


def revolute_transform(xyz, rpy, axis, q):
    tf_origin = make_transform_matrix(xyz, rpy).to(device=q.device)

    n_sample, n_horizen = q.size()

    if torch.linalg.norm(axis) < 1e-12:
        axis = tensor([1.0, 0.0, 0.0], device=q.device)
    else:
        axis = axis / torch.linalg.norm(axis)

    c = cos(q)
    s = sin(q)
    omc = 1 - c

    vx, vy, vz = axis

    R00 = c + vx * vx * omc
    R01 = vx * vy * omc - vz * s
    R02 = vx * vz * omc + vy * s

    R10 = vy * vx * omc + vz * s
    R11 = c + vy * vy * omc
    R12 = vy * vz * omc - vx * s

    R20 = vz * vx * omc - vy * s
    R21 = vz * vy * omc + vx * s
    R22 = c + vz * vz * omc

    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1)
    ], dim=-2)

    tf_rot = eye(4, device=q.device).expand(n_sample, n_horizen, 4, 4).clone()
    tf_rot[..., :3, :3] = R

    return tf_origin @ tf_rot


def prismatic_transform_cpu(xyz, rpy, axis, q):
    tf_origin = make_transform_matrix(xyz, rpy)

    if torch.linalg.norm(axis) < 1e-12:
        axis = tensor([1.0, 0.0, 0.0])
    else:
        axis = axis / torch.linalg.norm(axis)

    tf_slide = eye(4)
    tf_slide[:3, 3] = axis * q

    return tf_origin @ tf_slide


def revolute_transform_cpu(xyz, rpy, axis, q):
    tf_origin = make_transform_matrix(xyz, rpy)

    if torch.linalg.norm(axis) < 1e-12:
        axis = tensor([1.0, 0.0, 0.0])
    else:
        axis = axis / torch.linalg.norm(axis)

    c = cos(q)
    s = sin(q)
    omc = 1 - c

    vx, vy, vz = axis

    R00 = c + vx * vx * omc
    R01 = vx * vy * omc - vz * s
    R02 = vx * vz * omc + vy * s

    R10 = vy * vx * omc + vz * s
    R11 = c + vy * vy * omc
    R12 = vy * vz * omc - vx * s

    R20 = vz * vx * omc - vy * s
    R21 = vz * vy * omc + vx * s
    R22 = c + vz * vz * omc

    R = tensor([[R00, R01, R02],
                [R10, R11, R12],
                [R20, R21, R22]])

    tf_rot = eye(4)
    tf_rot[:3, :3] = R

    return tf_origin @ tf_rot


def transformation_matrix_from_xyzrpy(q):
    n_sample, n_horizen, n_action = q.size()

    T = torch.eye(4, device=q.device).expand(n_sample, n_horizen, 4, 4).clone()
    T[:, :, 0, 3] = q[:, :, 0]
    T[:, :, 1, 3] = q[:, :, 1]
    T[:, :, 2, 3] = q[:, :, 2]

    roll = q[:, :, 3]
    pitch = q[:, :, 4]
    yaw = q[:, :, 5]

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    R = torch.stack([
         torch.stack([r00, r01, r02], dim=-1),
         torch.stack([r10, r11, r12], dim=-1),
         torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)

    T[:, :, :3, :3] = R

    return T


def transformation_matrix_from_xyzrpy_cpu(q):
    T = torch.eye(4).clone()
    T[0, 3] = q[0]
    T[1, 3] = q[1]
    T[2, 3] = q[2]

    roll = q[3]
    pitch = q[4]
    yaw = q[5]

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    R00 = cy * cp
    R01 = cy * sp * sr - sy * cr
    R02 = cy * sp * cr + sy * sr

    R10 = sy * cp
    R11 = sy * sp * sr + cy * cr
    R12 = sy * sp * cr - cy * sr

    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr

    R = tensor([[R00, R01, R02],
                [R10, R11, R12],
                [R20, R21, R22]])

    T[:3, :3] = R

    return T