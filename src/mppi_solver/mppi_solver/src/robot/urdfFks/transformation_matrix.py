import torch
from torch import sin, cos, tensor, eye


@torch.jit.script
def rotation_matrix_rpy(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
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


@torch.jit.script
def make_transform_matrix(xyz: torch.Tensor, rpy: torch.Tensor) -> torch.Tensor:
    T = eye(4, dtype=xyz.dtype, device=xyz.device)

    cr = cos(rpy[0])
    sr = sin(rpy[0])
    cp = cos(rpy[1])
    sp = sin(rpy[1])
    cy = cos(rpy[2])
    sy = sin(rpy[2])

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
        torch.stack([r00, r01, r02]),
        torch.stack([r10, r11, r12]),
        torch.stack([r20, r21, r22])
    ])

    T[0:3, 0:3] = R
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    return T


@torch.jit.script
def prismatic_transform(tf_origin: torch.Tensor, axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    n_sample, n_horizon = q.size()

    tf_slide = torch.eye(4, device=q.device).expand(n_sample, n_horizon, 4, 4).clone()
    tf_slide[..., :3, 3] = axis.view(1, 1, 3) * q.unsqueeze(-1)

    tf = torch.einsum('ij,btjk->btik', tf_origin, tf_slide)
    return tf


@torch.jit.script
def revolute_transform(tf_origin: torch.Tensor, axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    n_sample, n_horizon = q.size()

    c = cos(q)
    s = sin(q)
    omc = 1 - c
    vx = axis[0]
    vy = axis[1]
    vz = axis[2]

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

    tf_rot = eye(4, device=q.device).expand(n_sample, n_horizon, 4, 4).clone()
    tf_rot[..., :3, :3] = R

    tf = torch.einsum('ij,btjk->btik', tf_origin, tf_rot)
    return tf


@torch.jit.script
def transformation_matrix_from_xyzrpy(q: torch.Tensor) -> torch.Tensor:
    n_sample, n_horizen = q.size()

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
