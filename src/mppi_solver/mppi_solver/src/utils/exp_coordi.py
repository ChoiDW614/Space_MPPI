import torch


def near_zero(z: torch.Tensor, eps: float = 1e-6) -> torch.BoolTensor:
    """
    Determines whether elements of a tensor are approximately zero.
    :param z: tensor
    :param eps: tolerance
    :return: boolean tensor
    """
    return torch.abs(z) < eps


def axis_angle(p: torch.Tensor):
    """
    Computes the axis-angle representation of exponential coordinates.
    :param p: 3D tensor of exponential coordinates [omega * theta]
    :return: (omega_unit_vector, theta)
    """
    theta = p.norm()
    if near_zero(theta):
        return torch.zeros_like(p), theta
    return p / theta, theta


def vec3_to_skew3(p: torch.Tensor) -> torch.Tensor:
    """
    Returns the 3x3 skew-symmetric matrix [p] for a 3D vector.
    :param p: 3D vector
    :return: 3x3 skew-symmetric matrix
    """
    zero = torch.zeros(1, dtype=p.dtype, device=p.device)
    return torch.tensor([
        [zero,        -p[2],     p[1]],
        [p[2],        zero,     -p[0]],
        [-p[1],       p[0],      zero]
    ], dtype=p.dtype, device=p.device)


def vec6_to_skew4(s: torch.Tensor) -> torch.Tensor:
    """
    Returns the 4x4 twist matrix [s] for a 6D twist vector.
    :param s: 6D twist vector [omega, v]
    :return: 4x4 skew-symmetric matrix
    """
    omega = s[:3]
    v = s[3:]
    p_sk = vec3_to_skew3(omega)
    top = torch.cat([p_sk, v.view(3,1)], dim=1)
    bottom = torch.zeros((1,4), dtype=s.dtype, device=s.device)
    return torch.cat([top, bottom], dim=0)


def skew3_to_vec3(p_skew: torch.Tensor) -> torch.Tensor:
    """
    Extracts the 3D vector from a 3x3 skew-symmetric matrix.
    :param p_skew: 3x3 skew-symmetric matrix
    :return: 3D vector
    """
    return torch.tensor([
        p_skew[2,1],
        p_skew[0,2],
        p_skew[1,0]
    ], dtype=p_skew.dtype, device=p_skew.device)


def skew4_to_vec6(s_skew: torch.Tensor) -> torch.Tensor:
    """
    Extracts the 6D twist vector from a 4x4 skew-symmetric matrix.
    :param s_skew: 4x4 twist matrix
    :return: 6D vector [omega, v]
    """
    omega = skew3_to_vec3(s_skew[:3,:3])
    v = s_skew[:3,3]
    return torch.cat([omega, v], dim=0)


def htm_to_rp(T: torch.Tensor):
    """
    Splits a homogeneous transform into rotation R and position p.
    :param T: 4x4 transform
    :return: (R 3x3, p 3-vector)
    """
    return T[:3,:3], T[:3,3]


def htm_inverse(T: torch.Tensor) -> torch.Tensor:
    """
    Computes the inverse of a homogeneous transformation matrix.
    :param T: 4x4 transform
    :return: 4x4 inverse transform
    """
    R, p = htm_to_rp(T)
    R_t = R.t()
    top = torch.cat([R_t, -R_t @ p.view(3,1)], dim=1)
    bottom = torch.tensor([[0,0,0,1]], dtype=T.dtype, device=T.device)
    return torch.cat([top, bottom], dim=0)


def htm_adj(T: torch.Tensor) -> torch.Tensor:
    """
    Computes the 6x6 adjoint representation of a transform.
    :param T: 4x4 transform
    :return: 6x6 adjoint matrix
    """
    R, p = htm_to_rp(T)
    p_sk = vec3_to_skew3(p)
    top = torch.cat([R, torch.zeros((3,3), dtype=T.dtype, device=T.device)], dim=1)
    bottom = torch.cat([p_sk @ R, R], dim=1)
    return torch.cat([top, bottom], dim=0)


def skew3_to_matrix_exp3(p_sk: torch.Tensor) -> torch.Tensor:
    """
    Computes exp([omega] * theta) via Rodrigues' formula.
    :param p_sk: 3x3 skew matrix representing omega*theta
    :return: 3x3 rotation matrix
    """
    ptheta = skew3_to_vec3(p_sk)
    theta = ptheta.norm()
    if near_zero(theta):
        return torch.eye(3, dtype=p_sk.dtype, device=p_sk.device)
    omega_sk = p_sk / theta
    I = torch.eye(3, dtype=p_sk.dtype, device=p_sk.device)
    return I + torch.sin(theta) * omega_sk + (1 - torch.cos(theta)) * (omega_sk @ omega_sk)


def skew4_to_matrix_exp4(s_sk: torch.Tensor) -> torch.Tensor:
    """
    Computes the matrix exponential of a 4x4 twist matrix.
    :param s_sk: 4x4 twist matrix [omega*theta, v*theta]
    :return: 4x4 homogeneous transform
    """
    omega_sk = s_sk[:3,:3]
    vtheta = s_sk[:3,3]
    omegatheta = skew3_to_vec3(omega_sk)

    if near_zero(omegatheta.norm()):
        # Prismatic
        top = torch.cat([torch.eye(3, dtype=s_sk.dtype, device=s_sk.device), vtheta.view(3,1)], dim=1)
        bottom = torch.tensor([[0,0,0,1]], dtype=s_sk.dtype, device=s_sk.device)
        return torch.cat([top, bottom], dim=0)

    # Revolute
    theta = omegatheta.norm()
    omega_unit = omega_sk / theta
    R = skew3_to_matrix_exp3(omega_sk)
    I = torch.eye(3, dtype=s_sk.dtype, device=s_sk.device)
    G = I * theta + (1 - torch.cos(theta)) * omega_unit + (theta - torch.sin(theta)) * (omega_unit @ omega_unit)
    v = G @ (vtheta / theta)
    top = torch.cat([R, v.view(3,1)], dim=1)
    bottom = torch.tensor([[0,0,0,1]], dtype=s_sk.dtype, device=s_sk.device)
    return torch.cat([top, bottom], dim=0)


def mat3_to_log3(R: torch.Tensor) -> torch.Tensor:
    """
    Computes the matrix logarithm of a 3x3 rotation.
    :param R: 3x3 rotation matrix
    :return: 3x3 skew matrix log(R)
    """
    tr = torch.trace(R).item()
    cos_th = (tr - 1) / 2.0
    if cos_th >= 1.0:
        return torch.zeros((3,3), dtype=R.dtype, device=R.device)
    if cos_th <= -1.0:
        # theta = pi
        # choose largest diagonal
        if not near_zero(1 + R[2,2]):
            omega = torch.tensor([R[0,2], R[1,2], 1 + R[2,2]], device=R.device)
        elif not near_zero(1 + R[1,1]):
            omega = torch.tensor([R[0,1], 1 + R[1,1], R[2,1]], device=R.device)
        else:
            omega = torch.tensor([1 + R[0,0], R[1,0], R[2,0]], device=R.device)
        omega = omega / omega.norm()
        return vec3_to_skew3(torch.pi * omega)
    theta = torch.acos(torch.tensor(cos_th, dtype=R.dtype, device=R.device))
    return (theta / (2 * torch.sin(theta))) * (R - R.t())


def htm_to_log6(T: torch.Tensor) -> torch.Tensor:
    """
    Computes the matrix logarithm of a homogeneous transform.
    :param T: 4x4 transform
    :return: 4x4 twist matrix log(T)
    """
    R, p = htm_to_rp(T)
    omega_sk = mat3_to_log3(R)
    if torch.allclose(omega_sk, torch.zeros(3,3, dtype=R.dtype, device=R.device), atol=1e-6):
        top = torch.cat([torch.zeros((3,3), dtype=T.dtype, device=T.device), p.view(3,1)], dim=1)
        bottom = torch.tensor([[0,0,0,0]], dtype=T.dtype, device=T.device)
        return torch.cat([top, bottom], dim=0)

    # general case
    tr = torch.trace(R).item()
    theta = torch.acos(torch.tensor((tr - 1) / 2.0, dtype=R.dtype, device=R.device))
    I = torch.eye(3, dtype=R.dtype, device=R.device)
    omega2 = omega_sk @ omega_sk
    G = (I - 0.5 * omega_sk + (1.0 / theta - 0.5 / torch.tan(theta / 2.0)) * omega2)
    v = G @ p
    top = torch.cat([omega_sk, v.view(3,1)], dim=1)
    bottom = torch.tensor([[0,0,0,0]], dtype=T.dtype, device=T.device)
    return torch.cat([top, bottom], dim=0)
