import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

def rot_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])


def rot_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])


def rot_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def rpy_to_rotation(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def make_transform(x, y, z, roll, pitch, yaw):
    T = np.eye(4)

    R = rpy_to_rotation(
        roll,
        pitch,
        yaw
    )

    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def joint_transform(
    x, y, z,
    roll, pitch, yaw,
    q
):
    """
    URDF revolute joint:

    T_parent_child =
        T_origin @ R_axis(q)

    当前 xArm6 所有关节 axis 都是局部 Z 轴。
    """

    T_origin = make_transform(
        x, y, z,
        roll, pitch, yaw
    )

    T_joint = np.eye(4)
    T_joint[:3, :3] = rot_z(q)

    return T_origin @ T_joint


def forward_kinematics(q, verbose=False):
    """
    q = [q1, q2, q3, q4, q5, q6]
    """

    joint_params = [
        # x,       y,        z,      roll,       pitch, yaw
        (0.0,      0.0,      0.267,   0.0,        0.0, 0.0),
        (0.0,      0.0,      0.0,    -1.5708,     0.0, 0.0),
        (0.0535,  -0.2845,    0.0,     0.0,        0.0, 0.0),
        (0.0775,   0.3425,    0.0,    -1.5708,     0.0, 0.0),
        (0.0,      0.0,       0.0,     1.5708,     0.0, 0.0),
        (0.076,    0.097,     0.0,    -1.5708,     0.0, 0.0),
    ]

    T = np.eye(4)

    for i in range(6):
        x, y, z, roll, pitch, yaw = joint_params[i]

        Ti = joint_transform(
            x, y, z,
            roll, pitch, yaw,
            q[i]
        )

        T = T @ Ti

        if verbose:
            print(f"\nT_base_link{i + 1}:")
            print(np.round(T, 6))

    return T


def rotation_matrix_to_quaternion(R):
    """
    输出格式：
    [qx, qy, qz, qw]
    """

    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0

        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(
            1.0 + R[0, 0] - R[1, 1] - R[2, 2]
        ) * 2.0

        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(
            1.0 + R[1, 1] - R[0, 0] - R[2, 2]
        ) * 2.0

        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

    else:
        s = np.sqrt(
            1.0 + R[2, 2] - R[0, 0] - R[1, 1]
        ) * 2.0

        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([
        qx,
        qy,
        qz,
        qw
    ])

def numerical_jacobian(q, eps=1e-6):

    J = np.zeros((3, 6))

    T0 = forward_kinematics(q)
    p0 = T0[:3, 3]

    for i in range(6):

        q_eps = q.copy()
        q_eps[i] += eps

        T_eps = forward_kinematics(q_eps)
        p_eps = T_eps[:3, 3]

        J[:, i] = (p_eps - p0) / eps

    return J

def analytic_jacobian(q):

    joint_params = [
        (0.0,     0.0,     0.267,  0.0,     0.0, 0.0),
        (0.0,     0.0,     0.0,   -1.5708,  0.0, 0.0),
        (0.0535, -0.2845,  0.0,    0.0,     0.0, 0.0),
        (0.0775,  0.3425,  0.0,   -1.5708,  0.0, 0.0),
        (0.0,     0.0,     0.0,    1.5708,  0.0, 0.0),
        (0.076,   0.097,   0.0,   -1.5708,  0.0, 0.0),
    ]

    T = np.eye(4)

    joint_positions = []
    joint_axes = []

    z_local = np.array([0.0, 0.0, 1.0])

    for i in range(6):

        x, y, z, roll, pitch, yaw = joint_params[i]

        # 先到达 joint_i 的 origin
        T_origin = make_transform(
            x, y, z,
            roll, pitch, yaw
        )

        T_joint_frame = T @ T_origin

        # 关节轴上一点，在 base 下的位置
        p_i = T_joint_frame[:3, 3]

        # 关节局部 Z 轴转换到 base 坐标系
        R_i = T_joint_frame[:3, :3]

        z_i = R_i @ z_local

        joint_positions.append(p_i)
        joint_axes.append(z_i)

        # 再加入关节实际旋转
        T_rotation = np.eye(4)
        T_rotation[:3, :3] = rot_z(q[i])

        T = T_joint_frame @ T_rotation

    # 最终末端位置
    p_e = T[:3, 3]

    Jv = np.zeros((3, 6))
    Jw = np.zeros((3, 6))

    for i in range(6):

        p_i = joint_positions[i]
        z_i = joint_axes[i]

        Jv[:, i] = np.cross(
            z_i,
            p_e - p_i
        )

        Jw[:, i] = z_i

    J = np.vstack([
        Jv,
        Jw
    ])

    return J

def cartesian_velocity_to_joint_velocity(q, v_des):

    J = analytic_jacobian(q)

    Jv = J[:3, :]

    Jv_pinv = np.linalg.pinv(Jv)

    dq = Jv_pinv @ v_des

    return dq

def cartesian_velocity_to_joint_velocity(q, v_des):

    J = analytic_jacobian(q)

    Jv = J[:3, :]

    Jv_pinv = np.linalg.pinv(Jv)

    dq = Jv_pinv @ v_des

    return dq

def simulate_cartesian_motion(
    q_init,
    v_des,
    dt=0.01,
    duration=2.0
):
    q = q_init.copy()

    steps = int(duration / dt)

    print("\n=== Cartesian Motion Simulation ===")

    for i in range(steps):

        J = analytic_jacobian(q)

        Jv = J[:3, :]

        dq = np.linalg.pinv(Jv) @ v_des

        q = q + dq * dt

        if i % 20 == 0:

            T = forward_kinematics(q)

            p = T[:3, 3]

            print(
                f"t={i*dt:.2f} s  "
                f"x={p[0]:.4f}  "
                f"y={p[1]:.4f}  "
                f"z={p[2]:.4f}"
            )

    return q


def main():

    q = np.array([
        0.3,
        -0.5,
        0.4,
        0.2,
        -0.3,
        0.1,
    ])

    T = forward_kinematics(q, verbose=True)

    print("\n==============================")
    print("Final T_base_eef:")
    print(np.round(T, 6))

    position = T[:3, 3]
    rotation = T[:3, :3]

    quaternion = rotation_matrix_to_quaternion(
        rotation
    )

    print("\nPosition:")
    print(f"x = {position[0]:.6f} m")
    print(f"y = {position[1]:.6f} m")
    print(f"z = {position[2]:.6f} m")

    print("\nQuaternion:")
    print(f"qx = {quaternion[0]:.6f}")
    print(f"qy = {quaternion[1]:.6f}")
    print(f"qz = {quaternion[2]:.6f}")
    print(f"qw = {quaternion[3]:.6f}")

    J = numerical_jacobian(q)

    print("\nPosition Jacobian:")
    print(np.round(J, 6))

    J_num = numerical_jacobian(q)

    J_ana = analytic_jacobian(q)

    print("\nNumerical Position Jacobian:")
    print(np.round(J_num, 6))

    print("\nAnalytic Position Jacobian:")
    print(np.round(J_ana[:3, :], 6))

    print("\nFull 6x6 Jacobian:")
    print(np.round(J_ana, 6))

    print("\nPosition Jacobian Error:")
    print(
        np.round(
            J_num - J_ana[:3, :],
            8
        )
    )

    v_des = np.array([
        0.01,
        0.0,
        0.0
    ])

    dq = cartesian_velocity_to_joint_velocity(
        q,
        v_des
    )

    print("\nDesired Cartesian Velocity:")
    print(v_des)

    print("\nComputed Joint Velocity:")
    print(np.round(dq, 6))

    J = analytic_jacobian(q)

    v_check = J[:3, :] @ dq

    print("\nCheck Cartesian Velocity:")
    print(np.round(v_check, 6))

    q_start = np.array([
        0.3,
        -0.5,
        0.4,
        0.2,
        -0.3,
        0.1
    ], dtype=float)

    v_des = np.array([
        0.01,
        0.0,
        0.0
    ])

    q_final = simulate_cartesian_motion(
        q_start,
        v_des,
        dt=0.01,
        duration=2.0
    )

    print("\nFinal Joint Position:")
    print(np.round(q_final, 6))

    T_final = forward_kinematics(q_final)

    print("\nFinal Cartesian Position:")
    print(np.round(T_final[:3, 3], 6))


if __name__ == "__main__":
    main()