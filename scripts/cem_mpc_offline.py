import numpy as np

import torch

import matplotlib.pyplot as plt


from world_model import WorldModel

from xarm_learning.manual_fk import (
    forward_kinematics
)


# ==========================================================
# Config
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = (
    "models/multi_step_world_model.pt"
)

DATA_PATH = (
    "data/xarm_world_model_15d.npz"
)


STATE_DIM = 9
ACTION_DIM = 6


# ==========================================================
# CEM Config
# ==========================================================

HORIZON = 15

NUM_SAMPLES = 1024

NUM_ELITES = 64

CEM_ITERATIONS = 5


ACTION_LIMIT = 0.035

INITIAL_STD = 0.025


# MPC最多执行多少步
MPC_STEPS = 40


# ==========================================================
# Load World Model
# ==========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


model = WorldModel().to(
    DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()


# ==========================================================
# Normalization Statistics
# ==========================================================

STATE_MEAN = torch.tensor(
    checkpoint["state_mean"],
    dtype=torch.float32,
    device=DEVICE
)

STATE_STD = torch.tensor(
    checkpoint["state_std"],
    dtype=torch.float32,
    device=DEVICE
)


ACTION_MEAN = torch.tensor(
    checkpoint["action_mean"],
    dtype=torch.float32,
    device=DEVICE
)

ACTION_STD = torch.tensor(
    checkpoint["action_std"],
    dtype=torch.float32,
    device=DEVICE
)


def normalize_state(state):

    return (
        state
        -
        STATE_MEAN
    ) / STATE_STD


def denormalize_state(state):

    return (
        state
        *
        STATE_STD
        +
        STATE_MEAN
    )


def normalize_action(action):

    return (
        action
        -
        ACTION_MEAN
    ) / ACTION_STD


# ==========================================================
# Load Dataset
#
# 用一个数据集中的真实目标点，
# 保证goal处于训练数据覆盖区域附近
# ==========================================================

data = np.load(
    DATA_PATH
)

states_15d = data[
    "states"
].astype(
    np.float32
)


# ----------------------------------------------------------
# 从一个没有用于训练的episode里选目标
#
# 训练episode = 0~23
#
# 这里使用episode 24
# ----------------------------------------------------------

GOAL_EPISODE = 24
GOAL_TIME = 25


goal_q = states_15d[
    GOAL_EPISODE,
    GOAL_TIME,
    0:6
]


goal_xyz = states_15d[
    GOAL_EPISODE,
    GOAL_TIME,
    12:15
]


GOAL_XYZ = torch.tensor(
    goal_xyz,
    dtype=torch.float32,
    device=DEVICE
)


print(
    "Goal q:"
)

print(
    goal_q
)


print(
    "\nGoal xyz:"
)

print(
    goal_xyz
)


# ==========================================================
# Fake "True Environment"
#
# 你的数据已经证明：
#
# q_next = q + action
#
# 然后通过真实FK算xyz
# ==========================================================

def true_environment_step(
    state,
    action
):

    q = state[
        0:6
    ]

    q_next = (
        q
        +
        action
    )


    T = forward_kinematics(
        q_next
    )


    xyz_next = (
        T[
            :3,
            3
        ]
        .astype(
            np.float32
        )
    )


    next_state = np.concatenate(
        [
            q_next,
            xyz_next
        ]
    ).astype(
        np.float32
    )


    return next_state


# ==========================================================
# CEM Planning
# ==========================================================

@torch.no_grad()
def cem_plan(
    current_state
):

    # ------------------------------------------------------
    # action sequence distribution
    #
    # [H, 6]
    # ------------------------------------------------------

    mean = torch.zeros(
        HORIZON,
        ACTION_DIM,
        device=DEVICE
    )


    std = torch.ones(
        HORIZON,
        ACTION_DIM,
        device=DEVICE
    ) * INITIAL_STD


    current_state_t = torch.tensor(
        current_state,
        dtype=torch.float32,
        device=DEVICE
    )


    # ======================================================
    # CEM iteration
    # ======================================================

    for iteration in range(
        CEM_ITERATIONS
    ):

        # --------------------------------------------------
        # Sample action sequences
        #
        # [N,H,6]
        # --------------------------------------------------

        noise = torch.randn(
            NUM_SAMPLES,
            HORIZON,
            ACTION_DIM,
            device=DEVICE
        )


        action_sequences = (
            mean.unsqueeze(0)
            +
            std.unsqueeze(0)
            * noise
        )


        action_sequences = torch.clamp(
            action_sequences,
            -ACTION_LIMIT,
            ACTION_LIMIT
        )


        # --------------------------------------------------
        # 所有sample从同一个当前state开始
        # --------------------------------------------------

        state = (
            current_state_t
            .unsqueeze(0)
            .repeat(
                NUM_SAMPLES,
                1
            )
        )


        state_norm = normalize_state(
            state
        )


        scores = torch.zeros(
            NUM_SAMPLES,
            device=DEVICE
        )


        # ==================================================
        # World Model Rollout
        # ==================================================

        for t in range(
            HORIZON
        ):

            action = (
                action_sequences[
                    :,
                    t
                ]
            )


            action_norm = (
                normalize_action(
                    action
                )
            )


            state_norm = model(
                state_norm,
                action_norm
            )


            # ----------------------------------------------
            # 回到真实物理量
            # ----------------------------------------------

            state_physical = (
                denormalize_state(
                    state_norm
                )
            )


            predicted_xyz = (
                state_physical[
                    :,
                    6:9
                ]
            )


            distance = torch.norm(
                predicted_xyz
                -
                GOAL_XYZ.unsqueeze(0),
                dim=-1
            )


            # ----------------------------------------------
            # 每一步都希望靠近goal
            #
            # score越大越好
            # ----------------------------------------------

            scores -= distance


            # ----------------------------------------------
            # 很小的动作惩罚
            # 避免无意义的大动作
            # ----------------------------------------------

            action_cost = (
                0.02
                *
                torch.sum(
                    action ** 2,
                    dim=-1
                )
            )


            scores -= action_cost


        # ==================================================
        # Elite Selection
        # ==================================================

        elite_indices = torch.topk(
            scores,
            NUM_ELITES
        ).indices


        elites = action_sequences[
            elite_indices
        ]


        # ==================================================
        # Update Gaussian Distribution
        # ==================================================

        mean = elites.mean(
            dim=0
        )


        std = elites.std(
            dim=0
        )


        std = torch.clamp(
            std,
            min=0.002
        )


    # ======================================================
    # 最终只返回第一步
    # ======================================================

    first_action = (
        mean[0]
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    return first_action


# ==========================================================
# Initial State
# ==========================================================

q0 = np.zeros(
    6,
    dtype=np.float32
)


T0 = forward_kinematics(
    q0
)


xyz0 = (
    T0[
        :3,
        3
    ]
    .astype(
        np.float32
    )
)


state = np.concatenate(
    [
        q0,
        xyz0
    ]
).astype(
    np.float32
)


print(
    "\nInitial xyz:"
)

print(
    xyz0
)


print(
    "\nInitial distance:"
)

print(
    np.linalg.norm(
        xyz0
        -
        goal_xyz
    )
)


# ==========================================================
# MPC Closed Loop
# ==========================================================

trajectory = [
    xyz0.copy()
]


distance_history = []


for step in range(
    MPC_STEPS
):

    # ======================================================
    # Plan
    # ======================================================

    action = cem_plan(
        state
    )


    # ======================================================
    # Execute only first action
    # ======================================================

    state = true_environment_step(
        state,
        action
    )


    xyz = state[
        6:9
    ]


    distance = np.linalg.norm(
        xyz
        -
        goal_xyz
    )


    trajectory.append(
        xyz.copy()
    )


    distance_history.append(
        distance
    )


    print(
        f"Step "
        f"{step:02d} | "
        f"distance="
        f"{distance:.5f} | "
        f"|action|="
        f"{np.linalg.norm(action):.5f}"
    )


    # ======================================================
    # Goal reached
    # ======================================================

    if distance < 0.015:

        print(
            "\nGoal reached!"
        )

        break


trajectory = np.array(
    trajectory
)


# ==========================================================
# Final Result
# ==========================================================

print(
    "\nFinal xyz:"
)

print(
    trajectory[-1]
)


print(
    "\nGoal xyz:"
)

print(
    goal_xyz
)


print(
    "\nFinal distance:"
)

print(
    np.linalg.norm(
        trajectory[-1]
        -
        goal_xyz
    )
)


# ==========================================================
# Plot XYZ trajectory
# ==========================================================

fig = plt.figure()

ax = fig.add_subplot(
    111,
    projection="3d"
)


ax.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    trajectory[:, 2],
    marker="o",
    label="CEM-MPC"
)


ax.scatter(
    trajectory[0, 0],
    trajectory[0, 1],
    trajectory[0, 2],
    marker="s",
    s=100,
    label="Start"
)


ax.scatter(
    goal_xyz[0],
    goal_xyz[1],
    goal_xyz[2],
    marker="*",
    s=200,
    label="Goal"
)


ax.set_xlabel(
    "X"
)

ax.set_ylabel(
    "Y"
)

ax.set_zlabel(
    "Z"
)

ax.legend()


plt.title(
    "xArm6 World Model + CEM-MPC"
)


plt.savefig(
    "results/cem_mpc_offline.png",
    dpi=200
)


plt.show()


# ==========================================================
# Distance Curve
# ==========================================================

plt.figure()


plt.plot(
    distance_history,
    marker="o"
)


plt.xlabel(
    "MPC Step"
)

plt.ylabel(
    "EE Distance to Goal (m)"
)

plt.title(
    "CEM-MPC Goal Distance"
)


plt.grid()


plt.savefig(
    "results/cem_mpc_distance.png",
    dpi=200
)


plt.show()