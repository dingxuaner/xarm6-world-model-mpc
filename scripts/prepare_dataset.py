import os
import numpy as np

from xarm_learning.manual_fk import forward_kinematics


# ==========================================================
# Path
# ==========================================================

INPUT_PATH = (
    "data/xarm_random_dataset.npz"
)

OUTPUT_PATH = (
    "data/xarm_world_model_15d.npz"
)


# ==========================================================
# Load
# ==========================================================

data = np.load(
    INPUT_PATH
)


states = data[
    "states"
].astype(
    np.float32
)


actions = data[
    "actions"
].astype(
    np.float32
)


print(
    "Original states:",
    states.shape
)

print(
    "Actions:",
    actions.shape
)


# ==========================================================
# Split State
#
# state =
#
# [q1...q6, dq1...dq6]
# ==========================================================

q = states[
    ...,
    0:6
]

dq = states[
    ...,
    6:12
]


# ==========================================================
# Basic Dataset Analysis
# ==========================================================

print(
    "\n=============================="
)

print(
    "Dataset Statistics"
)

print(
    "=============================="
)


print(
    "\nJoint position min:"
)

print(
    q.min(
        axis=(0, 1)
    )
)


print(
    "\nJoint position max:"
)

print(
    q.max(
        axis=(0, 1)
    )
)


print(
    "\nMean |velocity|:"
)

print(
    np.mean(
        np.abs(dq),
        axis=(0, 1)
    )
)


print(
    "\nMax |velocity|:"
)

print(
    np.max(
        np.abs(dq),
        axis=(0, 1)
    )
)


print(
    "\nMean |action|:"
)

print(
    np.mean(
        np.abs(actions),
        axis=(0, 1)
    )
)


print(
    "\nMax |action|:"
)

print(
    np.max(
        np.abs(actions),
        axis=(0, 1)
    )
)


# ==========================================================
# Commanded action vs realized joint motion
#
# q_(t+1) - q_t
# ==========================================================

realized_delta_q = (
    q[:, 1:, :]
    -
    q[:, :-1, :]
)


tracking_error = (
    realized_delta_q
    -
    actions
)


print(
    "\nMean absolute command tracking error:"
)

print(
    np.mean(
        np.abs(
            tracking_error
        ),
        axis=(0, 1)
    )
)


print(
    "\nOverall command tracking MAE:"
)

print(
    np.mean(
        np.abs(
            tracking_error
        )
    )
)


print(
    "\nExample:"
)

print(
    "commanded action:"
)

print(
    actions[
        0,
        0
    ]
)


print(
    "realized delta q:"
)

print(
    realized_delta_q[
        0,
        0
    ]
)


# ==========================================================
# Forward Kinematics
#
# q
#
# ->
#
# xyz
# ==========================================================

num_episodes = (
    states.shape[0]
)

num_states = (
    states.shape[1]
)


ee_positions = np.zeros(
    (
        num_episodes,
        num_states,
        3
    ),
    dtype=np.float32
)


print(
    "\nComputing FK..."
)


for episode in range(
    num_episodes
):

    for t in range(
        num_states
    ):

        q_t = q[
            episode,
            t
        ]

        T = forward_kinematics(
            q_t
        )

        ee_positions[
            episode,
            t
        ] = T[
            :3,
            3
        ]


print(
    "EE positions:",
    ee_positions.shape
)


print(
    "\nEE xyz min:"
)

print(
    ee_positions.min(
        axis=(0, 1)
    )
)


print(
    "\nEE xyz max:"
)

print(
    ee_positions.max(
        axis=(0, 1)
    )
)


# ==========================================================
# Build 15D State
#
# state =
#
# [
#   q       6
#   dq      6
#   xyz     3
# ]
#
# total = 15
# ==========================================================

states_15d = np.concatenate(
    [
        q,
        dq,
        ee_positions
    ],
    axis=-1
).astype(
    np.float32
)


print(
    "\nNew state shape:"
)

print(
    states_15d.shape
)


# ==========================================================
# Sanity Check
# ==========================================================

print(
    "\nFirst 15D state:"
)

print(
    states_15d[
        0,
        0
    ]
)


print(
    "\nFirst EE position:"
)

print(
    ee_positions[
        0,
        0
    ]
)


# ==========================================================
# Save
# ==========================================================

np.savez_compressed(
    OUTPUT_PATH,

    states=states_15d,

    actions=actions,

    joint_names=data[
        "joint_names"
    ],

    state_description=np.array(
        [
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",

            "dq1",
            "dq2",
            "dq3",
            "dq4",
            "dq5",
            "dq6",

            "ee_x",
            "ee_y",
            "ee_z",
        ]
    ),
)


print(
    "\nSaved:"
)

print(
    os.path.abspath(
        OUTPUT_PATH
    )
)
