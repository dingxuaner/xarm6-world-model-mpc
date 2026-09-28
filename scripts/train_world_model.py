import os
import copy

import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# ==========================================================
# Config
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DATA_PATH = (
    "data/xarm_world_model_15d.npz"
)

MODEL_DIR = "models"
RESULT_DIR = "results"

STATE_DIM = 9
ACTION_DIM = 6

BATCH_SIZE = 256

TRAIN_STEPS = 5000

MULTI_HORIZON = 10


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ==========================================================
# Load Dataset
# ==========================================================

data = np.load(
    DATA_PATH
)

states_15d = data[
    "states"
].astype(
    np.float32
)

actions = data[
    "actions"
].astype(
    np.float32
)


# ==========================================================
# Remove useless dq
#
# 15D:
#
# q   = 0:6
# dq  = 6:12
# xyz = 12:15
#
# ->
#
# 9D:
#
# [q, xyz]
# ==========================================================

states = np.concatenate(
    [
        states_15d[
            ...,
            0:6
        ],

        states_15d[
            ...,
            12:15
        ]
    ],
    axis=-1
)


print(
    "State shape:",
    states.shape
)

print(
    "Action shape:",
    actions.shape
)


# ==========================================================
# Train / Test Split
#
# 按episode切
#
# 不能把同一个episode随机拆散
# ==========================================================

NUM_EPISODES = (
    states.shape[0]
)

NUM_TRAIN = 24


train_states = states[
    :NUM_TRAIN
]

train_actions = actions[
    :NUM_TRAIN
]


test_states = states[
    NUM_TRAIN:
]

test_actions = actions[
    NUM_TRAIN:
]


print(
    "Train episodes:",
    train_states.shape[0]
)

print(
    "Test episodes:",
    test_states.shape[0]
)


# ==========================================================
# Statistics
# ==========================================================

state_mean = train_states.reshape(
    -1,
    STATE_DIM
).mean(
    axis=0
)


state_std = train_states.reshape(
    -1,
    STATE_DIM
).std(
    axis=0
)


action_mean = train_actions.reshape(
    -1,
    ACTION_DIM
).mean(
    axis=0
)


action_std = train_actions.reshape(
    -1,
    ACTION_DIM
).std(
    axis=0
)


state_std = np.maximum(
    state_std,
    1e-6
)


action_std = np.maximum(
    action_std,
    1e-6
)


# ==========================================================
# Convert Statistics to Torch
# ==========================================================

STATE_MEAN = torch.tensor(
    state_mean,
    dtype=torch.float32,
    device=DEVICE
)

STATE_STD = torch.tensor(
    state_std,
    dtype=torch.float32,
    device=DEVICE
)


ACTION_MEAN = torch.tensor(
    action_mean,
    dtype=torch.float32,
    device=DEVICE
)

ACTION_STD = torch.tensor(
    action_std,
    dtype=torch.float32,
    device=DEVICE
)


# ==========================================================
# Normalize
# ==========================================================

def normalize_state(
    state
):

    return (
        state
        -
        STATE_MEAN
    ) / STATE_STD


def normalize_action(
    action
):

    return (
        action
        -
        ACTION_MEAN
    ) / ACTION_STD


# ==========================================================
# World Model
# ==========================================================

class WorldModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                STATE_DIM
                +
                ACTION_DIM,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                STATE_DIM
            )
        )


    def forward(
        self,
        state,
        action
    ):

        x = torch.cat(
            [
                state,
                action
            ],
            dim=-1
        )

        delta_state = (
            self.net(
                x
            )
        )

        next_state = (
            state
            +
            delta_state
        )

        return next_state


# ==========================================================
# Tensor Dataset
# ==========================================================

train_states_t = torch.tensor(
    train_states,
    dtype=torch.float32,
    device=DEVICE
)


train_actions_t = torch.tensor(
    train_actions,
    dtype=torch.float32,
    device=DEVICE
)


test_states_t = torch.tensor(
    test_states,
    dtype=torch.float32,
    device=DEVICE
)


test_actions_t = torch.tensor(
    test_actions,
    dtype=torch.float32,
    device=DEVICE
)


# ==========================================================
# Sample One-step Batch
# ==========================================================

def sample_one_step_batch():

    E = train_states_t.shape[0]
    T = train_actions_t.shape[1]

    episode_idx = torch.randint(
        0,
        E,
        (BATCH_SIZE,),
        device=DEVICE
    )

    time_idx = torch.randint(
        0,
        T,
        (BATCH_SIZE,),
        device=DEVICE
    )


    state = train_states_t[
        episode_idx,
        time_idx
    ]

    action = train_actions_t[
        episode_idx,
        time_idx
    ]

    next_state = train_states_t[
        episode_idx,
        time_idx + 1
    ]


    return (
        normalize_state(
            state
        ),

        normalize_action(
            action
        ),

        normalize_state(
            next_state
        )
    )


# ==========================================================
# Sample Multi-step Batch
# ==========================================================

def sample_multi_step_batch(
    horizon
):

    E = train_states_t.shape[0]

    T = train_actions_t.shape[1]

    episode_idx = torch.randint(
        0,
        E,
        (BATCH_SIZE,),
        device=DEVICE
    )

    start_idx = torch.randint(
        0,
        T - horizon + 1,
        (BATCH_SIZE,),
        device=DEVICE
    )


    initial_states = []

    action_seq = []

    target_seq = []


    for b in range(
        BATCH_SIZE
    ):

        e = episode_idx[b]

        t = start_idx[b]

        initial_states.append(
            train_states_t[
                e,
                t
            ]
        )

        action_seq.append(
            train_actions_t[
                e,
                t:t + horizon
            ]
        )

        target_seq.append(
            train_states_t[
                e,
                t + 1:t + horizon + 1
            ]
        )


    initial_states = torch.stack(
        initial_states
    )

    action_seq = torch.stack(
        action_seq
    )

    target_seq = torch.stack(
        target_seq
    )


    return (
        normalize_state(
            initial_states
        ),

        normalize_action(
            action_seq
        ),

        normalize_state(
            target_seq
        )
    )


# ==========================================================
# Train One-step Model
# ==========================================================

def train_one_step():

    model = WorldModel().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()


    print(
        "\n===== One-step Training ====="
    )


    for step in range(
        TRAIN_STEPS
    ):

        (
            state,
            action,
            next_state
        ) = sample_one_step_batch()


        pred = model(
            state,
            action
        )


        loss = loss_fn(
            pred,
            next_state
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        if step % 250 == 0:

            print(
                f"OneStep "
                f"{step:05d} "
                f"Loss="
                f"{loss.item():.8f}"
            )


    return model


# ==========================================================
# Train Multi-step Model
# ==========================================================

def train_multi_step():

    model = WorldModel().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()


    print(
        "\n===== Multi-step Training ====="
    )


    for step in range(
        TRAIN_STEPS
    ):

        (
            state,
            actions_seq,
            target_seq
        ) = sample_multi_step_batch(
            MULTI_HORIZON
        )


        pred_state = state

        total_loss = 0.0


        for t in range(
            MULTI_HORIZON
        ):

            pred_state = model(
                pred_state,
                actions_seq[:, t]
            )


            total_loss += loss_fn(
                pred_state,
                target_seq[:, t]
            )


        loss = (
            total_loss
            /
            MULTI_HORIZON
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        if step % 250 == 0:

            print(
                f"MultiStep "
                f"{step:05d} "
                f"Loss="
                f"{loss.item():.8f}"
            )


    return model


# ==========================================================
# Rollout Evaluation
# ==========================================================

@torch.no_grad()
def evaluate_rollout(
    model
):

    model.eval()

    horizon = (
        test_actions_t.shape[1]
    )


    all_errors = []


    for episode in range(
        test_states_t.shape[0]
    ):

        state = (
            test_states_t[
                episode,
                0
            ]
        )

        pred_state = (
            normalize_state(
                state
            )
            .unsqueeze(0)
        )


        episode_errors = []


        for t in range(
            horizon
        ):

            action = (
                test_actions_t[
                    episode,
                    t
                ]
                .unsqueeze(0)
            )

            action_norm = (
                normalize_action(
                    action
                )
            )


            pred_state = model(
                pred_state,
                action_norm
            )


            true_state = (
                test_states_t[
                    episode,
                    t + 1
                ]
                .unsqueeze(0)
            )


            true_norm = (
                normalize_state(
                    true_state
                )
            )


            error = torch.mean(
                (
                    pred_state
                    -
                    true_norm
                ) ** 2
            )


            episode_errors.append(
                error.item()
            )


        all_errors.append(
            episode_errors
        )


    all_errors = np.array(
        all_errors
    )


    return all_errors.mean(
        axis=0
    )


# ==========================================================
# Train
# ==========================================================

one_step_model = (
    train_one_step()
)


multi_step_model = (
    train_multi_step()
)


# ==========================================================
# Evaluate
# ==========================================================

one_errors = evaluate_rollout(
    one_step_model
)


multi_errors = evaluate_rollout(
    multi_step_model
)


print(
    "\nFinal rollout normalized MSE"
)

print(
    "One-step:",
    one_errors[-1]
)

print(
    "Multi-step:",
    multi_errors[-1]
)


# ==========================================================
# Save
# ==========================================================

torch.save(
    {
        "model": one_step_model.state_dict(),

        "state_mean": state_mean,

        "state_std": state_std,

        "action_mean": action_mean,

        "action_std": action_std,
    },

    "models/one_step_world_model.pt"
)


torch.save(
    {
        "model": multi_step_model.state_dict(),

        "state_mean": state_mean,

        "state_std": state_std,

        "action_mean": action_mean,

        "action_std": action_std,
    },

    "models/multi_step_world_model.pt"
)


# ==========================================================
# Plot Error
# ==========================================================

plt.figure()


plt.plot(
    np.arange(
        1,
        len(one_errors) + 1
    ),

    one_errors,

    label="One-step Model"
)


plt.plot(
    np.arange(
        1,
        len(multi_errors) + 1
    ),

    multi_errors,

    label="Multi-step Model"
)


plt.xlabel(
    "Rollout Step"
)

plt.ylabel(
    "Normalized State MSE"
)

plt.title(
    "xArm6 World Model Rollout Error"
)

plt.legend()

plt.grid()


plt.savefig(
    "results/rollout_error.png",
    dpi=200
)


plt.show()
