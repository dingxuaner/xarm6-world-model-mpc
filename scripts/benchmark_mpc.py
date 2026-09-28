import os
import csv
import time

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

DATA_PATH = (
    "data/xarm_world_model_15d.npz"
)

ONE_STEP_MODEL_PATH = (
    "models/one_step_world_model.pt"
)

MULTI_STEP_MODEL_PATH = (
    "models/multi_step_world_model.pt"
)


RESULT_DIR = "results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


STATE_DIM = 9
ACTION_DIM = 6


# ==========================================================
# Planning
# ==========================================================

HORIZON = 15

NUM_SAMPLES = 1024

NUM_ELITES = 64

CEM_ITERATIONS = 5


# Random Shooting使用同样总sample budget
RANDOM_SAMPLES = (
    NUM_SAMPLES
    *
    CEM_ITERATIONS
)


ACTION_LIMIT = 0.035

INITIAL_STD = 0.025


MAX_MPC_STEPS = 30

GOAL_THRESHOLD = 0.015


# ==========================================================
# Benchmark
# ==========================================================

NUM_GOALS = 20

RANDOM_SEED = 42


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


q_dataset = states_15d[
    ...,
    0:6
]


q_min = q_dataset.min(
    axis=(0, 1)
)

q_max = q_dataset.max(
    axis=(0, 1)
)


Q_MIN_T = torch.tensor(
    q_min,
    dtype=torch.float32,
    device=DEVICE
)

Q_MAX_T = torch.tensor(
    q_max,
    dtype=torch.float32,
    device=DEVICE
)


# ==========================================================
# True Environment
#
# fake simulation已经证明：
#
# q_next = q + action
#
# xyz由真实FK计算
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


    # 与ROS版一样，
    # 限制在训练数据覆盖区域
    q_next = np.clip(
        q_next,
        q_min,
        q_max
    ).astype(
        np.float32
    )


    T = forward_kinematics(
        q_next
    )


    xyz = T[
        :3,
        3
    ].astype(
        np.float32
    )


    next_state = np.concatenate(
        [
            q_next,
            xyz
        ]
    ).astype(
        np.float32
    )


    return next_state


# ==========================================================
# Initial State
# ==========================================================

def create_initial_state():

    q = np.zeros(
        6,
        dtype=np.float32
    )


    T = forward_kinematics(
        q
    )


    xyz = T[
        :3,
        3
    ].astype(
        np.float32
    )


    state = np.concatenate(
        [
            q,
            xyz
        ]
    ).astype(
        np.float32
    )


    return state


# ==========================================================
# Learned Model Wrapper
# ==========================================================

class LearnedDynamics:

    def __init__(
        self,
        checkpoint_path
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False
        )


        self.model = (
            WorldModel()
            .to(DEVICE)
        )


        self.model.load_state_dict(
            checkpoint[
                "model"
            ]
        )


        self.model.eval()


        self.state_mean = torch.tensor(
            checkpoint[
                "state_mean"
            ],
            dtype=torch.float32,
            device=DEVICE
        )


        self.state_std = torch.tensor(
            checkpoint[
                "state_std"
            ],
            dtype=torch.float32,
            device=DEVICE
        )


        self.action_mean = torch.tensor(
            checkpoint[
                "action_mean"
            ],
            dtype=torch.float32,
            device=DEVICE
        )


        self.action_std = torch.tensor(
            checkpoint[
                "action_std"
            ],
            dtype=torch.float32,
            device=DEVICE
        )


    def normalize_state(
        self,
        state
    ):

        return (
            state
            -
            self.state_mean
        ) / self.state_std


    def denormalize_state(
        self,
        state
    ):

        return (
            state
            *
            self.state_std
            +
            self.state_mean
        )


    def normalize_action(
        self,
        action
    ):

        return (
            action
            -
            self.action_mean
        ) / self.action_std


# ==========================================================
# Score Action Sequences
#
# action_sequences:
#
# [N,H,6]
#
# return:
#
# [N]
# ==========================================================

@torch.no_grad()
def score_action_sequences(
    dynamics,
    current_state,
    goal_xyz,
    action_sequences
):

    num_samples = (
        action_sequences.shape[0]
    )


    goal_t = torch.tensor(
        goal_xyz,
        dtype=torch.float32,
        device=DEVICE
    )


    current_state_t = torch.tensor(
        current_state,
        dtype=torch.float32,
        device=DEVICE
    )


    state = (
        current_state_t
        .unsqueeze(0)
        .repeat(
            num_samples,
            1
        )
    )


    state_norm = (
        dynamics.normalize_state(
            state
        )
    )


    scores = torch.zeros(
        num_samples,
        device=DEVICE
    )


    # ======================================================
    # Imagined rollout
    # ======================================================

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
            dynamics.normalize_action(
                action
            )
        )


        state_norm = (
            dynamics.model(
                state_norm,
                action_norm
            )
        )


        state_physical = (
            dynamics.denormalize_state(
                state_norm
            )
        )


        predicted_q = (
            state_physical[
                :,
                0:6
            ]
        )


        predicted_xyz = (
            state_physical[
                :,
                6:9
            ]
        )


        # --------------------------------------------------
        # Goal distance
        # --------------------------------------------------

        distance = torch.norm(
            predicted_xyz
            -
            goal_t.unsqueeze(0),
            dim=-1
        )


        scores -= distance


        # --------------------------------------------------
        # Small action penalty
        # --------------------------------------------------

        action_cost = (
            0.02
            *
            torch.sum(
                action ** 2,
                dim=-1
            )
        )


        scores -= action_cost


        # --------------------------------------------------
        # OOD penalty
        # --------------------------------------------------

        below = torch.relu(
            Q_MIN_T
            -
            predicted_q
        )


        above = torch.relu(
            predicted_q
            -
            Q_MAX_T
        )


        ood_cost = torch.sum(
            below ** 2
            +
            above ** 2,
            dim=-1
        )


        scores -= (
            100.0
            *
            ood_cost
        )


    return scores


# ==========================================================
# CEM Planner
# ==========================================================

@torch.no_grad()
def cem_plan(
    dynamics,
    current_state,
    goal_xyz
):

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


    for iteration in range(
        CEM_ITERATIONS
    ):

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
            *
            noise
        )


        action_sequences = torch.clamp(
            action_sequences,
            -ACTION_LIMIT,
            ACTION_LIMIT
        )


        scores = score_action_sequences(
            dynamics,
            current_state,
            goal_xyz,
            action_sequences
        )


        elite_indices = torch.topk(
            scores,
            NUM_ELITES
        ).indices


        elites = (
            action_sequences[
                elite_indices
            ]
        )


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


    return (
        mean[0]
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


# ==========================================================
# Random Shooting Planner
# ==========================================================

@torch.no_grad()
def random_shooting_plan(
    dynamics,
    current_state,
    goal_xyz
):

    action_sequences = (
        torch.randn(
            RANDOM_SAMPLES,
            HORIZON,
            ACTION_DIM,
            device=DEVICE
        )
        *
        INITIAL_STD
    )


    action_sequences = torch.clamp(
        action_sequences,
        -ACTION_LIMIT,
        ACTION_LIMIT
    )


    scores = score_action_sequences(
        dynamics,
        current_state,
        goal_xyz,
        action_sequences
    )


    best_index = torch.argmax(
        scores
    )


    best_sequence = (
        action_sequences[
            best_index
        ]
    )


    return (
        best_sequence[
            0
        ]
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


# ==========================================================
# Run one MPC episode
# ==========================================================

def run_episode(
    dynamics,
    planner,
    goal_xyz
):

    state = create_initial_state()


    initial_distance = np.linalg.norm(
        state[
            6:9
        ]
        -
        goal_xyz
    )


    planning_times = []


    success = False


    steps_used = 0


    for step in range(
        MAX_MPC_STEPS
    ):

        current_xyz = (
            state[
                6:9
            ]
        )


        current_distance = (
            np.linalg.norm(
                current_xyz
                -
                goal_xyz
            )
        )


        if (
            current_distance
            <
            GOAL_THRESHOLD
        ):

            success = True

            steps_used = step

            break


        start_time = (
            time.perf_counter()
        )


        action = planner(
            dynamics,
            state,
            goal_xyz
        )


        # CUDA是异步执行的，
        # 为了准确计时需要同步
        if DEVICE == "cuda":

            torch.cuda.synchronize()


        planning_time = (
            time.perf_counter()
            -
            start_time
        )


        planning_times.append(
            planning_time
        )


        state = true_environment_step(
            state,
            action
        )


        steps_used = (
            step + 1
        )


    final_xyz = (
        state[
            6:9
        ]
    )


    final_distance = np.linalg.norm(
        final_xyz
        -
        goal_xyz
    )


    if (
        final_distance
        <
        GOAL_THRESHOLD
    ):

        success = True


    mean_planning_time = (
        np.mean(
            planning_times
        )
        if len(planning_times) > 0
        else 0.0
    )


    return {
        "success": success,

        "initial_distance": float(
            initial_distance
        ),

        "final_distance": float(
            final_distance
        ),

        "steps": int(
            steps_used
        ),

        "planning_time": float(
            mean_planning_time
        ),
    }


# ==========================================================
# Select Held-out Goals
#
# Episodes:
#
# 0 ~ 23  train
# 24 ~ 29 test
# ==========================================================

def select_test_goals():

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    candidates = []


    initial_state = (
        create_initial_state()
    )


    initial_xyz = (
        initial_state[
            6:9
        ]
    )


    for episode in range(
        24,
        30
    ):

        for t in range(
            1,
            61
        ):

            xyz = states_15d[
                episode,
                t,
                12:15
            ]


            distance = np.linalg.norm(
                xyz
                -
                initial_xyz
            )


            # 太近的goal没有测试价值
            if distance > 0.04:

                candidates.append(
                    (
                        episode,
                        t,
                        xyz.copy()
                    )
                )


    selected_indices = rng.choice(
        len(candidates),
        size=NUM_GOALS,
        replace=False
    )


    selected = [
        candidates[i]
        for i in selected_indices
    ]


    return selected


# ==========================================================
# Load Models
# ==========================================================

print(
    "Device:",
    DEVICE
)


print(
    "Loading models..."
)


one_step_dynamics = LearnedDynamics(
    ONE_STEP_MODEL_PATH
)


multi_step_dynamics = LearnedDynamics(
    MULTI_STEP_MODEL_PATH
)


# ==========================================================
# Methods
# ==========================================================

methods = {
    "OneStep+CEM": (
        one_step_dynamics,
        cem_plan
    ),

    "MultiStep+CEM": (
        multi_step_dynamics,
        cem_plan
    ),

    "MultiStep+Random": (
        multi_step_dynamics,
        random_shooting_plan
    ),
}


# ==========================================================
# Benchmark
# ==========================================================

goals = select_test_goals()


print(
    f"\nSelected {len(goals)} "
    f"held-out goals."
)


results = []


for goal_index, (
    episode,
    t,
    goal_xyz
) in enumerate(
    goals
):

    print(
        "\n======================================"
    )

    print(
        f"Goal "
        f"{goal_index + 1}/"
        f"{NUM_GOALS}"
    )

    print(
        f"Source: episode={episode}, "
        f"time={t}"
    )

    print(
        "Goal xyz:",
        goal_xyz
    )


    for method_name, (
        dynamics,
        planner
    ) in methods.items():

        result = run_episode(
            dynamics,
            planner,
            goal_xyz
        )


        row = {
            "goal_id": goal_index,

            "source_episode": episode,

            "source_time": t,

            "method": method_name,

            **result
        }


        results.append(
            row
        )


        print(
            f"{method_name:18s} | "
            f"success={result['success']} | "
            f"initial="
            f"{result['initial_distance']:.4f} | "
            f"final="
            f"{result['final_distance']:.4f} | "
            f"steps="
            f"{result['steps']:2d} | "
            f"time="
            f"{result['planning_time']*1000:.2f} ms"
        )


# ==========================================================
# Save CSV
# ==========================================================

csv_path = (
    "results/"
    "benchmark_results.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "goal_id",
            "source_episode",
            "source_time",
            "method",
            "success",
            "initial_distance",
            "final_distance",
            "steps",
            "planning_time",
        ]
    )


    writer.writeheader()

    writer.writerows(
        results
    )


print(
    "\nSaved:",
    csv_path
)


# ==========================================================
# Summary
# ==========================================================

print(
    "\n======================================"
)

print(
    "BENCHMARK SUMMARY"
)

print(
    "======================================"
)


summary = {}


for method_name in methods.keys():

    method_results = [
        r
        for r in results
        if r["method"]
        ==
        method_name
    ]


    success_rate = np.mean(
        [
            r["success"]
            for r in method_results
        ]
    )


    final_errors = np.array(
        [
            r["final_distance"]
            for r in method_results
        ]
    )


    steps = np.array(
        [
            r["steps"]
            for r in method_results
        ]
    )


    planning_times = np.array(
        [
            r["planning_time"]
            for r in method_results
        ]
    )


    summary[
        method_name
    ] = {
        "success_rate": success_rate,

        "mean_final_error": (
            final_errors.mean()
        ),

        "median_final_error": (
            np.median(
                final_errors
            )
        ),

        "mean_steps": (
            steps.mean()
        ),

        "mean_planning_time": (
            planning_times.mean()
        ),
    }


    print(
        f"\n{method_name}"
    )

    print(
        f"  Success rate: "
        f"{success_rate*100:.1f}%"
    )

    print(
        f"  Mean final error: "
        f"{final_errors.mean()*1000:.2f} mm"
    )

    print(
        f"  Median final error: "
        f"{np.median(final_errors)*1000:.2f} mm"
    )

    print(
        f"  Mean MPC steps: "
        f"{steps.mean():.2f}"
    )

    print(
        f"  Mean planning time: "
        f"{planning_times.mean()*1000:.2f} ms"
    )


# ==========================================================
# Plot 1:
# Final Error
# ==========================================================

method_names = list(
    methods.keys()
)


final_error_data = []


for method_name in method_names:

    values = [
        r["final_distance"]
        *
        1000.0

        for r in results

        if r["method"]
        ==
        method_name
    ]


    final_error_data.append(
        values
    )


plt.figure()


plt.boxplot(
    final_error_data,
    labels=method_names
)


plt.axhline(
    GOAL_THRESHOLD
    *
    1000.0,
    linestyle="--",
    label="Success Threshold"
)


plt.ylabel(
    "Final EE Error (mm)"
)

plt.title(
    "World Model MPC Benchmark"
)

plt.legend()

plt.grid()


plt.savefig(
    "results/"
    "benchmark_final_error.png",
    dpi=200,
    bbox_inches="tight"
)


# ==========================================================
# Plot 2:
# Success Rate
# ==========================================================

success_rates = [
    summary[
        name
    ][
        "success_rate"
    ]
    *
    100.0

    for name in method_names
]


plt.figure()


plt.bar(
    method_names,
    success_rates
)


plt.ylabel(
    "Success Rate (%)"
)

plt.ylim(
    0,
    105
)

plt.title(
    "Held-out Goal Success Rate"
)

plt.grid(
    axis="y"
)


plt.savefig(
    "results/"
    "benchmark_success_rate.png",
    dpi=200,
    bbox_inches="tight"
)


# ==========================================================
# Plot 3:
# Planning Time
# ==========================================================

planning_times_ms = [
    summary[
        name
    ][
        "mean_planning_time"
    ]
    *
    1000.0

    for name in method_names
]


plt.figure()


plt.bar(
    method_names,
    planning_times_ms
)


plt.ylabel(
    "Mean Planning Time (ms)"
)

plt.title(
    "Planner Runtime"
)

plt.grid(
    axis="y"
)


plt.savefig(
    "results/"
    "benchmark_planning_time.png",
    dpi=200,
    bbox_inches="tight"
)


print(
    "\nPlots saved in results/"
)


plt.show()