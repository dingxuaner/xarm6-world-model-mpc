import os
import time

import numpy as np

import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)

from xarm_learning.manual_fk import (
    forward_kinematics
)


# ==========================================================
# Paths
# ==========================================================

MODEL_PATH = os.path.expanduser(
    "~/embodied_projects/"
    "xarm_world_model/"
    "models/"
    "multi_step_world_model.pt"
)

DATA_PATH = os.path.expanduser(
    "~/embodied_projects/"
    "xarm_world_model/"
    "data/"
    "xarm_world_model_15d.npz"
)


# ==========================================================
# Device
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# Dimensions
# ==========================================================

STATE_DIM = 9

ACTION_DIM = 6


JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]


# ==========================================================
# MPC / CEM Config
# ==========================================================

HORIZON = 15

NUM_SAMPLES = 1024

NUM_ELITES = 64

CEM_ITERATIONS = 5


ACTION_LIMIT = 0.035

INITIAL_STD = 0.025


# ==========================================================
# Robot Execution Config
# ==========================================================

MOVE_TIME = 0.30

SETTLE_TIME = 0.40


MAX_MPC_STEPS = 30


GOAL_THRESHOLD = 0.015


# ==========================================================
# World Model
#
# 必须和训练时结构完全一致
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
            self.net(x)
        )

        next_state = (
            state
            +
            delta_state
        )

        return next_state


# ==========================================================
# Main ROS Node
# ==========================================================

class WorldModelMPC(Node):

    def __init__(self):

        super().__init__(
            "world_model_mpc"
        )


        # ==================================================
        # ROS Subscriber
        # ==================================================

        self.joint_sub = (
            self.create_subscription(
                JointState,
                "/joint_states",
                self.joint_state_callback,
                20
            )
        )


        # ==================================================
        # ROS Publisher
        # ==================================================

        self.traj_pub = (
            self.create_publisher(
                JointTrajectory,
                "/xarm6_traj_controller/"
                "joint_trajectory",
                10
            )
        )


        self.latest_q = None


        # ==================================================
        # Load World Model
        # ==================================================

        self.get_logger().info(
            f"Loading model from: "
            f"{MODEL_PATH}"
        )


        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )


        self.model = (
            WorldModel()
            .to(DEVICE)
        )


        self.model.load_state_dict(
            checkpoint["model"]
        )


        self.model.eval()


        # ==================================================
        # Normalization Statistics
        # ==================================================

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


        # ==================================================
        # Load Dataset
        #
        # 这里有两个用途：
        #
        # 1. 取一个dataset中的目标
        # 2. 得到训练数据覆盖的joint范围
        # ==================================================

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


        self.q_min = q_dataset.min(
            axis=(0, 1)
        )


        self.q_max = q_dataset.max(
            axis=(0, 1)
        )


        # ==================================================
        # Goal
        #
        # 与刚才offline测试使用同一个目标
        # ==================================================

        GOAL_EPISODE = 24

        GOAL_TIME = 25


        self.goal_xyz = states_15d[
            GOAL_EPISODE,
            GOAL_TIME,
            12:15
        ].copy()


        self.goal_xyz_t = torch.tensor(
            self.goal_xyz,
            dtype=torch.float32,
            device=DEVICE
        )


        self.get_logger().info(
            f"Device: {DEVICE}"
        )


        self.get_logger().info(
            f"Goal xyz: "
            f"{self.goal_xyz}"
        )


        self.get_logger().info(
            f"Dataset q min: "
            f"{self.q_min}"
        )


        self.get_logger().info(
            f"Dataset q max: "
            f"{self.q_max}"
        )


    # ======================================================
    # Joint State Callback
    # ======================================================

    def joint_state_callback(
        self,
        msg
    ):

        joint_map = {}

        for i, name in enumerate(
            msg.name
        ):

            joint_map[
                name
            ] = i


        for name in JOINT_NAMES:

            if name not in joint_map:

                return


        q = np.array(
            [
                msg.position[
                    joint_map[name]
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.float32
        )


        self.latest_q = q


    # ======================================================
    # ROS Wait
    # ======================================================

    def ros_wait(
        self,
        duration
    ):

        start = time.time()

        while (
            rclpy.ok()
            and
            time.time() - start
            <
            duration
        ):

            rclpy.spin_once(
                self,
                timeout_sec=0.02
            )


    # ======================================================
    # Wait for first joint state
    # ======================================================

    def wait_for_joint_state(
        self
    ):

        self.get_logger().info(
            "Waiting for /joint_states ..."
        )


        while (
            rclpy.ok()
            and
            self.latest_q is None
        ):

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )


        self.get_logger().info(
            "Joint state received."
        )


    # ======================================================
    # FK
    # ======================================================

    def get_xyz(
        self,
        q
    ):

        T = forward_kinematics(
            q
        )


        xyz = T[
            :3,
            3
        ].astype(
            np.float32
        )


        return xyz


    # ======================================================
    # Current 9D State
    #
    # [q1..q6, x, y, z]
    # ======================================================

    def get_current_state(
        self
    ):

        q = (
            self.latest_q.copy()
        )


        xyz = self.get_xyz(
            q
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


    # ======================================================
    # Normalization
    # ======================================================

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


    # ======================================================
    # CEM Planner
    # ======================================================

    @torch.no_grad()
    def cem_plan(
        self,
        current_state
    ):

        # --------------------------------------------------
        # Gaussian distribution over
        # future action sequence
        #
        # [H, 6]
        # --------------------------------------------------

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


        # ==================================================
        # CEM optimization
        # ==================================================

        for iteration in range(
            CEM_ITERATIONS
        ):

            # ==============================================
            # 1.
            #
            # Sample N future action sequences
            #
            # [N,H,6]
            # ==============================================

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


            # ==============================================
            # 2.
            #
            # Every imagined trajectory
            # starts from same real state
            # ==============================================

            state = (
                current_state_t
                .unsqueeze(0)
                .repeat(
                    NUM_SAMPLES,
                    1
                )
            )


            state_norm = (
                self.normalize_state(
                    state
                )
            )


            scores = torch.zeros(
                NUM_SAMPLES,
                device=DEVICE
            )


            # ==============================================
            # 3.
            #
            # Imagination rollout
            # ==============================================

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
                    self.normalize_action(
                        action
                    )
                )


                state_norm = (
                    self.model(
                        state_norm,
                        action_norm
                    )
                )


                state_physical = (
                    self.denormalize_state(
                        state_norm
                    )
                )


                # ------------------------------------------
                # predicted q
                # ------------------------------------------

                predicted_q = (
                    state_physical[
                        :,
                        0:6
                    ]
                )


                # ------------------------------------------
                # predicted xyz
                # ------------------------------------------

                predicted_xyz = (
                    state_physical[
                        :,
                        6:9
                    ]
                )


                # ==========================================
                # Goal distance cost
                # ==========================================

                distance = torch.norm(
                    predicted_xyz
                    -
                    self.goal_xyz_t
                    .unsqueeze(0),
                    dim=-1
                )


                scores -= distance


                # ==========================================
                # Action regularization
                #
                # 不鼓励无意义的大动作
                # ==========================================

                action_cost = (
                    0.02
                    *
                    torch.sum(
                        action ** 2,
                        dim=-1
                    )
                )


                scores -= action_cost


                # ==========================================
                # OOD penalty
                #
                # 如果predicted q超出dataset覆盖范围
                # 就罚它
                # ==========================================

                q_min_t = torch.tensor(
                    self.q_min,
                    dtype=torch.float32,
                    device=DEVICE
                )


                q_max_t = torch.tensor(
                    self.q_max,
                    dtype=torch.float32,
                    device=DEVICE
                )


                below = torch.relu(
                    q_min_t
                    -
                    predicted_q
                )


                above = torch.relu(
                    predicted_q
                    -
                    q_max_t
                )


                ood_cost = (
                    torch.sum(
                        below ** 2
                        +
                        above ** 2,
                        dim=-1
                    )
                )


                scores -= (
                    100.0
                    *
                    ood_cost
                )


            # ==============================================
            # 4.
            #
            # Select elite action sequences
            # ==============================================

            elite_indices = torch.topk(
                scores,
                NUM_ELITES
            ).indices


            elites = (
                action_sequences[
                    elite_indices
                ]
            )


            # ==============================================
            # 5.
            #
            # Update CEM distribution
            # ==============================================

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


        # ==================================================
        # MPC:
        #
        # only execute first action
        # ==================================================

        first_action = (
            mean[
                0
            ]
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )


        return first_action


    # ======================================================
    # Send Joint Target
    # ======================================================

    def send_joint_target(
        self,
        target_q
    ):

        msg = JointTrajectory()


        msg.joint_names = (
            JOINT_NAMES
        )


        point = (
            JointTrajectoryPoint()
        )


        point.positions = (
            target_q
            .astype(float)
            .tolist()
        )


        sec = int(
            MOVE_TIME
        )


        nanosec = int(
            (
                MOVE_TIME
                -
                sec
            )
            *
            1e9
        )


        point.time_from_start.sec = (
            sec
        )


        point.time_from_start.nanosec = (
            nanosec
        )


        msg.points = [
            point
        ]


        self.traj_pub.publish(
            msg
        )


    # ======================================================
    # Main MPC Loop
    # ======================================================

    def run_mpc(
        self
    ):

        self.wait_for_joint_state()


        print(
            "\n=============================="
        )

        print(
            "Starting World Model MPC"
        )

        print(
            "=============================="
        )


        for step in range(
            MAX_MPC_STEPS
        ):

            # ==============================================
            # 1. Observe REAL robot state
            # ==============================================

            current_state = (
                self.get_current_state()
            )


            current_q = (
                current_state[
                    0:6
                ]
            )


            current_xyz = (
                current_state[
                    6:9
                ]
            )


            distance_before = (
                np.linalg.norm(
                    current_xyz
                    -
                    self.goal_xyz
                )
            )


            print(
                f"\nMPC Step "
                f"{step:02d}"
            )


            print(
                "Current xyz:",
                current_xyz
            )


            print(
                "Distance before:",
                distance_before
            )


            # ==============================================
            # Goal already reached
            # ==============================================

            if (
                distance_before
                <
                GOAL_THRESHOLD
            ):

                print(
                    "\nGoal reached!"
                )

                break


            # ==============================================
            # 2.
            #
            # Plan completely inside
            # learned World Model
            # ==============================================

            plan_start = time.time()


            action = self.cem_plan(
                current_state
            )


            planning_time = (
                time.time()
                -
                plan_start
            )


            # ==============================================
            # 3.
            #
            # Convert delta q
            # into target q
            # ==============================================

            target_q = (
                current_q
                +
                action
            )


            # ==============================================
            # Safety:
            #
            # stay inside dataset support
            # ==============================================

            target_q = np.clip(
                target_q,
                self.q_min,
                self.q_max
            ).astype(
                np.float32
            )


            actual_command = (
                target_q
                -
                current_q
            )


            print(
                "Planned action:",
                action
            )


            print(
                "|action|:",
                np.linalg.norm(
                    action
                )
            )


            print(
                "Planning time:",
                planning_time
            )


            # ==============================================
            # 4.
            #
            # Execute on ROS controller
            # ==============================================

            self.send_joint_target(
                target_q
            )


            # ==============================================
            # 5.
            #
            # Wait and receive REAL /joint_states
            # ==============================================

            self.ros_wait(
                SETTLE_TIME
            )


            # ==============================================
            # 6.
            #
            # Observe again
            # ==============================================

            new_state = (
                self.get_current_state()
            )


            new_xyz = (
                new_state[
                    6:9
                ]
            )


            distance_after = (
                np.linalg.norm(
                    new_xyz
                    -
                    self.goal_xyz
                )
            )


            print(
                "Executed command:",
                actual_command
            )


            print(
                "New xyz:",
                new_xyz
            )


            print(
                "Distance after:",
                distance_after
            )


        # ==================================================
        # Final
        # ==================================================

        final_state = (
            self.get_current_state()
        )


        final_xyz = (
            final_state[
                6:9
            ]
        )


        final_distance = (
            np.linalg.norm(
                final_xyz
                -
                self.goal_xyz
            )
        )


        print(
            "\n=============================="
        )

        print(
            "MPC Finished"
        )

        print(
            "=============================="
        )


        print(
            "Final xyz:"
        )

        print(
            final_xyz
        )


        print(
            "Goal xyz:"
        )

        print(
            self.goal_xyz
        )


        print(
            "Final distance:"
        )

        print(
            final_distance
        )


# ==========================================================
# Main
# ==========================================================

def main(
    args=None
):

    rclpy.init(
        args=args
    )


    node = (
        WorldModelMPC()
    )


    try:

        node.run_mpc()


    except KeyboardInterrupt:

        print(
            "\nMPC interrupted."
        )


    finally:

        node.destroy_node()


        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()