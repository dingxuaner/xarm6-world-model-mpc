import os
import time

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)


# ==========================================================
# Config
# ==========================================================

NUM_JOINTS = 6

NUM_EPISODES = 30

STEPS_PER_EPISODE = 60


# 每一步最大随机关节变化
#
# rad
ACTION_LIMIT = 0.035


# 每个episode允许机械臂
# 在初始姿态附近活动的范围
#
# rad
LOCAL_RANGE = 0.45


# controller完成一次动作的时间
MOVE_TIME = 0.30


# 命令发出以后等待多久再读取最终状态
SETTLE_TIME = 0.40


DATASET_PATH = os.path.expanduser(
    "~/embodied_projects/"
    "xarm_world_model/"
    "data/"
    "xarm_random_dataset.npz"
)


JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]


# ==========================================================
# Dataset Collector
# ==========================================================

class DatasetCollector(Node):

    def __init__(self):

        super().__init__(
            "xarm_dataset_collector"
        )

        # --------------------------------------------------
        # Joint State
        # --------------------------------------------------

        self.joint_sub = (
            self.create_subscription(
                JointState,
                "/joint_states",
                self.joint_state_callback,
                20,
            )
        )

        # --------------------------------------------------
        # Trajectory Publisher
        # --------------------------------------------------

        self.traj_pub = (
            self.create_publisher(
                JointTrajectory,
                "/xarm6_traj_controller/"
                "joint_trajectory",
                10,
            )
        )

        self.latest_q = None

        self.latest_dq = None

        self.last_q = None

        self.last_time = None

        self.get_logger().info(
            "Dataset collector started."
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

            joint_map[name] = i

        # 必须保证6个joint都存在
        for joint_name in JOINT_NAMES:

            if joint_name not in joint_map:

                return

        q = np.array(
            [
                msg.position[
                    joint_map[name]
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.float32,
        )

        # --------------------------------------------------
        # 优先使用JointState里的velocity
        # --------------------------------------------------

        velocity_available = (
            len(msg.velocity)
            ==
            len(msg.name)
        )

        if velocity_available:

            dq = np.array(
                [
                    msg.velocity[
                        joint_map[name]
                    ]
                    for name in JOINT_NAMES
                ],
                dtype=np.float32,
            )

        else:

            # ----------------------------------------------
            # 如果仿真没有velocity，
            # 用q差分估计
            # ----------------------------------------------

            now = time.time()

            if (
                self.last_q is None
                or
                self.last_time is None
            ):

                dq = np.zeros(
                    NUM_JOINTS,
                    dtype=np.float32,
                )

            else:

                dt = (
                    now
                    -
                    self.last_time
                )

                if dt > 1e-4:

                    dq = (
                        q
                        -
                        self.last_q
                    ) / dt

                else:

                    dq = np.zeros(
                        NUM_JOINTS,
                        dtype=np.float32,
                    )

            self.last_q = q.copy()

            self.last_time = now

        self.latest_q = q

        self.latest_dq = dq


    # ======================================================
    # Wait for Joint State
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
    # Read Current State
    #
    # 第一版：
    #
    # state = [q, dq]
    #
    # 12D
    # ======================================================

    def get_state(
        self
    ):

        q = self.latest_q.copy()

        dq = self.latest_dq.copy()

        state = np.concatenate(
            [
                q,
                dq
            ]
        ).astype(
            np.float32
        )

        return state


    # ======================================================
    # Publish Joint Target
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
            * 1e9
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
    # Wait while ROS callbacks continue
    # ======================================================

    def ros_wait(
        self,
        duration
    ):

        start = time.time()

        while (
            rclpy.ok()
            and
            time.time() - start < duration
        ):

            rclpy.spin_once(
                self,
                timeout_sec=0.02
            )


    # ======================================================
    # Move and Wait
    # ======================================================

    def move_and_wait(
        self,
        target_q
    ):

        self.send_joint_target(
            target_q
        )

        self.ros_wait(
            SETTLE_TIME
        )


    # ======================================================
    # Collect Dataset
    # ======================================================

    def collect(
        self
    ):

        self.wait_for_joint_state()

        # --------------------------------------------------
        # 初始姿态
        #
        # 后面的随机探索都限制在
        # 这个姿态附近
        # --------------------------------------------------

        q_center = (
            self.latest_q.copy()
        )

        q_lower = (
            q_center
            -
            LOCAL_RANGE
        )

        q_upper = (
            q_center
            +
            LOCAL_RANGE
        )


        print(
            "\nInitial q_center:"
        )

        print(
            q_center
        )


        all_states = []

        all_actions = []


        # ==================================================
        # Episode Loop
        # ==================================================

        for episode in range(
            NUM_EPISODES
        ):

            print(
                f"\n========== "
                f"Episode "
                f"{episode + 1}/"
                f"{NUM_EPISODES} "
                f"=========="
            )

            # ----------------------------------------------
            # 每个episode重新回到center
            # ----------------------------------------------

            self.move_and_wait(
                q_center
            )

            episode_states = []

            episode_actions = []


            # s_0
            current_state = (
                self.get_state()
            )

            episode_states.append(
                current_state
            )


            # ==============================================
            # Time Step Loop
            # ==============================================

            for step in range(
                STEPS_PER_EPISODE
            ):

                current_q = (
                    self.latest_q.copy()
                )

                # ------------------------------------------
                # Random Delta q
                # ------------------------------------------

                raw_action = (
                    np.random.uniform(
                        low=-ACTION_LIMIT,
                        high=ACTION_LIMIT,
                        size=NUM_JOINTS,
                    )
                    .astype(
                        np.float32
                    )
                )

                target_q = (
                    current_q
                    +
                    raw_action
                )

                # ------------------------------------------
                # 保证不离初始姿态太远
                # ------------------------------------------

                target_q = np.clip(
                    target_q,
                    q_lower,
                    q_upper
                )

                # ------------------------------------------
                # 实际执行的action
                #
                # 注意：
                # clip后不能再直接使用raw_action
                # ------------------------------------------

                actual_action = (
                    target_q
                    -
                    current_q
                ).astype(
                    np.float32
                )

                # ------------------------------------------
                # Execute
                # ------------------------------------------

                self.move_and_wait(
                    target_q
                )

                next_state = (
                    self.get_state()
                )

                episode_actions.append(
                    actual_action
                )

                episode_states.append(
                    next_state
                )


                if step % 10 == 0:

                    print(
                        f"step "
                        f"{step:02d} | "
                        f"|action|="
                        f"{np.linalg.norm(actual_action):.4f}"
                    )


            # ==============================================
            # Episode finished
            # ==============================================

            episode_states = (
                np.stack(
                    episode_states,
                    axis=0
                )
            )

            episode_actions = (
                np.stack(
                    episode_actions,
                    axis=0
                )
            )

            print(
                "episode states:",
                episode_states.shape
            )

            print(
                "episode actions:",
                episode_actions.shape
            )


            all_states.append(
                episode_states
            )

            all_actions.append(
                episode_actions
            )


        # ==================================================
        # Stack all episodes
        # ==================================================

        all_states = np.stack(
            all_states,
            axis=0
        )

        all_actions = np.stack(
            all_actions,
            axis=0
        )


        print(
            "\n=============================="
        )

        print(
            "Dataset collection finished."
        )

        print(
            "states:",
            all_states.shape
        )

        print(
            "actions:",
            all_actions.shape
        )


        # ==================================================
        # Save
        # ==================================================

        os.makedirs(
            os.path.dirname(
                DATASET_PATH
            ),
            exist_ok=True
        )


        np.savez_compressed(
            DATASET_PATH,

            states=all_states,

            actions=all_actions,

            joint_names=np.array(
                JOINT_NAMES
            ),

            action_limit=np.array(
                ACTION_LIMIT,
                dtype=np.float32
            ),

            local_range=np.array(
                LOCAL_RANGE,
                dtype=np.float32
            ),

            move_time=np.array(
                MOVE_TIME,
                dtype=np.float32
            ),
        )


        print(
            "\nSaved:"
        )

        print(
            DATASET_PATH
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
        DatasetCollector()
    )

    try:

        node.collect()

    except KeyboardInterrupt:

        print(
            "\nCollection interrupted."
        )

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()