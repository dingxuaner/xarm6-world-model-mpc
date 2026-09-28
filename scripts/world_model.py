import torch
import torch.nn as nn


STATE_DIM = 9
ACTION_DIM = 6


class WorldModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                STATE_DIM + ACTION_DIM,
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

        delta_state = self.net(
            x
        )

        next_state = (
            state
            +
            delta_state
        )

        return next_state