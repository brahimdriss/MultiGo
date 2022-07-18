import sys

sys.path.append("../")

import h5py
import numpy as np

from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from MultiGo.src import scoring
from MultiGo.src import zero
from MultiGo.src.goboard_fast import GameState, Player, Point

board_size = 5
encoder = zero.ZeroEncoder(board_size)

board_input = Input(shape=encoder.shape(), name="board_input")
pb = board_input
for i in range(4):
    pb = Conv2D(
        64, (3, 3), padding="same", data_format="channels_first", activation="relu"
    )(pb)

policy_conv = Conv2D(2, (1, 1), data_format="channels_first", activation="relu")(pb)
policy_flat = Flatten()(policy_conv)
policy_output = Dense(encoder.num_moves(), activation="softmax", name="policy")(
    policy_flat
)

value_conv = Conv2D(1, (1, 1), data_format="channels_first", activation="relu")(pb)
value_flat = Flatten()(value_conv)
value_hidden = Dense(256, activation="relu")(value_flat)
value_output = Dense(3, activation="linear", name="value")(value_hidden)

model = Model(inputs=[board_input], outputs=[policy_output, value_output])

filename = "t_400rounds_400games.h5"


black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
exp = zero.load_experience(h5py.File(filename))
black_agent.train(exp, 0.01, 256, 2)
