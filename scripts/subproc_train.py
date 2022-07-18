import sys

sys.path.append("../")

import argparse
import h5py
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from MultiGo.src import scoring
from MultiGo.src import zero
from MultiGo.src.goboard_fast import GameState, Player, Point
from MultiGo.src import kerasutil
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_model():

    board_size = 5
    filters = 128
    encoder = zero.ZeroEncoder(board_size)

    board_input = Input(shape=encoder.shape(), name="board_input")
    x = Conv2D(
        filters,
        (3, 3),
        padding="same",
        data_format="channels_first",
        activation="relu",
        kernel_regularizer=regularizers.l2(0.0001),
    )(board_input)

    for i in range(8):
        x1 = layers.Conv2D(
            filters,
            3,
            activation="relu",
            padding="same",
            data_format="channels_first",
            kernel_regularizer=regularizers.l2(0.0001),
        )(x)
        x1 = layers.Conv2D(
            filters,
            3,
            padding="same",
            data_format="channels_first",
            kernel_regularizer=regularizers.l2(0.0001),
        )(x1)
        x = layers.add([x1, x])
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)

    policy_conv = Conv2D(
        1,
        (1, 1),
        data_format="channels_first",
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(0.0001),
    )(x)
    policy_conv2 = Conv2D(
        1,
        (1, 1),
        data_format="channels_first",
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(0.0001),
    )(x)

    policy_flat = Flatten(data_format="channels_first")(policy_conv)
    policy_flat2 = Flatten(data_format="channels_first")(policy_conv2)

    pass_layer = Dense(1)(policy_flat2)
    policy = layers.concatenate([policy_flat, pass_layer])
    policy_output = layers.Activation("softmax", name="policy")(policy)

    value_flat = layers.GlobalAveragePooling2D(data_format="channels_first")(x)

    value_hidden1 = Dense(
        128, activation="relu", kernel_regularizer=regularizers.l2(0.0001)
    )(value_flat)
    value_hidden2 = Dense(
        128, activation="relu", kernel_regularizer=regularizers.l2(0.0001)
    )(value_flat)
    value_hidden3 = Dense(
        128, activation="relu", kernel_regularizer=regularizers.l2(0.0001)
    )(value_flat)

    value_output1 = Dense(
        26,
        name="value1",
        activation="softmax",
        kernel_regularizer=regularizers.l2(0.0001),
    )(value_hidden1)
    value_output2 = Dense(
        26,
        name="value2",
        activation="softmax",
        kernel_regularizer=regularizers.l2(0.0001),
    )(value_hidden2)
    value_output3 = Dense(
        26,
        name="value3",
        activation="softmax",
        kernel_regularizer=regularizers.l2(0.0001),
    )(value_hidden3)

    model = Model(
        inputs=[board_input],
        outputs=[policy_output, value_output1, value_output2, value_output3],
    )

    return model


def main():

    gpu_frac = 0.9 / 10
    kerasutil.set_gpu_memory_target(gpu_frac)
    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = "1"  # Select gpu that will be used, can cause error if it's full

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--exp1", "-x1", required=True)
    parser.add_argument("--exp2", "-x2", required=True)
    parser.add_argument("--out", "-o", required=True)

    args = parser.parse_args()

    board_size = 5
    filters = 64
    encoder = zero.ZeroEncoder(board_size)

    if args.exp2 != "None":
        model = tf.keras.models.load_model(args.model)
    else:
        model_name = "nn/az/base_nn_5p.h5"
        model = tf.keras.models.load_model(model_name)

    # filename = "buffer/1k_nn_1.h5"
    filename = args.exp1
    exp = zero.load_experience(h5py.File(filename))

    # filename2 = "buffer/1k_nn_0.h5"
    if args.exp2 != "None":
        filename2 = args.exp2
        exp2 = zero.load_experience(h5py.File(filename))
        exp = zero.combine_experience([exp, exp2])

    black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
    black_agent.train(exp, 0.0001, 64, 1, softmax=True)

    model_out = args.out
    black_agent.model.save(model_out)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
