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
from MultiGo.src import descent
from MultiGo.src.goboard_fast import GameState, Player, Point
from MultiGo.src import kerasutil
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():

    gpu_frac = 0.9 / 10
    kerasutil.set_gpu_memory_target(gpu_frac)
    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = "1"  # Select gpu that will be used, can cause error if it's full

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--exp1", "-x1", required=True)
    parser.add_argument("--out", "-o", required=True)
    parser.add_argument("--iter", "-i", required=True)

    args = parser.parse_args()

    board_size = 5
    rounds = 180
    encoder = zero.ZeroEncoder(board_size)

    model = tf.keras.models.load_model(args.model)

    filename = args.exp1
    exp = descent.load_experience(h5py.File(filename))

    black_agent = descent.Descent_Agent(model, encoder, rounds)
    black_agent.train(exp, 0.0001, 64, 1)

    model_out = args.out
    black_agent.model.save(model_out)

    if (int(args.iter) % 5) != 0:
        os.remove(filename)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
