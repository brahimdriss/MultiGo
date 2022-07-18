import sys
import os

sys.path.append("../")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse

# import datetime
import multiprocessing
import random
import time
import tempfile
import subprocess
import gc
from collections import namedtuple

import h5py
import numpy as np
from contextlib import redirect_stdout
from MultiGo.src import agent
from MultiGo.src import kerasutil
from MultiGo.src import scoring
from MultiGo.src import zero
from MultiGo.src.goboard_fast import GameState, Player, Point
from MultiGo.src.mcts import scoreuct

# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class GameRecord(namedtuple("GameRecord", "moves result")):
    pass


def train(model_file, exp1_file, exp2_file, model_out):

    import tensorflow as tf

    gpu_frac = 0.9 / 9
    kerasutil.set_gpu_memory_target(gpu_frac)

    model = tf.keras.models.load_model(model_file)

    exp = zero.load_experience(h5py.File(exp1_file))

    exp2 = zero.load_experience(h5py.File(exp2_file))
    exp = zero.combine_experience([exp, exp2])

    black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
    black_agent.train(exp, 0.0001, 64, 1)

    model_out = "nn/az/" + model_out
    black_agent.model.save(model_out)


def simulate_game(black_player, white_player, red_player, board_size):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
        Player.red: red_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
        # game.add_history(game.board)

    game_result = scoring.compute_game_result(game)

    return GameRecord(
        moves=moves,
        result=game_result,
    )


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix="dlgo-train")
    os.close(fd)
    return fname


def do_self_play(
    board_size, agent_filename, num_games, experience_filename, gpu_frac, iteration_num
):

    import tensorflow as tf
    from tensorflow.keras import backend as K

    kerasutil.set_gpu_memory_target(gpu_frac)
    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    rounds = 180
    c = 0.8
    alpha = 0.03
    board_size = 5
    encoder = zero.ZeroEncoder(board_size)

    tf.compat.v1.disable_eager_execution()
    K.clear_session()
    model = tf.keras.models.load_model(agent_filename)

    if random.random() < max(0.5 - iteration_num / 100, 0.05):
        black_agent = scoreuct.MCTS_score_Agent(rounds, c)
        black_agent.set_encoder(encoder)
    else:
        black_agent = zero.ZeroAgent(
            model, encoder, rounds_per_move=rounds, c=c, alpha=alpha
        )

    if random.random() < max(0.5 - iteration_num / 100, 0.05):
        white_agent = scoreuct.MCTS_score_Agent(rounds, c)
        white_agent.set_encoder(encoder)
    else:
        white_agent = zero.ZeroAgent(
            model, encoder, rounds_per_move=rounds, c=c, alpha=alpha
        )

    if random.random() < max(0.5 - iteration_num / 100, 0.05):
        red_agent = scoreuct.MCTS_score_Agent(rounds, c)
        red_agent.set_encoder(encoder)
    else:
        red_agent = zero.ZeroAgent(
            model, encoder, rounds_per_move=rounds, c=c, alpha=alpha
        )

    c1 = zero.ZeroExperienceCollector()
    c2 = zero.ZeroExperienceCollector()
    c3 = zero.ZeroExperienceCollector()

    for i in range(num_games):
        if i % 5 == 0:
            print("Simulating game %d/%d..." % (i + 1, num_games), flush=True)

        if random.random() < 0.25:
            black_agent.use_policy = 1
            white_agent.use_policy = 1
            red_agent.use_policy = 1
        else:
            black_agent.use_policy = 0
            white_agent.use_policy = 0
            red_agent.use_policy = 0

        c1.begin_episode()
        black_agent.set_collector(c1)
        c2.begin_episode()
        white_agent.set_collector(c2)
        c3.begin_episode()
        red_agent.set_collector(c3)

        game_record = simulate_game(black_agent, white_agent, red_agent, board_size)

        c1.complete_episode(game_record.result.final_scores)
        c2.complete_episode(game_record.result.final_scores)
        c3.complete_episode(game_record.result.final_scores)

    experience = zero.combine_experience([c1, c2, c3])
    print("Saving experience buffer to %s\n" % experience_filename)
    with h5py.File(experience_filename, "w") as experience_outf:
        experience.serialize(experience_outf)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", "-n", type=int, default=1000)
    parser.add_argument("--num-workers", "-w", type=int, default=10)
    parser.add_argument("--board-size", "-b", type=int, default=5)

    args = parser.parse_args()

    for i in range(0, 100):
        print("Starting iteration", i)
        print(i)
        iteration = str(i)
        outfile = "logs/az/self_play/log_nn_iter_" + iteration + ".txt"

        with open(outfile, "w") as f:
            with redirect_stdout(f):

                start_buffer = time.time()
                experience_out = "data/buffer/1k_nn_" + iteration + ".h5"
                experience_files = []
                workers = []
                gpu_frac = 0.9 / 9
                games_per_worker = args.num_games // args.num_workers
                if i != 0:
                    learning_agent = "nn/az/iter_" + str(i - 1) + "_nn.h5"
                else:
                    learning_agent = "nn/az/base_nn_5p.h5"
                print("Starting workers...")
                for j in range(args.num_workers):
                    filename = get_temp_file()
                    experience_files.append(filename)
                    worker = multiprocessing.Process(
                        target=do_self_play,
                        args=(
                            args.board_size,
                            learning_agent,
                            games_per_worker,
                            filename,
                            gpu_frac,
                            i,
                        ),
                    )
                    worker.start()
                    workers.append(worker)

                print("Waiting for workers...")
                for worker in workers:
                    worker.join()

                print("Merging experience buffers...")
                other_filenames = experience_files[1:]
                combined_buffer = zero.load_experience(h5py.File(filename))
                for filename in other_filenames:
                    next_buffer = zero.load_experience(h5py.File(filename))
                    combined_buffer = zero.combine_experience(
                        [combined_buffer, next_buffer]
                    )
                print("Saving into %s..." % experience_out)
                with h5py.File(experience_out, "w") as experience_outf:
                    combined_buffer.serialize(experience_outf)

                for fname in experience_files:
                    os.unlink(fname)

                print("--- %s seconds ---" % (time.time() - start_buffer))

        outfile = "logs/az/training/iter_" + iteration + "_train.txt"
        with open(outfile, "w") as f:
            if i != 0:
                model_name = "nn/az/iter_" + str(i - 1) + "_nn.h5"
            else:
                model_name = "base"
            exp1_file = "data/buffer/1k_nn_" + iteration + ".h5"
            if i != 0:
                exp2_file = "data/buffer/1k_nn_" + str(i - 1) + ".h5"
            else:
                exp2_file = "None"
            model_out = "nn/az/iter_" + iteration + "_nn.h5"

            subprocess.call(
                [
                    "python",
                    "scripts/subproc_train.py",
                    "-m",
                    model_name,
                    "-x1",
                    exp1_file,
                    "-x2",
                    exp2_file,
                    "-o",
                    model_out,
                ],
                stdout=f,
            )

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
