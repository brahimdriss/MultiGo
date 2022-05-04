import sys
sys.path.append("../")

import argparse
# import datetime
import multiprocessing
import os
import random
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np

from MultiGo.src import agent
#from MultiGo.src import kerasutil
from MultiGo.src import scoring
from MultiGo.src import zero
from MultiGo.src.goboard_fast import GameState, Player, Point
from MultiGo.src.mcts import scoreuct



def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))




class GameRecord(namedtuple('GameRecord', 'moves result')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    elif player == Player.white:
        return 'W'
    return 'R'


def simulate_game(black_player, white_player, red_player, board_size):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
        Player.red: red_player
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    #print_board(game.board)
    game_result = scoring.compute_game_result(game)

    #return game_result
    return GameRecord(
        moves=moves,
        result=game_result,
    )


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train')
    os.close(fd)
    return fname


def do_self_play(board_size, agent_filename,
                 num_games, temperature,
                 experience_filename,
                 gpu_frac):
    
    #kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    rounds = 400
    temperature = 1.2
    board_size = 5
    encoder = zero.ZeroEncoder(board_size)


    black_agent = scoreuct.MCTS_score_Agent(rounds, temperature)
    white_agent = scoreuct.MCTS_score_Agent(rounds, temperature)
    red_agent = scoreuct.MCTS_score_Agent(rounds, temperature)
    black_agent.set_encoder(encoder)
    white_agent.set_encoder(encoder)
    red_agent.set_encoder(encoder)

    c1 = zero.ZeroExperienceCollector()
    c2 = zero.ZeroExperienceCollector()
    c3 = zero.ZeroExperienceCollector()

    #black_agent.set_collector(c1)
    #white_agent.set_collector(c2)
    #red_agent.set_collector(c3)

    #color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        c1.begin_episode()
        black_agent.set_collector(c1)
        c2.begin_episode()
        white_agent.set_collector(c2)
        c3.begin_episode()
        red_agent.set_collector(c3)

        # if color1 == Player.black:
        #     black_player, white_player = agent1, agent2
        # else:
        #     white_player, black_player = agent1, agent2

        game_record = simulate_game(black_agent, white_agent, red_agent, board_size)
        
        c1.complete_episode(game_record.result.final_scores)
        c2.complete_episode(game_record.result.final_scores)
        c3.complete_episode(game_record.result.final_scores)
    
    #color1 = color.other

    experience = zero.combine_experience([c1, c2, c3])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', required=False)
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--experience-out', '-o', required=True)
    parser.add_argument('--num-workers', '-w', type=int, default=1)
    parser.add_argument('--temperature', '-t', type=float, default=0.0)
    parser.add_argument('--board-size', '-b', type=int, default=5)

    args = parser.parse_args()

    experience_files = []
    workers = []
    gpu_frac = 0.95 / float(args.num_workers)
    games_per_worker = args.num_games // args.num_workers
    print('Starting workers...')
    for i in range(args.num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                args.board_size,
                args.learning_agent,
                games_per_worker,
                args.temperature,
                filename,
                gpu_frac,
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()

    # Merge experience buffers.
    print('Merging experience buffers...')
    # first_filename = experience_files[0]
    other_filenames = experience_files[1:]
    combined_buffer = zero.load_experience(h5py.File(filename))
    for filename in other_filenames:
        next_buffer = zero.load_experience(h5py.File(filename))
        combined_buffer = zero.combine_experience([combined_buffer, next_buffer])
    print('Saving into %s...' % args.experience_out)
    with h5py.File(args.experience_out, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


if __name__ == '__main__':
    main()
