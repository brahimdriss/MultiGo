import argparse
import sys
import multiprocessing
import numpy as np
import os
import random
import time

sys.path.append("../")

from MultiGo.src import goboard_fast as goboard
from MultiGo.src.mcts import mcts
from MultiGo.src.utils import print_board, print_move
from MultiGo.src import agent
from MultiGo.src import gotypes
from MultiGo.src import scoring
from MultiGo.src.gotypes import Player



def generate_game(board_size, rounds, max_moves, temperature):
    boards, moves = [], []
    game = goboard.GameState.new_game(board_size)
    bot = mcts.MCTSAgent(rounds, temperature)

    while not game.is_over():
        move = bot.select_move(game)
        game = game.apply_move(move)
        
        #print_board(game.board)
        
    #print(game.winner())
    winner = game.winner()
    game_result = scoring.compute_game_result(game)
    result = game_result.final_scores
    print(game_result)
    print(result)
    return winner, result


def play_games(args):
    board_size, rounds, num_games, max_moves, temperature = args

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    black_score, white_score, red_score = 0, 0, 0
    
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        winner, result = generate_game(board_size, rounds, max_moves, temperature)
        
        if winner == Player.black:
            print('Black wins')
        elif winner == Player.white :
            print('White wins')
        else:
            print('Red wins')
        
        black_score += result[0]
        white_score += result[1]
        red_score += result[2]

    return black_score, white_score, red_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", "-b", type=int, default=5)
    parser.add_argument("--rounds", "-r", type=int, default=25)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument(
        "--max-moves", "-m", type=int, default=60, help="Max moves per game."
    )
    parser.add_argument("--num-games", "-n", type=int, default=10)
    parser.add_argument('--num-workers', '-w', type=int, default=2)
    parser.add_argument("--board-out")
    parser.add_argument("--move-out")

    args = parser.parse_args()

    games_per_worker = args.num_games // args.num_workers
    pool = multiprocessing.Pool(args.num_workers)
    worker_args = [
        (args.board_size, args.rounds, args.num_games, args.max_moves, args.temperature)
        for _ in range(args.num_workers)
    ]
    game_results = pool.map(play_games, worker_args)

    total_black_scores, total_white_scores, total_red_scores = 0, 0, 0
    for bscore, wscore, rscore in game_results:
        total_black_scores += bscore
        total_white_scores += wscore
        total_red_scores += rscore

    print('FINAL RESULTS:')
    print('Black : %d' % total_black_scores)
    print('White : %d' % total_white_scores)
    print('Red : %d' % total_red_scores)
    print("______")
    print('Black : %d' % total_black_scores/(args.num_games*args.num_workers))
    print('White : %d' % total_white_scores/(args.num_games*args.num_workers))
    print('Red : %d' % total_red_scores/(args.num_games*args.num_workers))

if __name__ == "__main__":
    main()
