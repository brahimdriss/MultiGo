import argparse
import numpy as np

import sys

sys.path.append("../")

from MultiGo.src import goboard_fast as goboard
from MultiGo.src.mcts import mcts
from MultiGo.src.utils import print_board, print_move
from MultiGo.src import agent
from MultiGo.src import gotypes
from MultiGo.src.gotypes import Player
from MultiGo.src import scoring
import time



def generate_game(board_size, rounds, max_moves, temperature):
    boards, moves = [], []

    game = goboard.GameState.new_game(board_size)

    bot = mcts.MCTSAgent(rounds, temperature)
    naive = agent.naive.RandomBot()

    num_moves = 0

    while not game.is_over():

        #if game.next_player == gotypes.Player.black:
            #move = bot.select_move(game)
        #else:
            #move = naive.select_move(game)
        move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)
        print_board(game.board)
        num_moves += 1
        #if num_moves > max_moves:
            #break
    
    
    game_result = scoring.compute_game_result(game)
    print_board(game.board)
    b,w,r = scoring.compute_territory(game)
    print(f"Black territory = {b[0]},Black stones {b[1]}")
    print(f"White territory = {w[0]},White stones {w[1]}")
    print(f"Red territory = {r[0]},Red stones {r[1]}")
    #print(game_result.final_scores)
    print("#######################")
    print("\n")
    print("\n")
    print("\n")
    return game.winner()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", "-b", type=int, default=5)
    parser.add_argument("--rounds", "-r", type=int, default=25)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument(
        "--max-moves", "-m", type=int, default=60, help="Max moves per game."
    )
    parser.add_argument("--num-games", "-n", type=int, default=10)
    parser.add_argument("--board-out")
    parser.add_argument("--move-out")

    args = parser.parse_args()

    scores = [0, 0, 0]

    for i in range(args.num_games):
        print("Playing game %d/%d..." % (i + 1, args.num_games))
        winner = generate_game(
            args.board_size, args.rounds, args.max_moves, args.temperature
        )

        if winner == Player.black:
            scores[0] += 1
        elif winner == Player.white:
            scores[1] += 1
        else:
            scores[2] += 1

    #print(scores)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))