import argparse
import numpy as np
import copy
import sys

sys.path.append("../")

from MultiGo.src import goboard_fast as goboard
from MultiGo.src.mcts import scoreuct
from MultiGo.src.utils import print_board, print_move
from MultiGo.src import agent
from MultiGo.src import gotypes
from MultiGo.src.gotypes import Player
from MultiGo.src import zero

def generate_game(board_size, rounds, max_moves, temperature):
    boards, moves = [], []

    game = goboard.GameState.new_game(board_size)

    bot = scoreuct.MCTS_score_Agent(rounds, temperature)
    naive = agent.naive.RandomBot()
    encoder = zero.ZeroEncoder(board_size)
    num_moves = 0

    while not game.is_over():
        
        if game.next_player == gotypes.Player.black:
            move = bot.select_move(game)
        else:
            move = naive.select_move(game)

        print("#################  Board  #################")
        print_board(game.board)
        print("#################  Encoder  #################")
        encoded = encoder.encode(game)
        for idx, plane in enumerate(encoded):
            print(f"Plane {idx}")
            print(plane)
        
        print("\n")
        print("\n")
        print("\n")
        
        game = game.apply_move(move)
        game.add_history(game.board)
        num_moves += 1
        if num_moves > max_moves:
            break
    
    return game.winner()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", "-b", type=int, default=5)
    parser.add_argument("--rounds", "-r", type=int, default=500)
    parser.add_argument("--temperature", "-t", type=float, default=1.2)
    parser.add_argument(
        "--max-moves", "-m", type=int, default=6, help="Max moves per game."
    )
    parser.add_argument("--num-games", "-n", type=int, default=1)
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

    print(scores)


if __name__ == "__main__":
    main()
