import sys

sys.path.append("../")

from MultiGo.src import agent
from MultiGo.src import goboard_slow as goboard
from MultiGo.src import gotypes
from MultiGo.src import scoring
from MultiGo.src.gotypes import Player
from MultiGo.src.utils import print_board, print_move
import time

from MultiGo.src.agent import naive


def main():
    verbose = 1
    board_size = 5
    scores = [0, 0, 0]
    n_games = 1

    for i in range(n_games):
        print(f"Starting game {i}")
        game = goboard.GameState.new_game(board_size)
        bots = {
            gotypes.Player.black: agent.naive.RandomBot(),
            gotypes.Player.white: agent.naive.RandomBot(),
            gotypes.Player.red: agent.naive.RandomBot(),
        }
        while not game.is_over():
            if verbose:
                time.sleep(0.3)
                print(chr(27) + "[2J")
                print_board(game.board)
                bot_move = bots[game.next_player].select_move(game)
                print_move(game.next_player, bot_move)
                game = game.apply_move(bot_move)
            else:
                bot_move = bots[game.next_player].select_move(game)
                game = game.apply_move(bot_move)

        winner = game.winner()
        if winner == Player.black:
            scores[0] += 1
        elif winner == Player.white:
            scores[1] += 1
        else:
            scores[2] += 1

        if winner is None:
            print("It's a draw.")
        else:
            print("Winner: ", str(winner))


if __name__ == "__main__":
    main()
