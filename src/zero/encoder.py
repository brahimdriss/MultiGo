import numpy as np

from MultiGo.src.goboard_fast import Move
from MultiGo.src.gotypes import Player, Point


class ZeroEncoder:
    def __init__(self, board_size):
        self.board_size = board_size
        # 0 - 3. our stones with 1, 2, 3, 4+ liberties
        # 4 - 7. next opponent stones with 1, 2, 3, 4+ liberties
        # 8 - 11. last opponent stones with 1, 2, 3, 4+ liberties
        # 12. 1 if we get komi
        # 13. 1 if opponent gets komi
        # 14. 1 if second opponent gets komi
        # 15. move would be illegal due to ko
        self.num_planes = 11 + 4 + 1

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player

        colors = [Player.black, Player.white, Player.red]
        pos = colors.index(next_player)

        if game_state.next_player == Player.white:
            board_tensor[8 + 4] = 1
            board_tensor[9 + 4] = 1
        elif game_state.next_player == Player.red:
            board_tensor[8 + 4] = 1
            board_tensor[10 + 4] = 1
        else:
            board_tensor[9 + 4] = 1
            board_tensor[10 + 4] = 1

        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(next_player, Move.play(p)):
                        board_tensor[11 + 4][r][c] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    liberty_plane += ((pos + 2) % 3) * 4
                    board_tensor[liberty_plane][r][c] = 1

        return board_tensor

    def encode_move(self, move):
        if move.is_play:
            return self.board_size * (move.point.row - 1) + (move.point.col - 1)
        elif move.is_pass:
            return self.board_size * self.board_size
        raise ValueError("Cannot encode resign move")

    def decode_move_index(self, index):
        if index == self.board_size * self.board_size:
            return Move.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row=row + 1, col=col + 1))

    def num_moves(self):
        return self.board_size * self.board_size + 1

    def shape(self):
        return self.num_planes, self.board_size, self.board_size
