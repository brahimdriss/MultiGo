import numpy as np

from MultiGo.src.goboard_fast import Board
from MultiGo.src.goboard_fast import Move
from MultiGo.src.gotypes import Player, Point


class ZeroEncoder:
    def __init__(self, board_size):
        self.board_size = board_size
        # 0 - 3   - black previous and current stones
        # 4 - 7   - white previous and current stones
        # 8 - 11  - red previous and current stones
        # 12      - black to play
        # 13 	  - white to play
        # 14 	  - red to play
        # self.num_planes = 15
        self.num_planes = 6

    def board_to_planes(self, board):
        black = np.zeros((self.board_size, self.board_size))
        white = np.zeros((self.board_size, self.board_size))
        red = np.zeros((self.board_size, self.board_size))

        planes = [black, white, red]
        colors = [Player.black, Player.white, Player.red]

        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(row=r + 1, col=c + 1)
                go_string = board.get_go_string(p)
                if go_string is not None:
                    idx = colors.index(go_string.color)
                    planes[idx][r][c] = 1
        return planes

    def board_to_planes_vect(self, board):

        colors = [Player.black, Player.white, Player.red]

        def get_point_color(x):
            p = Point(row=x[0] + 1, col=x[1] + 1)
            go_string = board.get_go_string(p)
            if go_string is not None:
                idx = colors.index(go_string.color)
                return idx
            return -1

        def get_stones_from_indices(x):
            return np.array(list(map(get_point_color, x)))

        planes = np.zeros((3, 5, 5))
        grid = np.indices((5, 5)).transpose((1, 2, 0))
        grid_vect = np.array(list(map(get_stones_from_indices, grid)))
        planes[0] = grid_vect == 0
        planes[1] = grid_vect == 1
        planes[2] = grid_vect == 2

        return planes

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player

        empty = Board(self.board_size, self.board_size)

        colors = [Player.black, Player.white, Player.red]
        pos = colors.index(next_player)

        if game_state.next_player == Player.white:
            # board_tensor[13] = 1
            board_tensor[3] = 1
        elif game_state.next_player == Player.red:
            board_tensor[4] = 1
        else:
            board_tensor[5] = 1

        # for idx in range(1):
        idx = 0
        black, white, red = self.board_to_planes(game_state.board)

        # board_tensor[0+idx] = black
        board_tensor[0] = black
        board_tensor[1] = white
        board_tensor[2] = red
        # board_tensor[4+idx] = white
        # board_tensor[8+idx] = red

        return board_tensor

    def encode_vect(self, game_state):

        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
        empty = Board(self.board_size, self.board_size)

        colors = [Player.black, Player.white, Player.red]
        pos = colors.index(next_player)

        if game_state.next_player == Player.white:
            board_tensor[13] = 1
        elif game_state.next_player == Player.red:
            board_tensor[14] = 1
        else:
            board_tensor[12] = 1

        for idx in range(1):
            black, white, red = self.board_to_planes_vect(game_state.board)

            board_tensor[0 + idx] = black
            board_tensor[4 + idx] = white
            board_tensor[8 + idx] = red

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
        # return self.board_size * self.board_size + 1
        return self.board_size * self.board_size

    def shape(self):
        return self.num_planes, self.board_size, self.board_size
