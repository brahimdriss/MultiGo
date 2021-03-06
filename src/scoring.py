from __future__ import absolute_import
from collections import namedtuple

from MultiGo.src.gotypes import Player, Point


class Territory(object):
    def __init__(self, territory_map):
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_red_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_red_stones = 0
        self.num_dame = 0
        self.dame_points = []
        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1
            elif status == Player.white:
                self.num_white_stones += 1
            elif status == Player.red:
                self.num_red_stones += 1
            elif status == "territory_b":
                self.num_black_territory += 1
            elif status == "territory_w":
                self.num_white_territory += 1
            elif status == "territory_r":
                self.num_red_territory += 1
            elif status == "dame":
                self.num_dame += 1
                self.dame_points.append(point)


# change with class GameResult(namedtuple('GameResult', 'b w r komi')):
# class GameResult(namedtuple('GameResult', 'b w komi')):
class GameResult(namedtuple("GameResult", "b w r komi")):
    @property
    def winner(self):
        scores = [
            (Player.black, self.b),
            (Player.white, self.w + self.komi),
            (Player.red, self.r + self.komi + (self.komi) / 2),
        ]
        return max(scores, key=lambda x: x[1])[0]

    @property
    def winning_margin(self):
        w = self.w + self.komi
        return abs(self.b - w)

    @property
    def final_scores(self):
        return [self.b, self.w, self.r]

    def __str__(self):
        return f"B {self.b} W {self.w} R {self.r}"


def _collect_region(start_pos, board, visited=None):
    if visited is None:
        visited = {}
    if start_pos in visited:
        return [], set()
    all_points = [start_pos]
    all_borders = set()
    visited[start_pos] = True
    here = board.get(start_pos)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for delta_r, delta_c in deltas:
        next_p = Point(row=start_pos.row + delta_r, col=start_pos.col + delta_c)
        if not board.is_on_grid(next_p):
            continue
        neighbor = board.get(next_p)
        if neighbor == here:
            points, borders = _collect_region(next_p, board, visited)
            all_points += points
            all_borders |= borders
        else:
            all_borders.add(neighbor)
    return all_points, all_borders


def evaluate_territory(board):
    status = {}
    for r in range(1, board.num_rows + 1):
        for c in range(1, board.num_cols + 1):
            p = Point(row=r, col=c)
            if p in status:
                continue
            stone = board.get(p)
            if stone is not None:
                status[p] = board.get(p)
            else:
                group, neighbors = _collect_region(p, board)
                if len(neighbors) == 1:
                    neighbor_stone = neighbors.pop()
                    if neighbor_stone == Player.black:
                        stone_str = "b"
                    elif neighbor_stone == Player.white:
                        stone_str = "w"
                    else:
                        stone_str = "r"
                    fill_with = "territory_" + stone_str
                else:
                    fill_with = "dame"
                for pos in group:
                    status[pos] = fill_with
    return Territory(status)


def compute_game_result(game_state):
    territory = evaluate_territory(game_state.board)
    b = (territory.num_black_territory, territory.num_black_stones)
    w = (territory.num_white_territory, territory.num_white_stones)
    r = (territory.num_red_territory, territory.num_red_stones)
    return GameResult(
        territory.num_black_territory + territory.num_black_stones,
        territory.num_white_territory + territory.num_white_stones,
        territory.num_red_territory + territory.num_red_stones,
        komi=0.5,
    )


def compute_territory(game_state):
    territory = evaluate_territory(game_state.board)
    b = (territory.num_black_territory, territory.num_black_stones)
    w = (territory.num_white_territory, territory.num_white_stones)
    r = (territory.num_red_territory, territory.num_red_stones)
    return b, w, r
