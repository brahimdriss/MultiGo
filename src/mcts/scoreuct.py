import math
import random

from MultiGo.src import agent
from MultiGo.src.gotypes import Player
from MultiGo.src import scoring

# from MultiGo.src.utils import coords_from_point

__all__ = [
    "MCTS_score_Agent",
]


class MCTS_score_Node(object):
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
            Player.red: 0,
        }
        self.score = [0,0,0]
        self.num_rollouts = 0
        self.children = []
        self.univisted_moves = game_state.legal_moves()

    def add_random_child(self):
        index = random.randint(0, len(self.univisted_moves) - 1)
        new_move = self.univisted_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTS_score_Node(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, scores):
        self.score[0] += scores[0]
        self.score[1] += scores[1]
        self.score[2] += scores[2]
        self.num_rollouts += 1

    def can_add_child(self):
        return len(self.univisted_moves) > 0

    def is_terminal(self):
        return self.game_state.is_over()

    def score_frac(self, player):
        return float(self.score[player.value-1]) / float(self.num_rollouts)


class MCTS_score_Agent(agent.Agent):
    def __init__(self, num_rounds, temperature):
        agent.Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        root = MCTS_score_Node(game_state)

        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            if node.can_add_child():
                node = node.add_random_child()

            result = self.simulate_random_game(node.game_state)

            while node is not None:
                node.record_win(result)
                node = node.parent

        scored_moves = [
            (child.move, child.num_rollouts)
            for child in root.children
        ]
        scored_moves.sort(key=lambda x: x[1], reverse=True)        
        # for child in root.children:
        # #for idx, child in enumerate(root.children):
        #     #print(f"Child {idx}, N = {child.num_rollouts}, score_frac = {child.score_frac(game_state.next_player)}")
        #     child_pct = child.score_frac(game_state.next_player)
        #     if child_pct > best_pct:
        #         best_pct = child_pct
        #         best_move = child.move
        #         best_child = child
        # print('Select move %s with win pct %.3f' % (best_move, best_pct))
        best_move = scored_moves[0][0]
        return best_move

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        for child in node.children:
            win_percentage = child.score_frac(node.game_state.next_player)
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    @staticmethod
    def simulate_random_game(game):
        bots = {
            Player.black: agent.RandomBot(),
            Player.white: agent.RandomBot(),
            Player.red: agent.RandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        
        game_result = scoring.compute_game_result(game)
        result = game_result.final_scores
        return result
