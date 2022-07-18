import sys
import os

sys.path.append("../")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from MultiGo.src.goboard_fast import Move
from MultiGo.src import scoring
import time
import random
from ..agent import Agent


__all__ = [
    "ZeroAgent",
]


class Branch:
    def __init__(self, prior, num_players=3):
        self.prior = prior
        self.visit_count = 0
        self.total_value = np.zeros(num_players)


class ZeroTreeNode:
    def __init__(
        self, state, value, priors, parent, last_move, num_players=3, alpha=0.3
    ):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        self.num_players = num_players
        self.num_moves = len(priors)
        # self.alpha = alpha
        self.alpha = 0
        self.noise_weight = 0.2

        ## Change with only len of legal moves
        # if parent is None and self.alpha != 0:
        #     num_legal_moves = self.num_moves
        #     noise = np.random.dirichlet([self.alpha] * num_legal_moves)
        #     self.priors = priors

        idx = 0
        legal_moves = state.legal_moves()
        #### FIX MASKING
        for move, p in priors.items():
            # if state.is_valid_move(move):
            if move in legal_moves:
                # if parent is None and self.alpha != 0:
                #     self.branches[move] = Branch( (1-self.noise_weight)*p + self.noise_weight*noise[idx] )
                # else:
                # if (not move.is_pass) or (state.turn > 20) or (move.is_pass and idx == 0) :
                self.branches[move] = Branch(p)
                # idx += 1
        if self.branches == {}:
            move = Move.pass_turn()
            self.branches[move] = Branch(1)
        self.children = {}

    def moves(self):
        return self.branches.keys()

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return np.ones(self.num_players)
        # Normalizing value in [0,1] to avoid move selection being stuck
        return (branch.total_value / self.num_moves) / branch.visit_count

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0


class ZeroAgent(Agent):
    def __init__(
        self, model, encoder, rounds_per_move=1600, c=2.0, alpha=0.03, policy=0
    ):
        # alpha = dirichlet noise at root
        # c = cpuct
        self.model = model
        self.encoder = encoder
        self.alpha = alpha
        self.collector = None
        self.use_policy = policy

        self.num_rounds = rounds_per_move
        self.c = c

    def select_move(self, game_state):
        rec = True
        root, priors = self.create_node(game_state)
        self.root = root
        # priors /= priors.sum()

        # legal_moves = []
        legal_moves = game_state.legal_moves()
        if legal_moves[0] == Move.pass_turn():
            return Move.pass_turn()

        prior = []
        zrs = []

        if self.use_policy:
            ind = 0
            for move, p in priors.items():
                # if game_state.is_valid_move(move):
                #     legal_moves.append(move)
                #     prior.append(p)
                if move in legal_moves:
                    prior.append(p)
                else:
                    zrs.append(ind)
                ind += 1

            prior = np.array(prior)
            prior /= prior.sum()
            # assert len(legal_moves)==len(prior), f"Problem priors, len(prior)={len(prior)} and len(legal_moves)={len(legal_moves)} "
            if len(legal_moves) == len(prior):
                move_to_play = np.random.choice(
                    legal_moves, p=prior
                )  # Working Correctly
            elif len(legal_moves) == 1 and len(prior) == 0:
                move_to_play = legal_moves[0]
                rec = False
            # if True:
            if self.collector is not None and rec:
                root_state_tensor = self.encoder.encode(game_state)
                visit_counts = np.fromiter(priors.values(), dtype=np.float32)
                # print("Policy visit counts for collector : ")
                # print(visit_counts)
                # print("Zeros = ",zrs)
                for val in zrs:
                    visit_counts[val] = 0
                visit_counts = visit_counts * 25 / visit_counts.sum()
                # visit_counts = visit_counts[:25]
                # print("from use policy",visit_counts,"type = ",type(visit_counts),visit_counts.dtype)
                self.collector.record_decision(root_state_tensor, visit_counts)

            return move_to_play

        else:
            for i in range(self.num_rounds):
                node = root
                next_move = self.select_branch(node)
                while node.has_child(next_move):
                    node = node.get_child(next_move)
                    next_move = self.select_branch(node)

                new_state = node.state.apply_move(next_move)
                child_node, _ = self.create_node(new_state, move=next_move, parent=node)

                move = next_move
                value = child_node.value
                while node is not None:
                    node.record_visit(move, value)
                    move = node.last_move
                    node = node.parent

            if self.collector is not None:
                # if True:
                root_state_tensor = self.encoder.encode(game_state)
                visit_counts = np.array(
                    [
                        root.visit_count(self.encoder.decode_move_index(idx))
                        for idx in range(self.encoder.num_moves())
                    ],
                    dtype=np.float32,
                )
                # print(visit_counts,"type = ",type(visit_counts),visit_counts.dtype)
                self.collector.record_decision(root_state_tensor, visit_counts)

            #### WITH 0.2 probability, choose move only using policy prob  [ ]
            #### Try playing only against UCT, maybe for the first 10 iterations [ ]
            #### ILLEGAL ACTIONS ?

            if game_state.turn < 20:  #### Change with game length
                moves = [move for move in root.moves()]

                #### WITH 50% chance select move according to UCT probabilities ie. not greedy
                if random.random() < 0.8:
                    # if len(moves) > 1:
                    #     moves.pop()
                    visit_counts = np.array([root.visit_count(move) for move in moves])
                    # if visit_counts.sum() != 0:
                    visit_counts = visit_counts / visit_counts.sum()
                    move = np.random.choice(moves, p=visit_counts)
                    # else:
                    #     move = np.random.choice(moves)
                    return move
            # print([root.visit_count(move) for move in root.moves()])
            # print([root.expected_value(move) for move in root.moves()])
            # expected_value
            return max(root.moves(), key=root.visit_count)

    def get_game_result(self, state):
        game_result = scoring.compute_game_result(state)
        result = game_result.final_scores
        return result

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):

            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            result = q + self.c * p * np.sqrt(total_n) / (n + 1)
            return result[node.state.next_player.value - 1]

        return max(node.moves(), key=score_branch)

    def create_node(self, game_state, move=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values1, values2, values3 = self.model.predict(model_input)
        # print(self.model.predict_proba(model_input))
        # if parent is None:
        #     num = 10
        #     values1 /= np.sum(values1/100)
        #     #print(values1)
        #     ar = [(i[0],i[1]) for i in sorted(enumerate(values1[0]), reverse=True, key=lambda x:x[1])]
        #     print(ar)
        #     print("\n")
        #     values2 /= np.sum(values2/100)
        #     ar = [(i[0],i[1]) for i in sorted(enumerate(values2[0]), reverse=True, key=lambda x:x[1])]
        #     print(ar)
        #     print("\n")
        #     values3 /= np.sum(values3/100)
        #     ar = [(i[0],i[1]) for i in sorted(enumerate(values3[0]), reverse=True, key=lambda x:x[1])]
        #     print(ar)
        #     print("\n")
        priors = priors[0]
        value = np.zeros(3)
        value[0], value[1], value[2] = (
            np.argmax(values1),
            np.argmax(values2),
            np.argmax(values3),
        )
        move_priors = {
            self.encoder.decode_move_index(idx): p for idx, p in enumerate(priors)
        }
        if game_state.is_over():
            value = self.get_game_result(game_state)
        new_node = ZeroTreeNode(
            game_state, value, move_priors, parent, move, alpha=self.alpha
        )
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node, move_priors

    def train(self, experience, learning_rate, batch_size, epochs=10, softmax=False):

        num_examples = experience.states.shape[0]
        model_input = experience.states
        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((num_examples, 1))
        action_target = experience.visit_counts / visit_sums
        value_target = experience.rewards
        if softmax:
            new_value_target = np.zeros((3, value_target.shape[0], 26))
            for idx, target in enumerate(value_target):
                arr1 = np.zeros(26)
                arr2 = np.zeros(26)
                arr3 = np.zeros(26)
                arr1[target[0]] = 1
                arr2[target[1]] = 1
                arr3[target[2]] = 1
                new_value_target[0][idx] = arr1
                new_value_target[1][idx] = arr2
                new_value_target[2][idx] = arr3

            value_target = new_value_target

        # Remove this later
        self.action_target = action_target
        self.value_target = value_target
        self.model_input = model_input

        self.model.compile(
            SGD(learning_rate=learning_rate),
            loss=[
                "categorical_crossentropy",
                "categorical_crossentropy",
                "categorical_crossentropy",
                "categorical_crossentropy",
            ],
            metrics={
                "policy": "categorical_accuracy",
                "value1": "categorical_accuracy",
                "value2": "categorical_accuracy",
                "value3": "categorical_accuracy",
            },
        )
        self.model.fit(
            model_input,
            [action_target, value_target[0], value_target[1], value_target[2]],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
        )

        # [X] Network probabilities
        # [X] Descent algorithm to learn the value
        # Learning to Play Two-Player Perfect-Information Games without Knowledge
        # [X] Ordinal action distribution
        # [ ] Number of neural network evaluation instead of time
        # [ ] 40k games with descent
