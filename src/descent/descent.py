import math
import numpy as np
import random
import time

from MultiGo.src import agent
from MultiGo.src.gotypes import Player
from MultiGo.src.goboard_fast import Move
from MultiGo.src import scoring
from tensorflow.keras.optimizers import SGD, Adam

__all__ = [
    "Descent_Agent",
]


class Descent_Node(object):
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.checked = False
        self.recorded = False
        self.move = move
        self.estimate = [0, 0, 0]
        self.value = [0, 0, 0]
        self.children = []
        self.univisted_moves = game_state.legal_moves()

    def add_random_child(self):
        index = random.randint(0, len(self.univisted_moves) - 1)
        new_move = self.univisted_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = Descent_Node(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def add_all_children(self):
        while len(self.univisted_moves) > 0:
            index = random.randint(0, len(self.univisted_moves) - 1)
            new_move = self.univisted_moves.pop(index)
            new_game_state = self.game_state.apply_move(new_move)
            new_node = Descent_Node(new_game_state, self, new_move)
            self.children.append(new_node)

    def can_add_child(self):
        return len(self.univisted_moves) > 0

    def is_terminal(self):
        return self.game_state.is_over()

    def best_child_state(self):
        best_child_score = -10
        best_child = None
        player = self.game_state.next_player
        for child in self.children:
            score = child.estimate[player.value - 1]
            if score > best_child_score:
                best_child = child
                best_child_score = score
        return best_child


class Descent_Agent(agent.Agent):
    def __init__(self, model, encoder, rounds, iteration=12500, time_budget=1.5):
        agent.Agent.__init__(self)
        self.max_rounds = rounds
        self.collector = None
        self.encoder = encoder
        self.model = model
        self.iteration = iteration
        self.max_iterations = 12500
        self.time_budget = time_budget

    def set_collector(self, collector):
        self.collector = collector

    def descent_iteration(self, state_node):
        node = state_node
        if node.is_terminal():
            node.checked = True
            node.value = self.get_game_result(node)
        else:
            if node.checked == False:
                node.checked = True
                node.add_all_children()
                for child_node in node.children:
                    if child_node.is_terminal():
                        child_node.checked = True
                        child_node.estimate = self.get_game_result(child_node)
                        child_node.value = child_node.estimate
                    else:
                        child_node.estimate = self.model_evaluation(child_node)

            best_action_state = node.best_child_state()
            best_action_state.estimate = self.descent_iteration(best_action_state)
            best_action_state = node.best_child_state()
            node.value = best_action_state.estimate

        return node.value

    def model_evaluation(self, node):
        state_tensor = self.encoder.encode(node.game_state)
        model_input = np.array([state_tensor])
        values1, values2, values3 = self.model.predict(model_input)
        value = np.zeros(3)
        value[0], value[1], value[2] = (
            np.argmax(values1),
            np.argmax(values2),
            np.argmax(values3),
        )
        return value

    def get_game_result(self, node):
        state = node.game_state
        game_result = scoring.compute_game_result(state)
        result = game_result.final_scores
        return result

    def dfs_record(self, node):
        if node.recorded == False:
            node.recorded = True
            if node.checked == True:
                state_tensor = self.encoder.encode(node.game_state)
                if isinstance(node.value, list):
                    rec = node.value
                    if np.sum(rec) > 25:
                        rec = (rec / np.sum(rec)) * 25
                    self.collector.record_decision(state_tensor, rec)
            for child_node in node.children:
                self.dfs_record(child_node)

    def ordinal_action_distribution(self, state_node, epsilon):
        ep = epsilon
        player = state_node.game_state.next_player
        moves = [i.move for i in state_node.children]
        values = [i.value[player.value - 1] for i in state_node.children]
        num_moves = len(moves)
        zipped = zip(values, moves)
        zipped = sorted(zipped, key=lambda x: x[0])
        sorted_values = [i for (i, s) in zipped]
        sorted_moves = [s for (i, s) in zipped]

        p = np.zeros(num_moves)
        for i in range(num_moves):
            p[i] = (ep + (1 - ep) / (num_moves - i)) * (1 - np.sum(p[:i]))
        p /= np.sum(p)
        assert (
            len(p[p < 0]) == 0
        ), f"Problem in ordinal action distribution, p = {p}, number of moves = {num_moves}, epsilon = {ep}, sum = {np.sum(p)}, Number of negatives = {len(p[p < 0])}"
        return p, sorted_moves

    def select_move(self, game_state):
        root = Descent_Node(game_state)

        legal_moves = game_state.legal_moves()
        if legal_moves[0] == Move.pass_turn():
            return Move.pass_turn()
        # if len(root.univisted_moves) == 1:
        # return root.univisted_moves[0]
        start_time = time.time()
        while time.time() - start_time < self.time_budget:
            self.descent_iteration(root)

        if self.collector is not None:
            self.dfs_record(root)
        self.root = root

        epsilon = self.iteration / self.max_iterations
        p, moves = self.ordinal_action_distribution(root, epsilon)
        move_to_play = np.random.choice(moves, p=p)
        return move_to_play

    def set_collector(self, collector):
        self.collector = collector

    def set_encoder(self, encoder):
        self.encoder = encoder

    def train(self, experience, learning_rate, batch_size, epochs=10, softmax=False):
        num_examples = experience.states.shape[0]
        model_input = experience.states
        value_target = experience.values
        new_value_target = np.zeros((3, value_target.shape[0], 26))
        for idx, target in enumerate(value_target):
            arr1 = np.zeros(26)
            arr2 = np.zeros(26)
            arr3 = np.zeros(26)
            arr1[int(target[0])] = 1
            arr2[int(target[1])] = 1
            arr3[int(target[2])] = 1
            new_value_target[0][idx] = arr1
            new_value_target[1][idx] = arr2
            new_value_target[2][idx] = arr3
        value_target = new_value_target
        # Remove this later
        self.value_target = value_target
        self.model_input = model_input
        self.model.compile(
            SGD(learning_rate=learning_rate),
            loss=[
                "categorical_crossentropy",
                "categorical_crossentropy",
                "categorical_crossentropy",
            ],
            metrics={
                "value1": "categorical_accuracy",
                "value2": "categorical_accuracy",
                "value3": "categorical_accuracy",
            },
        )
        self.model.fit(
            model_input,
            [value_target[0], value_target[1], value_target[2]],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
        )
