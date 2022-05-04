import sys
sys.path.append("../")


from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from MultiGo.src import scoring
from MultiGo.src import zero
from MultiGo.src.goboard_fast import GameState, Player, Point
from MultiGo.src.mcts import scoreuct
import h5py
import numpy as np


def simulate_game(
    board_size,
    black_agent,
    black_collector,
    white_agent,
    white_collector,
    red_agent,
    red_collector,
):
    #print("Starting the game!")
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
        Player.red: red_agent,
    }

    black_collector.begin_episode()
    white_collector.begin_episode()
    red_collector.begin_episode()

    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    
    black_collector.complete_episode(game_result.final_scores)
    white_collector.complete_episode(game_result.final_scores)
    red_collector.complete_episode(game_result.final_scores)



board_size = 5
rounds = 100
temperature = 1.2

encoder = zero.ZeroEncoder(board_size)


black_agent = scoreuct.MCTS_score_Agent(rounds, temperature)
white_agent = scoreuct.MCTS_score_Agent(rounds, temperature)
red_agent = scoreuct.MCTS_score_Agent(rounds, temperature)

c1 = zero.ZeroExperienceCollector()
c2 = zero.ZeroExperienceCollector()
c3 = zero.ZeroExperienceCollector()

black_agent.set_collector(c1)
white_agent.set_collector(c2)
red_agent.set_collector(c3)

black_agent.set_encoder(encoder)
white_agent.set_encoder(encoder)
red_agent.set_encoder(encoder)

for i in range(1):
    simulate_game(board_size, black_agent, c1, white_agent, c2, red_agent, c3)

exp = zero.combine_experience([c1, c2, c3])

experience_filename = "test_file.h5"
print('Saving experience buffer to %s\n' % experience_filename)
with h5py.File(experience_filename, 'w') as experience_outf:
    exp.serialize(experience_outf)

#black_agent.train(exp, 0.01, 2048)

