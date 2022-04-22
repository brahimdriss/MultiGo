import sys

sys.path.append("../")


from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from MultiGo.src import scoring
from MultiGo.src import zero
from MultiGo.src.goboard_fast import GameState, Player, Point


def simulate_game(
    board_size,
    black_agent,
    black_collector,
    white_agent,
    white_collector,
    red_agent,
    red_collector,
):
    print("Starting the game!")
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
    if game_result.winner == Player.black:
        black_collector.complete_episode([1, 0, 0])
        white_collector.complete_episode([1, 0, 0])
        red_collector.complete_episode([1, 0, 0])
    elif game_result.winner == Player.white:
        black_collector.complete_episode([0, 1, 0])
        white_collector.complete_episode([0, 1, 0])
        red_collector.complete_episode([0, 1, 0])
    else:
        black_collector.complete_episode([0, 0, 1])
        white_collector.complete_episode([0, 0, 1])
        red_collector.complete_episode([0, 0, 1])


board_size = 5
encoder = zero.ZeroEncoder(board_size)

board_input = Input(shape=encoder.shape(), name="board_input")
pb = board_input
for i in range(4):
    pb = Conv2D(
        64, (3, 3), padding="same", data_format="channels_first", activation="relu"
    )(pb)

policy_conv = Conv2D(2, (1, 1), data_format="channels_first", activation="relu")(pb)
policy_flat = Flatten()(policy_conv)
policy_output = Dense(encoder.num_moves(), activation="softmax")(policy_flat)

value_conv = Conv2D(1, (1, 1), data_format="channels_first", activation="relu")(pb)
value_flat = Flatten()(value_conv)
value_hidden = Dense(256, activation="relu")(value_flat)
value_output = Dense(3, activation="softmax")(value_hidden)

model = Model(inputs=[board_input], outputs=[policy_output, value_output])

black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
white_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
red_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
c1 = zero.ZeroExperienceCollector()
c2 = zero.ZeroExperienceCollector()
c3 = zero.ZeroExperienceCollector()
black_agent.set_collector(c1)
white_agent.set_collector(c2)
red_agent.set_collector(c3)

for i in range(5):
    simulate_game(board_size, black_agent, c1, white_agent, c2, red_agent, c3)

exp = zero.combine_experience([c1, c2, c3])
black_agent.train(exp, 0.01, 2048)

