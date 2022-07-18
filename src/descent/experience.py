import numpy as np

__all__ = [
    "DescExperienceCollector",
    "DescExperienceBuffer",
    "combine_experience",
    "load_experience",
]


class DescExperienceCollector:
    def __init__(self):
        self.states = []
        self.values = []
        # self.rewards = []
        self._current_episode_states = []
        self._current_episode_values = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_values = []

    def record_decision(self, state, value):
        self._current_episode_states.append(state)
        self._current_episode_values.append(value)

    def complete_episode(self):
        # num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.values += self._current_episode_values
        # self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_values = []


class DescExperienceBuffer:
    def __init__(self, states, values):
        self.states = states
        self.values = values
        # self.rewards = rewards

    def serialize(self, h5file):
        h5file.create_group("experience")
        h5file["experience"].create_dataset("states", data=self.states)
        h5file["experience"].create_dataset("values", data=self.values)
        # h5file["experience"].create_dataset("rewards", data=self.rewards)


def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_values = np.concatenate([np.array(c.values) for c in collectors])
    # combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])

    return DescExperienceBuffer(combined_states, combined_values)


def load_experience(h5file):
    return DescExperienceBuffer(
        states=np.array(h5file["experience"]["states"]),
        values=np.array(h5file["experience"]["values"]),
    )
