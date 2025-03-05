import numpy as np
import random
from rlkit.data_management.replay_buffer import ReplayBuffer


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, random_seed=1995):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 0.01
        self.abs_err_upper = 1.0
        self._trajs = 0
        self._np_rand_state = np.random.RandomState(random_seed)

    def _get_priority(self, error):
        error = np.abs(error) + self.epsilon
        clipped_errors = np.minimum(error, self.abs_err_upper)
        pr = np.power(clipped_errors, self.alpha)
        return pr

    def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
        #没有对absorbing进行处理
        error = kwargs.get('error', 1.0)
        absorbing = kwargs.get('absorbing', np.array([0.0, 0.0]))
        priority = self._get_priority(error)
        self.tree.add(priority, (observation, action, reward, terminal, next_observation, absorbing))

    def update(self, idx, error):
        priority = self._get_priority(error)
        for i, pr in zip(idx, priority):
            self.tree.update(i, pr)

    def add_path(self, path, absorbing=False, env=None):
        if not absorbing:
            for (
                    ob,
                    action,
                    reward,
                    next_ob,
                    terminal,
                    # agent_info,
                    # env_info
            ) in zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                # path["agent_infos"],
                # path["env_infos"],
            ):
                self.add_sample(
                    observation=ob,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                    next_observation=next_ob,
                    # agent_info=agent_info,
                    # env_info=env_info,
                )
        else:
            for (
                    ob,
                    action,
                    reward,
                    next_ob,
                    terminal,
                    # agent_info,
                    # env_info
            ) in zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                # path["agent_infos"],
                # path["env_infos"],
            ):
                self.add_sample(
                    observation=ob,
                    action=action,
                    reward=reward,
                    terminal=np.array([False]),
                    next_observation=next_ob,
                    absorbing=np.array([0.0, 0.0]),
                    # agent_info=agent_info,
                    # env_info=env_info,
                )
                if terminal[0]:
                    print("add terminal")
                    self.add_sample(
                        observation=next_ob,
                        # action=action,
                        action=env.action_space.sample(),
                        reward=reward,
                        terminal=np.array([False]),
                        next_observation=np.zeros_like(next_ob),  # next_ob,
                        absorbing=np.array([0.0, 1.0]),
                        # agent_info=agent_info,
                        # env_info=env_info,
                    )
                    self.add_sample(
                        observation=np.zeros_like(next_ob),  # next_ob,
                        # action=action,
                        action=env.action_space.sample(),
                        reward=reward,
                        terminal=np.array([False]),
                        next_observation=np.zeros_like(next_ob),  # next_ob,
                        absorbing=np.array([1.0, 1.0]),
                        # agent_info=agent_info,
                        # env_info=env_info,
                    )

        self.terminate_episode()
        self._trajs += 1

    def random_batch(self, batch_size, keys=None):
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = self._np_rand_state.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            priorities.append(priority)
            batch.append(data)
            idxs.append(index)
        sampling_probabilities = priorities / self.tree.total_priority
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        if keys is None:
            keys = {"observations", "actions", "rewards", "terminals", "next_observations", "absorbing"}
        batch_dict = {key: [] for key in keys}
        for data in batch:
            observation, action, reward, terminal, next_observation, absorbing = data
            if "observations" in keys:
                batch_dict["observations"].append(observation)
            if "actions" in keys:
                batch_dict["actions"].append(action)
            if "rewards" in keys:
                batch_dict["rewards"].append(reward)
            if "terminals" in keys:
                batch_dict["terminals"].append(terminal)
            if "next_observations" in keys:
                batch_dict["next_observations"].append(next_observation)
            if "absorbing" in keys:
                batch_dict["absorbing"].append(absorbing)

        for key in batch_dict:
            batch_dict[key] = np.array(batch_dict[key])

        return batch_dict, idxs, is_weight

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return np.count_nonzero(self.tree.data)

    def get_traj_num(self):
        return self._trajs


