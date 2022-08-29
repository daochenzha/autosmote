import numpy as np
import gym
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics.pairwise import euclidean_distances

class Env:
    def __init__(
        self,
        train_X,
        train_y,
        val_X,
        val_y,
        clf,
        metric,
    ):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.clf = clf
        self.metric = metric

        unique, counts = np.unique(self.train_y, return_counts=True)
        self.min_label = unique[0] if counts[0] < counts[1] else unique[1]
        self.train_X_min = self.train_X[self.train_y==self.min_label]
        
    def step(self, samples):
        if len(samples) > 0:
            train_X = np.vstack([self.train_X, samples])
            train_y = np.hstack([self.train_y, np.hstack([self.min_label]*len(samples))])
        else:
            train_X, train_y = self.train_X, self.train_y
        self.clf.fit(train_X, train_y)
        pred_y = self.clf.predict(self.val_X)
        if self.metric == "macro_f1":
            result = f1_score(self.val_y, pred_y, average="macro")
        elif self.metric == "mcc":
            result = matthews_corrcoef(self.val_y, pred_y)
        return result

class GymWrapper(gym.Env):
    def __init__(self, env, flags, neighbors=None):
        self._env = env
        self.X_min = self._env.train_X_min
        self.length = len(self.X_min)
        self.num_instance_specific_actions = flags.num_instance_specific_actions
        self.feature_dim = self.X_min.shape[1]

        self.count = np.zeros(self.length, dtype=np.int32)
        count_features = np.zeros((self.length, 10))
        self.instance_features = np.concatenate((count_features, self.X_min), axis=1)
        if neighbors is None:
            self.neighbors = flags.neighbors
        else:
            self.neighbors = neighbors

        # Precompute the pairwise diffs.
        self.diffs = self.X_min[np.newaxis,:,:] - self.X_min[:,np.newaxis,:]
        self.diffs = self.diffs.reshape(-1,self.diffs.shape[-1])

        # Find neighbors
        self.distances = euclidean_distances(self.X_min, self.X_min)
        self.nearest_neighbors = np.argsort(self.distances)

        # For gym
        self.observation_space = gym.spaces.Dict({
            "src_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim+10,), dtype=np.float32),
            "neighbor_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.neighbors, self.feature_dim+10), dtype=np.float32),
        })
        self.action_space = gym.spaces.Dict({
            "dst_index": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
            "ratio": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "next": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=bool),
        })

    def reset(self):
        self.src_indices = []
        self.dst_indices = []
        self.ratio = []
        self.cur_index = 0
        self.count[:] = 0
        self.instance_features[:,0] = 1.0
        self.instance_features[:,1:10] = 0.0

        return self._get_obs()

    def step(self, action):
        # actions contain src, dst, ratio
        if action["dst_index"] >= 0:
            self.src_indices.append(self.cur_index)
            self.dst_indices.append(self.nearest_neighbors[self.cur_index][action["dst_index"]])
            dst_index = self.nearest_neighbors[self.cur_index][action["dst_index"]]
            self.count[self.cur_index] += 1
            if self.count[self.cur_index] % 20 == 0:
                col = min(self.count[self.cur_index] // 20, 9)
                if col > 0:
                    self.instance_features[self.cur_index][col-1] = 0.0
                self.instance_features[self.cur_index][col] = 1.0

            self.count[dst_index] += 1
            if self.count[dst_index] % 20 == 0:
                col = min(self.count[dst_index] // 20, 9)
                if col > 0:
                    self.instance_features[dst_index][col-1] = 0.0
                self.instance_features[dst_index][col] = 1.0

            self.ratio.append(action["ratio"])

        if action["next"]:
            self.cur_index += 1

        obs = self._get_obs()

        if self.cur_index < self.length:
            return obs, 0, False, {}
        else:
            samples = {
                "src_indices": np.array(self.src_indices),
                "dst_indices": np.array(self.dst_indices),
                "ratio": np.array(self.ratio),
            }
            reward = self.get_reward(samples)
            return obs, reward, True, {"samples": samples}

    def get_reward(self, samples):
        # Generating samples
        if len(samples["src_indices"]) == 0:
            samples = []
        else:
            src_indices = samples["src_indices"]
            dst_indices = samples["dst_indices"]
            ratio = samples["ratio"]
            flat_indices = src_indices * self.length + dst_indices
            samples = self.X_min[src_indices] + ratio.reshape(ratio.shape[0],1) * self.diffs[flat_indices]

        return self._env.step(samples)

    def _get_obs(self):
        if self.cur_index >= self.length:
            return None
        else:
            return {
                "src_features": self.instance_features[self.cur_index],
                "neighbor_features": self.instance_features[self.nearest_neighbors[self.cur_index][:self.neighbors]],
            }
