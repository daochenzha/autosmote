import torch
from torch import nn
from torch.nn import functional as F

class CrossInstanceNet(nn.Module):
    def __init__(self, observation_space, num_actions):
        super().__init__()
        self.state_dim = observation_space["src_features"].shape[0]
        self.state_hidden = 128
        self.num_actions = num_actions

        self.fc_0 = nn.Linear(self.state_dim, self.state_hidden)
        self.fc_1 = nn.Linear(self.state_hidden, self.state_hidden)

        self.policy = nn.Linear(self.state_hidden, num_actions)
        self.baseline = nn.Linear(self.state_hidden, 1)

    def forward(self, inputs, device="cpu"):
        x = inputs["src_features"]
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))

        policy_logits = self.policy(x)
        baseline = self.baseline(x)

        if self.training:
            #print(torch.argmax(policy_logits, dim=1))
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, action=action)

class InstanceSpecificNet(nn.Module):
    def __init__(self, observation_space, num_actions):
        super().__init__()
        self.state_dim = observation_space["src_features"].shape[0]
        self.state_hidden = 128
        self.num_actions = num_actions

        self.fc_0 = nn.Linear(self.state_dim, self.state_hidden)
        self.fc_1 = nn.Linear(self.state_hidden, self.state_hidden)

        self.policy = nn.Linear(self.state_hidden, num_actions)
        self.baseline = nn.Linear(self.state_hidden, 1)

    def forward(self, inputs, device="cpu"):
        x = inputs["src_features"]
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))

        policy_logits = self.policy(x)
        baseline = self.baseline(x)

        if self.training:
            #print(torch.argmax(policy_logits, dim=1))
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, action=action)

class LowLevelNet(nn.Module):
    def __init__(self, observation_space, num_ratios, device="cpu"):
        super().__init__()
        self.state_dim = observation_space["src_features"].shape[0]
        self.num_actions, self.action_dim = observation_space["neighbor_features"].shape
        self.num_actions *= num_ratios # The distance is 0, 0.25, 0.5, 0.75, 1.0
        self.hidden = 128
        self.num_ratios = num_ratios

        # Policy
        self.p_fc_0 = nn.Linear(self.state_dim+self.action_dim+num_ratios, self.hidden)
        self.p_fc_1 = nn.Linear(self.hidden, self.hidden)
        self.policy = nn.Linear(self.hidden, 1)

        # Baseline
        self.v_fc_0 = nn.Linear(self.state_dim, self.hidden)
        self.v_fc_1 = nn.Linear(self.hidden, self.hidden)
        self.baseline = nn.Linear(self.hidden, 1)

        self.distance_features = []
        for i in range(self.num_ratios):
            tmp = [0 for _ in range(self.num_ratios)]
            tmp[i] = 1
            self.distance_features.append(tmp)
        self.distance_features = torch.tensor(self.distance_features, dtype=torch.float32)
        self.distance_features = self.distance_features.to(device)
    def forward(self, inputs, considered_neighbors, device="cpu"):
        state_features = inputs["src_features"]
        T, B, *_ = state_features.shape
        state_features = torch.flatten(state_features, 0, 1)  # Merge time and batch.
        neighbor_features = inputs["neighbor_features"]
        neighbor_features = torch.flatten(neighbor_features, 0, 2)  # Merge time, batch, and num_actions.
        neighbor_features = neighbor_features.repeat(1, self.num_ratios).reshape(-1, self.action_dim)
        distance_features = self.distance_features.repeat(neighbor_features.shape[0]//self.num_ratios, 1)
        action_features = torch.cat((neighbor_features, distance_features), -1)
        # Policy
        x = state_features.repeat(1,self.num_actions).view(-1,state_features.shape[1])
        x = torch.cat((x, action_features), -1)
        x = F.relu(self.p_fc_0(x))
        x = F.relu(self.p_fc_1(x))
        policy_logits = self.policy(x)
        policy_logits = policy_logits.view(-1, self.num_actions)

        # Baseline
        baseline = F.relu(self.v_fc_0(state_features))
        baseline = F.relu(self.v_fc_1(baseline))
        baseline = self.baseline(baseline)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits[:, :considered_neighbors*self.num_ratios], dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, action=action)




