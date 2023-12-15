from stable_baselines3 import DQN
import numpy as np
import torch


class MaskedActionsDQN(DQN):
    def predict(self, observation, deterministic=False):
        # Extracting the middle 11x11 part of the board
        # Assuming the board is the last two dimensions of the tensor
        middle_start = 2  # 15x15 to 11x11, start from index 2
        # middle_end = middle_start + middle_board_3rd size - 2
        middle_end = observation.shape[1] - 2
        middle_board_3d = observation[
            :, middle_start:middle_end, middle_start:middle_end, :
        ]

        # Extract legal moves from board_3d
        legal_moves = (middle_board_3d[:, :, :, 0] == 0) & (
            middle_board_3d[:, :, :, 1] == 0
        )
        legal_moves_flat = legal_moves.flatten()
        # print("Shitty legal moves here", legal_moves_flat)

        board_3d_tensor = torch.tensor(observation, dtype=torch.float32)
        # print("Shitty tensor here", board_3d_tensor)
        # print(observation)
        # print(board_3d_tensor.shape)

        q_values = self.policy.q_net(board_3d_tensor).squeeze(0)
        q_values = q_values.cpu().detach().numpy()
        # print("Shitty q values here", q_values)

        

        # Apply mask to Q-values
        masked_q_values = q_values * legal_moves_flat - (1 - legal_moves_flat) * 1e8

        if (
            deterministic or np.random.rand() > self.exploration_rate
        ):  # Epsilon-greedy strategy
            action = np.argmax(masked_q_values)
        else:
            legal_indices = np.where(legal_moves_flat)[0]
            action = np.random.choice(legal_indices)

        action = np.array([action])
        return action, None
