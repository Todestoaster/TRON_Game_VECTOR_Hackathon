import tensorflow as tf
import numpy as np
import random

class MLAgent:
    def __init__(self, player_id, input_shape, num_actions=3):
        self.player_id = player_id
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model = self.build_model()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995


    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_actions, activation='linear')  # 3 Ausgänge für 3 Aktionen
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse')  # Für Q-Learning später wichtig
        return model

    def select_action(self, state):
        if state is None or state.shape[0] == 0:
            # Sofort stoppen wenn ungültiger Zustand
            raise ValueError("State input for select_action is None or empty.")

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Vorbereitung des Eingabebildes
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=-1)  # (H, W) -> (H, W, 1)
        state = np.expand_dims(state, axis=0)        # (H, W, 1) -> (1, H, W, 1)
        state = state.astype(np.float32)

        q_values = self.model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
        return action




    def train(self, states, actions, rewards, next_states, dones, gamma=0.99):
        next_qs = self.model.predict(next_states, verbose=0)
        target_qs = rewards + (1 - dones) * gamma * np.amax(next_qs, axis=1)

        target_f = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = target_qs[i]

        self.model.fit(states, target_f, epochs=1, verbose=0)
