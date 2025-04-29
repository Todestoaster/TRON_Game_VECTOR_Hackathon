import random
import numpy as np
import tensorflow as tf
import os
import json
import pygame
import matplotlib.pyplot as plt
from tqdm import tqdm
from game.env import TronEnv
from game.constants import GRID_WIDTH, GRID_HEIGHT, CELL_SIZE

# === Optionen ===#
SHOW_WINDOW = True  # True = Pygame und Matplotlib Fenster zeigen, False = nur Konsole

PLAYER_COLORS = [(0, 255, 0), (186, 85, 211), (30, 144, 255)]  # Grün, Lila, Blau
MODEL_DIRS = ["dqn_tron_tf_player1.keras", "dqn_tron_tf_player2.keras", "dqn_tron_tf_player3.keras"]
STATE_FILES = ["train_state_player1.json", "train_state_player2.json", "train_state_player3.json"]
HISTORY_FILES = ["reward_history_player1.json", "reward_history_player2.json", "reward_history_player3.json"]

# === Hilfsfunktionen ===
def save_training_state(episode, steps_done, filename):
    with open(filename, "w") as f:
        json.dump({"episode": episode, "steps_done": steps_done}, f)

def load_training_state(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            return data["episode"], data["steps_done"]
    else:
        return 0, 0

def save_reward_history(episode_numbers, rewards_plot, filename):
    with open(filename, "w") as f:
        json.dump({"episodes": episode_numbers, "rewards": rewards_plot}, f)

def load_reward_history(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            return data["episodes"], data["rewards"]
    else:
        return [], []

# === DQN Netzwerk ===
def build_model(input_shape, n_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape + (1,)),  # Kanal hinzufügen für Conv2D
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_actions)  # Q-Werte für jede Aktion
    ])
    return model

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# === Training ===
def train():
    if SHOW_WINDOW:
        pygame.init()
        window_width = GRID_WIDTH * CELL_SIZE + 300
        window_height = GRID_HEIGHT * CELL_SIZE
        screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Training Multi-Agent")
        clock = pygame.time.Clock()

        plt.ion()
        fig, ax = plt.subplots()

    env = TronEnv()
    input_shape = (GRID_WIDTH, GRID_HEIGHT)
    n_actions = 3

    models = [build_model(input_shape, n_actions) for _ in range(3)]
    optimizers = [tf.keras.optimizers.Adam(learning_rate=1e-4) for _ in range(3)]
    buffers = [ReplayBuffer(10000) for _ in range(3)]
    epsilons = [1.0, 1.0, 1.0]

    gamma = 0.99
    epsilon_final = 0.05
    epsilon_decay = 10000
    batch_size = 64

    starts = [load_training_state(file) for file in STATE_FILES]
    episode_starts, steps_done = zip(*starts)
    episode_starts = list(episode_starts)
    steps_done = list(steps_done)

    # === Korrigierte Initialisierung für History ===
    episode_numbers = [[] for _ in range(3)]
    rewards_plot = [[] for _ in range(3)]

    for idx, file in enumerate(HISTORY_FILES):
        if os.path.exists(file):
            episodes, rewards = load_reward_history(file)
            if episodes and rewards:
                episode_numbers[idx] = episodes
                rewards_plot[idx] = rewards

    # === Modell Laden ===
    for i in range(3):
        if os.path.exists(MODEL_DIRS[i]):
            models[i] = tf.keras.models.load_model(MODEL_DIRS[i])
            print(f"Modell für Spieler {i+1} geladen.")

    try:
        for episode in tqdm(range(max(episode_starts), 100000)):
            state = env.reset()
            state = np.array(state, dtype=np.float32)
            state = np.expand_dims(state, axis=-1)
            done = False
            episode_rewards = [0, 0, 0]
            step_rewards = [[] for _ in range(3)]

            while not done:
                actions = []
            
                for i in range(3):
                    epsilons[i] = epsilon_final + (1.0 - epsilon_final) * np.exp(-1. * steps_done[i] / epsilon_decay)
                    steps_done[i] += 1
                    if random.random() < epsilons[i]:
                        actions.append(random.randint(0, n_actions - 1))
                    else:
                        state_input = np.expand_dims(state, axis=0)
                        q_values = models[i](state_input, training=False)
                        actions.append(np.argmax(q_values[0]))

                next_state, rewards, done, _ = env.step(actions)

                # === Reward Shaping ===
                original_rewards = rewards.copy()  # Originale Rewards sichern!

                alive_players = [i for i in range(3) if rewards[i] > -1.0]  # Lebende Spieler

                for i in range(3):
                    if original_rewards[i] == -1.0:  # Spieler ist gestorben
                        rewards[i] -= 10.0  # Strafe für Tod

                        if env.death_reason[i] == "self":
                            rewards[i] -= 10.0  # Extra Strafe für Selbstkollision
                    else:
                        rewards[i] += 0.01  # Kleine Belohnung fürs Überleben

                # Bonus: Gegner sind gestorben
                num_dead = 3 - len(alive_players)
                for i in alive_players:
                    rewards[i] += 2.0 * num_dead
                    if len(alive_players) == 1:  # Nur einer lebt
                        rewards[i] += 5.0




                next_state = np.array(next_state, dtype=np.float32)
                next_state = np.expand_dims(next_state, axis=-1)

                for i in range(3):
                    buffers[i].push(state, actions[i], rewards[i], next_state, done)
                    step_rewards[i].append(rewards[i])

                state = next_state

                for i in range(3):
                    if len(buffers[i]) >= batch_size:
                        batch = buffers[i].sample(batch_size)
                        states, actions_b, rewards_b, next_states, dones = zip(*batch)

                        states = np.array(states)
                        actions_b = np.array(actions_b)
                        rewards_b = np.array(rewards_b)
                        next_states = np.array(next_states)
                        dones = np.array(dones)

                        next_q_values = models[i](next_states, training=False)
                        max_next_q = np.max(next_q_values, axis=1)
                        target_q = rewards_b + gamma * max_next_q * (1 - dones)

                        with tf.GradientTape() as tape:
                            q_values = models[i](states)
                            action_qs = tf.reduce_sum(q_values * tf.one_hot(actions_b, n_actions), axis=1)
                            loss = tf.keras.losses.MeanSquaredError()(target_q, action_qs)

                        grads = tape.gradient(loss, models[i].trainable_variables)
                        optimizers[i].apply_gradients(zip(grads, models[i].trainable_variables))

                if SHOW_WINDOW:
                    screen.fill((10, 10, 10))
                    env.draw(screen)
                    pygame.draw.rect(screen, (20, 20, 20), (GRID_WIDTH * CELL_SIZE, 0, 300, window_height))
                    font = pygame.font.SysFont(None, 24)
                    for idx in range(3):
                        wins = env.wins[idx]
                        info = f"P{idx+1} R:{episode_rewards[idx]:.2f} E:{epsilons[idx]:.2f} W:{wins}"
                        text_surface = font.render(info, True, PLAYER_COLORS[idx])
                        screen.blit(text_surface, (GRID_WIDTH * CELL_SIZE + 10, 20 + idx * 30))
                    pygame.display.flip()
                    clock.tick(15)

            for i in range(3):
                mean_reward = np.mean(step_rewards[i])  # <<< Mittelwert korrekt
                rewards_plot[i].append(mean_reward)      # <<< Korrekt mean_reward speichern
                episode_numbers[i].append(episode)

            # === Plot alle 10 Episoden ===
            if SHOW_WINDOW and episode % 10 == 0:
                ax.clear()
                for idx in range(3):
                    if len(episode_numbers[idx]) > 0:
                        ax.plot(episode_numbers[idx], rewards_plot[idx], label=f"P{idx+1}", color=np.array(PLAYER_COLORS[idx])/255)
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.set_title("Training Progress")
                ax.legend()
                plt.pause(0.01)

            # === Speichern alle 100 Episoden ===
            if episode % 100 == 0:
                for i in range(3):
                    models[i].save(MODEL_DIRS[i])
                    save_training_state(episode, steps_done[i], STATE_FILES[i])
                    save_reward_history(episode_numbers[i], rewards_plot[i], HISTORY_FILES[i])
                print(f"Episode {episode} gespeichert.")

    except KeyboardInterrupt:
        print("\nTraining abgebrochen. Speichere alle Modelle...")
        for i in range(3):
            models[i].save(MODEL_DIRS[i])
            save_training_state(episode, steps_done[i], STATE_FILES[i])
            save_reward_history(episode_numbers[i], rewards_plot[i], HISTORY_FILES[i])
        print("Alle Modelle gespeichert.")

if __name__ == "__main__":
    train()
