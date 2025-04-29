import pygame
import numpy as np
from game.game import Game
from game.ml_agent import MLAgent
from game.config import FIELD_HEIGHT, FIELD_WIDTH
from game.replay_memory import ReplayMemory

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Tron Lightcycles (MLAgent Training)")

    clock = pygame.time.Clock()
    game = Game(screen)

    input_shape = (FIELD_HEIGHT, FIELD_WIDTH, 1)
    agents = [MLAgent(1, input_shape), MLAgent(2, input_shape), MLAgent(3, input_shape)]

    memory = ReplayMemory(capacity=100_000)

    running = True
    fps = 60

    raw_state = game.get_game_state()
    normalized_state = raw_state / 3.0
    done = False

    episode = 1
    last_rewards = [0.0, 0.0, 0.0]
    last_winner = None
    font = pygame.font.SysFont("Arial", 20)
    step_in_episode = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not done:  # <<< NEU: nur wenn Spiel läuft!
            step_in_episode += 1

            actions = [agent.select_action(normalized_state) for agent in agents]
            next_state, rewards, done = game.step(actions)

            normalized_next_state = next_state / 3.0

            for i, agent in enumerate(agents):
                memory.push(normalized_state, actions[i], rewards[i], normalized_next_state, done)

            normalized_state = normalized_next_state

            if len(memory) > 1000:
                for agent in agents:
                    states, actions_batch, rewards_batch, next_states, dones_batch = memory.sample(batch_size=64)
                    agent.train(states, actions_batch, rewards_batch, next_states, dones_batch, gamma=0.99)
                    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Zeichnen immer erlaubt
        game.draw()
        draw_info(screen, episode, step_in_episode, agents, last_rewards, last_winner, font)
        pygame.display.flip()
        clock.tick(fps)

        if done:
            print(f"Episode {episode} beendet. Rewards: {rewards}")
            if game.winner:
                print(f"🏆 Gewinner: Agent {game.winner}")
            else:
                print("Unentschieden.")

            game.reset()
            raw_state = game.get_game_state()
            # NEU: Sicherstellen dass raw_state korrekt ist
            if raw_state is None or raw_state.shape[0] == 0:
                print("WARNUNG: Spielfeld-Reset fehlerhaft! State leer.")
                # Kleines Dummy-Feld erzeugen, damit kein Crash kommt
                raw_state = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=np.float32)

            normalized_state = raw_state / 3.0

            last_rewards = rewards
            last_winner = game.winner

            episode += 1
            step_in_episode = 0
            done = False  # <<< GANZ WICHTIG: Reset für neue Runde!
        

    pygame.quit()

def draw_info(screen, episode, step_in_episode, agents, last_rewards, winner, font):
    x = 10
    y = 10
    screen.blit(font.render(f"Episode: {episode} | Steps: {step_in_episode}", True, (255, 255, 255)), (x, y))
    y += 25

    for i, agent in enumerate(agents):
        status = "alive" if agent.player_id == (winner or 0) else "dead"
        screen.blit(font.render(
            f"Agent {i+1}: ε={agent.epsilon:.3f} | Reward={last_rewards[i]:.2f} | {status}",
            True,
            (255, 255, 255)
        ), (x, y))
        y += 20

    if winner:
        screen.blit(font.render(f"🏆 Winner: Agent {winner}", True, (255, 215, 0)), (x, y))

if __name__ == "__main__":
    main()
