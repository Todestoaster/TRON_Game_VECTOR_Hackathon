import numpy as np
from game.player import Player
from game.config import CELL_SIZE, FIELD_WIDTH, FIELD_HEIGHT
import pygame


class Game:
    
    def __init__(self, screen):
        self.screen = screen
        self.players = []
        self.running = True
        self.winner = None
        self.create_players()

    def create_players(self):
        player1 = Player((255, 0, 0), (100, 100), "RIGHT")
        player2 = Player((0, 255, 0), (700, 100), "LEFT")
        player3 = Player((0, 0, 255), (400, 500), "UP")
        self.players = [player1, player2, player3]

    def step(self, actions):
        for player, action in zip(self.players, actions):
            if player.alive:
                player.change_direction(action)

        all_trails = set()
        for player in self.players:
            all_trails.update(player.trail)

        for player in self.players:
            if player.alive:
                player.move()

        for player in self.players:
            if player.alive and tuple(player.position) in all_trails:
                player.alive = False

        state = self.get_game_state()
        rewards = self.get_rewards()

        alive_players = [player for player in self.players if player.alive]
        done = len(alive_players) <= 1

        if done and len(alive_players) == 1:
            self.winner = self.players.index(alive_players[0]) + 1

        return state, rewards, done


    def get_game_state(self):
        state = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=int)

        for idx, player in enumerate(self.players):
            player_id = idx + 1
            for x, y in player.trail:
                col = x // CELL_SIZE
                row = y // CELL_SIZE
                if 0 <= row < FIELD_HEIGHT and 0 <= col < FIELD_WIDTH:
                    state[row][col] = player_id

        return state

    def get_rewards(self):
        rewards = []
        for player in self.players:
            if player.alive:
                rewards.append(0.1)  # Überleben: kleine positive Belohnung
            else:
                rewards.append(-1.0)  # Tod: negative Belohnung
        return rewards

    def reset(self):
        self.players.clear()
        self.running = True
        self.winner = None
        self.create_players()
        return self.get_game_state()

    def update(self):
        pass  # update() wird nicht mehr direkt benutzt

    def draw(self):
        self.screen.fill((0, 0, 0))

        for player in self.players:
            player.draw(self.screen)
        
        self.draw_grid()
    
    def draw_grid(self):
        for x in range(0, 800, CELL_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, 600))
        for y in range(0, 600, CELL_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (800, y))

    
        

