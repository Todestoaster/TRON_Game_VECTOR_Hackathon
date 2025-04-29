import numpy as np
import random
import pygame

from .constants import GRID_WIDTH, GRID_HEIGHT, CELL_SIZE

class TronEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
        self.players = []
        self.directions = []
        self.done = False
        self.wins = [0, 0, 0]
        self.death_reason = ["alive" for _ in range(3)]  # <<<<< Neu: Gründe für Tod initialisieren
        self.reset()

    def reset(self):
        self.grid.fill(0)
        self.players = [
            [GRID_HEIGHT // 2, 1],                      # Spieler 1 links
            [GRID_HEIGHT // 2, GRID_WIDTH - 2],          # Spieler 2 rechts
            [1, GRID_WIDTH // 2],                        # Spieler 3 oben
        ]
        self.directions = [
            (0, 1),  # Rechts
            (0, -1), # Links
            (1, 0),  # Runter
        ]
        self.done = False
        self.death_reason = ["alive" for _ in range(3)]  # <<<<< Neu: reset death_reason bei Neustart
        return self.grid.copy()

    def step(self, actions):
        rewards = [0.0, 0.0, 0.0]

        for i, action in enumerate(actions):
            if self.death_reason[i] != "alive":
                continue  # Überspringe tote Spieler
            if action == 0:  # Geradeaus
                pass

            elif action == 1:  # Links drehen
                self.directions[i] = (-self.directions[i][1], self.directions[i][0])
            elif action == 2:  # Rechts drehen
                self.directions[i] = (self.directions[i][1], -self.directions[i][0])

        next_positions = []
        for i in range(3):
            if self.death_reason[i] != "alive":
                next_positions.append(self.players[i])  # Tote bleiben wo sie sind
                continue

            y, x = self.players[i]
            dy, dx = self.directions[i]
            next_positions.append((y + dy, x + dx))

        # Kollisionsprüfung
        deaths = [False, False, False]
        for i, (y, x) in enumerate(next_positions):
            # Wandkollision
            if not (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH):
                deaths[i] = True
                self.death_reason[i] = "wall"
            # Kollision mit sich selbst oder anderen Spielern
            elif self.grid[y, x] != 0:
                if self.grid[y, x] == i + 1:
                    self.death_reason[i] = "self"
                else:
                    self.death_reason[i] = "opponent"
                deaths[i] = True

        for i in range(3):
            if self.death_reason[i] != "alive":
                continue

            if not deaths[i]:
                y, x = next_positions[i]
                self.players[i] = [y, x]
                self.grid[y, x] = i + 1

        if all(deaths) or deaths.count(False) <= 1:
            self.done = True
            if deaths.count(False) == 1:
                winner = deaths.index(False)
                self.wins[winner] += 1

        return self.grid.copy(), rewards, self.done, {}  # Info kann leer bleiben

    def draw(self, screen):
        head_positions = {tuple(player): idx for idx, player in enumerate(self.players)}
        font = pygame.font.SysFont(None, CELL_SIZE)  # Kleine Schrift für X-Zeichen

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y, x] != 0:
                    player_idx = self.grid[y, x] - 1

                    # Standard Körperfarbe
                    if player_idx == 0:
                        color = (0, 200, 0)  # Dunkelgrün
                    elif player_idx == 1:
                        color = (160, 32, 240)  # Dunkellila
                    elif player_idx == 2:
                        color = (30, 144, 255)  # Blau bleibt gleich

                    # Ist dies der Kopf?
                    if (y, x) in head_positions:
                        idx = head_positions[(y, x)]
                        if not self.done or self.death_reason[idx] == "alive":
                            # Spieler lebt → heller Kopf zeichnen
                            if player_idx == 0:
                                color = (144, 238, 144)  # Hellgrün
                            elif player_idx == 1:
                                color = (221, 160, 221)  # Helles Lila
                            elif player_idx == 2:
                                color = (173, 216, 230)  # Hellblau
                        else:
                            # Spieler tot → Körper bleibt, aber extra "X" Zeichen drauf
                            color = (100, 100, 100)  # Graue Farbe für toten Körper

                    pygame.draw.rect(
                        screen,
                        color,
                        (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    )

                    # Wenn es ein toter Kopf ist → kleines "X" zeichnen
                    if (y, x) in head_positions:
                        idx = head_positions[(y, x)]
                        if self.death_reason[idx] != "alive":
                            text = font.render("X", True, (255, 0, 0))  # Rotes X
                            text_rect = text.get_rect(center=(
                                x * CELL_SIZE + CELL_SIZE // 2,
                                y * CELL_SIZE + CELL_SIZE // 2
                            ))
                            screen.blit(text, text_rect)
