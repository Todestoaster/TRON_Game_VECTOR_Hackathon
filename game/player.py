from game.config import CELL_SIZE
import pygame

class Player:
    MOVE_DISTANCE = CELL_SIZE
    DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]  # <- Diese Zeile darf NICHT fehlen und muss hier sein!

    def __init__(self, color, start_pos, direction):
        self.color = color
        self.position = list(start_pos)
        self.direction = direction
        self.trail = [tuple(start_pos)]
        self.alive = True

    def move(self):
        if not self.alive:
            return

        if self.direction == "RIGHT":
            self.position[0] += self.MOVE_DISTANCE
        elif self.direction == "LEFT":
            self.position[0] -= self.MOVE_DISTANCE
        elif self.direction == "UP":
            self.position[1] -= self.MOVE_DISTANCE
        elif self.direction == "DOWN":
            self.position[1] += self.MOVE_DISTANCE

        self.position[0] %= 800
        self.position[1] %= 600

        self.trail.append(tuple(self.position))

    def change_direction(self, action):
        idx = self.DIRECTIONS.index(self.direction)
        if action == 1:  # Links
            idx = (idx - 1) % 4
        elif action == 2:  # Rechts
            idx = (idx + 1) % 4
        # 0 = geradeaus (keine Änderung)
        self.direction = self.DIRECTIONS[idx]

    def draw(self, screen):
        if not self.trail:
            return

        # Trail zuerst
        trail_color = tuple(min(c + 100, 255) for c in self.color)  # Hellerer Farbton für Trail
        for point in self.trail[:-1]:  # Alle außer Kopf
            pygame.draw.rect(screen, trail_color, (point[0], point[1], CELL_SIZE, CELL_SIZE))

        # Kopf (immer starke Originalfarbe)
        head = self.trail[-1]
        pygame.draw.rect(screen, self.color, (head[0], head[1], CELL_SIZE, CELL_SIZE))

        if not self.alive:
            # Wenn tot: auf alle Trailpunkte ein schwarzes X malen
            for point in self.trail:
                x, y = point
                center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)
                offset = CELL_SIZE // 2 - 2
                pygame.draw.line(screen, (0, 0, 0), (center[0] - offset, center[1] - offset), (center[0] + offset, center[1] + offset), 2)
                pygame.draw.line(screen, (0, 0, 0), (center[0] - offset, center[1] + offset), (center[0] + offset, center[1] - offset), 2)
