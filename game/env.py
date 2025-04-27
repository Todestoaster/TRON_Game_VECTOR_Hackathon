import random
from game.constants import GRID_WIDTH, GRID_HEIGHT, CELL_SIZE

class Player:
    def __init__(self, id, x, y, direction):
        self.id = id
        self.x = x
        self.y = y
        self.direction = direction
        self.alive = True

    def move(self):
        if not self.alive:
            return
        if self.direction == 'UP':
            self.y = (self.y - 1) % GRID_HEIGHT
        elif self.direction == 'DOWN':
            self.y = (self.y + 1) % GRID_HEIGHT
        elif self.direction == 'LEFT':
            self.x = (self.x - 1) % GRID_WIDTH
        elif self.direction == 'RIGHT':
            self.x = (self.x + 1) % GRID_WIDTH

    def turn_left(self):
        dirs = ['UP', 'LEFT', 'DOWN', 'RIGHT']
        self.direction = dirs[(dirs.index(self.direction) + 1) % 4]

    def turn_right(self):
        dirs = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.direction = dirs[(dirs.index(self.direction) + 1) % 4]

class TronEnv:
    def __init__(self):
        self.wins = [0, 0, 0]  # Siege pro Spieler
        self.reset()

    def reset(self):
        self.field = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.players = []

        # Erzeuge zufällige Startpositionen
        start_positions = []
        while len(start_positions) < 3:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in start_positions:
                start_positions.append((x, y))

        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        for idx, (x, y) in enumerate(start_positions):
            direction = random.choice(directions)
            player = Player(id=idx + 1, x=x, y=y, direction=direction)
            self.players.append(player)
            self.field[y][x] = player.id

        self.head_positions = {player.id: (player.x, player.y) for player in self.players}
        return self.get_observation()

    def move_players(self, players):
        for player in players:
            if player.alive:
                player.move()

    def get_observation(self):
        return self.field

    def step(self, actions):
        for player, action in zip(self.players, actions):
            if player.alive:
                if action == 0:
                    player.turn_left()
                elif action == 1:
                    player.turn_right()

        self.move_players(self.players)

        rewards = []
        self.head_positions.clear()
        for player in self.players:
            if player.alive:
                x, y = player.x, player.y
                if self.field[y][x] is not None:
                    player.alive = False
                    rewards.append(-1.0)
                else:
                    self.field[y][x] = player.id
                    self.head_positions[player.id] = (x, y)
                    rewards.append(0.1)
            else:
                rewards.append(-1.0)

        alive_players = [p for p in self.players if p.alive]
        done = len(alive_players) <= 1

        # Gewinner erfassen
        if done and alive_players:
            winner_id = alive_players[0].id
            self.wins[winner_id - 1] += 1  # Spieler IDs beginnen bei 1

        return self.get_observation(), rewards, done, {}

    def draw(self, screen):
        import pygame

        COLORS = {
            1: (0, 255, 0),
            2: (186, 85, 211),
            3: (30, 144, 255)
        }
        PALE_COLORS = {
            1: (0, 150, 0),
            2: (140, 70, 180),
            3: (20, 100, 200)
        }

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                value = self.field[y][x]
                if value is not None:
                    player_id = abs(value)
                    player_alive = any(p.id == player_id and p.alive for p in self.players)
                    color = COLORS[player_id] if (x, y) in self.head_positions.values() else PALE_COLORS[player_id]
                    
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, color, rect)

                    if not player_alive:
                        pygame.draw.line(screen, (0, 0, 0), (x * CELL_SIZE, y * CELL_SIZE), ((x+1) * CELL_SIZE, (y+1) * CELL_SIZE), 2)
                        pygame.draw.line(screen, (0, 0, 0), ((x+1) * CELL_SIZE, y * CELL_SIZE), (x * CELL_SIZE, (y+1) * CELL_SIZE), 2)
