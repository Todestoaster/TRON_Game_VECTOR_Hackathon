import pygame
import random
from game.player import Player
from game.world import World
from game.constants import GRID_WIDTH, GRID_HEIGHT, CELL_SIZE

pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Lightcycle Simulation")
clock = pygame.time.Clock()

players = [
    Player(1, 5, 5, 'RIGHT'),
    Player(2, 15, 5, 'LEFT'),
    Player(3, 10, 15, 'UP')
]

world = World()
running = True

while running:
    clock.tick(10)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for player in players:
        if player.alive:
            action = random.choice(['straight', 'left', 'right'])
            if action == 'left':
                player.turn_left()
            elif action == 'right':
                player.turn_right()

    world.head_positions.clear()
    world.move_players(players)
    world.draw(screen)
    pygame.display.flip()

    alive = [p for p in players if p.alive]
    if len(alive) <= 1:
        pygame.time.wait(2000)
        running = False

pygame.quit()