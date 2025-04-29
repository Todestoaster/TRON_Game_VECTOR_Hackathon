import random

class Agent:
    def __init__(self, player_id):
        self.player_id = player_id  # z.B. 1, 2 oder 3

    def select_action(self, state):
        """
        Wähle eine Aktion basierend auf dem aktuellen Zustand.
        0 = geradeaus, 1 = links, 2 = rechts
        Aktuell: Zufallsaktion
        """
        return random.randint(0, 2)
