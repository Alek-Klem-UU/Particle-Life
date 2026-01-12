import numpy as np
from particleManager import ParticleManager
from simulation import Simulation

# All constant variables which influence the simulation

PARTICLE_COUNT = 20000
NUMBER_OF_TYPES = 6
INITIAL_SEED = 42

MAP_SIZE = 400
SIDEBAR_WIDTH = 300 
TOTAL_SCREEN_WIDTH = MAP_SIZE + SIDEBAR_WIDTH
TOTAL_SCREEN_HEIGHT = MAP_SIZE

MIN_ATTRACTION_RADIUS = 3
MAX_ATTRACTION_RADIUS = 21

CELL_SIZE = MAX_ATTRACTION_RADIUS 
GRID_DIMENSION = MAP_SIZE // CELL_SIZE + 1 

FRICTION = 0.4
DELTA_TIME = 0.1

BUFFER_CLEAR = True

def main():
    
    # The main class which handles the simulation

    sim = Simulation(
        particle_count=PARTICLE_COUNT,
        map_size=MAP_SIZE,
        min_r=MIN_ATTRACTION_RADIUS,
        max_r=MAX_ATTRACTION_RADIUS,
        cell_size=CELL_SIZE,
        screen_width=TOTAL_SCREEN_WIDTH,
        screen_height=TOTAL_SCREEN_HEIGHT,
        sidebar_width=SIDEBAR_WIDTH,
        initial_num_types=NUMBER_OF_TYPES,
        initial_seed=INITIAL_SEED,
        friction=FRICTION,
        delta_time=DELTA_TIME,
        buffer_clear=BUFFER_CLEAR
    )
    
    sim.run()

if __name__ == "__main__":
    main()
