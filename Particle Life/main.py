import numpy as np
from simulation import Simulation

# Simulation related variables
PARTICLE_COUNT         = 17000
NUMBER_OF_TYPES        = 6
INITIAL_SEED           = 42
FRICTION               = 0.4
MAX_SPEED              = 1000
DELTA_TIME             = 0.1


# Spatial variables - (KEEP MAP SIZE 300)
MAP_SIZE               = 300
MIN_ATTRACTION_RADIUS  = 3
MAX_ATTRACTION_RADIUS  = 20
CELL_SIZE              = MAX_ATTRACTION_RADIUS 

# Display setting (DO NOT CHANGE)
SIDEBAR_WIDTH          = 500 
TOTAL_SCREEN_WIDTH     = MAP_SIZE * 2 + SIDEBAR_WIDTH
TOTAL_SCREEN_HEIGHT    = MAP_SIZE * 2
BUFFER_CLEAR           = True

def main():
 
    
    # We pack alll the constants into a dictionary to keep
    # our code cleaner
    config = {
        "particle_count":    PARTICLE_COUNT,
        "map_size":          MAP_SIZE,
        "min_r":             MIN_ATTRACTION_RADIUS,
        "max_r":             MAX_ATTRACTION_RADIUS,
        "cell_size":         CELL_SIZE,
        "screen_width":      TOTAL_SCREEN_WIDTH,
        "screen_height":     TOTAL_SCREEN_HEIGHT,
        "sidebar_width":     SIDEBAR_WIDTH,
        "initial_num_types": NUMBER_OF_TYPES,
        "initial_seed":      INITIAL_SEED,
        "friction":          FRICTION,
        "delta_time":        DELTA_TIME,
        "max_speed":         MAX_SPEED,
        "buffer_clear":      BUFFER_CLEAR
    }

    # Initialize and run
    sim = Simulation(**config)
    sim.run()

if __name__ == "__main__":
    main()