import numpy as np
from numba import int16, njit, int32, float64, prange


@njit(cache=True)
def calculate_force(dist, R_min, R_max, alpha):
   
    # A piece wise function which repels particles which come to close to each other
    # when between min and max range is like a hill having strongest value in the middle
    # and weakening towards the real values and outside of that it returns 0. 

    # We use the sinoid version because its smoother then using ABS which creates
    # a very pointy and pyramid like shape. Off course this is just a preference with
    # no real value except smoother movement. 

    if dist < R_min:
        return alpha * (1 - dist / R_min)
    elif dist < R_max:
        normalized_dist = (dist - R_min) / (R_max - R_min)
        return alpha * np.sin(np.pi * normalized_dist)
    
    return 0.0

@njit(cache=True)
def map_particles_to_cells(pos, N, cell_size, grid_dim, grid_indices):
    # Maps all particles indexes to their corresponding grid position
    for particle_id in range(N):
      
        cell_x = int32(pos[particle_id, 0] / cell_size)
        cell_y = int32(pos[particle_id, 1] / cell_size)
        
        grid_indices[particle_id] = cell_y * grid_dim + cell_x

@njit(parallel=True, cache=True)
def update_particles_kernel_cpu(positions, velocities, types, N, R_min, R_max, matrix, friction, dt, map_size, 
                                grid_pos, grid_counts, cell_size, grid_dim):

    # We calculate the 'real' max distance once to check
    # Normally you would check if sqrt(dis * dis) < sqrt(max_dis) but
    # sqrt is a heavy operation and redundant
    

    # loop over each particle
    for particle_id in prange(N):
        
        f_x = 0.0
        f_y = 0.0
        
        pos_x = positions[particle_id, 0]
        pos_y = positions[particle_id, 1]

        particle_type = types[particle_id]

        cell_x = int32(pos_x / cell_size)
        cell_y = int32(pos_y / cell_size)
        
        # loop over all cells in a 3x3 neighborhood
        for i in range(-1, 2):
           

            for i2 in range(-1, 2):
                neighbor_cell_x = cell_x + i
                neighbor_cell_x = neighbor_cell_x % grid_dim

                neighbor_cell_y = cell_y + i2
                neighbor_cell_y = neighbor_cell_y % grid_dim
                
                cell_id = neighbor_cell_y * grid_dim + neighbor_cell_x
                
                start_i = grid_pos[cell_id]
                end_i = start_i + grid_counts[cell_id]

               
                for i3 in range(start_i, end_i):
                    # Check if particle interact with itself
                    if particle_id == i3: 
                        continue

                    pos_x_2, pos_y_2 = positions[i3, 0], positions[i3, 1]
                    type_2 = types[i3]

                    d_x = pos_x_2 - pos_x
                    d_y = pos_y_2 - pos_y
                    
                    # On edges we still calculate neighbors using wrapping
                    # boundaries. To actually get the correct distance we subtract
                    # or add the map_size
                    if d_x > map_size / 2.0:
                        d_x -= map_size 
                    elif d_x < -map_size / 2.0:
                        d_x += map_size 
                    if d_y > map_size / 2.0:
                        d_y -= map_size 
                    elif d_y < -map_size / 2.0:
                        d_y += map_size 
                    # Calculate the distance squared
                    dist = d_x*d_x + d_y*d_y
                    
                    # If outside MAX_ATTRACTION_RADIUS, skip
                    if dist > R_max * R_max + 1:
                        continue

                    # Get the actual distance for the correct force calculations
                    dist = np.sqrt(dist)
                   
                    # Get the type of relation between two types
                    interaction = matrix[type_2, particle_type]

                    # Calculate the magnitude of the force
                    force = calculate_force(dist, R_min, R_max, interaction)

                    # We apply the force in the right direction
                    if dist > 1e-6: 
                        f_x += force * (d_x / dist) # <- normalized 
                        f_y += force * (d_y / dist) # <- normalized

       
        # We update the velocities using euler intergrations and our found force
        # We also apply friction to take energy out of the system
        velocities[particle_id, 0] = (velocities[particle_id, 0] + f_x * dt) * friction
        velocities[particle_id, 1] = (velocities[particle_id, 1] + f_y * dt) * friction

        # Stablize the simulation though max speed
        vx = velocities[particle_id, 0]
        vy = velocities[particle_id, 1]
        current_speed_sq = vx*vx + vy*vy
        max_speed = 10
        if current_speed_sq > max_speed * max_speed:
            current_speed = np.sqrt(current_speed_sq)
            scale_factor = max_speed / current_speed
            velocities[particle_id, 0] *= scale_factor
            velocities[particle_id, 1] *= scale_factor

        # Update the positons and apply wrapping though the '%' operator
        positions[particle_id, 0] += velocities[particle_id, 0] * dt
        positions[particle_id, 1] += velocities[particle_id, 1] * dt

        positions[particle_id, 0] = positions[particle_id, 0] % map_size
        positions[particle_id, 1] = positions[particle_id, 1] % map_size


class ParticleManager:
    
    
    def __init__(self, particle_count, map_size, num_types, min_r, max_r, cell_size, friction, dt, interaction_matrix):
        self.particle_count = particle_count
        self.num_types = num_types
        self.map_size = map_size
        
        # These influence the simulation
        self.R_min = min_r
        self.R_max = max_r
        self.friction = friction  
        self.dt = dt   
        self.matrix = interaction_matrix.astype(np.float64)

        # Spatial information
        self.cell_size = cell_size
        self.GRID_DIM = map_size // cell_size + 1
        self.GRID_SIZE = self.GRID_DIM * self.GRID_DIM
        
        # Initialize the position, velocity and types array (last one at random)
        self.pos = np.random.uniform(0, map_size, size=(self.particle_count, 2)).astype(np.float64)
        self.vel = np.zeros((self.particle_count, 2), dtype=np.float64)
        self.types = np.random.randint(0, num_types, size=self.particle_count, dtype=np.int32)
        
        # Maps particles to which grid cell their in
        self.grid_indices = np.zeros(self.particle_count, dtype=np.int32)

        # The amount of particles in each cell
        self.grid_counts = np.zeros(self.GRID_SIZE, dtype=np.int32) 

        # The start position of all cells in the sorted array
        self.grid_pos = np.zeros(self.GRID_SIZE, dtype=np.int32)  

    def update_grid(self):
       
        # We map the particles to there corresponding cells
        map_particles_to_cells(
            self.pos, self.particle_count, self.cell_size, self.GRID_DIM, self.grid_indices
        )
        
        # We use np.argsort. This doesn't return the sorted array but
        # the INDICES that would sort the array. So all id's with cell index
        # 0 start in this array, cell index 1 after that, etc.
        sorted_indices = np.argsort(self.grid_indices)
        
        # We need to reorder our all ready initialized arrays using our
        # new indices
        self.pos = self.pos[sorted_indices]
        self.vel = self.vel[sorted_indices]
        self.types = self.types[sorted_indices]
        self.grid_indices = self.grid_indices[sorted_indices]
        
        # Initialize all elements in the grid_counts and grid_pos to 0
        self.grid_counts[:] = 0
        self.grid_pos[:] = 0
        
        # We use np.unique to count to retrieve all cells which contain particles
        # and their counts (amount of particles)
        unique_cells, counts = np.unique(self.grid_indices, return_counts=True)
        # Use the unique_cells (array) as indices and map to them the particle counts
        self.grid_counts[unique_cells] = counts
        
        # We use cumsum to set the correct starting positions for each cells
        # If we have the particle counts: [ 5, 10, 3, 7, ...] 
        # We get the corresponding starting positions: np.cumsum(self.grid_counts) -> [5, 15, 18, 25, ...] 
        self.grid_pos[1:] = np.cumsum(self.grid_counts)[:-1]

    def update(self):
    
        # This one one simulation step

        # We update the spatial grid. This takes compute time but makes
        # the main loop magnitudes faster
        self.update_grid()
        
        # We run the main physics kernel on the CPU using parallelized Numba
        update_particles_kernel_cpu(
            self.pos, self.vel, self.types, self.particle_count, 
            self.R_min, self.R_max, self.matrix, self.friction, self.dt, self.map_size,
            self.grid_pos, self.grid_counts, self.cell_size, self.GRID_DIM
        )
        
        return self.pos, self.types