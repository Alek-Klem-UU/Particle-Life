import numpy as np
from numba import njit, int32, float64, prange

#

@njit(cache=True)
def calculate_force(dist, R_min, R_max, alpha):
    
    # Piece-wise function:
    # - Repels particles if dist < R_min.
    # - Attracts/Repels if R_min < dist < R_max.
    # - Returns 0 otherwise.
    
    if dist < R_min:
        return alpha * (1.0 - dist / R_min)
    elif dist < R_max:

        # We use the sinoid version because its smoother then using ABS which creates
        # a very pointy and pyramid like shape. Off course this is just a preference with
        # no real value except smoother movement. 

        normalized_dist = (dist - R_min) / (R_max - R_min)
        return alpha * np.sin(np.pi * normalized_dist)
    
    return 0.0

@njit(parallel=True, cache=True)
def update_particles(
    positions, velocities, types, N, R_min, R_max, matrix, 
    friction, dt, max_speed, map_size, grid_pos, grid_counts, cell_size, grid_dim
):
    # Main simulation step: Spatial hashing lookup + Force accumulation + Integration
    
    # Boundary threshold for wrapping logic
    half_map = map_size / 2.0
    max_dist_sq = R_max * R_max + 1.0

    for i in prange(N):
        f_x, f_y = 0.0, 0.0
        pos_x, pos_y = positions[i]
        p_type = types[i]

        # Determine current cell coordinates
        cell_x = int32(pos_x / cell_size)
        cell_y = int32(pos_y / cell_size)
        
        # Search the 3x3 neighborhood of cells
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                # Wrapped grid coordinates
                nx = (cell_x + dx) % grid_dim
                ny = (cell_y + dy) % grid_dim
                
                cell_id = ny * grid_dim + nx
                start_idx = grid_pos[cell_id]
                end_idx = start_idx + grid_counts[cell_id]

                # Check particles within the cell
                for i2 in range(start_idx, end_idx):
                    if i == i2: 
                        continue

                    #  Relative vector
                    dx_val = positions[i2, 0] - pos_x
                    dy_val = positions[i2, 1] - pos_y

                    # Toroidal distance correction
                    if dx_val > half_map:     dx_val -= map_size 
                    elif dx_val < -half_map:  dx_val += map_size 
                    if dy_val > half_map:     dy_val -= map_size 
                    elif dy_val < -half_map:  dy_val += map_size 

                    dist_sq = dx_val**2 + dy_val**2
                    
                    if dist_sq > max_dist_sq or dist_sq < 1e-9:
                        continue

                    dist = np.sqrt(dist_sq)
                    interaction = matrix[types[i2], p_type]
                    force = calculate_force(dist, R_min, R_max, interaction)

                    # Accumulate normalized force vectors
                    f_x += force * (dx_val / dist)
                    f_y += force * (dy_val / dist)

      
        vx = (velocities[i, 0] + f_x * dt) * friction
        vy = (velocities[i, 1] + f_y * dt) * friction

        # Limit speeed 
       
        speed_sq = vx**2 + vy**2
        if speed_sq > max_speed**2:
            scale = max_speed / np.sqrt(speed_sq)
            vx *= scale
            vy *= scale

        velocities[i, 0], velocities[i, 1] = vx, vy

        # Update position with tordial wrapping
        positions[i, 0] = (pos_x + vx * dt) % map_size
        positions[i, 1] = (pos_y + vy * dt) % map_size



@njit(cache=True)
def map_particles_to_cells(pos, N, cell_size, grid_dim, grid_indices):
    # Maps all particle indices to their corresponding  grid index
    for i in range(N):
        cx = int32(pos[i, 0] / cell_size)
        cy = int32(pos[i, 1] / cell_size)
        grid_indices[i] = cy * grid_dim + cx


class ParticleManager:
    def __init__(self, **kwargs):
        # Configure all variables
        self.particle_count = kwargs.get('particle_count')
        self.num_types      = kwargs.get('num_types')
        self.map_size       = kwargs.get('map_size')
        self.R_min          = kwargs.get('min_r')
        self.R_max          = kwargs.get('max_r')
        self.friction       = kwargs.get('friction')
        self.dt             = kwargs.get('dt')
        self.max_speed      = kwargs.get('max_speed')
        self.matrix         = kwargs.get('interaction_matrix').astype(np.float64)

        # Spatial grid
        self.cell_size = kwargs.get('cell_size')
        self.GRID_DIM  = self.map_size // self.cell_size + 1
        self.GRID_SIZE = self.GRID_DIM ** 2
        
        # Particle state arrays
        self.pos   = np.random.uniform(0, self.map_size, (self.particle_count, 2)).astype(np.float64)
        self.vel   = np.zeros((self.particle_count, 2), dtype=np.float64)
        self.types = np.random.randint(0, self.num_types, self.particle_count, dtype=np.int32)
        
        # Spatial grid buffers
        self.grid_indices = np.zeros(self.particle_count, dtype=np.int32)
        self.grid_counts  = np.zeros(self.GRID_SIZE, dtype=np.int32) 
        self.grid_pos     = np.zeros(self.GRID_SIZE, dtype=np.int32)  

    def update_grid(self):
        # Reorders particles by grid cell
        map_particles_to_cells(self.pos, self.particle_count, self.cell_size, self.GRID_DIM, self.grid_indices)
        
        # Sort indices to group particles by cell
        sorted_indices = np.argsort(self.grid_indices)
        
        # Reorder all simulation arrays
        self.pos = self.pos[sorted_indices]
        self.vel = self.vel[sorted_indices]
        self.types = self.types[sorted_indices]
        self.grid_indices = self.grid_indices[sorted_indices]
        
       
        self.grid_counts[:] = 0
        self.grid_pos[:] = 0
        
        unique_cells, counts = np.unique(self.grid_indices, return_counts=True)
        self.grid_counts[unique_cells] = counts
        
        # Calculate offsets for each cell
        self.grid_pos[1:] = np.cumsum(self.grid_counts)[:-1]

    def update(self):
        # Update the whole simulation for one frame

        # Initialize the grid
        self.update_grid()
        
        # Compute the physics
        update_particles(
            self.pos, self.vel, self.types, self.particle_count, 
            self.R_min, self.R_max, self.matrix, self.friction, self.dt, self.max_speed, self.map_size,
            self.grid_pos, self.grid_counts, self.cell_size, self.GRID_DIM
        )
        
        return self.pos, self.types