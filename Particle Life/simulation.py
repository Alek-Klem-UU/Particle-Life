from numba.misc.appdirs import unicode
import pygame
import numpy as np
import random
from particleManager import ParticleManager
from visualization import draw_simulation, draw_matrix, draw_ui

class Simulation:
    def __init__(self, **kwargs):
        # Use passed variables to configure the simulation
        # and initialize graphical assests
        self.load_config(kwargs)
        self.init_graphics_assets()
        
        # Simulation state and objects related to the simulation
        self.running = True
        self.input_active = False
        self.ui_rects = {}
        self.interaction_matrix = None
        self.manager = None
        
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        
        # Start the simulation
        self.restart_simulation()

    def load_config(self, cfg):
        # Use the configuration variables to set all local variables
        self.particle_count = cfg.get('particle_count')
        self.map_size       = cfg.get('map_size')
        self.min_r          = cfg.get('min_r')
        self.max_r          = cfg.get('max_r')
        self.cell_size      = cfg.get('cell_size')
        self.friction       = cfg.get('friction')
        self.delta_time     = cfg.get('delta_time')
        self.max_speed      = cfg.get('max_speed')
        self.sidebar_width  = cfg.get('sidebar_width')
        self.screen_size    = (cfg.get('screen_width'), cfg.get('screen_height'))
        self.num_types      = cfg.get('initial_num_types')
        self.current_seed   = str(cfg.get('initial_seed'))
        self.buffer_clear   = cfg.get('buffer_clear')

    def init_graphics_assets(self):
        """Initializes all surfaces, buffers, and color palettes."""
        self.BG_COLOR = (0, 0, 0)
        self.pixel_buffer = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.particle_surface = pygame.Surface((self.map_size, self.map_size), depth=24)
        
        self.colors = [
            (251, 150, 72), (255, 255, 255), (250, 251, 255),
            (173, 198, 223), (170, 210, 160), (212, 0, 30),
            (212, 255, 0), (100, 150, 100), (200, 150, 100), (100, 150, 200)
        ]

   
    # --- Core Logic ---

    def restart_simulation(self):
        # Use the current seed to restart the simulation,
        # we check if the inputted seed is an integer
        try:
            seed_val = int(self.current_seed)
        except ValueError:
            seed_val = random.randint(0, 99999)
            self.current_seed = str(seed_val)
        
        np.random.seed(seed_val)
        random.seed(seed_val)
        
        # Using the seed initialize a new interaction matrix and Particle Manager
        self.interaction_matrix = np.random.uniform(-1.0, 1.0, (self.num_types, self.num_types)) * 1.5
        
        self.manager = ParticleManager(
            particle_count=self.particle_count,
            map_size=self.map_size,
            num_types=self.num_types,
            min_r=self.min_r,
            max_r=self.max_r,
            cell_size=self.cell_size,
            interaction_matrix=self.interaction_matrix,
            friction=self.friction,
            dt=self.delta_time,
            max_speed=self.max_speed
        )

        # Clear the buffer to remove previous coloured pixels
        self.pixel_buffer[:, :] = np.array(self.BG_COLOR, dtype=np.uint8)

    # --------------------------------------------------------------------------------------
    # Interactions

    def handle_events(self):
        # This function handles all the events, split into
        # mouse clicks and keyboard clicks.
        # This isn't the easiest in Pygame but it works 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.on_click(event.pos)
                
            elif event.type == pygame.KEYDOWN:
                self.on_keypress(event)

    def on_click(self, pos):
        mx, my = pos
        
        # A function which controls the actions of the UI buttons
        if self.ui_rects.get('seed_input', pygame.Rect(0,0,0,0)).collidepoint(mx, my):
            self.input_active = True
        else:
            self.input_active = False
            
        if self.ui_rects.get('type_minus', pygame.Rect(0,0,0,0)).collidepoint(mx, my):
            self.num_types = max(1, self.num_types - 1)
        
        if self.ui_rects.get('type_plus', pygame.Rect(0,0,0,0)).collidepoint(mx, my):
            self.num_types = min(len(self.colors), self.num_types + 1)

        if self.ui_rects.get('buffer_toggle', pygame.Rect(0,0,0,0)).collidepoint(mx, my):
            self.buffer_clear = not self.buffer_clear
        
        if self.ui_rects.get('rerun', pygame.Rect(0,0,0,0)).collidepoint(mx, my):
            self.restart_simulation()

    def on_keypress(self, event):

         # A function to handle all keyboard inputs. 

        if event.key == pygame.K_ESCAPE:
            self.running = False
        
        if self.input_active:
            if event.key == pygame.K_RETURN:
                self.restart_simulation()
                self.input_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.current_seed = self.current_seed[:-1]
            elif event.unicode.isdigit():
                self.current_seed += event.unicode

    # --------------------------------------------------------------------------------------
    # Main game loop

    def run(self):
        # The main game loop wrapped in a while loop
        while self.running:
            # Check for events
            self.handle_events()
            
            # Update particles
            positions, types = self.manager.update()
            
            # Render a new frame
            self.render_frame(positions, types)
            
            # Limit framerate and display it as
            # the window title
            self.clock.tick(60)
            pygame.display.set_caption(f"Particle Life | FPS: {self.clock.get_fps():.1f}")
            
        pygame.quit()

    def render_frame(self, positions, types):
        # Draws a new frame in correct order

        # 0. Clear the screen
        self.screen.fill((0, 0, 0)) 
        
        # 1. Particles
        draw_simulation(
            self.screen, self.particle_surface, self.pixel_buffer,
            positions, types, self.colors, self.map_size,
            self.buffer_clear, self.BG_COLOR
        )
        
        # 2. UI Panel
        self.ui_rects, matrix_y = draw_ui(
            self.screen, self.map_size, self.sidebar_width, 
            self.current_seed, self.input_active, self.num_types,
            self.buffer_clear
        )
        
        # 3. Interaction Matrix
        draw_matrix(
            self.screen, self.interaction_matrix, self.map_size, 
            self.sidebar_width, self.colors, matrix_y
        )
        
        pygame.display.flip()