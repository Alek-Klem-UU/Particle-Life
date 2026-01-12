import pygame
import numpy as np
import random
from particleManager import ParticleManager
from visualization import draw_simulation, draw_matrix, draw_ui

class Simulation:
    def __init__(self, particle_count, map_size, min_r, max_r, cell_size, 
                 screen_width, screen_height, sidebar_width, initial_num_types, initial_seed, friction, delta_time, buffer_clear):
        
       
        self.particle_count = particle_count
        self.map_size = map_size
        self.min_r = min_r
        self.max_r = max_r
        self.cell_size = cell_size

        self.friction = friction
        self.delta_time = delta_time
        
        self.sidebar_width = sidebar_width
        self.screen_size = (screen_width, screen_height)
        
        
        self.num_types = initial_num_types
        self.current_seed = str(initial_seed)
        self.input_active = False # 
        self.ui_rects = {} 

        self.manager = None
        self.interaction_matrix = None
        self.running = True
        self.screen = None
        self.clock = pygame.time.Clock()
        
        self.pixel_buffer = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        self.buffer_clear = buffer_clear
       
        self.particle_surface = pygame.Surface((map_size, map_size), depth=24)
        self.BG_COLOR = (0, 0, 0)

        self.colors = [
            (251, 150, 72),  
             
            (255, 255, 255),      
            (250, 251, 255),    
            (173, 198, 223),   
            (170, 210, 160),   
            (212, 0, 30),   
            (212, 255, 0), 
            (100, 150, 100), 
            (200, 150, 100), 
            (100, 150, 200), 
      
        ]

       
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Particle Life - Interactive")

       
        self.restart_simulation()

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
            dt=self.delta_time

        )

        # Clear the buffer to remove previous coloured pixels

        self.pixel_buffer[:, :] = np.array(self.BG_COLOR, dtype=np.uint8)
        
  

    def handle_events(self):

        # A function to handle all inputs. This isn't the easiest in Pygame but
        # it works 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # This handles all the mouse clicking
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                
                # Seed input
                if 'seed_input' in self.ui_rects and self.ui_rects['seed_input'].collidepoint(mx, my):
                    self.input_active = True
                else:
                    self.input_active = False
               
                if 'type_minus' in self.ui_rects and self.ui_rects['type_minus'].collidepoint(mx, my):
                    if self.num_types > 1:
                        self.num_types -= 1
               
                if 'type_plus' in self.ui_rects and self.ui_rects['type_plus'].collidepoint(mx, my):
                    if self.num_types < len(self.colors):
                        self.num_types += 1

                if 'buffer_toggle' in self.ui_rects and self.ui_rects['buffer_toggle'].collidepoint(mx, my):
                    self.buffer_clear = not self.buffer_clear
                
            
                if 'rerun' in self.ui_rects and self.ui_rects['rerun'].collidepoint(mx, my):
                    self.restart_simulation()

           
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                if self.input_active:
                    if event.key == pygame.K_RETURN:
                        self.restart_simulation()
                        self.input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.current_seed = self.current_seed[:-1]
                    else:
                        if event.unicode.isdigit():
                            self.current_seed += event.unicode

    def run(self):

        # The main game loop wrapped in a while loop

        while self.running:
            
            # Check for events

            self.handle_events()

            # Clear the screen
            self.screen.fill((0, 0, 0)) 
           
            # Update the particles
            new_positions, types = self.manager.update()
            
          
            
           
            draw_simulation(
                self.screen, 
                self.particle_surface,
                self.pixel_buffer,
                new_positions, 
                types, 
                self.colors, 
                self.map_size,
                self.buffer_clear,
                self.BG_COLOR
            )

         
         
            self.ui_rects, matrix_start_y = draw_ui(
                self.screen, self.map_size, self.sidebar_width, 
                self.current_seed, self.input_active, self.num_types,
                self.buffer_clear
            )
            
            
            draw_matrix(
                self.screen, 
                self.interaction_matrix, 
                self.map_size, 
                self.sidebar_width, 
                self.colors,
                matrix_start_y
            )
            
          
            pygame.display.flip()

            # Limit the framerate to 60 frames per second, to make the experience
            # less variable

            self.clock.tick(60)

            # Set the title of the windows and also display frame rate
            # which correlates to the efficiency of the grid based approach
            # (or the current average density of particles)

            pygame.display.set_caption(f"Particle Life | FPS: {self.clock.get_fps():.1f}")
            
        pygame.quit()