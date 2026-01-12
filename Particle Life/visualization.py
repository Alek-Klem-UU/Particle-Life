import pygame
import numpy as np
from numba import njit, prange

def lerp(start_color, end_color, t):

    # A function which takes two colors and lerps between them with a strength t

    r = int(start_color[0] + (end_color[0] - start_color[0]) * t)
    g = int(start_color[1] + (end_color[1] - start_color[1]) * t)
    b = int(start_color[2] + (end_color[2] - start_color[2]) * t)
    return (r, g, b)

def draw_button(screen, rect, text, font, bg_color=(100, 100, 100), text_color=(255, 255, 255)):

    # A helper function which you can draw a button with text with

    pygame.draw.rect(screen, bg_color, rect)
    pygame.draw.rect(screen, (200, 200, 200), rect, 2) 

    text = font.render(text, True, text_color)
    text_rect = text.get_rect(center=rect.center)
    
    screen.blit(text, text_rect)

def draw_ui(screen, map_size, sidebar_width, seed_text, input_active, num_types, buffer_clear):

    # The main function which 'draws' the whole UI. It returns all UI element
    # as list of rect objects which then can be drawn to the screen using a for loop
   

    x_start = map_size + 10
    y_start = 10
    width = sidebar_width - 20
    font = pygame.font.Font(None, 24)
    
    ui_rects = {}

    # The seed input
    title = font.render("Seed:", True, (255, 255, 255))
    screen.blit(title, (x_start, y_start))
    
    input_box_color = (200, 200, 200) if input_active else (100, 100, 100)
    input_rect = pygame.Rect(x_start + 60, y_start - 5, width - 60, 30)
    pygame.draw.rect(screen, input_box_color, input_rect)
    pygame.draw.rect(screen, (255, 255, 255), input_rect, 2)
    
    # Check if the text box is clicked and if true change the color and contrast
    input_ = font.render(seed_text, True, (0, 0, 0) if input_active else (255, 255, 255))
    screen.blit(input_, (input_rect.x + 5, input_rect.y + 7))
    
    ui_rects['seed_input'] = input_rect
  
    y_cursor = y_start + 50

    # Some display text
    type_ = font.render(f"Particle Types: {num_types}", True, (255, 255, 255))
    screen.blit(type_, (x_start, y_cursor))
    
    # The '-' button
    minus_rect = pygame.Rect(x_start, y_cursor + 25, 40, 30)
    draw_button(screen, minus_rect, "-", font, bg_color=(150, 50, 50))
    ui_rects['type_minus'] = minus_rect
    
    # The '+' button
    plus_rect = pygame.Rect(x_start + 50, y_cursor + 25, 40, 30)
    draw_button(screen, plus_rect, "+", font, bg_color=(50, 150, 50))
    ui_rects['type_plus'] = plus_rect

    # The toggle for the buffer clear
    buffer_color = (50, 150, 50)
    if not buffer_clear:
        buffer_color = (150, 50, 50)

    plus_rect = pygame.Rect(x_start + 100, y_cursor + 25, 150, 30)
    draw_button(screen, plus_rect, "CLEAR BUFFER", font, bg_color=buffer_color)
    ui_rects['buffer_toggle'] = plus_rect
    
    # The rerun Button
    y_cursor += 70
    rerun_rect = pygame.Rect(x_start, y_cursor, width, 40)
    draw_button(screen, rerun_rect, "RERUN / RESTART", font, bg_color=(50, 100, 200))
    ui_rects['rerun'] = rerun_rect
    
    # We return the ui_rects but also the y_position so that we can
    # can use it when drawing the interaction buffer
    return ui_rects, y_cursor + 60


def draw_matrix(screen, matrix, map_size, sidebar_width, particle_colors, start_y):
    
    # A function which draws the interaction matrix. For this function we
    # needed the lerp function

    num_types = matrix.shape[0]

    RED = (200, 50, 50)
    WHITE = (255, 255, 255)
    GREEN = (50, 200, 50)

    x_start = map_size
    start_y += 20

    MAX_MAG = np.max(np.abs(matrix))
    
    if MAX_MAG < 1e-6: MAX_MAG = 1.0
    
    matrix_width = sidebar_width - 80
    cell_size = min(17, matrix_width // num_types) 
    cell_margin = 1
    matrix_x_offset = x_start + 50 

    for i in range(num_types): 
        for i2 in range(num_types): 
            value = matrix[i, i2]

            # We colour a square green or red depending on
            # the sign of the value
            if value >= 0:
                t = value / MAX_MAG 
                color = lerp(WHITE, GREEN, t)
            else:
                t = -value / MAX_MAG
                color = lerp(RED, WHITE, t)

            

            rect = pygame.Rect(
                matrix_x_offset + i2 * cell_size + cell_margin,
                start_y + i * cell_size + cell_margin,
                cell_size - 2 * cell_margin,
                cell_size - 2 * cell_margin
            )

            pygame.draw.rect(screen, color, rect)
            
           
        # Draw the dots with their corresponding color above the matrix 
        # to see the interaction

        row_dot_center = (matrix_x_offset - 15, start_y + i * cell_size + cell_size // 2)
        pygame.draw.circle(screen, particle_colors[i % len(particle_colors)], row_dot_center, 5)

    # Column Dots
    for i2 in range(num_types):
        col_dot_center = (matrix_x_offset + i2 * cell_size + cell_size // 2, start_y - 10)
        pygame.draw.circle(screen, particle_colors[i2 % len(particle_colors)], col_dot_center, 5)



@njit(parallel=True, cache=True)
def draw_particles_fast(pixel_buffer, positions, types, colors, num_colors):

    # A function which uses multithreading to draw each pixel seperatly

    N = len(positions)
 
   
    for i in prange(N):
        x = positions[i, 0]
        y = positions[i, 1]
        p_type = types[i]
        
        x = int(x)
        y = int(y)
        
        color = colors[p_type % num_colors]
        
        pixel_buffer[y, x, 0] = color[0]
        pixel_buffer[y, x, 1] = color[1]
        pixel_buffer[y, x, 2] = color[2]


        
       


def draw_simulation(screen, surface, pixel_buffer, positions, types, colors, map_size, buffer_clear, bg_color=(20, 20, 20)):
     
    # The function which optionally clears the screen and then draws all particles to the surface
    # and blits it onto the screen

    if (buffer_clear):
        pixel_buffer[:, :] = np.array(bg_color, dtype=np.uint8)
    
    colors_np = np.array(colors, dtype=np.uint8)
    num_colors = len(colors)
    
    draw_particles_fast(
        pixel_buffer, 
        positions, 
        types, 
        colors_np, 
        num_colors
    )

    pygame.surfarray.blit_array(surface, pixel_buffer)
    screen.blit(surface, (0, 0))
    