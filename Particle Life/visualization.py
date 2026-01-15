from numba.cuda import grid
import pygame
import numpy as np
import cv2
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

# ---------------------------------------------------------------------------------
# Helper functions

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

# ---------------------------------------------------------------------------------
# UI drawing functions

def draw_ui(screen, map_size, sidebar_width, seed_text, input_active, num_types, buffer_clear):

    # The main function which 'draws' the whole UI. It returns all UI element
    # as list of rect objects which then can be drawn to the screen using a for loop
   

    x_start = map_size * 2 + 20
    y_start = 25
    width = sidebar_width - 20
    font = pygame.font.Font(None, 24)
    
    ui_rects = {}

    # The seed input
    title = font.render("Seed:", True, (255, 255, 255))
    screen.blit(title, (x_start, y_start))
    
    input_box_color = (200, 200, 200) if input_active else (100, 100, 100)
    input_rect = pygame.Rect(x_start + 60, y_start - 5, 190, 30)
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
    rerun_rect = pygame.Rect(x_start, y_cursor, 250, 40)
    draw_button(screen, rerun_rect, "RERUN / RESTART", font, bg_color=(50, 100, 200))
    ui_rects['rerun'] = rerun_rect
    
    # We return the ui_rects but also the y_position so that we can
    # can use it when drawing the interaction buffer


    return ui_rects, y_cursor + 60

def draw_ui_2(screen, map_size, sidebar_width, fancy):

    # Another UI drawer in another function

    x_start = map_size * 2 + 310
    y_cursor = 20
    
    font = pygame.font.Font(None, 24)
    
    ui_rects = {}

  
    fancy_rect = pygame.Rect(x_start, y_cursor, 120, 30)
    if (fancy): draw_button(screen, fancy_rect, "FANCY", font, bg_color=(170, 51, 106))
    else: draw_button(screen, fancy_rect, "FANCY", font, bg_color=(55, 55, 55))

    ui_rects['fancy'] = fancy_rect
    
    y_cursor += 45

    seed_1 = pygame.Rect(x_start, y_cursor, 120, 30)
    draw_button(screen, seed_1, "Worms", font, bg_color=(50, 100, 200))
    ui_rects['seed_1'] = seed_1
    
    y_cursor += 45

    seed_2 = pygame.Rect(x_start, y_cursor, 120, 30)
    draw_button(screen, seed_2, "Gliders", font, bg_color=(50, 100, 200))
    ui_rects['seed_2'] = seed_2
    
    y_cursor += 45

    seed_3 = pygame.Rect(x_start, y_cursor, 120, 30)
    draw_button(screen, seed_3, "Diggers", font, bg_color=(50, 100, 200))
    ui_rects['seed_3'] = seed_3
    
    y_cursor += 45

    seed_4 = pygame.Rect(x_start, y_cursor, 120, 30)
    draw_button(screen, seed_4, "Cool", font, bg_color=(50, 100, 200))
    ui_rects['seed_4'] = seed_4
    
    y_cursor += 45

 
    
    return ui_rects


def draw_matrix(screen, matrix, map_size, sidebar_width, particle_colors, start_y):
    
    # A function which draws the interaction matrix. For this function we
    # needed the lerp function

    num_types = matrix.shape[0]

    RED = (200, 50, 50)
    WHITE = (255, 255, 255)
    GREEN = (50, 200, 50)

    x_start = map_size * 2
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


# ---------------------------------------------------------------------------------
# Simulation drawing functions (optimized)

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


 
def draw_simulation(screen, surface, pixel_buffer, positions, types, colors, map_size, buffer_clear, fancy, bg_color=(20, 20, 20)):
     
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
    # Lot of experimenting with the values. Don't know why it works now
    # but looks fancier
    if fancy:
        glow_small = cv2.GaussianBlur(pixel_buffer, (55, 55), 1)
        glow_f = glow_small.astype(np.float32) / 255.0
        # Increase glow fall off and intensity with the constant value
        dense_glow_f = np.power(glow_f, 2.13) * 4
        # Clip it again to be in the range (0, 255)
        dense_glow_final = np.clip(dense_glow_f * 255, 0, 255).astype(np.uint8)

        upscaled_glow = cv2.resize(dense_glow_final, (map_size * 2, map_size * 2), interpolation=cv2.INTER_CUBIC)
        pygame.surfarray.blit_array(surface, upscaled_glow)
        screen.blit(surface, (0, 0))

    else:
        upscaled_buffer = cv2.resize(pixel_buffer, (map_size * 2, map_size * 2), interpolation=cv2.INTER_AREA)
        pygame.surfarray.blit_array(surface, upscaled_buffer)
        screen.blit(surface, (0, 0))

    
def draw_graph(screen, checks, time_step, map_size, y_pos, width=450, height=200):

    x_pos = map_size * 2 + 25
   
    if len(checks) < 2:
        return 

    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    
    
    fig.patch.set_facecolor((0.08, 0.08, 0.08))
    ax.set_facecolor((0.15, 0.15, 0.15))
    
    time_axis = np.arange(len(checks))
    ax.plot(time_axis, checks, color='#3296fa', linewidth=2)

    # Make the graph pretty
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white', labelsize=8)
    ax.tick_params(axis='y', colors='white', labelsize=8)
    ax.set_title(f"Neighbour checks", color='white', fontsize=10)
    
    # Convert the graph to pygame compatible surface

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    
    raw_data = renderer.buffer_rgba().tobytes() 
    size = canvas.get_width_height()
    
    graph_surface = pygame.image.fromstring(raw_data, size, "RGBA")
    screen.blit(graph_surface, (x_pos, y_pos))

    plt.close(fig)
    return graph_surface
   

   
  

    






  



