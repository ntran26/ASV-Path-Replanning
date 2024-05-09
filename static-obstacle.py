import pygame
import sys

# Initialize Pygame
pygame.init()

# Define the dimensions of the 2D environment
width = 700
height = 500
step = 100
speed = 0.1
lidar_radius = 50

# # Set up the Pygame window
# screen = pygame.display.set_mode((width*1.2, height*1.2))
# pygame.display.set_caption('ASV Path Following Simulation')

# Define colors
white = (255,255,255)
red = (255,0,0)
blue = (0,0,255)
black = (0,0,0)
green = (0,255,0)

# Initialize Pygame
pygame.init()

# Define the original dimensions of the coverage path
original_width = 700
original_height = 500

# Define the larger scaled dimensions for the display
scaled_screen_width = 800  # Adjust as desired
scaled_screen_height = 600  # Adjust as desired

# Calculate the offsets for centering the coverage path on the scaled screen
x_offset = (scaled_screen_width - original_width) / 2
y_offset = (scaled_screen_height - original_height) / 2
# Set up the Pygame window with the larger scaled resolution
screen = pygame.display.set_mode((scaled_screen_width, scaled_screen_height))
pygame.display.set_caption('ASV Path Following Simulation')

# Initialize coverage points and ASV position
x_points = []
y_points = []
current_point_index = 0

# Implement the modified Boustrophedon algorithm
for x in range(0, width+1, step):
    for y in range(0, height+1, step):
        if x % (2*step) == 0:
            x_points.append(x)
            y_points.append(y)
        else:
            x_points.append(x)
            y_points.append(height - y)

# Create ASV class for simulating movement
class ASV:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Scale the coverage path to fit the new screen dimensions while maintaining their relative position on the larger screen
x_points = [x + x_offset for x in x_points]
y_points = [y + y_offset for y in y_points]

# Initialize the ASV at the start point
asv = ASV(x_points[0], y_points[0])

# Generate static obstacles
obstacles = [
    (0, (200, 100), 100, 50),  # Rectangle obstacle at (200, 100) with width 100 and height 50
    (0, (400, 300), 80, 80),   # Rectangle obstacle at (400, 300) with width and height 80
    (1, (600, 200), 30)        # Circle obstacle at (600, 200) with radius 30
]

# Function to display obstacles on the screen
def draw_obstacles():
    for obstacle in obstacles:
        shape = obstacle[0]
        if shape == 0:  # Rectangle obstacle
            pygame.draw.rect(screen, white, pygame.Rect(obstacle[1][0], obstacle[1][1], obstacle[2], obstacle[3]))
        elif shape == 1:  # Circle obstacle
            pygame.draw.circle(screen, white, obstacle[1], obstacle[2])

# Function to check if an obstacle is within the lidar range
def is_obstacle_visible(obstacle):
    obstacle_x, obstacle_y = obstacle[1]
    if obstacle[0] == 0:  # Rectangle obstacle
        # Check if any corner of the rectangle is within the lidar range
        corners = [
            (obstacle_x, obstacle_y),
            (obstacle_x + obstacle[2], obstacle_y),
            (obstacle_x, obstacle_y + obstacle[3]),
            (obstacle_x + obstacle[2], obstacle_y + obstacle[3])
        ]
        for corner in corners:
            dx = corner[0] - asv.x
            dy = corner[1] - asv.y
            distance_to_corner = (dx ** 2 + dy ** 2) ** 0.5
            if distance_to_corner <= lidar_radius:
                return True
    elif obstacle[0] == 1:  # Circle obstacle
        # Check if the center of the circle is within the lidar range
        dx = obstacle_x - asv.x
        dy = obstacle_y - asv.y
        distance_to_center = (dx ** 2 + dy ** 2) ** 0.5
        if distance_to_center <= lidar_radius + obstacle[2]:
            return True
    return False

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(black)  # Clear the screen

    # Draw the coverage path
    for i in range(1, len(x_points)):
        pygame.draw.line(screen, blue, (x_points[i-1], y_points[i-1]), (x_points[i], y_points[i]))
    
    # Draw obstacles that are within the lidar range
    for obstacle in obstacles:
        if is_obstacle_visible(obstacle):
            if obstacle[0] == 0:  # Rectangle obstacle
                pygame.draw.rect(screen, white, pygame.Rect(obstacle[1][0], obstacle[1][1], obstacle[2], obstacle[3]))
            elif obstacle[0] == 1:  # Circle obstacle
                pygame.draw.circle(screen, white, obstacle[1], obstacle[2])

    # Draw obstacles
    draw_obstacles()

    # Draw the lidar sensor circle (representing the ASV's sensor range)
    pygame.draw.circle(screen, green, (asv.x, asv.y), lidar_radius, 1)

    # Draw the ASV
    pygame.draw.circle(screen, red, (asv.x, asv.y), 5)

    # Update ASV's position towards the next point
    if current_point_index < len(x_points):
        next_point = (x_points[current_point_index], y_points[current_point_index])
        dx = next_point[0] - asv.x
        dy = next_point[1] - asv.y
        distance_to_next_point = (dx ** 2 + dy ** 2) ** 0.5
        if distance_to_next_point > speed:
            unit_dx = speed * dx / distance_to_next_point
            unit_dy = speed * dy / distance_to_next_point
            asv.x += unit_dx
            asv.y += unit_dy
        else:
            current_point_index += 1

    pygame.display.flip()  # Update the display

# Quit the game
pygame.quit()
