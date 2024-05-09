import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the dimensions of the 2D environment
width = 50
height = 30
step = 3
speed = 0.01

# Initialize the coverage points as empty lists
x_points = []
y_points = []
path = []

# Implement the modified Boustrophedon algorithm
def boustrophedon_coverage(width, height):
    for x in range(0, width+1, step):
        for y in range(0, height+1, step):
            if x % (2*step) == 0:
                x_points.append(x)
                y_points.append(y)
                path.append((x,y))
            else:
                x_points.append(x)
                y_points.append(height-y)
                path.append((x,height-y))


# Call the function to generate coverage path points
boustrophedon_coverage(width, height)
print(path)

# # Create a figure and axis
# fig, ax = plt.subplots()
# line, = ax.plot([], [], 'bo-')

# # Set the axis limits
# ax.set_xlim(-1, width+1)
# ax.set_ylim(-1, height+1)

# # Function to initialize the plot
# def init():
#     line.set_data([], [])
#     return line,

# Function to update the plot for each frame of the animation
# def animate(i):
#     if i < len(x_points):
#         line.set_data(x_points[:i+1], y_points[:i+1])
#     else:
#         ani.event_source.stop()  # Stop the animation when it reaches the end
#     return line,

# # Set up the animation
# ani = animation.FuncAnimation(fig, animate, frames=len(x_points)+10, init_func=init, blit=True)

# # Display the animation
# plt.title('Boustrophedon Coverage Path Animation')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# plt.show()

# Define the ASV class for simulating movement
class ASV:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Initialize the ASV at the start point
asv = ASV(x_points[0], y_points[0])

# Create a figure and axis with background color
fig, ax = plt.subplots()
fig.patch.set_facecolor('lightgray')  # Setting background color

# Plot the coverage path
ax.plot(x_points, y_points, 'bo-')

# Function to update the ASV's position in the animation
def update(frame):
    if frame < len(x_points):
        # Update ASV's position
        asv.x = x_points[frame]
        asv.y = y_points[frame]
        ax.clear()  # Clear the previous plot
        ax.plot(x_points, y_points, 'bo-')  # Replot the coverage path
        ax.plot(asv.x, asv.y, 'ro', markersize=10)  # Plot ASV's position
        ax.set_xlim(-1, width+1)
        ax.set_ylim(-1, height+1)
        plt.title('ASV Movement on Boustrophedon Coverage Path')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
    else:
        ani.event_source.stop()  # Stop the animation when it reaches the end

# Set up animation
ani = animation.FuncAnimation(fig, update, frames=len(x_points)+1, interval=1/speed, blit=False)

plt.show()

