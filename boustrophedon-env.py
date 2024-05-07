import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the dimensions of the 2D environment
width = 100
height = 50
step = 10

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

# Create a figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], 'bo-')

# Set the axis limits
ax.set_xlim(-1, width+1)
ax.set_ylim(-1, height+1)

# Function to initialize the plot
def init():
    line.set_data([], [])
    return line,

# Function to update the plot for each frame of the animation
def animate(i):
    if i < len(x_points):
        line.set_data(x_points[:i+1], y_points[:i+1])
    else:
        ani.event_source.stop()  # Stop the animation when it reaches the end
    return line,

# Set up the animation
ani = animation.FuncAnimation(fig, animate, frames=len(x_points)+10, init_func=init, blit=True)

# Display the animation
plt.title('Boustrophedon Coverage Path Animation')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
