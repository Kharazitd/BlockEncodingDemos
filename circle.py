import numpy as np
import matplotlib.pyplot as plt

def rasterized_circle(N, r, cx, cy):
    """
    Generates a right-angle polygonal approximation of a circle on an N x N grid.
    
    Parameters:
        N  - Grid size (N x N)
        r  - Radius of the circle
        cx - X-coordinate of center
        cy - Y-coordinate of center
    
    Returns:
        A set of boundary points defining the rasterized circle.
    """
    grid = np.zeros((N, N), dtype=int)
    
    # Iterate over all grid points
    for x in range(N):
        for y in range(N):
            # Compute the Euclidean distance to the circle center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # If the distance is within the circle's radius, mark it
            if r - 0.5 <= dist <= r + 0.5:
                grid[x, y] = 1
    
    return grid


def points_inside_circle(N, r, cx, cy):
    """
    Returns a list of all (x, y) grid points inside a circle of radius r.
    
    Parameters:
        N  - Grid size (N x N)
        r  - Radius of the circle
        cx - X-coordinate of center
        cy - Y-coordinate of center
    
    Returns:
        List of (x, y) tuples inside the circle.
    """
    inside_points = []
    outside_points =[]
    bndry_points = []
    for x in range(N):
        for y in range(N):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < r-.5:  # Check if inside circle
                inside_points.append((x, y))
            if dist > r+.5:
                outside_points.append((x,y))
            else:
                bndry_points.append((x,y))
    
    return inside_points, bndry_points, outside_points

    
# Parameters
N = 64+1   # Grid size
r = N//2    # Circle radius
cx, cy = N // 2, N // 2  # Center at grid middle

# Generate rasterized circle
grid = rasterized_circle(N, r, cx, cy)
"""
# Plot
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap='gray_r', origin='lower')
plt.grid(visible=True, color='gray', linestyle='--')
plt.xticks(range(N))
plt.yticks(range(N))
plt.title("Rasterized Circle on Grid")
plt.show()
"""