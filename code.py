"""
Title: 2D generalised stereo disparity flow implementation for displacement estimation
Final Version - Updated 2025.04.22 (had issues with boundary conditions)
Name: Rahul Rustagi
GTID: 904024521
"""

import numpy as np
from matplotlib import pyplot as plt
from time import time
import cv2

### Helper functions section ###

def get_interpolated(image, x, y):
    """Bilinear interpolation with zero-pad for OOB 
    (Tried different padding first, zero worked best)
    """
    H, W = image.shape
    x_floor = np.floor(x).astype('int32')
    y_floor = np.floor(y).astype('int32')
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    
    # Weights calculation
    dx = x - x_floor
    dy = y - y_floor
    wt_tl = (1-dx)*(1-dy)  # top-left weight
    wt_tr = dx*(1-dy)
    wt_bl = (1-dx)*dy
    wt_br = dx*dy
    
    # Initialize with zeros (handles OOB automatically)
    interpolated = np.zeros_like(x)
    
    # Check each corner with boundary constraints
    # (This mask approach was faster than np.clip)
    valid_tl = (x_floor >= 0) & (x_floor < W) & (y_floor >= 0) & (y_floor < H)
    interpolated[valid_tl] += wt_tl[valid_tl] * image[y_floor[valid_tl], x_floor[valid_tl]]
    
    valid_tr = (x_ceil >= 0) & (x_ceil < W) & (y_floor >= 0) & (y_floor < H)
    interpolated[valid_tr] += wt_tr[valid_tr] * image[y_floor[valid_tr], x_ceil[valid_tr]]
    
    valid_bl = (x_floor >= 0) & (x_floor < W) & (y_ceil >= 0) & (y_ceil < H)
    interpolated[valid_bl] += wt_bl[valid_bl] * image[y_ceil[valid_bl], x_floor[valid_bl]]
    
    valid_br = (x_ceil >= 0) & (x_ceil < W) & (y_ceil >= 0) & (y_ceil < H)
    interpolated[valid_br] += wt_br[valid_br] * image[y_ceil[valid_br], x_ceil[valid_br]]
    
    return interpolated

def apply_laplacian(matrix):
    """5-point stencil laplacian with reflection BCs
    (Experimented with zero-padding first, reflection gave better edge behavior)

    The formulation is: ∇²f = f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) -4f(x,y)
    """
    # Add reflective padding to handle borders
    padded = np.pad(matrix, pad_width=1, mode='reflect')
    
    # Stencil implementation
    laplacian = (padded[:-2, 1:-1]   # i-1,j
               + padded[2:, 1:-1]    # i+1,j
               + padded[1:-1, :-2]   # i,j-1
               + padded[1:-1, 2:]    # i,j+1
               - 4*padded[1:-1, 1:-1])
    return laplacian

def calculate_gradients(img):
    """Central difference gradients with simple edge handling
    (Tried forward diff first but central worked better)
    """
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)
    
    # X-direction (columns)
    grad_x[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    # Y-direction (rows)
    grad_y[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    
    # Handle edges by replication (simple approach)
    grad_x[:, 0] = grad_x[:, 1]
    grad_x[:, -1] = grad_x[:, -2]
    grad_y[0, :] = grad_y[1, :]
    grad_y[-1, :] = grad_y[-2, :]
    
    return grad_x, grad_y

### Main algorithm ###

# Experiment parameters
GRID_SIZE = 64  # Changed from 128 to 64 for faster testing
LAMBDA = 0.5  # Regularization parameter
MAX_ITERS = 20  # Max iterations (usually converges faster)
TRUE_SHIFT_X = 0  # Ground truth X shift
TRUE_SHIFT_Y = 10   # Ground truth Y shift

# I1 = cv2.imread('/home/rrustagi7/Desktop/view0_1.png', cv2.IMREAD_GRAYSCALE) / 255
# I2 = cv2.imread('/home/rrustagi7/Desktop/view6_1.png', cv2.IMREAD_GRAYSCALE) /255
# # cv2.imshow('I1', I1)
# # Resize images to N x N
# I1 = cv2.resize(I1, (GRID_SIZE, GRID_SIZE))
# I2 = cv2.resize(I2, (GRID_SIZE, GRID_SIZE))

# base_img = I1
# shifted_img = I2

# Create test images with square pattern
base_img = np.zeros((GRID_SIZE, GRID_SIZE))
square_coords = slice(20,40), slice(20,40)
# base_img[square_coords] = 1.0
base_img[20:30, 20:40] = 1.0  # Centered box
base_img[35:45, 20:40] = 1.0  # Centered box
base_img[5:15, 10:20] = 1.0  # Centered box
# Shifted image (simulate translation)
shifted_img = np.roll(base_img, TRUE_SHIFT_Y, axis=0)
shifted_img = np.roll(shifted_img, TRUE_SHIFT_X, axis=1)

# Precompute gradients of shifted image
grad_x, grad_y = calculate_gradients(shifted_img)

# Initialize displacement fields (a=horizontal, b=vertical)
disp_x = np.zeros_like(base_img)
disp_y = np.zeros_like(base_img)

a_history = []  # Store horizontal displacement history
b_history = []  # Store vertical displacement history
E_history = []  # Store energy history
valid_points = [] # Store valid points for visualization


# Create coordinate grids (switched to meshgrid for readability)
col_grid, row_grid = np.meshgrid(np.arange(GRID_SIZE),

                                 np.arange(GRID_SIZE))
# Convert to float for displacement updates
col_grid = col_grid.astype('float64')
row_grid = row_grid.astype('float64')

# Optimization loop
start_time = time()

for iteration in range(MAX_ITERS):
    # Current displaced coordinates
    warped_cols = col_grid + disp_x
    warped_rows = row_grid + disp_y
    
    # Interpolate shifted image and gradients
    I2_warped = get_interpolated(shifted_img, warped_cols, warped_rows)
    I2_wx = get_interpolated(grad_x, warped_cols, warped_rows)
    I2_wy = get_interpolated(grad_y, warped_cols, warped_rows)

    # Calculate valid points percentage
    valid_mask = ((warped_cols >= 0) & (warped_cols < GRID_SIZE) & 
                  (warped_rows >= 0) & (warped_rows < GRID_SIZE))
    valid_count = np.mean(valid_mask) * 100  # Percentage
    valid_points.append(valid_count)
    
    # Compute residual and gradient terms
    residual = I2_warped - base_img
    Rx = residual * I2_wx
    Ry = residual * I2_wy
    
    # Regularization terms
    lap_x = apply_laplacian(disp_x)
    lap_y = apply_laplacian(disp_y)
    
    # Dynamic time step calculation
    max_grad = max(np.abs(Rx).max(), np.abs(Ry).max())
    denominator = 4*LAMBDA
    dt = 2.0 / (denominator + 1e-12)  # Prevent division by zero
    dt = min(min(dt, 2/max_grad),0.25)  # Stability limit
    # print(dt)
    
    # Update displacement fields
    disp_x += dt * (LAMBDA * lap_x - Rx)
    disp_y += dt * (LAMBDA * lap_y - Ry)

    # Store history
    if iteration % 10 == 0:
        a_history.append(disp_x.copy())
        b_history.append(disp_y.copy())
        # Compute energy for convergence check
        E_history.append(np.mean(residual**2) + np.mean(lap_x**2) + np.mean(lap_y**2))

    # Optional: Add convergence check here
    # if iteration % 100 == 0:
    #     print(f"Iter {iteration}: max displacement {np.hypot(disp_x, disp_y).max():.3f}")

print(f"Computation time: {time()-start_time:.2f}s")

### Visualization ###
plt.figure(figsize=(10, 8))

# Original images
plt.subplot(2, 2, 1), plt.imshow(base_img, cmap='gray')
plt.title('Reference Image (I1)')

plt.subplot(2, 2, 2), plt.imshow(shifted_img, cmap='gray')
plt.title(f'Shifted Image (I2)\n({TRUE_SHIFT_X}px right, {TRUE_SHIFT_Y}px down)')

# Displacement field (every 4th arrow)
skip = 4
plt.subplot(2, 2, 3)
plt.quiver(col_grid[::skip, ::skip],
           row_grid[::skip, ::skip],
           disp_x[::skip, ::skip],
           -disp_y[::skip, ::skip],  # Y-axis flip for display
           scale=25, color='#FF3300', width=0.003)
plt.gca().invert_yaxis()  # Match image coordinates
plt.title('Estimated Displacement Vectors')

# Convergence plots
plt.subplot(2, 2, 4)
plt.plot(np.arange(0, MAX_ITERS, 10), [disp_x[GRID_SIZE//2, GRID_SIZE//2] for disp_x in a_history], 'b-o', label='Horizontal (a)')
plt.plot(np.arange(0, MAX_ITERS, 10), [disp_x[GRID_SIZE//2, GRID_SIZE//2] for disp_x in b_history], 'r-o', label='Vertical (b)')
# plt.axhline(TRUE_SHIFT_X, color='b', linestyle='--', label='True horizontal')
# plt.axhline(TRUE_SHIFT_Y, color='r', linestyle='--', label='True vertical')
plt.xlabel('Iteration')
plt.ylabel('Disparity Value at Center')
plt.title('Convergence History')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))

# plot the evolution of the energy
plt.subplot(1, 2, 1),
plt.plot(np.arange(0, MAX_ITERS, 10), E_history, 'g-o')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('Energy Evolution')
plt.grid(True)

# Valid Points Tracking
plt.subplot(1, 2, 2),
plt.plot(valid_points)
plt.ylim(0, 100)
plt.xlabel('Iteration')
plt.ylabel('Valid Points (%)')
plt.title('Percentage of Valid Data Points')
plt.grid(True)

plt.tight_layout()

# Print center values for verification
center = GRID_SIZE//2
print(f"Center displacement - X: {disp_x[center,center]:.2f} (true {TRUE_SHIFT_X})")
print(f"Center displacement - Y: {disp_y[center,center]:.2f} (true {TRUE_SHIFT_Y})")

# Save figure for report
# plt.savefig('displacement_results.png', dpi=150)
plt.show()
