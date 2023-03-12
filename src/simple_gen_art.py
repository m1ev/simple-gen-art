import random

import matplotlib.pyplot as plt
import matplotlib.path as mplp
from matplotlib.transforms import Bbox
import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from sklearn.datasets import make_blobs

# Line intersection check functions to ensure that each shape lies inside
# the corresponding voronoi cell
def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = _det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (_det(*line1), _det(*line2))
    x = _det(d, xdiff) / div
    y = _det(d, ydiff) / div
    return x, y

# Chaikin's corner cutting algorithm
def chaikins_corner_cutting(coords, refinements=5):
    coords = np.array(coords)
    for _ in range(refinements):
        l = coords.repeat(2, axis=0)
        r = np.empty_like(l)
        r[0] = l[0]
        r[2::2] = l[1:-1:2]
        r[1:-1:2] = l[2::2]
        r[-1] = l[-1]
        coords = l * 0.75 + r * 0.25
    return coords

#==============================================================
# Setting up random parameters to make each picture unique
#==============================================================
# Number of random points on the plane to build voronoi diagram from
# More points = more voronoi cells = more shapes
n_points = np.random.randint(10, 600, 1)

# Random color of shapes
shapes_clr = ("#%06x" % random.randint(0, 0xFFFFFF))

# How many random points needed in total to draw the shapes:
# 1) n_samples is an integer:
#    samples will be divided equally for each shape
#    Each shape requires minimum 4 points (4 * n_points)
# 2) n_samples is an array of integers with length = n_points
#    assign n_samples[i] points to each shape (should be minimum
#    4 for each shape)
#
# More points = More roundness of each shape and closer to ellipse shape
# Less points = More edgy shapes with sharp edges

# Make it random when the program decides which of these 4 options to apply
opt = np.random.choice([1, 2, 3, 4])
LB_CONST = 4        # Set lower and
UB_CONST = 150      # upper bounds

PERCENTAGE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
lb = np.random.randint(LB_CONST, UB_CONST, 1)

if opt == 1:
    n_samples = np.random.randint(LB_CONST * n_points[0],
                                  UB_CONST * n_points[0], 1)
elif opt == 2:
    n_samples = np.random.randint(LB_CONST * n_points[0],
                                  UB_CONST * n_points[0], 1)
elif opt == 3:
    ub = np.random.randint(lb[0], UB_CONST, 1)
    n_samples = np.random.randint((lb[0]), (ub), n_points[0])
else:
    ub = lb[0] + (((UB_CONST - lb[0]) / 100) * np.random.choice(PERCENTAGE))
    n_samples = np.random.randint((lb[0]), (ub), n_points[0])

#==============================================================

# Maximum values of x and y coordinates
# Resolution of no less than 8000 x 6400px is required, so add an extra
# 500 px for each side for further upscaling
X_MAX = 9000
Y_MAX = 7400

# Create random n_points on the plane to build voronoi diagram from
xx = np.random.randint(0, X_MAX, n_points[0])
yy = np.random.randint(0, Y_MAX, n_points[0])
#plt.scatter(xx, yy, s=2, c='r')

# Symmetrically reflect points in the central region to the left, right,
# up and down for purposes of building voronoi cells with finite boundaries
symm_left = xx.copy()
symm_left = 0 - symm_left

symm_right = xx.copy()
symm_right = X_MAX + (X_MAX - symm_right)

symm_up = yy.copy()
symm_up = Y_MAX + (Y_MAX - symm_up)

symm_down = yy.copy()
symm_down = 0 - symm_down

# Draw symmetrically reflected points
# psize = np.full(n_points, 30.01)    # Setting up points size
# plt.scatter(symm_left, yy, s=psize, c='b')
# plt.scatter(symm_right, yy, s=psize, c='g')
# plt.scatter(xx, symm_up, s=psize, c='pink')
# plt.scatter(xx, symm_down, s=psize, c='grey')

# Concatenate original and symmetrically reflected points
vor_centers_x = np.concatenate([xx, symm_left, symm_right, xx, xx])
vor_centers_y = np.concatenate([yy, yy, yy, symm_up, symm_down])
# plt.scatter(vor_centers_x, vor_centers_y, s=2, c='r')

# Unite the two aformentioned 1D arrays into a 2D array of X and Y coordinates
# and build a voronoi diagram
vor_centers = np.vstack((vor_centers_x, vor_centers_y)).T
vor = Voronoi(vor_centers)
vor_diagram = voronoi_plot_2d(vor)

vor_diagram, axes = plt.subplots()

# Select points (voronoi centers) only from the central area
vor_centers_x = vor_centers_x[: len(xx)]
vor_centers_y = vor_centers_y[: len(yy)]
# vor_diagram = plt.figure()
# plt.scatter(vor_centers_x, vor_centers_y,
#             s=21.1, c='blue', cmap='Set1', alpha=1)
# plt.show()

# Indices of the Voronoi regions in the central area
point_region = vor.point_region[: n_points[0]]

# Select Voronoi regions only in the central area
central_regions = []
for i in range(len(point_region)):
    central_regions.append(vor.regions[vor.point_region[i]])

# Calculate standard deviation of vertices of each Voronoi cell
# in the central area
central_regions_stds = []
for i, central_region in enumerate(central_regions):
    # Coordinates of vertices of each Voronoi cell
    cell_vertices = []
    for j, cell_vertex in enumerate(central_region):
        cell_vertices.append(vor.vertices[cell_vertex])
    central_regions_stds.append(np.std(cell_vertices))

# Find the centroids of each voronoi cell by using the formulae of
# finding the centroid of a non-self-intersecting closed polygon defined by
# n vertices (en.wikipedia.org/wiki/Centroid)
centroids_x = []
centroids_y = []
for i in range(len(central_regions)):
    a = 0.0
    c_x = 0.0
    c_y = 0.0

    for j in range(len(central_regions[i])):
        x_j = vor.vertices[central_regions[i][j]][0]
        y_j = vor.vertices[central_regions[i][j]][1]

        if j < (len(central_regions[i]) - 1):
            x_j_plus_1 = vor.vertices[central_regions[i][j+1]][0]
            y_j_plus_1 = vor.vertices[central_regions[i][j+1]][1]
        elif j == (len(central_regions[i]) - 1):
            x_j_plus_1 = vor.vertices[central_regions[i][0]][0]
            y_j_plus_1 = vor.vertices[central_regions[i][0]][1]

        a = a + (x_j*y_j_plus_1) - (x_j_plus_1*y_j)
        c_x = c_x + (x_j+x_j_plus_1) * ((x_j*y_j_plus_1) - (x_j_plus_1*y_j))
        c_y = c_y + (y_j+y_j_plus_1) * ((x_j*y_j_plus_1) - (x_j_plus_1*y_j))

    a = a / 2
    if a > 0:
        central_regions[i] = central_regions[i][::-1]

    c_x = c_x * (1/(6*a))
    c_y = c_y * (1/(6*a))

    centroids_x.append(c_x)
    centroids_y.append(c_y)

centroids = np.vstack((centroids_x, centroids_y)).T

# Standard deviation (STD) of clusters (shapes): max - 150, min - 20,
# regulates the size of each shape, 4 random options available
STD_MIN = 20
STD_MAX = 150
std = np.random.choice([1, 2, 3, 4])

if std == 1:    # Random array of stds per shape
    clusters_stds = (np.random.randint(STD_MIN, STD_MAX, n_points[0]))
elif std == 2:   # Every shape has the same STD
    clusters_stds = np.random.randint(STD_MIN, STD_MAX, 1)
elif std == 3:      # Using lower and upper bounds
    lb = np.random.randint(STD_MIN, STD_MAX, 1)[0]
    ub = np.random.randint(lb, STD_MAX+1, 1)[0]
    clusters_stds = np.random.randint(lb, ub, n_points[0])
else:     # Using percentage values
    PERCENTAGE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    lb = np.random.randint(STD_MIN, STD_MAX, 1)[0]
    ub = lb + (((150 - lb) / 100) * np.random.choice(PERCENTAGE))
    clusters_stds = np.random.randint(lb, ub, n_points[0])

if len(n_samples) == 1:
    n_shapes_sample = n_samples[0]
else:
    n_shapes_sample = n_samples

if len(clusters_stds) == 1:
    n_shapes_std = clusters_stds[0]
else:
    n_shapes_std = clusters_stds

# Using sklearn dataset function "make_blobs" to generate Gaussian blobs
# as a future foundation of shapes
# Returns: X - generated samples, y - labels for cluster membership
# of each sample
X, y = make_blobs(n_samples=n_shapes_sample, centers=centroids,
                  cluster_std=n_shapes_std, n_features=2)
blobs = np.around(np.column_stack((X, y)), 2)

# Extract the coordinates of vertices of each voronoi cell
central_regions_coords = []
num_of_regions = len(central_regions)
for i in range(num_of_regions):
    cell_coords = []
    for j in range(len(central_regions[i])):
        cell_coords.append(vor.vertices[central_regions[i][j]])
    central_regions_coords.append(cell_coords)

# Going through each Voronoi cell and shrink it according to a pre-defined
# coefficient to make sure that each shape doesn't intersect with others
shrink_coef = np.random.randint(20, 50)
shrink_value_x = shrink_coef
shrink_value_y = shrink_coef

for i in range(num_of_regions):
    coords = central_regions_coords[i]
    lines = [[coords[i-1], coords[i]] for i in range(len(coords))]

    new_lines = []
    for j in lines:
        dx = j[1][0] - j[0][0]
        dy = j[1][1] - j[0][1]

        # Taking into account slopes
        factor = 1 / (dx*dx + dy*dy)**0.5
        new_dx = dy*shrink_value_x * factor
        new_dy = dx*shrink_value_y * factor

        new_lines.append([[j[0][0] + new_dx, j[0][1] - new_dy],
                          [j[1][0] + new_dx, j[1][1] - new_dy]])

    # Find position of intersection of all the lines
    new_coords = []
    for j in range(len(new_lines)):
        new_coords.append(line_intersection(new_lines[j-1], new_lines[j]))

    central_regions_coords[i] = np.asarray(new_coords)

# Make sure that each point of each shape lies strictly inside
# the corresponding Voronoi cell which was shrunk on previous stage.
# If some shapes don't meet this criteria - exclude it from the canvas
selected_blobs = []
path = []
for i in range(num_of_regions):
    path.append(mplp.Path(central_regions_coords[i]))

for i in range(len(blobs)):
    check_blob = path[int(blobs[i][2])].contains_points([[blobs[i][0],
                                                        blobs[i][1]]])
    if check_blob[0]:
        selected_blobs.append(blobs[i])

blobs = selected_blobs.copy()

# Create a new array of points
central_regions_coords = np.asarray(central_regions_coords, dtype=object)

# Update the array of labels
blobs = np.asarray(blobs, dtype=object)
blobs_and_labels = np.hsplit(blobs, np.array([2, 3]))
X_new = blobs_and_labels[0].astype(np.double)
y_new = np.ndarray.flatten(np.asarray(blobs_and_labels[1])).astype(int)

# Smooth generated shapes
for i in range(len(centroids)):
    curr_labels = np.where(y_new == i)[0]

    # If a shape has less than 3 points - remove it entirely
    if len(curr_labels) <= 3:
        continue

    # Apply Chaikin's corner cutting algorithm
    points = X_new[curr_labels]
    hull = ConvexHull(points)
    # Get x and y coordinates, repeat last point to close the polygon
    x_hull = np.append(points[hull.vertices,0], points[hull.vertices,0])
    y_hull = np.append(points[hull.vertices,1], points[hull.vertices,1])
    points = chaikins_corner_cutting(np.vstack((x_hull, y_hull)).T,
                                     refinements=np.random.choice([1, 5]))
    hull = ConvexHull(points)

    # Interpolate
    x_hull = np.append(points[hull.vertices,0], points[hull.vertices,0])
    y_hull = np.append(points[hull.vertices,1], points[hull.vertices,1])
    coef_k = np.random.choice([1, 3])
    dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2
                   + (y_hull[:-1] - y_hull[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull],
                                    u=dist_along, s=0, k=coef_k)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)

    # Apply Chaikin's corner cutting algorithm once more
    ref_rand2 = np.random.randint(2, 11, 1)
    points = np.vstack((interp_x, interp_y)).T
    points = chaikins_corner_cutting(points, refinements=ref_rand2[0])
    x_and_y = np.hsplit(points, np.array([1, 2]))
    x_split = np.ndarray.flatten(x_and_y[0])
    y_split = np.ndarray.flatten(x_and_y[1])

    # Plot filled polygons (generated shapes)
    plt.fill(x_split, y_split, c=shapes_clr, alpha=1)

# Select the central area of the required size with black solid line
# and save it as a separate image
border_x = [500, 500, 500 + X_MAX - 1000, 500 + X_MAX - 1000, 500]
border_y = [500, 500 + Y_MAX - 1000, 500 + Y_MAX - 1000, 500, 500]
# plt.plot(border_x, border_y, '-', c='black', linewidth=1, alpha=1)
plt.gca().set_aspect(aspect=1)
plt.savefig('output with scale.png', dpi=420,
            bbox_inches="tight", transparent=True)

# Save the selected central area only
crop_box = Bbox([[500,500],[(500 + X_MAX - 1000),(500 + Y_MAX - 1000)]])
crop_box = crop_box.transformed(axes.transData).transformed(
    vor_diagram.dpi_scale_trans.inverted())
vor_diagram.savefig('output.png', dpi=2500,
                    bbox_inches=crop_box, transparent=True)
vor_diagram.canvas.draw()
