import time
import argparse
import sys
import os
import numpy as np
import yaml
import math

from sklearn.cluster import DBSCAN
from skimage.feature import canny
from skimage.morphology import skeletonize
from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.segmentation import flood_fill
from scipy.ndimage.morphology import distance_transform_edt
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from tkinter import Tk
from tkinter.filedialog import askopenfile, askdirectory

from graph_util import FindCycle
from pixel2point import pixels2points
from pixel2point import points2pixels

def imageFromFile(pgmf):

    if pgmf is None:
        pgmf = askopenfile(mode='rb', filetypes=[('PGM Files', '*.pgm')], title='Select PGM Track Image File')

    if pgmf is None:
        sys.exit("Invalid PGM File")

    header = []
    while len(header) < 4:
        line = pgmf.readline()
        words = line.split()
        if len(words) > 0 and words[0] != b'#':
            header.extend(words)
    
    if len(header) != 4 or header[0] != b'P5' or not header[1].isdigit() or not header[2].isdigit() or not header[3].isdigit():
        raise ValueError("Error Reading PGM File")
    
    width = int(header[1])
    height = int(header[2])

    image = []
    while len(image) < height:
        row = []
        while len(row) < width:
            word = ord(pgmf.read(1))
            row.append(word)
        image.append(row)

    return np.array(image), pgmf.name

def infoFromFile(yamlf):

    if yamlf is None:
        yamlf = askopenfile(mode='r', filetypes=[('YAML Files', '*.yaml *.yml')], title='Select Track Info YAML File')

    if yamlf is None:
        sys.exit("Invalid YAML File")

    data = yaml.safe_load(yamlf)
    resolution = None
    origin = None
    try:
        resolution = data['resolution']
        origin = data['origin']
    except KeyError:
        sys.exit("Error Reading YAML File")

    return resolution, origin

def outputDirFromFile(output_dir):

    if output_dir is None:
        output_dir = askdirectory(title='Select Output Directory')

    if output_dir is None:
        sys.exit("Invalid Output Directory")

    return output_dir

def plot(image, title, plot_mode):
    if plot_mode > 0:
        fig = plt.figure()
        fig.canvas.set_window_title(title)
        plt.imshow(image, interpolation='nearest', cmap='Greys')

def plotInColor(image, title, plot_mode):
    if plot_mode > 0:
        fig = plt.figure()
        fig.canvas.set_window_title(title)
        plt.imshow(image, interpolation='nearest')

def plotExtra(image, title, plot_mode):
    if plot_mode > 1:
        plot(image, title, plot_mode)

def pixelsInCircle(radius):
    num = 0
    for i in range(100000):
        num += math.floor(radius**2. / (4.*i + 1.)) - math.floor(radius**2. / (4.*i + 3.))
    num *= 4
    num += 1
    return num

def denoise(image, plot_mode):

    radius = 7
    db_scan = DBSCAN(eps=radius, min_samples=pixelsInCircle(radius))

    points = pixels2points(image, False)

    clusters = db_scan.fit(points).labels_
    cluster_sizes = np.bincount(clusters + 1)
    if len(cluster_sizes) < 2:
        raise RuntimeError('No clusters found')
    cluster_sizes = cluster_sizes[1:] # ignore noise at first index
    main_cluster = np.argmax(cluster_sizes)

    points = [x for i, x in enumerate(points) if clusters[i] == main_cluster]

    height, width = image.shape
    image = points2pixels(points, width, height, False)

    inverted_points = pixels2points(image, True)

    db_scan = DBSCAN(eps=1, min_samples=3)

    clusters = db_scan.fit(inverted_points).labels_
    cluster_sizes = np.bincount(clusters + 1)
    if len(cluster_sizes) < 3:
        raise RuntimeError('Not enough clusters found while denoising')
    cluster_sizes = cluster_sizes[1:] # ignore noise at first index
    main_clusters = np.argsort(cluster_sizes)[-2:]

    inverted_points = [x for i, x in enumerate(inverted_points) if clusters[i] in main_clusters]

    image = points2pixels(inverted_points, width, height, True)

    image = erosion(image)

    return image

def addToGraphFromImage(g, image):
    height, width = image.shape
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            if pixel > 0:
                for j in range(y-1, y+2):
                    for i in range(x-1, x+2):
                        if i >= 0 and i < width and j >= 0 and j < height and image[j][i] > 0:
                            g.addEdge((x, y), (i, j))

def findTrackWalls(image):
    # Canny Edge Filter finds edge pixels
    edges = canny(image * 255, sigma = 2, low_threshold=0.25, high_threshold=0.6)
    edge_points = pixels2points(edges, False)

    plotExtra(edges, "Edges", plot_mode)

    # DBSCAN helps distinguish between pixels on the inner wall and pixels on the outer wall
    db_scan = DBSCAN(eps=3, min_samples=2)
    db_scan.fit(edge_points)

    clusters = db_scan.labels_
    cluster_sizes = np.bincount(clusters + 1)
    if len(cluster_sizes) < 3:
        raise RuntimeError("Not enough clusters found while finding track walls")
    cluster_sizes = cluster_sizes[1:] # ignore noise at first index
    main_clusters = np.argsort(cluster_sizes)[-2:]

    innerwall_points = [point for i, point in enumerate(edge_points) if clusters[i] == main_clusters[0]]
    outerwall_points = [point for i, point in enumerate(edge_points) if clusters[i] == main_clusters[1]]

    height, width = image.shape
    innerwall_pixels = points2pixels(innerwall_points, width, height, False)
    outerwall_pixels = points2pixels(outerwall_points, width, height, False)
    
    inner_graph = FindCycle()
    outer_graph = FindCycle()

    addToGraphFromImage(inner_graph, innerwall_pixels)
    addToGraphFromImage(outer_graph, outerwall_pixels)

    inner_graph.findCycle()
    outer_graph.findCycle()

    return np.array(inner_graph.cycle), np.array(outer_graph.cycle)

def prune(image, plot_mode):
    height, width = image.shape
    g = FindCycle()
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            if pixel > 0:
                for j in range(y-1, y+2):
                    for i in range(x-1, x+2):
                        if i >= 0 and i < width and j >= 0 and j < height and image[j][i] > 0:
                            g.addEdge((x, y), (i, j))
    
    g.findCycle()

    image = np.zeros((height, width))
    for i, (x, y) in enumerate(g.cycle):
        image[y, x] = i / len(g.cycle) * (155) + 100
    return image, np.array(g.cycle)

def subsample(cycle, period):
    subsampled_cycle = []
    for i, point in enumerate(cycle):
        if i % period == 0 and len(cycle) - i >= period / 2:
            subsampled_cycle.append(point)
    
    return np.array(subsampled_cycle)

def generate_sd_field(track_wall_pixels, contained_pixel_position):
    track_pixels = flood_fill(track_wall_pixels, contained_pixel_position, 1, connectivity=1) # fill in the pixels between the track walls

    occupied_pixels = np.where(track_wall_pixels == 1, 0, 1) # prepare pixels for distance transform by making occupied pixels 0 and uoccupied pixels 1
    d_field = np.array(distance_transform_edt(occupied_pixels))

    # negate distances outside of the track walls
    for y, row in enumerate(d_field):
        for x, distance in enumerate(row):
            if track_pixels[y][x] == 0:
                d_field[y][x] = -distance

    return d_field

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Fits a racetrack to an image if possible.')
    parser.add_argument('--pgm_path', help='file path to the pgm track image file', nargs='?', type=argparse.FileType('rb'), default=None)
    parser.add_argument('--yaml_path', help='file path to the yaml track info file', nargs='?', type=argparse.FileType('r'), default=None)
    parser.add_argument('--subsample_period', help='defines the subsampling rate of the center-line pixels for the output points', nargs='?', type=int, default=6)
    parser.add_argument('--plot_mode', help='0: no plots, 1: basic plots, 2: all plots', nargs='?', type=int, default=1)
    parser.add_argument('--out_dir', help='directory path to write output to', nargs='?', type=argparse.FileType('w'), default=None)
    args = parser.parse_args()

    plot_mode = args.plot_mode
    subsample_period = args.subsample_period
    
    # Get track image and info from file
    Tk().withdraw()
    image, name = imageFromFile(args.pgm_path)
    height, width = image.shape
    resolution, origin = infoFromFile(args.yaml_path)

    start_time = time.time()

    # Make the image binary
    image = np.where(image > 0, 1, 0)
    original = image

    plot(image, "Original Image", plot_mode)

    # Denoise image
    print("Denoising")
    denoised = denoise(image, plot_mode)
    image = denoised

    plotExtra(image, "Denoised", plot_mode)

    ### Find Track Walls ###
    print("Finding Track Walls")
    innerwall_points, outerwall_points = findTrackWalls(image)
    wall_points = np.concatenate((innerwall_points, outerwall_points))
    track_walls = points2pixels(wall_points, width, height, False)

    plot(track_walls, "Track Walls", plot_mode)

    ### Find Center Line ###
    # Skeletonize Image
    print("Skeletonizing")
    image = skeletonize(image)
    skeleton_overlay = np.where(image == 1, 0, denoised)
    image = image

    plotExtra(image, "Skeletonized", plot_mode)
    plotExtra(skeleton_overlay, "Skeletonized Overlayed on Denoised", plot_mode)

    # Prune Image
    print("Pruning")
    image, centerline_points = prune(image, plot_mode)
    overlayed = np.where(image > 0, 0, denoised)

    plotExtra(image, "Pruned", plot_mode)
    plotExtra(overlayed, "Pruned Overlayed on Denoised", plot_mode)

    ### Generate Signed Distance Field ###
    print("Generating Signed Distance Field")
    contained_point = centerline_points[0] # an arbitrary point that falls between the track walls
    sd_field = generate_sd_field(track_walls, (contained_point[1], contained_point[0]))
    max_absolute_distance = np.max(abs(sd_field))

    sd_field_points = []
    for y, row in enumerate(sd_field):
        for x, signed_dist in enumerate(row):
            sd_field_points.append([x, y, signed_dist])
    sd_field_points = np.array(sd_field_points)

    ### Make a Finalized Plot ###
    results_pixels = []
    for y, row in enumerate(sd_field):
        newrow = []
        for x, pixel in enumerate(row):
            if pixel < 0:
                hsv = (0, 1, (1 - abs(pixel) / max_absolute_distance)**8)
            else:
                hsv = (0.5, 1, (1 - abs(pixel) / max_absolute_distance)**8)
            if track_walls[y][x] == 1.0 or image[y][x] > 0.0:
                hsv = (1.0, 0.0, 1.0)
            newrow.append(hsv_to_rgb(hsv))
        results_pixels.append(newrow)

    plotInColor(results_pixels, "Signed Depth Field", plot_mode)

    # Subsample
    print("Subsampling")
    centerline_points = subsample(centerline_points, subsample_period)
    innerwall_points = subsample(innerwall_points, subsample_period)
    outerwall_points = subsample(outerwall_points, subsample_period)
    all_points = np.concatenate((centerline_points, innerwall_points, outerwall_points))
    image = points2pixels(all_points, width, height, False)

    image = dilation(image)
    plot(image, "Walls and Centerline", plot_mode)

    ### Transform to Real-World Coordinates ###
    print("Transforming to Real-World Coordinates")
    centerline_points = centerline_points * resolution - origin[0:2]
    innerwall_points = innerwall_points * resolution - origin[0:2]
    outerwall_points = outerwall_points * resolution - origin[0:2]
    sd_field_points[:,0:2] = sd_field_points[:,0:2] * resolution - origin[0:2]
    sd_field_points[:,2] *= resolution

    print("Processing Time:", time.time() - start_time, "seconds")

    ### Save Results to CSV ###
    print("Writing Output CSV")
    output_dir = outputDirFromFile(args.out_dir)
    output_path = os.path.join(output_dir, os.path.splitext(name)[0] + '-centerline.csv')
    np.savetxt(output_path, centerline_points, delimiter=",")

    output_path = os.path.join(output_dir, os.path.splitext(name)[0] + '-innerwall.csv')
    np.savetxt(output_path, innerwall_points, delimiter=",")

    output_path = os.path.join(output_dir, os.path.splitext(name)[0] + '-outerwall.csv')
    np.savetxt(output_path, outerwall_points, delimiter=",")

    output_path = os.path.join(output_dir, os.path.splitext(name)[0] + '-signeddistfield.csv')
    np.savetxt(output_path, sd_field_points, delimiter=",")

    print("Finished. Output saved to", output_dir)

    # Display Plots
    if plot_mode > 0:
        plt.show()