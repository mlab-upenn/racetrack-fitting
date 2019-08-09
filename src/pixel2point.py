import numpy as np

def pixels2points(image, invert):
    points = []
    for y in range(len(image)):
        for x in range(len(image[y])):
            if invert ^ (image[y][x] > 0):
                points.append([x, y])
    return np.array(points)

def points2pixels(points, width, height, invert):
    image = []
    for _ in range(height):
        row = []
        for _ in range(width):
            if invert:
                row.append(1)
            else:
                row.append(0)
        image.append(row)
    
    for point in points:
        if invert:
            image[point[1]][point[0]] = 0
        else:
            image[point[1]][point[0]] = 1
    
    return np.array(image)