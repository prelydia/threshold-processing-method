from cProfile import label
import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import math
import imutils
from collections import defaultdict

# Upload image
image = cv2.imread('path\to\your\file') 

# Coordinates of all pixels and centroids for each particle
particles_pixels = []
particle_centers = []

def find_particle_distances(x1, y1, x2, y2):
    """
    The function finds the distance between two particles

    where (x1, y1) are centroids of the first particle,
          (x2, y2) are centroids of the second particle.
          
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def find_particle_radius(particle_area):

    return math.sqrt(particle_area / math.pi)

def is_dimer(p1, p2):
    """
    The function to check whether two particles are dimmer. 
    
    The connected component is a dimmer if each of the two particles in 
    the component has only one neighbor at a distance R + some error.

    The error is calculated using the formula error = (r1 + r2)/10, 

    where r1 and r2 are the radii of the particles.

    """
    # extract the coordinates of the center and the area of the particles
    x1, y1, area1 = p1
    x2, y2, area2 = p2

    # calculating the radii of two particles
    r1 = find_particle_radius(area1)
    r2 = find_particle_radius(area2)

    distance = find_particle_distances(x1, y1, x2, y2)
    sum_radii = r1 + r2
    error = sum_radii / 10

    # сhecking the proximity of two particles
    return distance <= sum_radii + error

def is_chain(graph, dict):
    """
    The function to check whether a connected component is a chain. 
    
    The connected component is a chain if two particles on its 
    edges have one neighbor, and all others have two neighbors. Otherwise, it is a cluster.

    """
    # searching for particles with 1 neighbor (on the edges)
    endpoints = [item for item in graph if len(dict[item]) == 1]
    # search for particles with 2 neighbors (inside)
    internal = [item for item in graph if len(dict[item]) == 2]

    if (len(endpoints) == 2 and len(internal) == len(graph) - 2):
        # then it's a chain
        return True
    # otherwise it's a cluster
    return False

# https://en.wikipedia.org/wiki/Depth-first_search
def find_connected_components(graph):
    """
    Formation of connected components based on found particle neighbors (Depth-First Search)

    Returns the found related components

    """
    visited = set()  # a list for storing visited particles
    components = []  # a list for storing connected components

    def dfs(node, component):
        # until we have passed all the particles
        if node not in visited:
            # adding it to the visited ones
            visited.add(node)
            # adding it to a connected component
            component.append(node)

            # for all the neighbors of this particle (adjacent)
            for neighbor in graph[node]:
                # do the same for the neighbors of the neighbor
                dfs(neighbor, component)

    # go through all the particles with the neighbors
    for node in graph:
        # if the particle is not visited
        if node not in visited:
            # form a connected component (from neighbors)
            component = []
            # using the dfs function
            dfs(node, component)

            # adding the final connected component
            components.append(component)

    return components

# https://www.geeksforgeeks.org/image-segmentation-with-watershed-algorithm-opencv-python/
def watershed_algorithm(image, type):
    """
    Image segmentation function by watershed algorithm

    Returns identified particles and source image in grayscale

    """
    # Converting the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Using Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # We apply threshold processing (Otsu's method)
    _, thresh = cv2.threshold(blurred, 0, 255, type)

    # Calculates the distance to the nearest black pixel (background) for each white pixel
    dist = ndimage.distance_transform_edt(thresh)

    # Looks for points on the distance map that are local maxima (centers of objects)
    coords = peak_local_max(dist, min_distance=15, labels=thresh)

    # Marker mask
    mask = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # Numbering of objects
    markers, _ = ndimage.label(mask)

    # Using the Watershed algorithm
    labels = watershed(-dist, markers, mask=thresh)

    return labels, gray

def define_thresholding_type(labels, gray, default_type):
    """
    The function for determining the threshold type for the input image

    Returns the threshold type

    """
    # the threshold types
    types = [cv2.THRESH_BINARY | cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU]

    # the index of the current type
    index = types.index(default_type) if default_type in types else None

    # list for storing the circularities of the found particles
    circularities = []
    # for all identified particles
    for label in np.unique(labels):

        if label == 0: continue # ignoring the background

        # Creating the mask for the current placemark
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # Finding the contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour) # area
            perimeter = cv2.arcLength(contour, True) # perimeter

            # https://learnopencv.com/blob-detection-using-opencv-python-c/

            # if the default type is chosen incorrectly, the chance of 
            # identifying noises increases, the area and radius of 
            # which may be insignificant, which causes an error in 
            # executing the following lines (we skip such noises)
            try:
                circulatiry = round((4*math.pi*area)/(perimeter**2)) 
                circularities.append(circulatiry)
            except:
                continue

    # if the particles were identified correctly, we keep the type of threshold value
    # otherwise, change it
    return types[abs(index - 1)] if np.mean(circularities) <= 0.78 else types[index]

def find_mean_area(labels):
    """
    The function for finding the minimum area for filtering recognized particles
    
    Returns the median of the area array with some error

    """
    # the areas of the found particles
    areas = []

    # mark the boundaries of the objects in the original image
    for label in np.unique(labels):

        if label == 0:  # ignoring the background
            continue
        
        # creating the mask for the current placemark
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # finding the contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # adding the area of all found particles to the list
        areas.append([cv2.contourArea(cnt) for cnt in contours][0]) 

        # in the mask we find the largest contour (this will be a separate particle)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

    # from the areas of all the particles, find the median (to filter out incorrectly recognized particles)
    # During the search and analysis of various options, it was decided to use a value of 10% of the median of the array as the return parameter.
    return np.median(areas) * 0.1

# In case of an image error
if image is None:
    raise ValueError("The image has not been uploaded. Check the file path.")

# default threshold type
default_type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU

# identifying the particles in the image
labels, gray = watershed_algorithm(image, default_type)

# find the threshold type
type = define_thresholding_type(labels, gray, default_type)

print(f'выбранный тип порогового преобразования: {type}')

# if the threshold type was initially selected incorrectly
if (type != default_type):  
    # then change the type and find new particles
    labels, gray = watershed_algorithm(image, type)

# get threshold area parameter
mean_area = find_mean_area(labels)

# for all recognized particles
for label in np.unique(labels):

    if label == 0: continue # ignoring the background

    # Creating a mask for the current placemark
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # Finding the contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # particle area
        area = cv2.contourArea(contour)

        # if the area is filtered
        if area > mean_area:
            # get the coordinates of all the white pixels in the mask
            pixels = np.column_stack(np.where(mask == 255))
            particles_pixels.append(pixels)

            M = cv2.moments(contour)
            if M["m00"] != 0:  # avoiding division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                particle_centers.append((cX, cY, area))

# dict for storing found neighbors of particles
mainNeighbors = defaultdict(list)

# list for storing ALL found particles
particles = []

# Searching for nearest neighbors
for i in range(len(particle_centers)):
    for j in range(len(particle_centers)):
        # if particles are dimer (i.e the neighbours)
        if is_dimer(particle_centers[i], particle_centers[j]) and i != j:
            # numbering the neighbor
            cv2.putText(image, str(j), (particle_centers[j][0], particle_centers[j][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            mainNeighbors[i].append(j)
    # we number the initial particle
    cv2.putText(image, str(i), (particle_centers[i][0], particle_centers[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    particles.append(i)


# The final dict of connected components
cleaned_neighbors = find_connected_components(mainNeighbors)

# Dict for distributed clusters
clusters = defaultdict(list)
# Each cluster has its own color
colors = {'isolated': (0, 0, 255), 'chain': (107, 142, 35), 'dimer': (127, 255, 212), 'cluster': (148, 0, 211) }

# Predicting the results
for value in cleaned_neighbors:
    # if there are 2 elements in a connected component
    if (len(value) == 2):
        # then it's a dimer
        for item in value: clusters[item].append('dimer')
    else:
        # otherwise, check for a chain
        if (is_chain(value, mainNeighbors)): 
            for item in value: clusters[item].append('chain')
        # otherwise, it's a cluster
        else:
            for item in value: clusters[item].append('cluster')

# Output of particles to an image without neighbors (i.e. isolated)
if ([str(value) for value in particles - mainNeighbors.keys()]):
     
        # from the list of all particles, we subtract those that have neighbors
        for value in particles - mainNeighbors.keys(): clusters[value].append('isolated')

# Сolor the particles according to their cluster
for i, particle in enumerate(particles_pixels):
    for y, x in particle[:, :2]:
        value = clusters[i][0]
        image[y, x] = colors[value] #(0, 0, 255) 
        
    cv2.putText(image, str(i), (particle_centers[i][0], particle_centers[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Adding the cluster names
for i, value in enumerate(['isolated', 'chain', 'dimer', 'cluster']):
    cv2.putText(image, value, (i + 5, (i + 1) * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[value], 2)

cv2.imshow("result", image)
cv2.waitKey(0)


