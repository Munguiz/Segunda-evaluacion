import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

FILE = 'Jit1.jpg'
ITERACIONES = 15

def kernel(size,sigma):

    borders = int((size-1) / 2);

    kernel = np.zeros([size, size], dtype=np.float64)
    norm_kernel = np.zeros([size, size], dtype=np.float64)

    for x in range(-1 * borders, borders + 1):
        for y in range(-1 * borders, borders + 1):
            kernel[x + borders][y + borders] = (1 / (2 * np.pi * pow(sigma, 2))) * math.exp(-1 * (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
   
    gauss_mean = 0

    for x in range(size):
        for y in range(size):
            gauss_mean += kernel[x][y]

    for i in range(size):
        for j in range(size):
            norm_kernel[i,j] = kernel[i,j] / gauss_mean;

    return kernel

def add_zeros(channel,size):

    border = int((size-1) / 2);
    rows,cols = channel.shape
    channel_w_zeros = np.zeros([size-1+rows, size-1+cols], dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            channel_w_zeros[i+border,j+border] = channel[i,j]

    return channel_w_zeros

def gauss_filter(kernel,img_zeros,size,ch,filtered_ch):
    rows, cols = ch.shape
    temp_matrix = np.empty((size, size), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            
                sum = 0
                temp_matrix = img_zeros[i:i+size,j:j+size]

                for k in range(0,size):
                    for l in range(0,size):
                        sum +=  (temp_matrix[k,l] * kernel[k,l])

                filtered_ch[i,j] = sum

# asignacion de clusters mediante la suma del error cuadratico
def cluster_assignment(centroids, vec):
    return ((vec - centroids[:, np.newaxis]) ** 2).sum(axis=2).argmin(axis=0)

# actualizacion del centroide mediante el promedio de los pixeles del cluster
def move_centroid(centroids, clusters, vec):
    for i in range(len(centroids)):
        vec_sub = vec[clusters==i]
        centroids[i] = np.mean(vec_sub, axis=0)
    return centroids

# algoritmo k means
def k_means(img):
    rows, cols, channels = img.shape
    vec_img = img.reshape(-1, channels).astype(int)
    i = 0
    centroids = np.array([[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255]])
    clusters = cluster_assignment(centroids, vec_img)
    while (i < ITERACIONES):
        clusters = cluster_assignment(centroids, vec_img)
        centroids = move_centroid(centroids, clusters, vec_img)    
        i += 1
    
    img_compressed = centroids[clusters].reshape(img.shape)
    
    kmeans = img_compressed.astype(np.uint8)
    cv2.imwrite("Jit1kmeans.jpg", kmeans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return kmeans
    
def binarize(image):
    rows,columns = image.shape
    binarized = np.zeros([rows,columns], dtype=np.uint8)
    
    for i in range(rows):
        for j in range(columns):
            if(image[i,j] < 40):
                binarized[i,j] = 0
            else:
                binarized[i,j] = 255
    return binarized

def fill_spaces(img):

    th, img_th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV);

    img_spaces = img_th.copy()
    
    h, w = img_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    cv2.floodFill(img_spaces, mask, (0,0), 255);

    img_spaces_inv = cv2.bitwise_not(img_spaces)
    
    img_filled = img_th | img_spaces_inv

    img_filled = cv2.bitwise_not(img_filled)

    return img_filled

def tag(image):

    rows, cols = image.shape

    colors = []
    same = True

    for i in range(rows):
        for j in range(cols):

            if(image[i][j]==0):
                same = True
                gray = np.random.randint(50,205)
                while same:
                    if gray not in colors:
                        colors.append(gray)
                        same =  False
                    else:
                        gray = np.random.randint(50,205)
                
                cv2.floodFill(image,None,(j,i),gray)

    tags = dict()
            
    for i in range(rows):

        for j in range(cols):

            if image[i][j] in tags and image[i][j] != 0 and image[i][j] != 255:
                tags[image[i][j]]["points"].append([i, j, 1])
                tags[image[i][j]]["count"] += 1
            else:
                tags[image[i][j]] = {
                    "points": [[i, j, 1]],
                    "count": 1
                }

    cv2.imwrite("Jit1colored.jpg", image)

    print(tags.keys())

    return image, tags

def euclidean(p1, p2):
    p1 = np.array([p1[0],p1[1],p1[2]])
    p2 = np.array([p2[0],p2[1],p2[2]])
    return np.sqrt(np.sum((p1 - p2)**2))

def longest_distance(tags):

    longest_distance_points = []

    for tag in tags:

        if tags[tag]["count"] < 1000: continue

        points = tags[tag]["points"]
        distance = 0
        point1 = None
        point2 = None

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                temp_distance = euclidean(points[i], points[j])
                if temp_distance > distance:
                    distance = temp_distance
                    point1 = points[i]
                    point2 = points[j]

        longest_distance_points.append([point1, point2, distance])

    return longest_distance_points

def main():

    img = cv2.imread(FILE)

    kernel_size = 7
    sigma = 1.8

    rows, cols, channels = img.shape

    b,g,r = cv2.split(img)

    kernel_gauss =  kernel(kernel_size,sigma)

    b_zeros_border = add_zeros(b,kernel_size)
    g_zeros_border = add_zeros(g,kernel_size)
    r_zeros_border = add_zeros(r,kernel_size)
    blur = np.zeros([rows,cols,channels], dtype=np.uint8)
    b_gauss = np.empty((rows,cols), dtype=np.float64)
    g_gauss = np.empty((rows,cols), dtype=np.float64)
    r_gauss = np.empty((rows,cols), dtype=np.float64)
    gauss_filter(kernel_gauss,b_zeros_border,kernel_size,b,b_gauss)
    gauss_filter(kernel_gauss,g_zeros_border,kernel_size,g,g_gauss)
    gauss_filter(kernel_gauss,r_zeros_border,kernel_size,r,r_gauss)

    blur[:, :, 0] = b_gauss
    blur[:, :, 1] = g_gauss
    blur[:, :, 2] = r_gauss

    cv2.imwrite("Jit1gauss.jpg", blur)

    print("Gaussiano realizado...")

    kmeans = k_means(blur)

    print("Kmeans realizado...")

    b,g,r = cv2.split(kmeans)

    cv2.imwrite("Jit1g.jpg", g)

    img_bin = binarize(g)

    img_bin = fill_spaces(img_bin)

    print("Binarización realizada...")

    cv2.imwrite("Jit1bin.jpg", img_bin)

    tagged, tags = tag(img_bin)

    print("Etiquetado de objetos realizado...")

    ld_points = longest_distance(tags)

    print("Puntos con mayor distancia en objeto 2 y 4 encontrados ....")

    for points in ld_points:
        
        center = [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2, 1]
        center[0] = int(math.floor(center[0]))
        center[1] = int(math.floor(center[1]))

        if center[0] < rows / 2 and center[0] > cols / 2: 

            left_y_pos = center[1]
            right_y_pos = center[1]

            while tagged[center[0]][left_y_pos] != 0: 
                left_y_pos -= 1
            while tagged[center[0]][right_y_pos] != 0: 
                right_y_pos += 1

            left_point = [center[0], left_y_pos, 1]
            right_point = [center[0], right_y_pos, 1]
            print(f"Objeto 2: P1({left_point[1]}, {left_point[0]})) P2({right_point[1]}, {right_point[0]})")
            print(f'Distancia entre puntos: {points[2]}')
            cv2.line(img, (left_point[1], left_point[0]), (right_point[1], right_point[0]), (0, 255, 255), 1)

        elif center[0] > rows/ 2:

            left_y_pos = center[1]
            right_y_pos = center[1]
            left_x_pos = center[0]
            right_x_pos = center[0]

            while tagged[left_x_pos][left_y_pos] != 0:
                left_y_pos -= 1
                left_x_pos += 1

            while tagged[right_x_pos][right_y_pos] != 0:
                right_y_pos += 1
                right_x_pos -= 1

            left_point = [left_x_pos, left_y_pos, 1]
            right_point = [right_x_pos, right_y_pos, 1]
            print(f"Objeto 4: P1({left_point[1]}, {left_point[0]})) P2({right_point[1]}, {right_point[0]})")
            print(f'Distancia entre puntos: {points[2]}')
            cv2.line(img, (left_point[1], left_point[0]), (right_point[1], right_point[0]), (0, 255, 255), 1)

    cv2.imwrite("Jit1lines.jpg", img)
    cv2.waitKey(0)

if __name__ == "__main__":

    main()