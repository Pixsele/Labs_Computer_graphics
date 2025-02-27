import math
import numpy as np
from PIL import Image, ImageOps

def parse(file_name,v_cord,f_cord):
    file = open(file_name, 'r')

    for file_line in file:
        temp = file_line.split()
        if(temp[0] == 'v'):
            v_cord.append([float(temp[1]),float(temp[2]),float(temp[3])])

        if(temp[0] == "f"):
            a = temp[1].split('/')
            b = temp[2].split('/')
            c = temp[3].split('/')
            f_cord.append([int(a[0]),int(b[0]),int(c[0])])

def baritser(x0,y0,x1,y1,x2,y2,x,y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0,lambda1,lambda2]

def draw_triangle(img_mat,z_buffer,x0,y0,z0,x1,y1,z1,x2,y2,z2,n,l,h,w):
    xmin = round(min(x0,x1,x2))
    ymin = round(min(y0,y1,y2))
    xmax = round(max(x0,x1,x2))
    ymax = round(max(y0,y1,y2))
    if(xmin < 0):
        xmin = 0
    if(ymin < 0):
        ymin = 0
    if(ymax > w):
        ymax = w-1
    if(xmax > h):
        xmax = h-1
    for i in range(xmin,xmax+1):
        for j in range(ymin,ymax+1):
            lambdas = baritser(x0, y0, x1, y1, x2, y2, i, j)
            if lambdas[0] >= 0 and lambdas[1] >= 0 and lambdas[2] >= 0:
                z_new = lambdas[0]*z0 + lambdas[1]*z1 + lambdas[2]*z2
                if(z_buffer[j,i] > z_new):
                    img_mat[j, i] = (-255 * face_or_no(n, l), -255 * face_or_no(n, l), -255 * face_or_no(n, l))
                    z_buffer[j,i] = z_new

def normal_vector(x0,y0,z0,x1,y1,z1, x2,y2,z2):
    v1 = np.array((x1,y1,z1)) - np.array((x0,y0,z0))
    v2 = np.array((x2,y2,z2)) - np.array((x0,y0,z0))
    normal = np.cross(v1, v2)
    return normal

def face_or_no(n,l):
    dot_product = np.dot(n, l)
    norm_n = np.linalg.norm(n)
    norm_l = np.linalg.norm(l)

    cos_theta = dot_product / (norm_n * norm_l)
    return cos_theta

def show_image(file_name):
    h = 1000
    w = 1000
    img_mat = np.zeros((h, w, 3), np.uint8)
    img_mat[0:h, 0:w] = [255, 255, 255]
    z_buffer = np.full((h,w),1000)
    v_cord = []
    f_cord = []
    parse(file_name,v_cord,f_cord)

    for float in f_cord:
        m = 5_000
        dx = 500
        dy = 300
        dz = 0
        x0 = (v_cord[float[0]-1][0]*m+dx)
        y0 = (v_cord[float[0]-1][1]*m+dy)
        z0 = (v_cord[float[0]-1][2]*m+dz)
        x1 = (v_cord[float[1]-1][0]*m+dx)
        y1 = (v_cord[float[1]-1][1]*m+dy)
        z1 = (v_cord[float[1]-1][2]*m+dz)
        x2 = (v_cord[float[2]-1][0]*m+dx)
        y2 = (v_cord[float[2]-1][1]*m+dy)
        z2 = (v_cord[float[2]-1][2]*m+dz)

        normal = normal_vector(x0,y0,z0,x1,y1,z1,x2,y2,z2)

        l = [0,0,1]

        if(face_or_no(normal,l) < 0):
            draw_triangle(img_mat,z_buffer, x0, y0, z0,x1, y1,z1, x2, y2,z2,normal,l,h,w)

    img = Image.fromarray(img_mat,mode = 'RGB')
    img = ImageOps.flip(img)
    img.show()

show_image("model_1.obj")


