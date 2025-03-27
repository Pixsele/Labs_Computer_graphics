from operator import indexOf

import numpy as np
from PIL import Image, ImageOps

def parse(file_name,v_cord,f_cord,vt_cord,ft_cord):
    file = open(file_name, 'r')

    for file_line in file:
        temp = file_line.split()
        if(len(temp)>0):
            if temp[0] == 'v':
                v_cord.append([float(temp[1]), float(temp[2]), float(temp[3])])

            if temp[0] == "f":
                a = temp[1].split('/')
                b = temp[2].split('/')
                c = temp[3].split('/')
                f_cord.append([int(a[0]), int(b[0]), int(c[0])])
                ft_cord.append([int(a[1]), int(b[1]), int(c[1])])

            if temp[0] == "vt":
                vt_cord.append([float(temp[1]), float(temp[2])])




def baritser(x0,y0,x1,y1,x2,y2,x,y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0,lambda1,lambda2]

def draw_triangle(img_mat,z_buffer,v0,v1,v2,n,l,h,w,vn,float,vt_cord,vt,texture):

    x0, y0, z0 = v0[0], v0[1], v0[2]
    x1, y1, z1 = v1[0], v1[1], v1[2]
    x2, y2, z2 = v2[0], v2[1], v2[2]

    n0 = vn[float[0]-1]/np.linalg.norm(vn[float[0]-1])
    n1 = vn[float[1]-1]/np.linalg.norm(vn[float[1]-1])
    n2 = vn[float[2]-1]/np.linalg.norm(vn[float[2]-1])

    ax = 300000
    ay = 300000

    x0_proj = ax * x0 / z0 + w / 2
    y0_proj = ay * y0 / z0 + h / 2
    x1_proj = ax * x1 / z1 + w / 2
    y1_proj = ay * y1 / z1 + h / 2
    x2_proj = ax * x2 / z2 + w / 2
    y2_proj = ay * y2 / z2 + h / 2

    xmin = round(min(x0_proj, x1_proj, x2_proj))
    ymin = round(min(y0_proj, y1_proj, y2_proj))
    xmax = round(max(x0_proj, x1_proj, x2_proj))
    ymax = round(max(y0_proj, y1_proj, y2_proj))


    vt0 = vt_cord[vt[0]-1]
    vt1 = vt_cord[vt[1]-1]
    vt2 = vt_cord[vt[2]-1]

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if ymax > w:
        ymax = w - 1
    if xmax > h:
        xmax = h - 1
    for i in range(xmin, xmax + 1):
        for j in range(ymin, ymax + 1):
            lambdas = baritser(x0_proj, y0_proj, x1_proj, y1_proj, x2_proj, y2_proj, i, j)
            if lambdas[0] >= 0 and lambdas[1] >= 0 and lambdas[2] >= 0:
                z_new = lambdas[0] * z0 + lambdas[1] * z1 + lambdas[2] * z2
                if z_buffer[j, i] > z_new:
                    I0 = np.dot(n0,l)/(np.linalg.norm(n0)*np.linalg.norm(l))
                    I1 = np.dot(n1,l)/(np.linalg.norm(n1)*np.linalg.norm(l))
                    I2 = np.dot(n2,l)/(np.linalg.norm(n2)*np.linalg.norm(l))

                    I = -255*(lambdas[0]*I0+lambdas[1]*I1+lambdas[2]*I2)

                    texture_cord = [round(1024*(lambdas[0]*vt0[0] + lambdas[1]*vt1[0] + lambdas[2]*vt2[0])),
                                    round(1024*(lambdas[0]*vt0[1]+lambdas[1]*vt1[1]+lambdas[2]*vt2[1]))]

                    color = texture.getpixel((texture_cord[0], texture_cord[1]))


                    img_mat[j, i] = (-color[0]*(lambdas[0]*I0+lambdas[1]*I1+lambdas[2]*I2),-color[1]*(lambdas[0]*I0+lambdas[1]*I1+lambdas[2]*I2),-color[2]*(lambdas[0]*I0+lambdas[1]*I1+lambdas[2]*I2))
                    z_buffer[j, i] = z_new

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


def rotate_vertex(vertex, alfa, beta, gamma,t):
    x, y, z = vertex

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alfa), -np.sin(alfa)],
        [0, np.sin(alfa), np.cos(alfa)]
    ])

    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    rotated_vertex = R @ np.array([x, y, z])
    return rotated_vertex + t

def rotate_all(v_cord, alfa, beta, gamma,t):
    result = []
    for v in v_cord:
        v = rotate_vertex(v, alfa, beta, gamma,t)
        result.append(v)
    return result


def vn_calc(n,v_cord,f_cord):
    result = np.zeros((n, 3), dtype=float)

    for poligon in f_cord:

        v0 = (v_cord[poligon[0] - 1])
        v1 = (v_cord[poligon[1] - 1])
        v2 = (v_cord[poligon[2] - 1])

        x0, y0, z0 = v0[0], v0[1], v0[2]
        x1, y1, z1 = v1[0], v1[1], v1[2]
        x2, y2, z2 = v2[0], v2[1], v2[2]

        normal = normal_vector(x0,y0,z0,x1,y1,z1,x2,y2,z2)

        result[poligon[0]-1] += normal
        result[poligon[1]-1] += normal
        result[poligon[2]-1] += normal

    return result







def show_image(file_name):

    texture = Image.open("bunny-atlas.jpg")
    texture = ImageOps.flip(texture)

    h = 3000
    w = 3000
    img_mat = np.zeros((h, w, 3), np.uint8)
    img_mat[0:h, 0:w] = [255, 255, 255]
    z_buffer = np.full((h,w),np.inf)
    v_cord = []
    f_cord = []
    vt_cord = []
    ft_cord = []
    parse(file_name,v_cord,f_cord,vt_cord,ft_cord)


    alfa = np.radians(0) ##X
    beta = np.radians(90) ##Y
    gamma = np.radians(0) ##Z

    tx = 0
    ty = -0.09
    tz = 20
    t = np.array([tx, ty, tz])
    v_cord = rotate_all(v_cord,alfa,beta,gamma,t)

    vn = vn_calc(len(v_cord),v_cord,f_cord)
    i = 0

    for float in f_cord:
        if(i%1000 == 0):
            print(i)

        v0 = (v_cord[float[0] - 1])
        v1 = (v_cord[float[1] - 1])
        v2 = (v_cord[float[2] - 1])

        x0, y0, z0 = v0[0], v0[1], v0[2]
        x1, y1, z1 = v1[0], v1[1], v1[2]
        x2, y2, z2 = v2[0], v2[1], v2[2]

        vt = ft_cord[i]

        normal = normal_vector(x0,y0,z0,x1,y1,z1,x2,y2,z2)

        l = [0,0,1]

        if face_or_no(normal, l) < 0:
            draw_triangle(img_mat,z_buffer, v0,v1, v2,normal,l,h,w,vn,float,vt_cord,vt,texture)

        i+=1

    img = Image.fromarray(img_mat,mode = 'RGB')
    img = ImageOps.flip(img)
    img.save('output2.png')
    img.show()

show_image("model_1.obj")


