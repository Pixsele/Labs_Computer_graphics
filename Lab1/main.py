import math
import numpy as np
from PIL import Image, ImageOps

img_mat = np.zeros((10000, 10000,3), np.uint8)
img_mat[0:10000, 0:10000] = [23,23,23]

def dotted_line(image,x0,y0,x1,y1,count):
    step = 1.0/count
    for t in np.arange(0,1,step):
        x = round((1.0 - t)*x0 + t*x1)
        y = round((1.0 - t)*y0 + t*y1)
        image[y,x] = 255

def dotted_line_fix1(image,x0,y0,x1,y1):
    count = math.sqrt((x0-x1)**2 + (y0-y1)**2)
    step = 1.0/count
    for t in np.arange(0,1,step):
        x = round((1.0 - t)*x0 + t*x1)
        y = round((1.0 - t)*y0 + t*y1)
        image[y,x] = 255

def x_loop_line(image,x0,y0,x1,y1):
    for x in range (x0, x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)*y0 + t*y1)
        image[y,x] = 255

def x_loop_line_fix1(image,x0,y0,x1,y1):
    if(x0>x1):
        x0,x1 = x1,x0
        y0,y1 = y1,y0
    for x in range (x0, x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)*y0 + t*y1)
        image[y,x] = 255


def x_loop_line_fix2(image,x0,y0,x1,y1):
    xchange = False

    if (abs(x0-x1)<abs(y0-y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if(x0>x1):
        x0,x1 = x1,x0
        y0,y1 = y1,y0


    for x in range (x0, x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x,y] = 255
        else:
            image[y,x] = 255

def x_loop_line_v2(image,x0,y0,x1,y1):

    xchange = False

    if (abs(x0-x1)<abs(y0-y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if(x0>x1):
        x0,x1 = x1,x0
        y0,y1 = y1,y0

    y = y0
    dy = abs(y1-y0)/(x1-x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range (x0, x1):
        if (xchange):
            image[x, y] = 255
        else:
            image[y, x] = 255
        derror += dy
        if(derror > 0.5):
            derror -= 1.0
            y += y_update

def x_loop_line_no_y_calc_v2_for_some_unknow_reason(image,x0,y0,x1,y1):

    xchange = False

    if (abs(x0-x1)<abs(y0-y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if(x0>x1):
        x0,x1 = x1,x0
        y0,y1 = y1,y0

    y = y0
    dy = 2.0*(x1-x0)*abs(y1-y0)/(x1-x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range (x0, x1):
        if (xchange):
            image[x, y] = 255
        else:
            image[y, x] = 255
        derror += dy
        if(derror > 2.0*(x1-x0)*0.5):
            derror -= 2.0*(x1-x0)*1.0
            y += y_update

def  bresenham_line(image,x0,y0,x1,y1,color):
    xchange = False

    if (abs(x0-x1)<abs(y0-y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if(x0>x1):
        x0,x1 = x1,x0
        y0,y1 = y1,y0

    y = y0
    dy = 2.0*abs(y1-y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range (x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if(derror > (x1-x0)):
            derror -= 2.0*(x1-x0)
            y += y_update


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


def image_v(color):
    v_cord = []
    f_cord = []
    parse("model_1.obj", v_cord, f_cord)

    for v in v_cord:
        img_mat[round(v[1]*5000+300),round(v[0]*5000+600)] = color

    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.show()
#pink [248,24,148]

def image_f(color):
    v_cord = []
    f_cord = []
    parse("model_1.obj", v_cord, f_cord)


    for float in f_cord:

        m = 50_000
        dx = 600*10
        dy = 300*10
        x0 = round(v_cord[float[0]-1][0]*m+dx)
        y0 = round(v_cord[float[0]-1][1]*m+dy)
        x1 = round(v_cord[float[1]-1][0]*m+dx)
        y1 = round(v_cord[float[1]-1][1]*m+dy)
        x2 = round(v_cord[float[2]-1][0]*m+dx)
        y2 = round(v_cord[float[2]-1][1]*m+dy)

        bresenham_line(img_mat,x0,y0,x1,y1,color)
        bresenham_line(img_mat,x1,y1,x2,y2,color)
        bresenham_line(img_mat,x2,y2,x0,y0,color)

    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save("model_1.png")

image_f([255,20,147])

# image_v([255,20,147])

# for i in range(13):
#     x0 = 100
#     y0 = 100
#     x1 = int(100 + 95*math.cos(i*2*math.pi/13))
#     y1 = int(100 + 95*math.sin(i*2*math.pi/13))
#     bresenham_line(img_mat,x0,y0,x1,y1)

# v_cord  = []
# f_cord = []
# parse("model_1.obj",v_cord, f_cord)
# print(f_cord)
# img = Image.fromarray(img_mat,mode = 'RGB')
# img.save('image_bresenham.png')