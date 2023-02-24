import os
import cv2
import csv
import math
import glob
import random
import operator
import itertools
import scipy.misc
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

def get_inertia_parameter(img_array):
    try:
        y, x = np.nonzero(img_array)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x = x - x_mean
        y = y - y_mean
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        return x_v1, y_v1, x_mean, y_mean, len(x)
    except Exception as e: 
        print(e)
        return 4,3,2,1,0
def inertia_detection_cv_line(img, scale, width, ignore_slope=False):
    x_v1, y_v1, x_mean, y_mean, len_non_zero = get_inertia_parameter(img)
    # if ignore_slope:
    cv2.line(img, (int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean)), (255, 0, 255), width)
    return img,[(int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean))]
    # elif slope(-x_v1,-y_v1,x_v1,y_v1) > 1.2:
    #     cv2.line(img, (int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean)), (255, 0, 255), width)
    #     return img,[(int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean))]
    # else:
    #     return img, [(0,0),(1,1)]
def point_in_left_side_of_line(point,line_edge1,line_edge2):
    dy = line_edge2[1]-line_edge1[1]
    dx = line_edge2[0]-line_edge1[0]
    if(dx==0):
        if(line_edge2[0]>point[0]):
            return True
        else:
            return False
    if(dy==0):
        dy = 1   
    try:
        m = (dy/dx)
    except:
        print('gradient error')
    c = line_edge1[1]-m*line_edge1[0]
    if (m < 0):
        if (point[1] - (m*point[0]) - c) < 0:
            return True
        else:
            return False
    if ((m > 0)):
        if (point[1] - (m*point[0]) - c) > 0:
            return True
        else:
            return False
def get_molar_line(fill_up_img, line_of_inertia, type_):
    y_list,x_list = np.nonzero(fill_up_img)
    
    L_fill_up = fill_up_img.copy()
    if type_ == 'L':
        point_in_left = [point_in_left_side_of_line([x_,y_],line_of_inertia[0],line_of_inertia[1]) for (x_, y_) in zip(x_list,y_list)]
        for bool_, x_, y_ in zip(point_in_left, x_list, y_list):
            if (not bool_):
                L_fill_up[y_,x_] = 0
        _, line_of_inertia = inertia_detection_cv_line(L_fill_up)
#         plt.imshow(L_fill_up)
#         plt.show()
        return line_of_inertia

    if type_ == 'R':
        R_fill_up = fill_up_img.copy()
        point_in_left = [point_in_left_side_of_line([x_,y_],line_of_inertia[0],line_of_inertia[1]) for (x_, y_) in zip(x_list,y_list)]
        for bool_, x_, y_ in zip(point_in_left, x_list, y_list):
            if bool_:
                R_fill_up[y_,x_] = 0          
        _, line_of_inertia = inertia_detection_cv_line(R_fill_up)
#         plt.imshow(R_fill_up)
#         plt.show()
        return line_of_inertia
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)
def crosspoint_point2line(point,line_edge1,line_edge2):
    #hyparameter is three points in type np.array([x,y])
    m = -999
    try:
        m = (line_edge2[1]-line_edge1[1])/(line_edge2[0]-line_edge1[0])
        new_m = -1/m
        c = point[1]-new_m*point[0]
        point_2 = [point[0]+1,(point[0]+1)*new_m+c]
    except:
        if m == 0:
            point_2 = [point[0],point[1]+1]
        elif m == -999:
            point_2 = [point[0]+1,point[1]]

    return line_intersection((point,point_2), (line_edge1, line_edge2))
def root_point_detect(fill_up_img, value):
    # zero_img = np.zeros((fill_up_img.shape[0],fill_up_img.shape[1],3), np.uint8)
    inertia, line_of_inertia = inertia_detection_cv_line(fill_up_img.copy(), scale=500,width=3, ignore_slope=True)
    # root_contour = draw_approx_hull_polygon(tooth_mask11class.astype('uint8'))
    # print('here')
    # plt.subplot(1, 2, 1)
    # plt.imshow(fill_up_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(inertia)
    # plt.show()
    i, j = np.where(fill_up_img != 0)
    point_list = [[i_, j_] for i_, j_ in zip(i, j)]
    # y, x
    # y,x = find_roots_by_countour(root_contour, inertia, value)
    if point_list != []:
        if value == 1:
            y, x  = min(point_list)
        elif value == 0:
            y, x  = max(point_list)
        pt1, pt2 = line_of_inertia
        x, y = crosspoint_point2line((x, y), pt1, pt2)
    else:
        y, x = [np.nan, np.nan]
    return [x],[y],line_of_inertia
def string_to_pair(string_pair):
    '''
    string_pair: "(1, 3)"
    return: [1, 3]
    '''
    # if isinstance(string_pair, float) and np.isnan(string_pair):
    #     return (np.nan, np.nan)
    num_pair = string_pair[1:-1].split(',')
    return [int(float(x)) for x in num_pair]
def point_resize(x, y, old_shape, new_shape):
    newX = new_round(x * (new_shape[0] / old_shape[0]))
    newY = new_round(y * (new_shape[1] / old_shape[1]))
    return newX, newY
def new_round(x):
    return int(round(x))
def extend_line(s,e,img):
    s = np.array(s)
    e = np.array(e)
#     scale = 1
    vec = e-s
    
#     print(vec)
    if vec[0]!=0:
        if vec[1]!=0:
            scale = min(abs(img.shape[0]/vec[0]),abs(img.shape[1]/vec[1]))
        else:
            scale = abs(img.shape[0]/vec[0])
    else:
        if vec[1]==0:
            scale = 10
        else:
            scale = abs(img.shape[1]/vec[1])
    
    scale = new_round(scale)+1
    e = e + scale*vec
    s = s - scale*vec
#     print(e)
    return tuple(s), tuple(e)
def extend_to_border(p1, p2, shape):
    # 2021/10/28 add by plchao, if work please write note
    x1, y1 = p1
    x2, y2 = p2
    boardx, boardy, _ = shape
    boardy = max(boardx, boardy)
    if x1 - x2 == 0 or y1 - y2 == 0:
        return p1, p2
    slope =  (y1 - y2) / (x1 - x2)
    assert type(slope) == float
    # x/y
    new_start_x, new_start_y = (boardy - y1) / slope + x1, boardy
    new_end_x, new_end_y = (0 - y1)/slope + x1, 0
    if new_start_x < 0:
        new_start_x, new_start_y = 0, (0 - x1) * slope + y1
    elif new_start_x > boardx:
        new_start_x, new_start_y = boardx, (boardx - x1) * slope + y1
    
    if new_end_x < 0:
        new_end_x, new_end_y = 0, (0 - x1) * slope + y1
    elif new_end_x > boardx:
        new_end_x, new_end_y = boardx, (boardx - x1) * slope + y1
    return (new_round(new_end_x), new_round(new_end_y)), (new_round(new_start_x), new_round(new_start_y))
def show_pixel_set(img_nparray):
    a = img_nparray
    unique, counts = np.unique(a, return_counts=True)
    return dict(zip(unique, counts))
def biggest_components(mask, root_mask): 
    out_image = np.zeros((mask.shape[0],mask.shape[1]), np.uint8).astype('uint8')
    mask_255 = mask*255
    mask_threshold = cv2.threshold(mask_255, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels_im = cv2.connectedComponents(mask_threshold)
    pixel_num_dict = show_pixel_set(labels_im)
    key_list = list(pixel_num_dict.keys())[1:]
    for ele in key_list:
        target_mask = np.where(labels_im==ele, 1, 0).astype('uint8')
        if (np.any((target_mask*root_mask)==1)):
            out_image+=target_mask
    try:
        return out_image
    except:
        return mask
    

def remove_pieces_1000(mask):
    out_image = np.zeros((mask.shape[0],mask.shape[1]), np.uint8)
    mask_255 = mask*255
    mask_threshold = cv2.threshold(mask_255, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels_im = cv2.connectedComponents(mask_threshold)
    pixel_num_dict = show_pixel_set(labels_im)
    pixel_num_list = list(pixel_num_dict)
    for ele in pixel_num_list[1:]:
        if pixel_num_dict[ele] > 1000:
            out_image+=np.where(labels_im==ele,1,0).astype('uint8')
    return out_image

def get_boundary_line_by_inertia(root_mask, tooth_mask):
    root_x, root_y, line_of_inertia = root_point_detect(root_mask.copy(), 1)
    blank_image = np.zeros((root_mask.shape[0],root_mask.shape[1]), np.uint8)
    cv2.line(blank_image, line_of_inertia[0], line_of_inertia[1], (255), 5)
    y_, x_ = np.nonzero(blank_image)
    inertia_point_list = [(x,y) for x, y in zip(x_,y_)]
    inertia_point_list = list(sorted(inertia_point_list, key = lambda x:x[1]))
    min_x = min(inertia_point_list[0][0],inertia_point_list[-1][0])
    max_x = max(inertia_point_list[0][0],inertia_point_list[-1][0])
    final_mask = np.zeros((root_mask.shape[0],root_mask.shape[1]), np.uint8)
    for i in range(-max_x, root_mask.shape[1]-min_x, 1):
        line_n = np.concatenate((blank_image[:,-i:], blank_image[:,:-i]), axis=1)
        if (np.any(line_n*root_mask!=0)):
            final_mask+=line_n
    and_mask = np.where(final_mask*tooth_mask > 0,1,0).astype('uint8')
    target_mask = biggest_components(and_mask, np.where(root_mask > 0,1,0).astype('uint8'))
    # print('final', show_pixel_set(final_mask))
    # visualize([target_mask, and_mask, root_mask, final_mask])
    target_mask = remove_pieces_1000(target_mask)
    return target_mask