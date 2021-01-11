#! /usr/bin/env python
import cv2
import numpy as np
import scipy.spatial as spatial
import logging
import os
import time
from flask import Flask,request, jsonify
from flask_cors import CORS
from face_detection import *
from PIL import Image
import statistics
from color_matching import *

app = Flask(__name__, static_url_path='')
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/api/GetDetectedFace": {"origins": "*"}})## 3D Transform

def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T

def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points
    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None


def triangular_affine_matrices(vertices, src_points, dst_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dst_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dst_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


def warp_image_3d(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
    rows, cols = dst_shape[:2]
    result_img = np.zeros((rows, cols, 3), dtype=dtype)

    delaunay = spatial.Delaunay(dst_points)
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dst_points)))

    process_warp(src_img, result_img, tri_affines, dst_points, delaunay)

    return result_img


## 2D Transform
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
    R = (np.dot(U, Vt)).T

    return np.vstack([np.hstack([s2 / s1 * R,
                                (c2.T - np.dot(s2 / s1 * R, c1.T))[:, np.newaxis]]),
                      np.array([[0., 0., 1.]])])


def warp_image_2d(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)

    return output_im


## Generate Mask
def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask


## Color Correction
def correct_colours(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


## Copy-and-paste
def apply_mask(img, mask):
    """ Apply mask to supplied image
    :param img: max 3 channel image
    :param mask: [0-255] values in mask
    :returns: new image with mask applied
    """
    masked_img=cv2.bitwise_and(img,img,mask=mask)

    return masked_img


## Alpha blending
def alpha_feathering(src_img, dest_img, img_mask, blur_radius=15):
    mask = cv2.blur(img_mask, (blur_radius, blur_radius))
    mask = mask / 255.0

    result_img = np.empty(src_img.shape, np.uint8)
    for i in range(3):
        result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1-mask)

    return result_img


def check_points(img,points):
    # Todo: I just consider one situation.
    if points[8,1]>img.shape[0]:
        logging.error("Jaw part out of image")
    else:
        return True
    return False


def best_skin(frame):
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    frame = cv2.resize(frame, (400, 400))
    # print(mode(frame1, axis=0))
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    skin1 = Image.fromarray(skin)
    c = []
    for pixel in skin1.getdata():
        if pixel != (0, 0, 0):
            c.append(pixel)

    r, g, b = statistics.mode(c)
    return r,g,b

@app.route("/face_swap",methods=['POST'])
def face_swap():
    if request.method == "POST":
        random_No_Image = time.time()
        if request.files:
            image = request.files["image"]
            print(type(image))
            image.save("InputImage_"+str(random_No_Image)+".jpg")
        src_img = cv2.imread("InputImage_" + str(random_No_Image) + ".jpg")
#        response = df.get_detectedFace(random_No_Image)
        correct_color = False
        warp_2d = False
        end = 48
        dst_img = cv2.imread("download.jpg")
        dst_img = color_transfer(src_img, dst_img)
        # Select src face
        src_points, src_shape, src_face = select_face(src_img)

        # Select dst face
        dst_img = color_transfer(src_face, dst_img)
        dst_points, dst_shape, dst_face = select_face(dst_img)
        if src_points is None or dst_points is None:
            print('Detect 0 Face !!!')
            exit(-1)
        h, w = dst_face.shape[:2]

        ## 3d warp
        warped_src_face = warp_image_3d(src_face, src_points[:end], dst_points[:end], (h, w))
        ## Mask for blending
        mask = mask_from_points((h, w), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask * mask_src, dtype=np.uint8)
        ## Correct color
        if correct_color:
            warped_src_face = apply_mask(warped_src_face, mask)
            dst_face_masked = apply_mask(dst_face, mask)
            warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
        ## 2d warp
        if warp_2d:
            unwarped_src_face = warp_image_3d(warped_src_face, dst_points[:end], src_points[:end], src_face.shape[:2])
            warped_src_face = warp_image_2d(unwarped_src_face, transformation_from_points(dst_points, src_points),
                                            (h, w, 3))

            mask = mask_from_points((h, w), dst_points)
            mask_src = np.mean(warped_src_face, axis=2) > 0
            mask = np.asarray(mask * mask_src, dtype=np.uint8)

        ## Shrink the mask
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        ##Poisson Blending
        r = cv2.boundingRect(mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

        x, y, w, h = dst_shape
        dst_img_cp = dst_img.copy()
        dst_img_cp[y:y + h, x:x + w] = output
        out_image_name = "OutputImage_"+str(random_No_Image)+".png"
        print("image done")
        #import pdb; pdb.set_trace()
        #skin_dict = best_skin(src_img)
        out_points, out_shape, out_face = select_face(dst_img_cp)
        skinTone_B, skinTone_G , skinTone_R = best_skin(out_face)
        im = Image.fromarray(dst_img_cp)
        for i in range(int(np.asarray(im).shape[0] * 2.5 / 3), np.asarray(im).shape[0]):
            for j in range(np.asarray(im).shape[1]):
                y = i
                x = j
                im.putpixel((x, y), (round(skinTone_B), round(skinTone_G), round(skinTone_R)))
        final_img = np.array(im)
        #final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        #import pdb; pdb.set_trace()
        #x1, y1, x2, y2 = faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
        alpha, beta = 0.8,-70
        final_img = np.clip(alpha * final_img + beta, 0, 255)
        # for y in range(final_img.shape[0]):
        #     for x in range(final_img.shape[1]):
        #         for c in range(final_img.shape[2]):
        #             new_image[y, x, c] = np.clip(alpha * final_img[y, x, c] + beta, 0, 255)
        cv2.imwrite(out_image_name, final_img)
#        response.headers.add('Access-Control-Allow-Origin', '*')
        return jsonify({
	     'done': 'True',
             'FileName':out_image_name,
             #'ByTeArray':encoded_string
             'Message':'',
             'skinToneRGB': str(round(skinTone_R)) + ',' + str(round(skinTone_G)) + ',' + str(round(skinTone_B))
           })
    else:
        dct = {'code':400, 'Message':"Pleaase change the method to POST"}
        return dict

if __name__=="__main__":
    app.run()

