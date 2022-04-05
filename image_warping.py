import numpy as np
import cv2
# to import
# compute_homography
# load_image, show_image

# # THIS VERSION OF WARPING LOOPS OVER THE SOURCE IMAGE AND MAPS IT TO THE DESTINATION IMAGE USING H-MATRIX
#
# def warp_img(src_img, H, dst_img_size):
#     dst_img = np.zeros([dst_img_size[0], dst_img_size[1], 3])
#
#     M = src_img.shape[0]
#     N = src_img.shape[1]
#     C = src_img.shape[2]
#
#     print(dst_img_size[0])
#     print(dst_img_size[1])
#
#     #print(src_img.shape)
#     #print(dst_img_size)
#
#     for i in range(N):
#     for j in range(M):
#       coords = [i, j, 1]
#       coords = np.asarray(coords)
#       coords = coords.transpose()
#       new_pts = np.matmul(H, coords)
#
#       new_x = round(new_pts[0]/new_pts[2])
#       new_y = round(new_pts[1]/new_pts[2])
#       #if i == 0 and j == 511:
#         #print(new_x, new_y)
#
#       #print(i, j)
#       if new_y <= dst_img_size[0] and new_x <= dst_img_size[0]:
#         for c in range(C):
#           dst_img[new_y, new_x, c] = src_img[j, i, c]
#
#     return dst_img[:dst_img_size[0]][:dst_img_size[1]]

# THIS VERSION OF WARPING LOOPS OVER THE DESTINATION IMAGE AND MAPS THE SOURCE IMAGE TO THE DESTINATION IMAGE USING THE INVERSE OF H-MATRIX

def warp_img(src_img, H, dst_img_size):
    dst_img = np.zeros([dst_img_size[0], dst_img_size[1], 3])

    M = dst_img.shape[0]
    N = dst_img.shape[1]

    for i in range(N):
    for j in range(M):
      coords = [i, j, 1]
      coords = np.asarray(coords)
      coords = coords.transpose()
      H_inv = np.linalg.inv(H)
      new_pts = np.matmul(H_inv, coords)

      src_x = round(new_pts[0]/new_pts[2])
      src_y = round(new_pts[1]/new_pts[2])

      if (src_y < 0 or src_x < 0) or (src_x > src_img.shape[1] or src_y > src_img.shape[0]):
        dst_img[j, i] = 0

      elif (src_y > 0 and src_x > 0) and (src_x < src_img.shape[1] and src_y < src_img.shape[0]):
        dst_img[j, i] = src_img[src_y, src_x]

    return dst_img[:M, :N]


def binary_mask(img):
    # Helper function for multiplying the warped image with binary mask
    mask = (img[:, :, 0] > 0) | (img[:, :, 1] > 0) | (img[:, :, 2] > 0)
    mask = mask.astype("int")
    return mask


def test_warp():
    src_img = load_image('mandrill.tif')
    canvas_img = load_image('Rubiks_cube.jpg')

    # The following are corners of the mandrill image
    src_pts = np.matrix('0, 0; 0, 511; 511, 511; 511, 0')
    # The following are corners of the blue face of the Rubik's cube
    canvas_pts = np.matrix('218, 238; 225, 560; 490, 463; 530, 178')

    H = compute_homography(src_pts, canvas_pts)
    print(H)

    dst_img = warp_img(src_img, H, [canvas_img.shape[0], canvas_img.shape[1]])

    dst_mask = 1 - binary_mask(dst_img)
    dst_mask = np.stack((dst_mask,) * 3, -1)
    out_img = np.multiply(canvas_img, dst_mask) + dst_img

    dsize = (600, 600)  # width and height of canvas_im
    src_smaller = cv2.resize(src_img, dsize, interpolation=cv2.INTER_AREA)

    warped_img = np.concatenate((src_smaller, canvas_img, out_img), axis=1)
    show_image(np.clip(warped_img, 0, 1))


test_warp()