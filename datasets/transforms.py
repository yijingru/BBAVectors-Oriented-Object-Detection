import numpy as np
import cv2

def random_flip(image, gt_pts, crop_center=None):
    # image: h x w x c
    # gt_pts: num_obj x 4 x 2
    h,w,c = image.shape
    if np.random.random()<0.5:
        image = image[:,::-1,:]
        if gt_pts.shape[0]:
            gt_pts[:,:,0] = w-1 - gt_pts[:,:,0]
        if crop_center is not None:
            crop_center[0] = w-1 - crop_center[0]
    if np.random.random()<0.5:
        image = image[::-1,:,:]
        if gt_pts.shape[0]:
            gt_pts[:,:,1] = h-1 - gt_pts[:,:,1]
        if crop_center is not None:
            crop_center[1] = h-1 - crop_center[1]
    return image, gt_pts, crop_center


def _get_border(size, border):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

def random_crop_info(h, w):
    if np.random.random() < 0.3:
        max_wh = max(h, w)
        random_size = max_wh * np.random.choice(np.arange(0.9, 1.1, 0.1))
        w_border = _get_border(size=w, border=32)
        h_border = _get_border(size=h, border=32)
        random_center_x = np.random.randint(low=w_border, high=w - w_border)
        random_center_y = np.random.randint(low=h_border, high=h - h_border)
        return [random_size, random_size], [random_center_x, random_center_y]
    else:
        return None, None



def Rotation_Transform(src_point, degree):
    radian = np.pi * degree / 180
    R_matrix = [[np.cos(radian), -np.sin(radian)],
                [np.sin(radian), np.cos(radian)]]
    R_matrix = np.asarray(R_matrix, dtype=np.float32)
    R_pts = np.matmul(R_matrix, src_point)
    return R_pts


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def load_affine_matrix(crop_center, crop_size, dst_size, inverse=False, rotation=False):
    dst_center = np.array([dst_size[0]//2, dst_size[1]//2], dtype=np.float32)
    if rotation and np.random.rand(1)>0.5:
        random_degree = np.random.rand(1)[0]*90
    else:
        random_degree = 0.

    src_1 = crop_center
    src_2 = crop_center + Rotation_Transform([0, -crop_size[0]//2], degree=random_degree)
    src_3 = get_3rd_point(src_1, src_2)
    src = np.asarray([src_1, src_2, src_3], np.float32)

    dst_1 = dst_center
    dst_2 = dst_center + [0, -dst_center[0]]
    dst_3 = get_3rd_point(dst_1, dst_2)
    dst = np.asarray([dst_1, dst_2, dst_3], np.float32)
    if inverse:
        M = cv2.getAffineTransform(dst, src)
    else:
        M = cv2.getAffineTransform(src, dst)
    return M

def ex_box_jaccard(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    inter_x1 = np.maximum(np.min(a[:,0]), np.min(b[:,0]))
    inter_x2 = np.minimum(np.max(a[:,0]), np.max(b[:,0]))
    inter_y1 = np.maximum(np.min(a[:,1]), np.min(b[:,1]))
    inter_y2 = np.minimum(np.max(a[:,1]), np.max(b[:,1]))
    if inter_x1>=inter_x2 or inter_y1>=inter_y2:
        return 0.
    x1 = np.minimum(np.min(a[:,0]), np.min(b[:,0]))
    x2 = np.maximum(np.max(a[:,0]), np.max(b[:,0]))
    y1 = np.minimum(np.min(a[:,1]), np.min(b[:,1]))
    y2 = np.maximum(np.max(a[:,1]), np.max(b[:,1]))
    mask_w = np.int(np.ceil(x2-x1))
    mask_h = np.int(np.ceil(y2-y1))
    mask_a = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    mask_b = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    a[:,0] -= x1
    a[:,1] -= y1
    b[:,0] -= x1
    b[:,1] -= y1
    mask_a = cv2.fillPoly(mask_a, pts=np.asarray([a], 'int32'), color=1)
    mask_b = cv2.fillPoly(mask_b, pts=np.asarray([b], 'int32'), color=1)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(inter)/(float(union)+1e-12)
    # cv2.imshow('img1', np.uint8(mask_a*255))
    # cv2.imshow('img2', np.uint8(mask_b*255))
    # k = cv2.waitKey(0)
    # if k==ord('q'):
    #     cv2.destroyAllWindows()
    #     exit()
    return iou