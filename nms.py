import numpy as np
import cv2

def calc_IoU(a, b):
    # step1:
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
    if x1>=x2 or y1>=y2 or (x2-x1)<2 or (y2-y1)<2:
        return 0.
    else:
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
        # print(iou)
        # cv2.imshow('img1', np.uint8(mask_a*255))
        # cv2.imshow('img2', np.uint8(mask_b*255))
        # k = cv2.waitKey(0)
        # if k==ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        return iou

def draw_image(pts, image):
    cen_pts = np.mean(pts, axis=0)
    tt = pts[0, :]
    rr = pts[1, :]
    bb = pts[2, :]
    ll = pts[3, :]
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255), 2, 1)
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255), 2, 1)
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0), 2, 1)
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0), 2, 1)
    return image


def NMS_numpy_exboxes(exboxes, conf, nms_thresh=0.5, image=None):
    if len(exboxes)==0:
        return None
    sorted_index = np.argsort(conf)      # Ascending order
    keep_index = []
    while len(sorted_index)>0:
        curr_index = sorted_index[-1]
        keep_index.append(curr_index)
        if len(sorted_index)==1:
            break
        sorted_index = sorted_index[:-1]
        IoU = []
        for index in sorted_index:
            iou = calc_IoU(exboxes[index,:,:].copy(), exboxes[curr_index,:,:].copy())
            IoU.append(iou)
        IoU = np.asarray(IoU, np.float32)
        sorted_index = sorted_index[IoU<=nms_thresh]
    return keep_index



def NMS_numpy_bbox(bboxes, nms_thresh=0.5):
    """
    bboxes: num_insts x 5 [x1,y1,x2,y2,conf]
    """
    if len(bboxes)==0:
        return None
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    conf = bboxes[:,4]
    area_all = (x2-x1)*(y2-y1)
    sorted_index = np.argsort(conf)      # Ascending order
    keep_index = []

    while len(sorted_index)>0:
        # get the last biggest values
        curr_index = sorted_index[-1]
        keep_index.append(curr_index)
        if len(sorted_index)==1:
            break
        # pop the value
        sorted_index = sorted_index[:-1]
        # get the remaining boxes
        yy1 = np.take(y1, indices=sorted_index)
        xx1 = np.take(x1, indices=sorted_index)
        yy2 = np.take(y2, indices=sorted_index)
        xx2 = np.take(x2, indices=sorted_index)
        # get the intersection box
        yy1 = np.maximum(yy1, y1[curr_index])
        xx1 = np.maximum(xx1, x1[curr_index])
        yy2 = np.minimum(yy2, y2[curr_index])
        xx2 = np.minimum(xx2, x2[curr_index])
        # calculate IoU
        w = xx2-xx1
        h = yy2-yy1
        w = np.maximum(0., w)
        h = np.maximum(0., h)
        inter = w*h
        rem_areas = np.take(area_all, indices=sorted_index)
        union = (rem_areas-inter)+area_all[curr_index]
        IoU = inter/union
        sorted_index = sorted_index[IoU<=nms_thresh]

    return keep_index
