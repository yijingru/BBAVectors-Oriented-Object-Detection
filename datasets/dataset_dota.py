from .base import BaseDataset
import os
import cv2
import numpy as np
from .DOTA_devkit.ResultMerge_multi_process import mergebypoly

class DOTA(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(DOTA, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['plane',
                         'baseball-diamond',
                         'bridge',
                         'ground-track-field',
                         'small-vehicle',
                         'large-vehicle',
                         'ship',
                         'tennis-court',
                         'basketball-court',
                         'storage-tank',
                         'soccer-ball-field',
                         'roundabout',
                         'harbor',
                         'swimming-pool',
                         'helicopter'
                         ]
        self.color_pans = [(204,78,210),
                           (0,192,255),
                           (0,131,0),
                           (240,176,0),
                           (254,100,38),
                           (0,0,255),
                           (182,117,46),
                           (185,60,129),
                           (204,153,255),
                           (80,208,146),
                           (0,0,204),
                           (17,90,197),
                           (0,255,255),
                           (102,255,102),
                           (255,255,0)]
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'images')
        self.label_path = os.path.join(data_dir, 'labelTxt')

    def load_img_ids(self):
        if self.phase == 'train':
            image_set_index_file = os.path.join(self.data_dir, 'trainval.txt')
        else:
            image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.png')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.txt')

    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []
        with open(self.load_annoFolder(self.img_ids[index]), 'r') as f:
            for i, line in enumerate(f.readlines()):
                obj = line.split(' ')  # list object
                if len(obj)>8:
                    x1 = min(max(float(obj[0]), 0), w - 1)
                    y1 = min(max(float(obj[1]), 0), h - 1)
                    x2 = min(max(float(obj[2]), 0), w - 1)
                    y2 = min(max(float(obj[3]), 0), h - 1)
                    x3 = min(max(float(obj[4]), 0), w - 1)
                    y3 = min(max(float(obj[5]), 0), h - 1)
                    x4 = min(max(float(obj[6]), 0), w - 1)
                    y4 = min(max(float(obj[7]), 0), h - 1)
                    # TODO: filter small instances
                    xmin = max(min(x1, x2, x3, x4), 0)
                    xmax = max(x1, x2, x3, x4)
                    ymin = max(min(y1, y2, y3, y4), 0)
                    ymax = max(y1, y2, y3, y4)
                    if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
                        valid_pts.append([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
                        valid_cat.append(self.cat_ids[obj[8]])
                        valid_dif.append(int(obj[9]))
        f.close()
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)
        # pts0 = np.asarray(valid_pts, np.float32)
        # img = self.load_image(index)
        # for i in range(pts0.shape[0]):
        #     pt = pts0[i, :, :]
        #     tl = pt[0, :]
        #     tr = pt[1, :]
        #     br = pt[2, :]
        #     bl = pt[3, :]
        #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
        #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
        #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
        #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
        #     cv2.putText(img, '{}:{}'.format(valid_dif[i], self.category[valid_cat[i]]), (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
        #                 (0, 0, 255), 1, 1)
        # cv2.imshow('img', np.uint8(img))
        # k = cv2.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        return annotation


    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)