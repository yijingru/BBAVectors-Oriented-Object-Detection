from .base import BaseDataset
import os
import cv2
import numpy as np
import sys
from .hrsc_evaluation_task1 import voc_eval


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class HRSC(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(HRSC, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['ship']
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'AllImages')
        self.label_path = os.path.join(data_dir, 'Annotations')

    def load_img_ids(self):
        image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists


    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.bmp')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.xml')

    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []
        target = ET.parse(self.load_annoFolder(self.img_ids[index])).getroot()
        for obj in target.iter('HRSC_Object'):
            difficult = int(obj.find('difficult').text)
            box_xmin = int(obj.find('box_xmin').text)  # bbox
            box_ymin = int(obj.find('box_ymin').text)
            box_xmax = int(obj.find('box_xmax').text)
            box_ymax = int(obj.find('box_ymax').text)
            mbox_cx = float(obj.find('mbox_cx').text)  # rbox
            mbox_cy = float(obj.find('mbox_cy').text)
            mbox_w = float(obj.find('mbox_w').text)
            mbox_h = float(obj.find('mbox_h').text)
            mbox_ang = float(obj.find('mbox_ang').text)*180/np.pi
            rect = ((mbox_cx, mbox_cy), (mbox_w, mbox_h), mbox_ang)
            pts_4 = cv2.boxPoints(rect)  # 4 x 2
            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]
            valid_pts.append([bl, tl, tr, br])
            valid_cat.append(self.cat_ids['ship'])
            valid_dif.append(difficult)
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)

        # img = self.load_image(index)
        # for rect in annotation['rect']:
        #     pts_4 = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))  # 4 x 2
        #     bl = pts_4[0,:]
        #     tl = pts_4[1,:]
        #     tr = pts_4[2,:]
        #     br = pts_4[3,:]
        #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
        #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
        #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
        #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
        # cv2.imshow('img', np.uint8(img))
        # k = cv2.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        return annotation

    def dec_evaluation(self, result_path):
        detpath = os.path.join(result_path, 'Task1_{}.txt')
        annopath = os.path.join(self.label_path, '{}.xml')  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
        imagesetfile = os.path.join(self.data_dir, 'test.txt')
        classaps = []
        map = 0
        for classname in self.category:
            if classname == 'background':
                continue
            print('classname:', classname)
            rec, prec, ap = voc_eval(detpath,
                                     annopath,
                                     imagesetfile,
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap
            # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
            print('{}:{} '.format(classname, ap*100))
            classaps.append(ap)
            # umcomment to show p-r curve of each category
            # plt.figure(figsize=(8,4))
            # plt.xlabel('recall')
            # plt.ylabel('precision')
            # plt.plot(rec, prec)
        # plt.show()
        map = map / len(self.category)
        print('map:', map*100)
        # classaps = 100 * np.array(classaps)
        # print('classaps: ', classaps)
        return map

