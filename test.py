import torch
import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
import func_utils

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class TestModule(object):
    def __init__(self, dataset, num_classes, model, decoder):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.num_classes = num_classes
        self.model = model
        self.decoder = decoder

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=True)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def imshow_heatmap(self, pr_dec, images):
        wh = pr_dec['wh']
        hm = pr_dec['hm']
        cls_theta = pr_decs['cls_theta']
        wh_w = wh[0, 0, :, :].data.cpu().numpy()
        wh_h = wh[0, 1, :, :].data.cpu().numpy()
        hm = hm[0, 0, :, :].data.cpu().numpy()
        cls_theta = cls_theta[0, 0, :, :].data.cpu().numpy()
        images = np.transpose((images.squeeze(0).data.cpu().numpy() + 0.5) * 255, (1, 2, 0)).astype(np.uint8)
        wh_w = cv2.resize(wh_w, (images.shape[1], images.shape[0]))
        wh_h = cv2.resize(wh_h, (images.shape[1], images.shape[0]))
        hm = cv2.resize(hm, (images.shape[1], images.shape[0]))
        fig = plt.figure(1)
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_xlabel('width')
        ax1.imshow(wh_w)
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_xlabel('height')
        ax2.imshow(wh_h)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_xlabel('center hm')
        ax3.imshow(hm)
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_xlabel('input image')
        ax5.imshow(cls_theta)
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_xlabel('input image')
        ax6.imshow(images)
        plt.savefig('heatmap.png')


    def test(self, args, down_ratio):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=down_ratio)
        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        total_time = []
        for cnt, data_dict in enumerate(data_loader):
            image = data_dict['image'][0].to(self.device)
            img_id = data_dict['img_id'][0]
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))
            begin_time = time.time()
            with torch.no_grad():
                pr_decs = self.model(image)

            #self.imshow_heatmap(pr_decs[2], image)

            torch.cuda.synchronize(self.device)
            decoded_pts = []
            decoded_scores = []
            predictions = self.decoder.ctdet_decode(pr_decs)
            pts0, scores0 = func_utils.decode_prediction(predictions, dsets, args, img_id, down_ratio)
            decoded_pts.append(pts0)
            decoded_scores.append(scores0)
            #nms
            results = {cat:[] for cat in dsets.category}
            for cat in dsets.category:
                if cat == 'background':
                    continue
                pts_cat = []
                scores_cat = []
                for pts0, scores0 in zip(decoded_pts, decoded_scores):
                    pts_cat.extend(pts0[cat])
                    scores_cat.extend(scores0[cat])
                pts_cat = np.asarray(pts_cat, np.float32)
                scores_cat = np.asarray(scores_cat, np.float32)
                if pts_cat.shape[0]:
                    nms_results = func_utils.non_maximum_suppression(pts_cat, scores_cat)
                    results[cat].extend(nms_results)

            end_time = time.time()
            total_time.append(end_time-begin_time)

            #"""
            ori_image = dsets.load_image(cnt)
            height, width, _ = ori_image.shape
            # ori_image = cv2.resize(ori_image, (args.input_w, args.input_h))
            # ori_image = cv2.resize(ori_image, (args.input_w//args.down_ratio, args.input_h//args.down_ratio))
            #nms
            for cat in dsets.category:
                if cat == 'background':
                    continue
                result = results[cat]
                for pred in result:
                    score = pred[-1]
                    tl = np.asarray([pred[0], pred[1]], np.float32)
                    tr = np.asarray([pred[2], pred[3]], np.float32)
                    br = np.asarray([pred[4], pred[5]], np.float32)
                    bl = np.asarray([pred[6], pred[7]], np.float32)

                    tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
                    rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
                    bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
                    ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

                    box = np.asarray([tl, tr, br, bl], np.float32)
                    cen_pts = np.mean(box, axis=0)
                    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0,0,255),1,1)
                    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255,0,255),1,1)
                    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0,255,0),1,1)
                    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255,0,0),1,1)

                    # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tl[0]), int(tl[1])), (0,0,255),1,1)
                    # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tr[0]), int(tr[1])), (255,0,255),1,1)
                    # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(br[0]), int(br[1])), (0,255,0),1,1)
                    # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bl[0]), int(bl[1])), (255,0,0),1,1)
                    ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (255,0,255),1,1)
                    # box = cv2.boxPoints(cv2.minAreaRect(box))
                    # ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (0,255,0),1,1)
                    cv2.putText(ori_image, '{:.2f} {}'.format(score, cat), (box[1][0], box[1][1]),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1,1)

            if args.dataset == 'hrsc':
                gt_anno = dsets.load_annotation(cnt)
                for pts_4 in gt_anno['pts']:
                    bl = pts_4[0, :]
                    tl = pts_4[1, :]
                    tr = pts_4[2, :]
                    br = pts_4[3, :]
                    cen_pts = np.mean(pts_4, axis=0)
                    box = np.asarray([bl, tl, tr, br], np.float32)
                    box = np.int0(box)
                    cv2.drawContours(ori_image, [box], 0, (255, 255, 255), 1)

            cv2.imshow('pr_image', ori_image)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                exit()
            #"""

        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1./np.mean(total_time)))
