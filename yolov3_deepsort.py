from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import time
import argparse
import torch
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

r"""Generate tracking results for videos using Siamese Model"""


import os.path as osp
import sys

import cv2
#import trackDrone

import datetime
from SiameseTracker import SiameseTracker




# CURRENT_DIR = osp.dirname(__file__)
# sys.path.append(CURRENT_DIR)



def preprocess(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return res


def postprocess(img):
    res = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return res



class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        #if not use_cuda:
        #    raise UserWarning("Running in cpu mode!")

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names


    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        assert self.vdo.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        
    from goto import with_goto
    @with_goto
    def run(self):
        idx_frame = 0
        start_tracking = False
        roi = None
        
        diffX = 0
        diffY = 0
        count = 1

        while self.vdo.grab(): 
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            if(start_tracking == False):
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

                # do detection
                bbox_xywh, cls_conf, cls_ids = self.detector(im)
                if bbox_xywh is not None:
                    # select person class
                    mask = cls_ids==0

                    bbox_xywh = bbox_xywh[mask]
                    bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                    cls_conf = cls_conf[mask]

                    # do tracking
                    outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:,:4]
                        identities = outputs[:,-1]
                        ori_im, roi, start_tracking = draw_boxes(ori_im, bbox_xyxy, identities)
                        if(start_tracking == True):
                            tracker = SiameseTracker(debug=0)
                            time_per_frame = 0
                            frame = ori_im
                            frame = preprocess(frame)
                            r = roi
                            print('ROI:', r)
                            tracker.set_first_frame(frame, r)
                            centerX= r[0] + 0.5 * r[2]
                            centerY= r[1] + 0.5 * r[3]
                            goto .start_the_tracking
            else:
                label .start_the_tracking
                #r = postprocess(roi)

                while self.vdo.grab():
                    
                    if idx_frame % self.args.frame_interval:
                        continue
                    _, ori_im = self.vdo.retrieve()
                    idx_frame += 1

                    frame = preprocess(ori_im)

                    start_time = datetime.datetime.now()
                    reported_bbox = tracker.track(frame)
                    end_time = datetime.datetime.now()

                    if(count == 40):
                        diffX = (reported_bbox[0] + 0.5 * reported_bbox[2]) - centerX
                        diffY = (reported_bbox[1] + 0.5 * reported_bbox[3]) - centerY
            
                        centerX = reported_bbox[0] + 0.5 * reported_bbox[2]
                        centerY = reported_bbox[1] + 0.5 * reported_bbox[3]

                        count = 1

                        print("(",diffX,",", diffY,") ,")
                        #trackDrone.track_person(diffX,diffY)
                    
                    count = count + 1


                    cv2.rectangle(frame, (int(reported_bbox[0]), int(reported_bbox[1])),
                                (
                                    int(reported_bbox[0]) + int(reported_bbox[2]),
                                    int(reported_bbox[1]) + int(reported_bbox[3])),
                                (0, 0, 255), 2)
                    
                    duration = end_time - start_time
                    time_per_frame = 0.9 * time_per_frame + 0.1 * duration.microseconds

                    cv2.putText(frame, 'FPS ' + str(round(1e6 / time_per_frame, 1)),
                                (30, 50), 0, 1, (0, 0, 255), 3)

                    if self.args.save_path:
                        self.writer.write(postprocess(frame))
                



            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
