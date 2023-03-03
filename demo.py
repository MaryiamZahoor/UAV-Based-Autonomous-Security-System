#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

r"""Generate tracking results for videos using Siamese Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

import cv2
#import trackDrone


# CURRENT_DIR = osp.dirname(__file__)
# sys.path.append(CURRENT_DIR)
import datetime
from SiameseTracker import SiameseTracker


def preprocess(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return res


def postprocess(img):
    res = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return res


def main():
    # debug = 0 , no log will produce
    # debug = 1 , will produce log file
    tracker = SiameseTracker(debug=0)
    time_per_frame = 0
    writer = None

    if len(sys.argv) <= 1:
        print('[ERROR]: File path error!')
        return

    if sys.argv[1] == "cam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(sys.argv[1])

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = preprocess(frame)
        cv2.imshow('frame', postprocess(frame))
        if cv2.waitKey(1500) & 0xFF == ord('o'):
            break

    # select ROI and initialize the model
    r = cv2.selectROI(postprocess(frame))
    cv2.destroyWindow("ROI selector")
    print('ROI:', r)
    tracker.set_first_frame(frame, r)
    
    centerX= r[0] + 0.5 * r[2]
    centerY= r[1] + 0.5 * r[3]

    diffX = 0
    diffY = 0 
    count = 1

    while True:
        ret, frame = cap.read()
        frame = preprocess(frame)
        start_time = datetime.datetime.now()
        reported_bbox = tracker.track(frame)
        end_time = datetime.datetime.now()

        # Display the resulting frame
        #print(reported_bbox)
        
        if(count == 40):
            diffX = (reported_bbox[0] + 0.5 * reported_bbox[2]) - centerX
            diffY = (reported_bbox[1] + 0.5 * reported_bbox[3]) - centerY
        
            centerX = reported_bbox[0] + 0.5 * reported_bbox[2]
            centerY = reported_bbox[1] + 0.5 * reported_bbox[3]

            count = 1

            print("(",diffX,",", diffY,") ,")
            #trackDrone.track_person(diffX,diffY)

        count = count + 1
    


        #cv2.putText(frame, ".", (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (int(reported_bbox[0]), int(reported_bbox[1])),
                      (
                          int(reported_bbox[0]) + int(reported_bbox[2]),
                          int(reported_bbox[1]) + int(reported_bbox[3])),
                      (0, 0, 255), 2)

        duration = end_time - start_time
        time_per_frame = 0.9 * time_per_frame + 0.1 * duration.microseconds
        cv2.putText(frame, 'FPS ' + str(round(1e6 / time_per_frame, 1)),
                    (30, 50), 0, 1, (0, 0, 255), 3)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output/demo_output.avi", fourcc, 20, (frame.shape[1], frame.shape[0]), True)
        
        if writer is not None:
            writer.write(postprocess(frame))
        
        cv2.imshow('frame', postprocess(frame))

        #trackDrone.track_person(diffX,diffY)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


main()
