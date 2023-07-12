import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from scipy.ndimage import label
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import time
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
import sys


def set_disconnected_blocks_to_true(array):
    labeled_array, num_labels = label(~array)
    boundary_labels = np.unique(np.concatenate((labeled_array[0], labeled_array[-1], labeled_array[:, 0], labeled_array[:, -1])))
    connected_to_boundary_mask = np.isin(labeled_array, boundary_labels)
    array[~connected_to_boundary_mask] = True

def expand_mask(pred_mask):
    t = time.time()
    pred_mask = binary_dilation(pred_mask > 0, iterations = 10, structure = np.ones((3, 3), dtype = bool))
    set_disconnected_blocks_to_true(pred_mask)
    return np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)


# choose good parameters in sam_args based on the first frame segmentation result
# other arguments can be modified in model_args.py
# note the object number limit is 255 by default, which requires < 10GB GPU memory with amp
sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    }

# Set Text args
'''
parameter:
    grounding_caption: Text prompt to detect objects in key-frames
    box_threshold: threshold for box 
    text_threshold: threshold for label(text)
    box_size_threshold: If the size ratio between the box and the frame is larger than the box_size_threshold, the box will be ignored. This is used to filter out large boxes.
    reset_image: reset the image embeddings for SAM
'''
grounding_caption = "bluecable"
box_threshold, text_threshold, box_size_threshold, reset_image = 0.25, 0.25, 0.5, True

# For every sam_gap frames, we use SAM to find new objects and add them for tracking
# larger sam_gap is faster but may not spot new objects in time
segtracker_args = {
    'sam_gap': 100, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
}


folder_dir = '/cephfs/dataset/'
input_dir = os.path.join('/cephfs/dataset/', sys.argv[1])
output_dir = os.path.join('/cephfs/dataset/', sys.argv[1]+'_processed')


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for file in os.listdir(input_dir):
    if not 'mp4' in file or os.path.exists(os.path.join(output_dir, file)):
        continue

    # source video to segment
    cap = cv2.VideoCapture(os.path.join(input_dir, file))

    # output masks
    pred_list = []
    masked_pred_list = []

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = segtracker_args['sam_gap']
    frame_idx = 0
    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()

    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if frame_idx == 0:
                pred_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                # pred_mask = cv2.imread('./debug/first_frame_mask.png', 0)
                torch.cuda.empty_cache()
                gc.collect()
                segtracker.add_reference(frame, pred_mask)
            elif (frame_idx % sam_gap) == 0:
                seg_mask, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = segtracker.track(frame)
                # find new objects, and update tracker with new objects
                if seg_mask is not None:
                    new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
                else:
                    new_obj_mask = np.zeros_like(track_mask)
                if np.sum(new_obj_mask > 0) >  frame.shape[0] * frame.shape[1] * 0.4:
                    new_obj_mask = np.zeros_like(new_obj_mask)
                pred_mask = track_mask + new_obj_mask
                # segtracker.restart_tracker()
                segtracker.add_reference(frame, pred_mask)
            else:
                pred_mask = segtracker.track(frame,update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()
            
            pred_list.append(pred_mask)
            
            
            print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')
            frame_idx += 1
        cap.release()
        print('\nfinished')

    # draw pred mask on frame and save as a video
    cap = cv2.VideoCapture(os.path.join(input_dir, file))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = imageio.get_writer(os.path.join(output_dir, file), fps=fps, macro_block_size=8)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pred_mask = pred_list[frame_idx]
        # masked_frame = frame * expand_mask(pred_mask)
        masked_frame = frame * np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)
        rgb_image = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        out.append_data(rgb_image)
        print('frame {} writed'.format(frame_idx),end='\r')
        frame_idx += 1
    out.close()
    cap.release()
    print("\n{} saved".format(os.path.join(output_dir, file)))
    print('\nfinished')
    # manually release memory (after cuda out of memory)
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()