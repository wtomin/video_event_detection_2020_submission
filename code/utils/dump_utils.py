# from model outputs to json file
import numpy as np
import os
import json
def dump(output_array, label_list, video_id, threshold = 0.5, min_seq_len = 2,
    max_interval = 2):
    """
    {
    "videoId": [
        {
            "label": "打电话",
            "score": 0.64,
            "segment": [24.25,38.08]
        },
        {
            "label": "行走",
            "score": 0.77,
            "segment": [11.25, 19.37]
        }
    ],
    ...
}
    min_seq_len: the minimum sequence length (in feature frames: floor(T/8), 2 corresponds to 16 actual
    frames, in FPS=15, it's approximately 1 second long)
    max_interval: if two segments have the same label, and their interval is shorter than the max_interval, 
    the two segments should be merged to one segment
    """
    output_list = []
    num_frames, num_classes = output_array.shape
    for i_label in range(num_classes):
        output_preds = output_array[:, i_label]
        label_name = label_list[i_label]
        output_preds = np.where(output_preds>threshold, output_preds, 0.)
        # filter out too short segment
        segments = [[]]
        for i_feature in range(len(output_preds)):
            if output_preds[i_feature]!=0.:
                segments[-1].append(i_feature)
            else:
                if len(segments[-1])!=0:
                    segments.append([])
        if len(segments[-1]) == 0:
            segments = segments[:-1]
        if len(segments)!=0:
            segments = [segment for segment in segments if len(segment)>min_seq_len]
        if len(segments)>1:
            new_segments = [segments[0]]
            i_old = 1
            i_new = 0
            # merge too close segments
            while i_old < len(segments):
                last_end = new_segments[i_new][-1]
                next_start = segments[i_old][0]
                if next_start - last_end < max_interval:
                    last_start = new_segments[i_new][0]
                    next_end = segments[i_old][-1]
                    new_segments[i_new] = list(np.arange(last_start, next_end+1))
                    i_old += 1
                else:
                    new_segments.append(segments[i_old])
                    i_new +=1
                    i_old +=1
            segments = new_segments
        if len(segments)!=0:
            # translate to segment label, [start, end] and score
            for segment in segments:
                start, end = segment[0], segment[-1]
                assert end-start >= min_seq_len
                start_second, end_second = np.around(start*8*(1/15), decimals=2), np.around( end*8*(1/15), decimals = 2)
                output_list.append({"label":label_name, "segment":[start_second, end_second],
                    "score":  output_array[start:end+1, i_label].sum()/(end+1-start)})

    content =  {"{}".format(video_id): output_list}
    return content













