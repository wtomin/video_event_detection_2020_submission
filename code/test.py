# test
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from data.dataloader import DataLoader
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable as Variable
from models.MLP_RNN import MLP_RNN
from sklearn.metrics import f1_score
from tqdm import tqdm
from data.video_dataset import Video_Dataset
from utils.dump_utils import dump
from data_balancing_sampler.sampler import balancing_sampler
import json
label_list_file = '../user_data/label_list.txt'
with open(label_list_file, 'r') as output:
    label_list = output.readlines()[0]
    label_list = label_list.strip().split(',')

parser = argparse.ArgumentParser(description="PyTorch implementation of video event detection")
parser.add_argument('--output_path', type=str)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--val_list_file', type=str, default = '../data/Round1_Test/val_video_ids.txt')
# ========================= Model Configs ==========================
parser.add_argument('--videos_dir', type=str, default ='../data/Round1_Test/i3d_features/')
parser.add_argument('--hidden_units', default=[1024, 256, 256], type=int, nargs="+",
                    help='hidden units set up')
parser.add_argument('--checkpoint', type=str, help="the checkpoint to be evaluated")
parser.add_argument('--gpus', nargs='+', type=int, default=None)
args = parser.parse_args()

def read_video_list(file):
    with open(file, 'r') as f:
        videos = f.readlines()
    videos = [v.strip() for v in videos]
    return videos
def test_on_video(model, dataset):
    model.eval()
    video_preds = []
    video_preds_ids = []
    for i, data_seq in enumerate(dataset):
        image = data_seq['image']
        video_preds_ids.append(data_seq['IDs'])
        image_var = Variable(image.type('torch.FloatTensor'))
        if args.gpus is not None:
            image_var = image_var.cuda()
        output = model(image_var.unsqueeze(0))
        output = torch.sigmoid(output)
        video_preds.append(output.squeeze(0).data.cpu().numpy())
    prev_id = -1
    for ids in video_preds_ids:
        assert prev_id +1 == ids[0], "ids not correct, might have missing features in video loading"
        prev_id = ids[1]
    assert prev_id == len(dataset.video_feature)-1
    video_preds = np.concatenate(video_preds, axis=0)
    return video_preds

def test():
    num_class = 53
    setattr(args, 'num_class', num_class)
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    ###########################  Create the classifier ###################       
    model = MLP_RNN(args.hidden_units, args.num_class)  
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params: {}".format(pytorch_total_params))
    if args.gpus is not None:
        if len(args.gpus)!=1:
            model = nn.DataParallel(model)
        model.cuda()
    data_dict = torch.load(args.checkpoint)
    model.load_state_dict(data_dict['state_dict'])
    print("checkpoint loaded from {}th epoch".format(data_dict['epoch']))

    ########################### Load validation dataset videos ###############
    val_videos = read_video_list(args.val_list_file)
    val_videos = [s for s in val_videos if len(s)!=0]
    print("{} videos to test.".format(len(val_videos)))
    video_outs = {}
    for video_name in tqdm(val_videos, total = len(val_videos)):
        dataset = Video_Dataset(video_name, os.path.join(args.videos_dir, video_name+'.npy'), args.seq_len, transform = lambda x: torch.Tensor(x))
        video_preds = test_on_video(model, dataset)
        video_content = dump(video_preds, label_list, video_name, threshold=0.5)
        video_outs.update(video_content)

    json_file = args.output_path
    with open(json_file, 'w', encoding='utf-8') as outfile:
        json.dump(video_outs, outfile, ensure_ascii=False)
if __name__ == '__main__':
    test()
    # save to submit/
    json_file = args.output_path
    import datetime
    from zipfile import ZipFile
    zip_file = '../submit/submit_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.zip'
    if not os.path.exists('../submit'):
        os.makedirs('../submit')
    with ZipFile(zip_file, 'w') as zipObj:
        # Add multiple files to the zip
        zipObj.write(json_file, os.path.basename(json_file))





