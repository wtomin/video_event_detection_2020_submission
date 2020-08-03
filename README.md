# video_event_detection_2020_submission
This repository contains the code to be submitted.

# Test Round1

There exists one trained model saved in `user_data/val_ratio_0.2_threshold_0.5_pos_weight_25.0_epochs_25/model/MLP_RNN_best_acc.pth.tar`. 

Simply run `sh test_example.sh` you can have the prediction json file in `user_data/val_ratio_0.2_threshold_0.5_pos_weight_25.0_epochs_25/output/`. This process will take about five minutes on GTX1080Ti.


# Train

1. `cd code/` and run `python download_dataset.py`. So that the i3d feature of Training set and Validation set are downloaded to `data/`

2. In 'code/' and 'cd data_balancing_sampler/', run 'python sampler.py'. This file will generate two files: one is 'user_data/Train/i3d_features.lmdb', which saves all training samples into a big binary file. The other one is `user_data/label_list.txt`, which saves all label names in a fixed order.

3. run 'sh train_example.sh', after the program stopped, in the `user_data/val_ratio_0.2_threshold_0.5_pos_weight_25.0_epochs_25` folder, there will be one subfolder named 'log' containing the training log, one subfolder named `model` containing all saved models.

# Method

The model is a two-layer MLP stacked with a bidirectional GRU model, the activation function applied to the final fully connected layer is sigmoid function. The input to this MLP-RNN model is a sequence of i3d features.

The highlight of this method is to handle the data balancing problem in this dataset. We print out the data distribution:

```
Print data distribution ...
The total number of videos: 44611
The total number of segments: 246269
Average number of segments in a video: 5.52
The total number of labels: 53
Label Number Total_Duration(s)
持枪 18769 189064.35
炒菜 246 1671.77
吸烟 4450 23658.87
使用计算机 2946 26566.71
手术 114 1790.50
唱歌 11271 204028.35
哭 13717 177961.74
跳水 204 929.28
滑冰 51 360.20
打斗 11797 178670.48
打电话 10682 168528.10
拖地 140 1104.13
使用手机 4536 39070.00
跳舞 10460 200667.47
健身 349 4884.07
亲吻 2588 16091.35
踢足球 530 3919.83
拥抱 6705 56658.74
骑自行车 1592 12216.05
跑 19521 131677.29
刷牙 123 893.82
扫地 200 1482.40
牵手 4805 36487.38
宴会 299 6627.20
拉小提琴 685 6266.78
挥手 6237 40312.83
现代会议 783 24379.16
打篮球 1247 16967.25
打排球 69 561.16
打架子鼓 1640 14022.57
骑摩托车 2034 17302.38
游泳 405 3996.53
弹吉他 3306 31468.16
婚礼 292 6067.91
战争对抗 1056 33908.01
写作 4695 33586.51
吃饭/吃东西 7512 70398.67
化妆 411 2635.29
打高尔夫球 52 319.93
饮酒/喝液体 9093 35813.69
玩纸牌类游戏 269 4314.32
挽手臂/搀扶 9312 75950.18
上课 336 10961.38
回头 1184 3300.64
弹钢琴 916 10022.04
阅读 15827 128900.68
驾驶汽车 5779 52855.66
鼓掌 23296 113825.22
射击 5226 57692.73
骑马 3409 36030.36
行走 6031 40055.50
持刀 9018 97167.44
太极 54 878.27
Average segment duration :9.97 s
Max duration 448.12s, min duration: 1.00s,  median duraton: 4.88s
label 吸烟 with median duartion: 23658.87s
```

We find that label `吸烟` has the median duration. We downsample the other labels if the label has longer duration than `吸烟`, otherwise we oversample the labels if the label has shorter duration than `吸烟`. The data sampling is executed in `code/data_balancing_sampler/sampler.py`

