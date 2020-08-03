import os
import wget
import json

with open("train_annotations.json") as f:
    ltrai = [_ for _ in json.loads(f.readlines()[0]).keys()]
with open("val_video_ids.txt") as f:
    lvals = [k.strip("\n") for k in f]
title = "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531798"

def getit(lid, subset, subtyp, subout):
    """@lid:ID;@subset:train,val;@subtyp:video,i3d_feature,vggish_feature;@subout:OUTPUT"""

    for kid in lid:
        url = "{}/{}/{}/{}.npy".format(title, subset, subtyp, kid)
        if not os.path.isdir('{}'.format(subout)):
            os.makedirs('{}'.format(subout))
        if not os.path.exists("{}/{}_{}.npy".format(subout, subtyp, kid)):
            try:
                wget.download(url, out="{}/{}.npy".format(subout, kid))
            except Exception as e:
                print(n, kid, e)
getit(ltrai, "train", "i3d_feature", "../data/Train/i3d_features")
getit(lvals, "val", "i3d_feature", "../data/Round1_Test/i3d_features")
