import matplotlib.pyplot as plt; 
import cv2
import os
import statistics 
import multiprocessing as mp
import torch.nn.functional as F
import torch
import random
import imageio
import numpy as np
from glob import glob
from collections import defaultdict
from pdb import set_trace as pdb
from itertools import combinations
from random import choice
import matplotlib.pyplot as plt
import imageio.v3 as iio

from torchvision import transforms

import sys

from glob import glob
import os
import gzip
import json
import numpy as np

from einops import rearrange, repeat
ch_sec = lambda x: rearrange(x,"... c x y -> ... (x y) c")
hom = lambda x, i=-1: torch.cat((x, torch.ones_like(x.unbind(i)[0].unsqueeze(i))), i)

def make_sample(sample,aspect,budget=192*640/4,hires_factor=2,med_factor=1,low_res=None,hi_res=None):
    
    y=np.sqrt(budget/aspect)
    x=budget/y
    low_res_=[int(y),int(x)]
    mult32=lambda x:x-(x%32)+32
    if low_res is None: low_res=[mult32(x) for x in low_res_]
    if hi_res is None: hi_res=[mult32(int(hires_factor*x)) for x in low_res_]
    med_res=[mult32(int(med_factor*x)) for x in low_res_]

    #print("making sample")
    uv = np.mgrid[0 : low_res[0], 0 : low_res[1]].astype(float).transpose(1, 2, 0)
    uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
    uv = uv / torch.tensor([low_res[1]-1, low_res[0]-1])  # uv in [0,1]

    uv_hires = np.mgrid[0 : hi_res[0], 0 : hi_res[1]].astype(float).transpose(1, 2, 0)
    uv_hires = torch.from_numpy(np.flip(uv_hires, axis=-1).copy()).long()
    uv_hires = uv_hires / torch.tensor([hi_res[1]-1, hi_res[0]-1])  # uv in [0,1]

    model_input,gt={},{}
    model_input["rgb"]= F.interpolate(sample["rgb"],low_res,antialias=True,mode="bilinear")

    if "dino_pca" in sample: model_input["dino_pca"]= F.interpolate(sample["dino_pca"],low_res,antialias=True,mode="bilinear")

    if "bwd_flow" in sample: model_input["bwd_flow"]= F.interpolate(sample["bwd_flow"],low_res,antialias=True,mode="bilinear")
    model_input["rig_flow_masks"]= F.interpolate(sample["rig_flow_masks"].flatten(0,1)[:,None].float(),low_res,mode="nearest").squeeze(1).unflatten(0,sample["rig_flow_masks"].shape[:2])

    if "pred_tracks" in sample:
        model_input["pred_tracks"]= sample["pred_tracks"]
        model_input["pred_visibility"]= sample["pred_visibility"]

    model_input["x_pix"]=uv[None].flatten(1,2).expand(len(model_input["rgb"]),-1,-1)
    gt["rgb"]=ch_sec(model_input["rgb"])*.5+.5

    if "intrinsics" in sample: model_input["gt_intrinsics"]=model_input["intrinsics"]=sample["intrinsics"]
    if "depth_inp" in sample:
        gt["depth_inp"]=model_input["depth_inp"]= ch_sec(F.interpolate(sample["depth_inp"][:,None],low_res,mode="nearest"))
    if "seg_imgs" in sample:
        gt["seg_imgs"]=model_input["seg_imgs"]= ch_sec(F.interpolate(sample["seg_imgs"].float(),low_res,mode="nearest"))
    if "c2w" in sample: model_input["c2w"]=sample["c2w"]
    if "org_ratio" in sample: model_input["org_ratio"]=sample["org_ratio"]
    #print("done making sample")
    return model_input,gt


class ImageFolder(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(
        self,
        n_skip=1,
        num_trgt=1,
        low_res=(96,112),
        path=".",
        val=False,
        sf=1,# img scale factor (fractional makes it cheaper)
    ):

        self.n_trgt=num_trgt-1
        self.val=val
        self.num_skip=n_skip
        self.low_res=torch.tensor(low_res)
        self.sf=sf

        print("Loading data")
        self.path=path
        #try:self.dino_feats = torch.load(path+"/dino_feats.pt")
        #except:print("no dino feats")
        try: self.tracks = list(torch.load(path+"/pred_tracks_offline.pt"))
        except: 
            self.tracks = list(torch.load(path+"/pred_tracks_more.pt"))
            #except: self.tracks = list(torch.load(path+"/pred_tracks.pt"))
        self.imgs = torch.load(path+"/imgs.pt")
        try:self.seg_imgs = torch.load(path+"/seg_imgs.pt")
        except:pass
        try:self.dino_feats = torch.load(path+"/dino_feats.pt",map_location="cpu")
        except:pass
        self.bwd_flow = torch.load(path+"/bwd_flow.pt")
        self.rig_flow_masks = torch.load(path+"/rig_flow_masks.pt")[:,:1]
        self.tracks[0] = rearrange(self.tracks[0],"g b t p c -> b t (p g) c")[0]
        self.tracks[1] = rearrange(self.tracks[1],"g b t p -> b t (p g)")[0]
        self.mdepths = torch.load(path+"/depth_ests.pt")
        self.depths=self.mdepths[0]
        
        print("Done loading data")
        self.poses = None
        if os.path.exists(path+"/poses.pt"): self.poses = torch.load(path+"/poses.pt")

        self.f= torch.load(path+"/intrinsics.pt")

    def __len__(self): return 1

    def collate_fn(self, batch_list):
        keys = batch_list[0].keys()
        result = defaultdict(list)

        for entry in batch_list:
            # make them all into a new dict
            for key in keys: result[key].append(entry[key])

        for key in keys:
            try: result[key] = torch.stack(result[key], dim=0)
            except: continue
        return result

    def __getitem__(self, idx,seq_query=None):

        idx=0

        context = []
        trgt = []
        post_input = []

        frames = self.imgs
        f=self.depths[1]
        depth_frames = self.depths

        if frames.max()>2: frames=frames/255

        intrinsics = repeat(torch.eye(3), "i j -> b i j", b=len(depth_frames)).clone()
        intrinsics[:, :2, 2] = 0.5
        f=self.f
        intrinsics[:, 0, 0] = f 
        intrinsics[:, 1, 1] = f * depth_frames.size(-1) / depth_frames.size(-2)

        org_ratio=frames[0].size(-2)/frames[0].size(-1)
        h,s=3,1
        hi_res=[640, 1024]

        pred_tracks = self.tracks[0][:self.n_trgt*self.num_skip:self.num_skip]
        pred_visibility = self.tracks[1][:self.n_trgt*self.num_skip:self.num_skip]
        #downsampling until more scalable approach
        #s=4
        gs=1
        track_sl=64
        pred_tracks = rearrange( rearrange(pred_tracks,"t (x y s) c -> (t s) c x y",y=track_sl,x=track_sl)[...,::gs,::gs], "(t s) c x y -> t (x y s) c",t=self.n_trgt)
        pred_visibility = rearrange( rearrange(pred_visibility,"t (x y s) -> (t s) x y",y=track_sl,x=track_sl)[...,::gs,::gs], "(t s) x y -> t (x y s)",t=self.n_trgt)

        #self.rig_flow_masks=torch.ones_like(self.rig_flow_masks[:,:])
        sample = {
                "intrinsics":intrinsics[:self.n_trgt*self.num_skip:self.num_skip],
                "rgb":frames[:self.n_trgt*self.num_skip:self.num_skip]* 2-1,
                "dino_pca":self.dino_feats[:self.n_trgt*self.num_skip:self.num_skip],
                "depth_inp":depth_frames[:self.n_trgt*self.num_skip:self.num_skip],"org_ratio":org_ratio,
                "bwd_flow":self.bwd_flow[:self.n_trgt*self.num_skip:self.num_skip][:-1], 
                "rig_flow_masks":self.rig_flow_masks[:self.n_trgt*self.num_skip:self.num_skip][:-1], "pred_tracks":pred_tracks,
                "pred_visibility":pred_visibility,
                }
        if self.poses is not None: sample["c2w"]=self.poses[:self.n_trgt*self.num_skip:self.num_skip]
        switch=[1,-1][0]
        return make_sample(sample, 1/org_ratio,hires_factor=h,budget=192*640/(8//s),
                low_res=[int(128*self.sf),int(224*self.sf)][::switch],#[::[-1,1][frames.size(-1)>frames.size(-2)]],
                hi_res=hi_res[::-1]#[::[-1,1][frames.size(-1)>frames.size(-2)]])
                )
class MultiImageFolder(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(
        self,
        n_skip=1,
        num_trgt=1,
        low_res=(96,112),
        path=".",
        val=False,
        sf=1,# img scale factor (fractional makes it cheaper)
    ):
        #self.paths=glob("/data/cameron/monocular_ests/pets_dogs/*/lowrespkg.pt")
        self.paths=glob("sample_data/*lowrespkg.pt")
        self.step=0
        
    def __len__(self): return len(self.paths)

    def collate_fn(self, batch_list):
        keys = batch_list[0].keys()
        result = defaultdict(list)

        for entry in batch_list:
            # make them all into a new dict
            for key in keys: result[key].append(entry[key])

        for key in keys:
            print(key)
            try: result[key] = torch.stack(result[key], dim=0)
            except: continue
        return result

    def __getitem__(self, idx,seq_query=None):
        self.step+=1
        data= list(torch.load(self.paths[idx]))
        data=[{k:v for k,v in x.items() if type(v)!=float} for x in data]
        if any([x in self.paths[idx] for x in ["re10k","hydrant"]]):data[0]["rig_flow_masks"]=torch.ones_like(data[0]["rig_flow_masks"])
        if data[0]["rig_flow_masks"].size(0)!=9:return self[random.randint(0,len(self)-1)]
        return data
