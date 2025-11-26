import os,io,shutil
import geometry
import wandb
from matplotlib import cm
import cv2
from tqdm import tqdm
import torchvision
import time
from torchvision.utils import make_grid,draw_keypoints
import torch.nn.functional as F
import kornia
import numpy as np
import torch
import flow_vis
import flow_vis_torch
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import models
import piqa
import imageio
from PIL import Image
#import splines.quaternion
#from torchcubicspline import (natural_cubic_spline_coeffs, NaturalCubicSpline)
from scipy import spatial
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

ch_fst = lambda src,x=None:rearrange(src,"... (x y) c -> ... c x y",x=int(src.size(-2)**(.5)) if x is None else x)
ch_sec = lambda x: rearrange(x,"... c x y -> ... (x y) c")

def wandb_summary(loss, model_output, model_input, ground_truth, resolution,prefix="",suffix="",step=0,losses_agg=[]):
    model_output,model_input,ground_truth = [{k:(v[:32] if len(v.shape) else v) for k,v in x.items()} for x in (model_output,model_input,ground_truth)]

    resolution = list(model_input["rgb"].flatten(0,1).permute(0,2,3,1).shape)
    resolution[0]=ground_truth["rgb"].size(1)*ground_truth["rgb"].size(0)
    nrow=model_input["rgb"].size(1)
    imsl=model_input["rgb"].shape[-2:]
    inv = lambda x : 1/(x+1e-8)

    # Convert depths to colormapped 3-channel images:
    for k,v in list(model_output.items()): # magma colormap for depth -- todo change to depth_colored instead of depth to avoid ambiguity
        if type(v)!=list and len(v.shape): v=v.clip(min=.001)
        if "depth" in k: model_output[k+"_raw"] = v
        #if "depth" in k and "raw" not in k: model_output[k+"vis"] = v.expand(-1,-1,-1,3)#torch.from_numpy(cm.get_cmap('magma')(v.min().item()/v.cpu().numpy())).squeeze(-2)[...,:3]
        if "depth" in k and "raw" not in k: model_output[k+"vis"] = torch.from_numpy(cm.get_cmap('magma')(v.min().item()/v.cpu().numpy())).squeeze(-2)[...,:3]

    wandb_out = {}

    #if step%50==0:
    wandb_out["ref/rgb_gt"]= make_grid(model_input["rgb"].cpu().flatten(0,1).detach()*.5+.5,nrow=nrow)

    if "lie_perpix" in model_output:
        rot_vis=kornia.geometry.conversions.quaternion_to_axis_angle(model_output["lie_perpix"][...,:4]).flatten(0,1).permute(0,3,1,2)*.5+.5
        trans_vis = model_output["lie_perpix"][...,-3:].flatten(0,1).permute(0,3,1,2)/5+.5
        wandb_out["est/poses_lie_rot_perpix"]= make_grid(rot_vis.detach(),nrow=nrow,normalize=False)
        wandb_out["est/poses_lie_trans_perpix"]= make_grid(trans_vis.detach(),nrow=nrow,normalize=False)
    if "rig_masks" in model_output:
        low_res=imsl#(64,64)
        wandb_out["est/rig_masks"]= make_grid(rearrange(model_output["rig_masks"],"b t o (x y) 1 -> (b t o) 1 x y",x=low_res[0]).detach(),nrow=model_output["rig_masks"].size(2))
        wandb_out["est/rig_masks_corr_weighted"]= make_grid((F.interpolate(model_output["corr_weights"][0],low_res).unsqueeze(2)*ch_fst(model_output["rig_masks"][0,1:],low_res[0])
                                                                    ).flatten(0,1).detach(),nrow=model_output["rig_masks"].size(2))
        wandb_out["est/rig_masks_corr_weighted_rgb"]= make_grid((F.interpolate(model_input["rgb"][0,1:],low_res).unsqueeze(1)*ch_fst(model_output["rig_masks"][0,1:],low_res[0])*
                                                                    F.interpolate(model_output["corr_weights"][0],low_res).unsqueeze(2)
                                                                    ).flatten(0,1).detach(),nrow=model_output["rig_masks"].size(2))
        wandb_out["est/rig_masks_rgb"]= make_grid(rearrange(model_output["rig_masks"].flatten(0,1)*(F.interpolate(model_input["rgb"].flatten(0,1),low_res).flatten(-2,-1).permute(0,2,1).unsqueeze(1)*.5+.5),
                                                "bt o (x y) c -> (bt o) c x y",x=low_res[0]).detach(),nrow=model_output["rig_masks"].size(2))
    if "depth_inpvis" in model_output: wandb_out["est/depth_inp"]=make_grid(model_output["depth_inpvis"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow)
    if "res_depthvis" in model_output: wandb_out["est/res_depth"]=make_grid(model_output["res_depthvis"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow)
    if "depthvis" in model_output: wandb_out["est/depth"]=make_grid(model_output["depthvis"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow)
    if "depth_raw" in model_output: 
        wandb_out["est/depth_raw"]=make_grid(model_output["depth_raw"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow,normalize=True)
        wandb_out["est/depth_raw_inv"]=make_grid(inv(model_output["depth_raw"]).cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow,normalize=True)

    if "corr_weights" in model_output: wandb_out["est/corr_weights"] = make_grid(model_output["corr_weights"].flatten(0,1).cpu().detach(),normalize=True,nrow=nrow)
    if "corr_weights_static" in model_output: wandb_out["est/corr_weights_static"] = make_grid(model_output["corr_weights_static"].flatten(0,1).cpu().detach(),normalize=True,nrow=nrow)
    if "bwd_flow" in model_input: wandb_out["ref/flow_gt_bwd"]= flow_vis_torch.flow_to_color(make_grid(model_input["bwd_flow"].flatten(0,1),nrow=nrow-1))/255
    if "rig_flow_masks" in model_input: 
        wandb_out["ref/rig_flow_masks"]= make_grid(rearrange(model_input["rig_flow_masks"],"b t o x y -> (b t o) 1 x y"),nrow=model_input["rig_flow_masks"].size(1))
    if "flow_from_pose" in model_output and not torch.isnan(model_output["flow_from_pose"]).any(): 
        wandb_out["est/flow_est_pose"] = flow_vis_torch.flow_to_color(make_grid(model_output["flow_from_pose"].clip(-.1,.1).flatten(0,1).permute(0,2,1).unflatten(-1,imsl),nrow=nrow-1))/255

    if "affinity_emb" in model_output: 
        for suff in ["","_unnorm"][:1]:
            aff_emb = model_output["affinity_emb"+suff]
            if aff_emb.size(2)<3: aff_emb = torch.cat((aff_emb,torch.zeros_like(aff_emb[:,:,[0]]).expand(-1,-1,3-aff_emb.size(2),-1,-1)),2)
            features=rearrange(aff_emb.flatten(0,1),"bt c x y -> 1 c (bt x) y")
            B, C, H, W = features.shape
            features = features.view(B, C, -1)
            # Center the data
            features_mean = features.mean(dim=2, keepdim=True)
            features = features - features_mean
            covariance = torch.bmm(features, features.transpose(1, 2)) / (H * W - 1)
            # Perform SVD
            U, S, V = torch.svd(covariance)
            # Project the data onto the top principal components
            num_components=min(6,C)
            transformed_features = torch.bmm(U[:, :, :num_components].transpose(1, 2), features)
            # Reshape back to original spatial dimensions
            wandb_out["est/affinity_emb0to3"+suff]= make_grid(rearrange(transformed_features[:,:3].detach(),"1 c (bt x y) -> (bt) c x y",y=aff_emb.size(-1),x=aff_emb.size(-2)), nrow=model_output["affinity_emb"].size(1),normalize=True)
            wandb_out["est/affinity_emb3to6"+suff]= make_grid(rearrange(transformed_features[:,3:].detach(),"1 c (bt x y) -> (bt) c x y",y=aff_emb.size(-1),x=aff_emb.size(-2)), nrow=model_output["affinity_emb"].size(1),normalize=True)


    # Visualize point track reprojection error
    if "pred_tracks" in model_input:
        # Plot tracks as flow image

        sl=64 if model_input["pred_tracks"].size(-2)%64**2==0 else 42
        nrow_=int(model_input["pred_tracks"].size(-2)//sl**2)
        low_res=(sl,sl)
        uv = np.mgrid[0 : low_res[0], 0 : low_res[1]].astype(float).transpose(1, 2, 0)
        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        uv = uv / torch.tensor([low_res[1]-1, low_res[0]-1])  # uv in [0,1]
        track_unp = lambda x: rearrange(x,"b t (x y s) c -> (b t s) c x y",y=sl,x=sl)
        wandb_out["ref/track_flow_gt"] = flow_vis_torch.flow_to_color(make_grid((track_unp(model_input["pred_tracks"]*model_input["pred_visibility"].unsqueeze(-1)) -
                                                                            uv.permute(2,0,1)[None].cuda())*track_unp(model_input["pred_visibility"].unsqueeze(-1)),nrow=nrow_))/255

    if "aff_sim_grid" in model_output:
        wandb_out["est/aff_grid"] = make_grid(model_output["aff_sim_grid"].flatten(0,2)[:,None],nrow=16)
    if "point_track_reproj" in model_output:
        # First just error image
        point_track_err = ( (model_output["point_track_reproj"] - model_input["pred_tracks"]) * model_input["pred_visibility"].unsqueeze(-1) ).abs()
        err_img = make_grid(rearrange(point_track_err,"b t (x y s) c -> (b t s) c x y",y=sl,x=sl),nrow=nrow_)*4#/255
        wandb_out["metrics/track_err_x"] = err_img[[0]].expand(3,-1,-1)
        wandb_out["metrics/track_err_y"] = err_img[[1]].expand(3,-1,-1)

        wandb_out["est/track_flow_est"] = flow_vis_torch.flow_to_color(make_grid((track_unp(model_output["point_track_reproj"]*model_input["pred_visibility"].unsqueeze(-1)) -
                                                                            uv.permute(2,0,1)[None].cuda())*track_unp(model_input["pred_visibility"].unsqueeze(-1)),nrow=nrow_))/255
        if "rig_pertrack" in model_output: wandb_out["est/rig_samps"] = make_grid(track_unp(model_output["rig_pertrack"][None,...,None]),nrow=nrow_)

        if "aff_emb_pertrack" in model_output: 
            features = rearrange(model_output["aff_emb_pertrack"],"b (x y s) c -> s c (b x y)",y=sl,x=sl)[[0]]
            # Center the data
            features_mean = features.mean(dim=2, keepdim=True)
            features = features - features_mean
            covariance = torch.bmm(features, features.transpose(1, 2)) / (features.size(-1) - 1)
            # Perform SVD
            U, S, V = torch.svd(covariance)
            # Project the data onto the top principal components
            num_components=min(3,features.size(1))
            transformed_features = torch.bmm(U[:, :, :num_components].transpose(1, 2), features)
            # Reshape back to original spatial dimensions
            wandb_out["est/affinity_emb_track"]= make_grid(rearrange(transformed_features.detach(),"1 c (b x y) -> b c x y",y=sl,x=sl), normalize=True)

        if "poses_all" in model_output: 
            poses_lie = torch.cat((kornia.geometry.conversions.rotation_matrix_to_quaternion(model_output["poses_all"][...,:3,:3],eps=1e-5),model_output["poses_all"][...,:3,-1]),-1)
            rot_vis=kornia.geometry.conversions.quaternion_to_axis_angle(poses_lie[...,:4])*.5+.5
            trans_vis = poses_lie[...,-3:]/5+.5
            wandb_out["est/poses_lie_rot_all"]= make_grid( rearrange(rot_vis.detach(),"b (x y s) t c -> (b s t) c x y ",y=sl,x=sl), nrow=poses_lie.size(2),normalize=False)
            wandb_out["est/poses_lie_trans_all"]= make_grid( rearrange(trans_vis.detach(),"b (x y s) t c -> (b s t) c x y ",y=sl,x=sl), nrow=poses_lie.size(2),normalize=False)

            rgbcamimgs = torch.stack((
                rearrange(model_output["rgb_pertrack"].detach(),"b (x y s) c -> b s c x y ",y=sl,x=sl)[:,0]*.5+.5,
                rearrange(rot_vis.detach(),"b (x y s) t c -> b s t c x y ",y=sl,x=sl)[:,0,-1],
                rearrange(trans_vis.detach(),"b (x y s) t c -> b s t c x y ",y=sl,x=sl)[:,0,-1],
                ),1).flatten(0,1)
            wandb_out["est/rgb_camimgs"]= make_grid( rgbcamimgs, nrow=3,normalize=False)
            #plt.imsave("/home/cameronsmith/tmp.png",wandb_out["est/rgb_camimgs"].permute(1,2,0).cpu().numpy())

    if 0: # save locally
        for k,v in wandb_out.items(): print(k,v.max(),v.min())
        for k,v in wandb_out.items():
            print(k,v.shape)
            plt.imsave("output/img/%s.png"%k,v.float().permute(1,2,0).detach().cpu().numpy().clip(0,1));
        print("saving locally")
        zz

    print("logging images",print(len(wandb_out)))
    for k,v in wandb_out.items():print(k,v.shape)
    wandb.log({prefix+k:wandb.Image(v.permute(1, 2, 0).float().detach().clip(0,1).cpu().numpy()) for k,v in wandb_out.items()})
    print("done logging images")
    return wandb_out
