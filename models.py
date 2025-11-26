import torch,torchvision
from torch import nn
import kornia
import functools
from einops import rearrange, repeat
#import torchvision.ops as ops
from torch.nn import functional as F
import numpy as np
import sys, random, time, os
from copy import deepcopy
from matplotlib import cm
import wandb
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Generator, Dict
from collections import defaultdict
#import torchvision.transforms as T

sys.path.append("./third_party/co-tracker/")
from cotracker.utils.visualizer import Visualizer, read_video_from_path

import geometry

import torch_kmeans
from torch_kmeans import KMeans

# Lambda helpers
ch_sec = lambda x: rearrange(x, "... c x y -> ... (x y) c")
ch_fst = lambda src, x=None: rearrange( src, "... (x y) c -> ... c x y", x=int(src.size(-2) ** (0.5)) if x is None else x)
hom = lambda x: torch.cat((x, torch.ones_like(x[..., [0]])), -1)
unhom = lambda x: x[..., :-1] / (1e-5 + x[..., -1:])
grid_samp_ = lambda x, y, pad,mode: F.grid_sample( x, y * 2 - 1, mode=mode, padding_mode=pad)  # assumes y in [0,1] and moves to [-1,1]
grid_samp = lambda x, y, pad="border",mode="bilinear": ( grid_samp_(x, y, pad,mode) if len(x.shape) == 4 else grid_samp_(x.flatten(0, 1), y.flatten(0, 1),pad,mode).unflatten(0, x.shape[:2]))  
project = lambda crds, K: unhom(torch.einsum("b...cij,b...ckj->b...cki", K, crds))
warp = lambda crds, poses, K: project( torch.einsum("b...cij,b...ckj->b...cki", poses, hom(crds))[..., :3], K)
shuffle = lambda x: x[torch.randperm(len(x)).to(x)]

def make_net(dims):
    def init_weights_normal(m):
        if type(m) == nn.Linear:
            if hasattr(m, "weight"): nn.init.kaiming_normal_( m.weight, a=0.0, nonlinearity="relu", mode="fan_in")
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    net = nn.Sequential(*layers[:-1])
    net.apply(init_weights_normal)
    return net

static_solve = True
use_depth_inp = True
scratch_model=True

class SIRE(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

        # Our actual model
        self.time_stride = 1
        fdim = 64
        self.depth_est = make_net([fdim, fdim, 1])
        self.depth_conv = nn.Conv2d(fdim, 1, 3, padding=1)
        affinity_dim = 8
        self.affinities_conv = nn.Conv2d(fdim, affinity_dim, 3, padding=1)
        self.general_confidence_conv = nn.Conv2d(fdim, 1, 3, padding=1)
        self.general_confidence_conv_static = nn.Conv2d(fdim, 1, 3, padding=1)

        self.corr_weighter_perpoint = make_net([fdim * 2, 16, 1])

        #self.img_enc = nn.Sequential( ResnetFPN(in_ch=3 * self.time_stride, use_first_pool=True), nn.Conv2d(512, fdim * self.time_stride, 3, padding=1),)
        self.img_enc = ResnetFPN(in_ch=(3+use_depth_inp*0) * self.time_stride, use_first_pool=False)#, nn.Conv2d(512, fdim * self.time_stride, 3, padding=1),)
        #self.img_enc = smp.Unet( encoder_name="mobileone_s2",encoder_weights="imagenet", in_channels=3+use_depth_inp,classes=fdim) # from s0 to s4 from 4-13M param   
        
        self.step=0

        #self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=not scratch_model)
        #self.midas_out=self.midas.scratch.output_conv
        #self.midas.scratch.output_conv=nn.Identity()

    def forward_allpts(
        self, model_input, sample_pts=None, out={}
    ):  # run forward pass over all point tracks

        track_idxs_all = torch.arange(model_input["pred_tracks"].size(-2))

        outs = []
        # Run model over all track queries iteratively
        print("collecting perpoint queries")
        for i, track_idxs in enumerate( tqdm( track_idxs_all.chunk(len(track_idxs_all) // 300), leave=False, desc="collecting perpoint queries",)):
            # for i,track_idxs in enumerate(track_idxs_all.chunk(len(track_idxs_all)//300)):
            out = self(model_input, track_idxs=track_idxs)
            if len(outs): # for memory sake just save values to aggregate
               out={k:v for k,v in out.items() if k in ["poses_all","point_track_reproj","worldcrds_pertrack","aff_sim",][:]}
            if "aff_sim" in out:
                if i in torch.linspace(0, len(track_idxs_all.chunk(len(track_idxs_all)//100)), 50).long(): out["aff_sim"] = out["aff_sim"][:,:1]
                else: del out["aff_sim"]
            outs.append(out)
        print("done collecting perpoint queries")
        # Aggregate outputs
        out = outs[0]
        if "aff_sim" in out:
            out["point_track_reproj"] = torch.cat([ x["point_track_reproj"] for x in outs ], 2)
            out["aff_sim"] = torch.cat([ x["aff_sim"] for x in outs if "aff_sim" in x], 1)
            out["poses_all"] = torch.cat([ x["poses_all"] for x in outs ], 1)
            out["worldcrds_pertrack"] = torch.cat([ x["worldcrds_pertrack"] for x in outs ],1)
            out["pose_clusters"] = geometry.cluster_and_represent(out["poses_all"][0],n_clusters=15) # cluster poses

        return out

    def forward( self, model_input, track_idxs=None, out={}):  # same as below but slight cleanup
        if torch.is_grad_enabled():self.step+=1

        imsize = model_input["rgb"].shape[-2:]
        (b, _), n_trgt = model_input["rgb"].shape[:2], model_input["rgb"].size(1)
        n_samp = imsize[0] * imsize[1]  # min(40000,imsize[0]*imsize[1])
        rand_subset = torch.linspace(0, model_input["pred_tracks"].size(-2)-1, n_samp).long()
        low_imres = (64, 64)

        # Process masks for static training losses
        rig_samp = ( grid_samp( model_input["rig_flow_masks"][:, :, [0]], model_input["pred_tracks"][:, 1:].unsqueeze(-2),) .squeeze(2) .squeeze(-1) .round())
        rig_samp = rig_samp_allframe = torch.where( model_input["pred_visibility"][:, 1:], rig_samp, torch.ones_like(rig_samp))
        rig_samp = rig_samp.min(dim=1)[0]

        # pick n random points if not provided
        if track_idxs is None: track_idxs = torch.randperm(model_input["pred_tracks"].size(-2))[:200]

        # General feature map backbone prediction
        img_inp = model_input["rgb"]# if not use_depth_inp else torch.cat((model_input["rgb"],depth_inp.log()-1),2)
        img_inp = rearrange( img_inp, "b (t s) c x y -> b t (s c) x y", s=self.time_stride)
        if "fmap" not in model_input or torch.is_grad_enabled():
            # FPN
            fmap_out = F.interpolate( self.img_enc(img_inp.flatten(0, 1) * 0.5 + 0.5), imsize, mode="bilinear",)
            model_input["fmap"] = rearrange( fmap_out, "(b t) (s c) x y -> b (t s) c x y", s=self.time_stride, b=b)

        depth = res_depth = F.softplus( self.depth_conv(model_input["fmap"].flatten(0, 1)).unflatten(0, (b, n_trgt)) + 1)+1 

        # Lift point track into 2.5D image-aligned surface
        rds_track = geometry.get_world_rays( model_input["pred_tracks"], model_input["intrinsics"], None)[1]
        eye_surf_track = rds_track * grid_samp( depth, model_input["pred_tracks"].unsqueeze(-2)).squeeze(2)

        # Do static solve first just for static training losses
        general_conf_static = ( self.general_confidence_conv_static(model_input["fmap"].flatten(0, 1)) .unflatten(0, (b, n_trgt)) .sigmoid() .clip(min=1e-4))
        general_conf_track_static = grid_samp( general_conf_static, model_input["pred_tracks"].unsqueeze(-2)).squeeze(2) * model_input["pred_visibility"].unsqueeze(-1)

        static_poses = geometry.efficient_procrustes( eye_surf_track[:, None, 1:, rand_subset], eye_surf_track[:, None, :-1, rand_subset], general_conf_track_static[:, None, :-1, rand_subset].clip(min=1e-4),)[1]
        for i in range(n_trgt - 1, 0, -1): static_poses = torch.cat( (static_poses[:, :, :i], static_poses[:, :, [i - 1]] @ static_poses[:, :, i:]), -3)  # aggregate adjacent poses
        static_poses = torch.cat( ( torch.eye(4).to(static_poses)[None, None, None].expand(static_poses.size(0), static_poses.size(1), -1, -1, -1), static_poses,), -3,)  # add identity for starting pose
        static_poses = static_poses.expand( -1, len(track_idxs), -1, -1, -1)  # if static solve, just use single pose as pose for all points

        # Compute static point track reprojection
        static_poses_all_to_all = repeat( static_poses.inverse(), "b p t x y -> b p s t x y", s=n_trgt) @ repeat(static_poses, "b p t x y -> b p t s x y", s=n_trgt)
        static_point_track_surf_reproj = torch.einsum( "bpstij,bstpj->bstpi", static_poses_all_to_all, hom( repeat( eye_surf_track[:, :, track_idxs], "b t p c -> b t s p c", s=n_trgt)),)[..., :3]
        static_point_track_reproj = project( static_point_track_surf_reproj, model_input["intrinsics"]).clip(0, 1)
        vis_and_rig_mask = (model_input["pred_visibility"] * torch.cat((torch.ones_like(rig_samp_allframe[:,:1]),rig_samp_allframe),1))
        static_point_track_loss = ( ( ( static_point_track_reproj - model_input["pred_tracks"][:, None, :, track_idxs]) * vis_and_rig_mask[:, None, :, track_idxs, None]).square().flatten().mean())

        out |= {
            "corr_weights_static": general_conf_static,
            "point_track_loss_static": static_point_track_loss,
            "point_track_reproj_static": static_point_track_reproj[:, 0],
            "poses": static_poses[:,0],
            #"poses_all": static_poses,
            "res_depth": ch_sec(res_depth),
            "depth": ch_sec(depth),
        }

        # Below is dynamic point track est

        # Est affinity weights and similarities for each source point -- these are correspondence weights from each point to each other point
        affinity_emb_unnorm = self.affinities_conv( model_input["fmap"].flatten(0, 1)).unflatten(0, (b, n_trgt))

        # Take affinity emb as mean over frames masked by visibility
        affinity_emb = F.normalize(affinity_emb_unnorm, dim=2)
        aff_emb_pertrack_allframe = ch_sec( grid_samp(affinity_emb, model_input["pred_tracks"].unsqueeze(-2)))
        aff_emb_pertrack = ( aff_emb_pertrack_allframe * model_input["pred_visibility"].unsqueeze(-1)).sum(dim=1) / model_input["pred_visibility"].unsqueeze(-1).sum(dim=1).clip( min=1)
        aff_sim = torch.einsum( "b p c, b q c -> b p q", aff_emb_pertrack[:, track_idxs], aff_emb_pertrack)  # from all source pix to all other source pix

        # Predict general correspondence weights -- how reliable is this track generally at each frame
        general_conf = ( self.general_confidence_conv(model_input["fmap"].flatten(0, 1)) .unflatten(0, (b, n_trgt)) .sigmoid() .clip(min=1e-4))
        general_conf_track = grid_samp( general_conf, model_input["pred_tracks"].unsqueeze(-2)).squeeze(2) * model_input["pred_visibility"].unsqueeze(-1)

        solve_stride = ( model_input["pred_tracks"].size(-2) // 3000)  # use every nth point in the solve
        aff_sim_rig = torch.where( rig_samp.bool()[:, track_idxs, None].expand(-1, -1, aff_sim.size(-1)), torch.ones_like(aff_sim), aff_sim,) # replace points in rigid mask with 1s
        # Estimate adjacent poses and chain adjacent poses
        poses = geometry.efficient_procrustes( eye_surf_track[:, None, 1:, ::solve_stride].expand(-1, aff_sim.size(1), -1, -1, -1), 
                                               eye_surf_track[:, None, :-1, ::solve_stride].expand(-1, aff_sim.size(1), -1, -1, -1),
                            ( general_conf_track[:, None, :-1, ::solve_stride].expand(-1, aff_sim.size(1), -1, -1, -1) * aff_sim_rig[:, :, None, ::solve_stride, None]).clip(min=1e-4),)[1]
        for i in range(n_trgt - 1, 0, -1): poses = torch.cat( (poses[:, :, :i], poses[:, :, [i - 1]] @ poses[:, :, i:]), -3)  # aggregate adjacent poses
        poses = torch.cat( ( torch.eye(4).to(poses)[None, None, None].expand(poses.size(0), poses.size(1), -1, -1, -1), poses,), -3,)  # add identity for starting pose

        # Compute point track reprojection
        poses_all_to_all = repeat( poses.inverse(), "b p t x y -> b p s t x y", s=n_trgt) @ repeat(poses, "b p t x y -> b p t s x y", s=n_trgt)
        point_track_surf_reproj = torch.einsum( "bpstij,bstpj->bstpi", poses_all_to_all, hom( repeat( eye_surf_track[:, :, track_idxs], "b t p c -> b t s p c", s=n_trgt)),)[..., :3]
        point_track_reproj = project( point_track_surf_reproj, model_input["intrinsics"]).clip(0, 1)
        point_track_loss = ( ( ( point_track_reproj - model_input["pred_tracks"][:, None, :, track_idxs]) * model_input["pred_visibility"][:, None, :, track_idxs, None]).square().flatten().mean())

        with torch.no_grad(): # just for visualization
            track_sl,dsl=(64,4) if model_input["pred_tracks"].size(-2)%64**2==0 else (42,3) # some datasets ran tracks at 42gridsize and some 64 (just for vis)
            src_tracks_0=rearrange(aff_emb_pertrack,"b (x y s) c -> b s c x y",y=track_sl,x=track_sl)[:,0]
            aff_sim_grid = torch.einsum( "b p c, b q c -> b p q", ch_sec(src_tracks_0[...,::dsl,::dsl]), ch_sec(src_tracks_0)).unflatten(1,(track_sl//dsl,track_sl//dsl)).unflatten(-1,(track_sl,track_sl))  # from all source pix to all other source pix
            # mean color and crds per track for vis
            rgb_pertrack = ch_sec( grid_samp(model_input["rgb"], model_input["pred_tracks"].unsqueeze(-2)))
            rgb_pertrack = ( rgb_pertrack * model_input["pred_visibility"].unsqueeze(-1)).sum(dim=1) / model_input["pred_visibility"].unsqueeze(-1).sum(dim=1).clip( min=1)
            worldcrds_pertrack = torch.einsum( "bptij,btpj->btpi", poses, hom(eye_surf_track[:, :, track_idxs]))[..., :3]
            worldcrds_pertrack = ( worldcrds_pertrack * model_input["pred_visibility"][:, :, track_idxs].unsqueeze(-1)
                                    ).sum(dim=1) / model_input["pred_visibility"][:, :, track_idxs].unsqueeze( -1).sum( dim=1).clip( min=1)

        return out | {
            "worldcrds_pertrack": worldcrds_pertrack,
            "rgb_pertrack": rgb_pertrack,
            "rig_pertrack": rig_samp,
            "poses_all": poses,
            #"depth_inp": model_input["depth_inp"],
            "point_track_loss": point_track_loss,
            "point_track_reproj": point_track_reproj[:, 0],
            "corr_weights": general_conf,
            "aff_sim": aff_sim,
            "aff_sim_grid": aff_sim_grid,
            "affinity_emb": affinity_emb,
            "affinity_emb_unnorm": affinity_emb_unnorm,
            "aff_emb_pertrack": aff_emb_pertrack,
            "depth": ch_sec(depth),
        }

class ResnetFPN(nn.Module): # from pixelnerf code
    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        in_ch=3,
    ):
        super().__init__()

        def get_norm_layer(norm_type="instance", group_norm_groups=32):
            """Return a normalization layer
            Parameters:
                norm_type (str) -- the name of the normalization layer: batch | instance | none
            For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
            For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
            """
            if norm_type == "batch":
                norm_layer = functools.partial(
                    nn.BatchNorm2d, affine=True, track_running_stats=True
                )
            elif norm_type == "instance":
                norm_layer = functools.partial(
                    nn.InstanceNorm2d, affine=False, track_running_stats=False
                )
            elif norm_type == "group":
                norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
            elif norm_type == "none":
                norm_layer = None
            else:
                raise NotImplementedError(
                    "normalization layer [%s] is not found" % norm_type
                )
            return norm_layer

        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = get_norm_layer(norm_type)

        print("Using torchvision", backbone, "encoder")
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained, norm_layer=norm_layer)

        if in_ch != 3:
            self.model.conv1 = nn.Conv2d(
                in_ch,
                self.model.conv1.weight.shape[0],
                self.model.conv1.kernel_size,
                self.model.conv1.stride,
                self.model.conv1.padding,
                padding_mode=self.model.conv1.padding_mode,
            )

        # Following 2 lines need to be uncommented for older configs
        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )

        self.out = nn.Sequential(
            nn.Conv2d(self.latent_size, 512, 1),
        )

        self.out_dim=64#32 # todo make arg
        #self.combs = nn.ModuleList([ nn.Sequential(nn.Conv2d(256, 128, 1),nn.ReLU(),nn.Conv2d(128,128,1)) for d1,d2 in [(256,128)]])#[4,64,64,128,256]])
        self.combs_1 = nn.ModuleList([ nn.Conv2d(d1, d2, 1) for d1,d2 in [(256,128),(128,64),(64,64),(64,64),(64,self.out_dim)]]).cuda()#[4,64,64,128,256]])
        self.combs_2 = nn.ModuleList([ nn.Conv2d(d, d, 1) for d in [128,64,64,64,self.out_dim]]).cuda()#[4,64,64,128,256]])
        self.last_conv_up=nn.Conv2d(in_ch, self.out_dim, 1).cuda()

    def forward(self, x, custom_size=None):

        if len(x.shape) > 4: return self(x.flatten(0, 1), custom_size).unflatten(0, x.shape[:2])

        if self.feature_scale != 1.0:
            x = F.interpolate( x, scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area", align_corners=True if self.feature_scale > 1.0 else None, recompute_scale_factor=True,)
        latents = [x]

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents.append(x)

        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)

        align_corners = None if self.index_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]

        up_latent = self.combs_2[0]( (self.combs_1[0](F.interpolate(latents[-1],latents[-2].shape[-2:],mode="bilinear"))+latents[-2]).relu() )
        up_latent = self.combs_2[1]( (self.combs_1[1](F.interpolate(up_latent,latents[-3].shape[-2:],mode="bilinear"))+latents[-3]).relu() )
        up_latent = self.combs_2[3]( (self.combs_1[3](F.interpolate(up_latent,latents[-4].shape[-2:],mode="bilinear"))+latents[-4]).relu() )
        up_latent = self.combs_2[4]( (self.combs_1[4](F.interpolate(up_latent,latents[-5].shape[-2:],mode="bilinear"))+self.last_conv_up(latents[-5])).relu() )
        return up_latent
        #for i in range(len(latents)):
        #    latents[i] = F.interpolate(
        #        latents[i],
        #        latent_sz if custom_size is None else custom_size,
        #        mode=self.upsample_interp,
        #        align_corners=align_corners,
        #    )
        #self.latent = torch.cat(latents, dim=1)
        #self.latent_scaling[0] = self.latent.shape[-1]
        #self.latent_scaling[1] = self.latent.shape[-2]
        #self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        #return self.out(self.latent)

    def forward_(self, x, custom_size=None):

        if len(x.shape) > 4:
            return self(x.flatten(0, 1), custom_size).unflatten(0, x.shape[:2])

        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        #latents.append(x)
        latents = [x]

        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)

        align_corners = None if self.index_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz if custom_size is None else custom_size,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        return self.out(self.latent)
