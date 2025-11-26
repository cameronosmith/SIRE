import os,time
import torch,wandb
from tqdm import trange
from einops import rearrange
import vis,geometry
from copy import deepcopy
import numpy as np
import piqa,kornia
from torchvision.utils import make_grid
from einops import rearrange, repeat
from models import ch_sec
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import getpass
from glob import glob

import data,models

def to_gpu(ob): return {k: to_gpu(torch.tensor(v)) for k, v in ob.items()} if isinstance(ob, dict) else ob.cuda()

def train_flowmap(run,train_dataset,until_img=25,until_vid=100,until_save=500,optim=None,single_data=None):

    def loss_fn(model_out, gt, model_input,model,step):

        losses = { }

        if "point_track_loss" in model_out: losses["metrics/tracks_err"] = model_out["point_track_loss"]*1e4
        if "point_track_loss_static" in model_out: losses["metrics/tracks_err_static"] = model_out["point_track_loss_static"]*1e4
        if step%10==0:print(losses,step)

        return losses

    losses_agg=[]
    optim = torch.optim.Adam(lr=run.args.lr, params=run.model.parameters())

    dataset=data.MultiImageFolder( num_trgt=run.args.vid_len+1,n_skip = run.args.n_skip,sf=run.args.sf)
    dataloader= iter(torch.utils.data.DataLoader(dataset, batch_size=run.args.batch_size, num_workers=min(run.args.n_workers,run.args.batch_size) if 1 else 0,shuffle=True,pin_memory=True))

    # Train loop
    step=0
    for step_ in trange(run.args.n_train_steps, desc="Fitting"): # train until user interruption

        try: model_input, ground_truth = next(dataloader)
        except StopIteration:
            print("done w dataset")
            dataloader= iter(torch.utils.data.DataLoader(dataset, batch_size=run.args.batch_size, num_workers=min(4,run.args.batch_size),shuffle=True,pin_memory=True))
            continue
        model_input, ground_truth = to_gpu(model_input), to_gpu(ground_truth)

        # Run model and calculate losses
        total_loss = 0.
        out=run.model(model_input)

        losses = loss_fn(out, ground_truth, model_input,run.model,step)
        for loss_name, loss in losses.items():
            wandb.log({loss_name: loss.item()}, step=step)
            total_loss += loss
        wandb.log({"loss": total_loss.item()}, step=step)

        total_loss.backward();optim.step();optim.zero_grad(); 

        if step==0:run.time=time.time()
        with torch.no_grad(): 
            wandb_imgs=None
            if step%until_img==0 and 1: 
                if (step%(until_img*3)==0 or 1): out=run.model.forward_allpts(model_input) # collect predictions for all point tracks 
                wandb_imgs=vis.wandb_summary( 0, out, model_input, ground_truth, None,step=step)
        if step%until_save == 0 and step and run.args.save_model: # save model
            print(f"Saving to {run.save_dir}"); torch.save({ 'step': step, 'model_state_dict': run.model.state_dict(), }, os.path.join(run.save_dir, f"checkpoint.pt")) 
            #save(run.splat_vars)
        step+=1

# Data/args setup and run
import argparse
parser = argparse.ArgumentParser(description='simple training job')
# logging parameters
parser.add_argument('-n','--name', type=str,default="",required=False,help="wandb training name")
parser.add_argument('-c','--init_ckpt', type=str,default=None,required=False,help="File for checkpoint loading. If folder specific, will use latest .pt file")
parser.add_argument('-o','--online', default=False, action='store_true')
parser.add_argument('-s','--save_model', default=True, action='store_true')
parser.add_argument('--viser', default=False, action='store_true')
parser.add_argument('--save_opt_vis', default=False, action='store_true')
# data/training parameters
parser.add_argument('-d','--dataset', type=str,default="hydrant")
parser.add_argument('--imgpath', type=str,default="")
parser.add_argument('-b','--batch_size', type=int,default=1,help="number of videos/sequences per training step")
parser.add_argument('-v','--vid_len', type=int,default=6,help="video length or number of images per batch")
parser.add_argument('--n_workers',type=int,default=4,help="number of workers per dataloader")
parser.add_argument('--until_save',type=int,default=500,help="number of steps until model save")
parser.add_argument('--lr',type=float,default=1e-4,help="learning rate")
parser.add_argument('--n_train_steps',type=int,default=int(1e8),help="learning rate")
parser.add_argument('--overfit', default=True, action='store_true',help="Whether to overfit on a single scene")
parser.add_argument('--until_img', type=int,default=50,help="Number of steps until image summary. ")
parser.add_argument('--sf', type=float,default=1,help="Image resolution scale factor (fractional is cheaper)")
# model parameters
parser.add_argument('--n_skip', type=int,default=1,help="Number of frames to skip between adjacent frames in dataloader. ")
parser.add_argument('--use_gt_intrinsics', default=True, action='store_true',help="Whether to use GT intrinsics instead of predicting them. Useful for pretraining scene rep.")
parser.add_argument('--point_track', default=True, action='store_true',help="Whether to use point tracking")

def make_run(args=None,val=False):
    args = parser.parse_args(args)
    self = argparse.Namespace()
    user = getpass.getuser()
    print(f"user={user}")

    # Wandb init
    #run = wandb.init(entity="cameronsmithbusiness",project="biasing",mode="online" if args.online else "disabled",name=args.name,dir=f"/tmp/wandb")
    run = wandb.init(
        entity=getattr(args, "entity", None),
        project=getattr(args, "project", None),
        mode="online" if args.online else "disabled",
        name=args.name,
    )
    wandb.run.log_code(".")
    self.save_dir = "/tmp/"+args.name#os.path.join(os.environ.get('LOGDIR', "") , run.name)
    os.makedirs(self.save_dir,exist_ok=True)
    wandb.save(os.path.join(self.save_dir, "checkpoint*"))
    wandb.save(os.path.join(self.save_dir, "video*"))

    self.args=args
    self.wandb=run
    if args.viser: self.viser_server=viser.ViserServer()

    imgfolders = []
    glob_str = "/data/cameron/monocular_ests/pets_dogs/*"
    self.dataset=None
    print("dummy multivid test")
    self.model = models.SIRE(args).cuda()
    if args.init_ckpt is not None:
        ckpt_file = args.init_ckpt if os.path.isfile(os.path.expanduser(args.init_ckpt)) else max(glob(os.path.join(args.init_ckpt,"*.pt")), key=os.path.getctime)
        self.model.load_state_dict(torch.load(ckpt_file)["model_state_dict"],strict=False)

    return self

run = make_run()
torch.autograd.set_detect_anomaly(False)
train_flowmap(run,run.dataset,until_save=run.args.until_save, until_vid=100 if not run.args.overfit else 300, until_img=run.args.until_img)
