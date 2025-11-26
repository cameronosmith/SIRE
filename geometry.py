"""Multi-view geometry & proejction code.. (most from vincent, some old nerf stuff hanging around) """ 
import torch
from einops import rearrange, repeat
from torch.nn import functional as F
import numpy as np

from math import ceil, log2
import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor
import kornia

from gsplat import rasterization

hom       = lambda x: torch.cat((x,torch.ones_like(x[...,[0]])),-1)
unhom     = lambda x: x[...,:-1]/(1e-5+x[...,-1:])
project   = lambda crds,K: unhom(torch.einsum("b...cij,b...ckj->b...cki",K, crds))


def compute_flow(pose_perpix,view_i,means,K):
    pos_i   = torch.einsum("pij,pj->pi",pose_perpix[:,view_i].inverse(),hom(means))[...,:3]
    pos_adj = torch.einsum("pij,pj->pi",pose_perpix[:,max(0,view_i-1)].inverse(),hom(means))[...,:3]
    pos_i_2d   = project(pos_i  [None,None],K[:1,None])[0,0]
    pos_adj_2d = project(pos_adj[None,None],K[:1,None])[0,0]
    flow_2d = pos_adj_2d-pos_i_2d
    return flow_2d

# clusters n poses into top n using kmeans (from chatgpt)

from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R

def cluster_and_represent(poses, n_clusters=3,return_labels=False):
    # Flatten trajectories (NxTx4x4 -> Nx(T * features))
    translations = poses[:, :, :3, 3].reshape(poses.size(0), -1)  # Nx(T*3)
    rotations = poses[:, :, :3, :3].reshape(-1, 3, 3).cpu().numpy()
    quaternions = torch.tensor(R.from_matrix(rotations).as_quat(), device=poses.device)
    quaternions = quaternions.reshape(poses.size(0), -1)  # Nx(T*4)
    features = torch.cat([translations, quaternions], dim=1).cpu().numpy()  # NxD

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    labels = kmeans.labels_
    centers = torch.tensor(kmeans.cluster_centers_, device=poses.device)  # Cluster centers

    # Find representative trajectories
    representatives = []
    for i in range(n_clusters):
        cluster_indices = torch.where(torch.tensor(labels) == i)[0]
        if cluster_indices.numel() == 0: continue
        try:
            cluster_features = features[cluster_indices]
            distances = torch.norm(torch.tensor(cluster_features, device=poses.device) - centers[i], dim=1)
            representatives.append(poses[cluster_indices[distances.argmin()]])
        except:continue
    return torch.stack(representatives) if not return_labels else (torch.stack(representatives),labels)

# start with just rgb then add pose-induced flow too
def do_render(pose,timestep,imsize,K,splat_vars):

    pose_perpix = torch.eye(4)[None,None].expand(*splat_vars["lie_perpix"].shape[:2],-1,-1).cuda()
    pose_perpix[...,:3,:3] = kornia.geometry.conversions.quaternion_to_rotation_matrix(splat_vars["lie_perpix"][...,:4])
    pose_perpix[...,:3,-1] = splat_vars["lie_perpix"][...,4:]

    pose = (pose[None] if len(pose.shape)==2 else pose) @ pose_perpix[:,timestep].inverse()

    if type(timestep)==float:
        # todo interpolate
        pass
        #from pdb import set_trace as pdb_;pdb_() 

    flow = compute_flow(pose_perpix,timestep,splat_vars["means"],K)
    colors_i = torch.cat((splat_vars["colors"],flow),-1)
    means_i=torch.einsum("kij,kj->ki",pose,hom(splat_vars["means"]))[...,:3]
    quats_i=kornia.geometry.conversions.rotation_matrix_to_quaternion(pose[:,:3,:3],eps=1e-5)*splat_vars["quats"]
    return rasterization( means_i, quats_i, splat_vars["scales"].clip(max=.1), splat_vars["opacities"], colors_i, torch.eye(4).cuda()[None], K, imsize[1], imsize[0],render_mode="RGB+D",
            backgrounds=torch.zeros_like(colors_i)[:1]+1)

def format_splat_vars(scene):
    stride=max(1,len(scene["world_crds"][0])//20)

    # todo just uses poses instead of this
    poses_lie = torch.cat((kornia.geometry.conversions.rotation_matrix_to_quaternion(scene["poses"][...,:3,:3],eps=1e-4),scene["poses"][...,:3,-1]),-1)
    scene["lie_crds"] = poses_lie.expand(len(scene["world_crds"][0].flatten(0,1)),-1,-1)

    return { 
            "means":      torch.nn.Parameter(scene["world_crds"][0].flatten(0,1)[::stride]),
            "colors":     torch.nn.Parameter(scene["rgb_crds"][0].flatten(0,1)[::stride]*.5+.5 ),
            "quats":      torch.nn.Parameter(torch.ones(len(scene["world_crds"][0].flatten(0,1)[::stride]),4).cuda() ),
            "opacities":  torch.nn.Parameter(torch.ones(len(scene["world_crds"][0].flatten(0,1)[::stride])).cuda()*.05 ),
            "scales":     torch.nn.Parameter(torch.ones(len(scene["world_crds"][0].flatten(0,1)[::stride]),3).cuda()*.01 ),
            #"lie_poses":  torch.nn.Parameter(poses_lie),
            "lie_perpix":  torch.nn.Parameter(scene["lie_crds"][::stride]),
    }
     #= torch.nn.Parameter(scene["lie_crds"].flatten(1,2))

# Idea is take in one point cloud with an additional ND tensor affinity_embedding used for weight estimation, still use general confidence weights too
def efficient_nonrig_procrustes(S1, S2,weights,aff_emb):

    # todo change to while shape>4
    while len(S1.shape)>3:
        out = efficient_nonrig_procrustes(S1.flatten(0,1),S2.flatten(0,1),weights.flatten(0,1),aff_emb.flatten(0,1))
        return out[0],out[1].unflatten(0,S1.shape[:2])
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (BxNx3) closest to a set of 3D points, S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale. / mod : assuming scale is 1
    i.e. solves the orthogonal Procrutes problem.
    '''
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        #if weights is not None:
        weights=weights.permute(0,2,1)
        transposed = True

        #if weights is None: weights = torch.ones_like(S1[:,:1])

        eps=1e-6
        weights=weights.clip(min=eps)

        # 1. Remove mean.
        weights_norm = weights/(weights.sum(-1,keepdim=True)+eps)
        mu1 = (S1*weights_norm).sum(2,keepdim=True)
        mu2 = (S2*weights_norm).sum(2,keepdim=True)

        weights_norm=weights_norm.clip(min=eps)

        X1 = S1 - mu1
        X2 = S2 - mu2

        #diags = torch.diag_embed(weights.squeeze(1))
        # 3. The outer product of X1 and X2.
        #K = (X1*weights).bmm(X2.permute(0,2,1))
        #K = (X1@torch.diag_embed(weights.squeeze(1))).bmm(X2.permute(0,2,1))

        # the expensive version to einsum replace with a single expression, for now just use first 2 dim to keep memory tractable
        aff_emb =aff_emb[...,:2]
        from pdb import set_trace as pdb_;pdb_() 
        tmp = torch.einsum('bik,bjk->bij',aff_emb,aff_emb) # == (aff_emb[...,:2].unsqueeze(-2)*aff_emb.unsqueeze(1)[...,:2]).sum(-1)

        #K = torch.einsum('bij,bij,bjk->bik', X1, weights, X2.permute(0, 2, 1))

        # I think this is correct below but instantiates tmp
        K = torch.einsum('bpij,bij,bjk->bpik', tmp.unsqueeze(2)*X1.unsqueeze(1), weights, X2.permute(0, 2, 1))
        K = torch.einsum('bik,bpij,bij,bjk->bpik', aff_emb, X1, weights, X2.permute(0, 2, 1))

        #result = torch.einsum( 'bik,bjk,bpk,bcp,bcq->bpik', aff_emb, aff_emb, X1, weights, X2)
        torch.einsum( 'bik,bjk,bpk,bcp,bcq->bpik', aff_emb, aff_emb, X1, weights, X2.permute(0, 2, 1))
        

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
        #scaled_X1 = X1 * weights  
        #K = torch.einsum('bin,bjn->bji', X1*weights, X2)

        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 6. Recover translation.
        t = mu2 - ((R.bmm(mu1)))

        # 7. Error:
        S1_hat = R.bmm(S1) + t

        # Combine recovered transformation as single matrix
        R_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        R_[:,:3,:3]=R
        T_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        T_[:,:3,-1]=t.squeeze(-1)
        S_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        transf = T_@S_@R_

        return (S1_hat-S2).square().mean(),transf



def efficient_procrustes(S1, S2,weights=None):

    # todo change to while shape>4
    if len(S1.shape)==5: 
        out = efficient_procrustes(S1.flatten(0,1),S2.flatten(0,1),weights.flatten(0,1) if weights is not None else None)
        return out[0],out[1].unflatten(0,S1.shape[:2])
    if len(S1.shape)==4:
        out = efficient_procrustes(S1.flatten(0,1),S2.flatten(0,1),weights.flatten(0,1) if weights is not None else None)
        return out[0],out[1].unflatten(0,S1.shape[:2])
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (BxNx3) closest to a set of 3D points, S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale. / mod : assuming scale is 1
    i.e. solves the orthogonal Procrutes problem.
    '''
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        if weights is not None:
            weights=weights.permute(0,2,1)
        transposed = True

        if weights is None: weights = torch.ones_like(S1[:,:1])

        eps=1e-6
        weights=weights.clip(min=eps)

        # 1. Remove mean.
        weights_norm = weights/(weights.sum(-1,keepdim=True)+eps)
        mu1 = (S1*weights_norm).sum(2,keepdim=True)
        mu2 = (S2*weights_norm).sum(2,keepdim=True)

        weights_norm=weights_norm.clip(min=eps)

        X1 = S1 - mu1
        X2 = S2 - mu2

        #diags = torch.diag_embed(weights.squeeze(1))
        # 3. The outer product of X1 and X2.
        K = (X1*weights).bmm(X2.permute(0,2,1))
        #K = (X1@torch.diag_embed(weights.squeeze(1))).bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
        #scaled_X1 = X1 * weights  
        #K = torch.einsum('bin,bjn->bji', scaled_X1, X2)

        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 6. Recover translation.
        t = mu2 - ((R.bmm(mu1)))

        # 7. Error:
        S1_hat = R.bmm(S1) + t

        # Combine recovered transformation as single matrix
        R_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        R_[:,:3,:3]=R
        T_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        T_[:,:3,-1]=t.squeeze(-1)
        S_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        transf = T_@S_@R_

        return (S1_hat-S2).square().mean(),transf




def procrustes(S1, S2,weights=None):

    # todo change to while shape>4
    if len(S1.shape)==5: 
        out = procrustes(S1.flatten(0,1),S2.flatten(0,1),weights.flatten(0,1) if weights is not None else None)
        return out[0],out[1].unflatten(0,S1.shape[:2])
    if len(S1.shape)==4:
        out = procrustes(S1.flatten(0,1),S2.flatten(0,1),weights.flatten(0,1) if weights is not None else None)
        return out[0],out[1].unflatten(0,S1.shape[:2])
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (BxNx3) closest to a set of 3D points, S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale. / mod : assuming scale is 1
    i.e. solves the orthogonal Procrutes problem.
    '''
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        if weights is not None:
            weights=weights.permute(0,2,1)
        transposed = True

        if weights is None: weights = torch.ones_like(S1[:,:1])

        eps=1e-6
        weights=weights.clip(min=eps)

        # 1. Remove mean.
        weights_norm = weights/(weights.sum(-1,keepdim=True)+eps)
        mu1 = (S1*weights_norm).sum(2,keepdim=True)
        mu2 = (S2*weights_norm).sum(2,keepdim=True)

        weights_norm=weights_norm.clip(min=eps)

        X1 = S1 - mu1
        X2 = S2 - mu2

        diags = torch.diag_embed(weights.squeeze(1))

        # 3. The outer product of X1 and X2.
        K = (X1@diags).bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 6. Recover translation.
        t = mu2 - ((R.bmm(mu1)))

        # 7. Error:
        S1_hat = R.bmm(S1) + t

        # Combine recovered transformation as single matrix
        R_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        R_[:,:3,:3]=R
        T_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        T_[:,:3,-1]=t.squeeze(-1)
        S_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        transf = T_@S_@R_

        return (S1_hat-S2).square().mean(),transf


def homogenize_points(points: torch.Tensor):
    """Appends a "1" to the coordinates of a (batch of) points of dimension DIM.

    Args:
        points: points of shape (..., DIM)

    Returns:
        points_hom: points with appended "1" dimension.
    """
    ones = torch.ones_like(points[..., :1], device=points.device)
    return torch.cat((points, ones), dim=-1)


def homogenize_vecs(vectors: torch.Tensor):
    """Appends a "0" to the coordinates of a (batch of) vectors of dimension DIM.

    Args:
        vectors: vectors of shape (..., DIM)

    Returns:
        vectors_hom: points with appended "0" dimension.
    """
    zeros = torch.zeros_like(vectors[..., :1], device=vectors.device)
    return torch.cat((vectors, zeros), dim=-1)


def unproject(
    xy_pix: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Unproject (lift) 2D pixel coordinates x_pix and per-pixel z coordinate
    to 3D points in camera coordinates.

    Args:
        xy_pix: 2D pixel coordinates of shape (..., 2)
        z: per-pixel depth, defined as z coordinate of shape (..., 1)
        intrinscis: camera intrinscics of shape (..., 3, 3)

    Returns:
        xyz_cam: points in 3D camera coordinates.
    """
    xy_pix_hom = homogenize_points(xy_pix)
    xyz_cam = torch.einsum("...ij,...kj->...ki", intrinsics.inverse(), xy_pix_hom)
    xyz_cam *= z
    return xyz_cam


def transform_world2cam(
    xyz_world_hom: torch.Tensor, cam2world: torch.Tensor
) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_world_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_cam: points in camera coordinates.
    """
    world2cam = torch.inverse(cam2world)
    return transform_rigid(xyz_world_hom, world2cam)


def transform_cam2world(
    xyz_cam_hom: torch.Tensor, cam2world: torch.Tensor
) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_cam_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_world: points in camera coordinates.
    """
    return transform_rigid(xyz_cam_hom, cam2world)


def transform_rigid(xyz_hom: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a rigid-body transform to a (batch of) points / vectors.

    Args:
        xyz_hom: homogenized 3D points of shape (..., 4)
        T: rigid-body transform matrix of shape (..., 4, 4)

    Returns:
        xyz_trans: transformed points.
    """
    return torch.einsum("...ij,...kj->...ki", T, xyz_hom)


def get_unnormalized_cam_ray_directions(
    xy_pix: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    return unproject(
        xy_pix,
        torch.ones_like(xy_pix[..., :1], device=xy_pix.device),
        intrinsics=intrinsics,
    )


def get_world_rays_(
    xy_pix: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2world: torch.Tensor,
) -> torch.Tensor:

    if cam2world is None: 
        cam2world = torch.eye(4)[None].expand(xy_pix.size(0),-1,-1).to(xy_pix)

    # Get camera origin of camera 1
    cam_origin_world = cam2world[..., :3, -1]

    # Get ray directions in cam coordinates
    ray_dirs_cam = get_unnormalized_cam_ray_directions(xy_pix, intrinsics)
    ray_dirs_cam = ray_dirs_cam / ray_dirs_cam.norm(dim=-1, keepdim=True)

    # Homogenize ray directions
    rd_cam_hom = homogenize_vecs(ray_dirs_cam)

    # Transform ray directions to world coordinates
    rd_world_hom = transform_cam2world(rd_cam_hom, cam2world)

    cam_origin_world = repeat( cam_origin_world, "... ch -> ... num_rays ch", num_rays=ray_dirs_cam.size(-2) )

    # Return tuple of cam_origins, ray_world_directions
    return cam_origin_world, rd_world_hom[..., :3]

def get_world_rays(
    xy_pix: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2world: torch.Tensor,
) -> torch.Tensor:
    if len(xy_pix.shape)==4:
        out = get_world_rays_(xy_pix.flatten(0,1),intrinsics.flatten(0,1),cam2world.flatten(0,1) if cam2world is not None else None)
        return [x.unflatten(0,xy_pix.shape[:2]) for x in out]
    return get_world_rays_(xy_pix,intrinsics,cam2world)

def numpy_procrustes(X, Y, scaling=True, reflection='best'):

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    #R_=torch.eye(4).numpy()
    #R_[:3,:3]=T
    #T_=torch.eye(4).numpy()
    #T_[:3,-1]=c
    #S_=torch.eye(4).numpy()*b
    #transf = T_@S_@R_

    return d, Z, tform
