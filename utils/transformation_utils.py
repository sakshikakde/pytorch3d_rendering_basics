import torch

def batch_orth_proj(X: torch.tensor, camera: torch.tensor) -> torch.tensor:
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn

def translate(X: torch.tensor, trans: list) -> torch.tensor:
    trans = torch.tensor(trans, device=X.device)
    return X + trans[None, None, ...]