import torch


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features.
    
    Args:
        feats: List of feature tensors
        strides: Stride tensor
        grid_cell_offset: Offset for grid cells
        
    Returns:
        anchor_points: Anchor points tensor
        stride_tensor: Stride tensor
    """
    anchor_points, stride_tensor = [], []
    dtype = feats[0].dtype
    device = feats[0].device
    
    # If strides is a tensor, convert to list
    if isinstance(strides, torch.Tensor):
        strides_list = strides.tolist()
    else:
        strides_list = strides
    
    for i, feat in enumerate(feats):
        _, _, h, w = feat.shape
        stride = strides_list[i] if i < len(strides_list) else strides_list[-1]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

