import torch
import torch.nn as nn
import torch.nn.functional as F


class PairedContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, projections, labels, pair_indices):
        device = projections.device
        batch_size = projections.shape[0]
        
        sorted_indices = torch.argsort(pair_indices)  # [B]
        sorted_projections = projections[sorted_indices]  # [B, emb_dim]
        sorted_labels = labels[sorted_indices]  # [B]
        
        if batch_size % 2 != 0:
            sorted_projections = sorted_projections[:-1]
            sorted_labels = sorted_labels[:-1]
            batch_size = batch_size - 1
        
        num_pairs = batch_size // 2
        paired_projections = sorted_projections.view(num_pairs, 2, -1)  # [N, 2, D]
        paired_labels = sorted_labels.view(num_pairs, 2)  # [N, 2]
        
        real_mask = (paired_labels == 1)  # [N, 2]
        
        real_first = real_mask[:, 0]  # [N]
        
        real_first_expanded = real_first.unsqueeze(1)  # [N, 1]
        
        z_real = torch.where(
            real_first_expanded,  # [N, 1]
            paired_projections[:, 0, :],  # [N, D]
            paired_projections[:, 1, :]   # [N, D]
        )  # [N, D]
        
        z_fake = torch.where(
            real_first_expanded,  # [N, 1]
            paired_projections[:, 1, :],  # [N, D]
            paired_projections[:, 0, :]   # [N, D]
        )  # [N, D]
        
        cos_sim = (z_real * z_fake).sum(dim=1)  # [N]
        
        target_threshold = 1.0 - self.margin
        losses = F.relu(cos_sim - target_threshold)  # [N]
        
        return losses.mean()
