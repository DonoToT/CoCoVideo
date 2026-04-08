import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PairedContrastiveModel(nn.Module):
    def __init__(self, backbone_name='r3d_18', emb_dim=128, pretrained=False, freeze_backbone=False):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone

        if backbone_name == 'r3d_18':
            if pretrained:
                try:
                    from torchvision.models.video import R3D_18_Weights
                    self.backbone = torchvision.models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
                except ImportError:
                    self.backbone = torchvision.models.video.r3d_18(pretrained=True)
            else:
                self.backbone = torchvision.models.video.r3d_18(pretrained=False)
            feat_dim = self.backbone.fc.in_features  # 512 for R3D-18
            self.backbone.fc = nn.Identity()
            self.backbone_type = 'r3d'
        else:
            raise NotImplementedError(
                f"Unsupported backbone: {backbone_name}\n"
                f"Supported options: r3d_18 (else backbones will be added)"
            )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
    
    def forward_single(self, x):
        if self.backbone_type == 'r3d':
            # [B, C, T, H, W]
            if x.ndim == 5 and x.shape[1] == 3:
                inp = x
            elif x.ndim == 5 and x.shape[1] != 3:
                inp = x.permute(0, 2, 1, 3, 4)
            else:
                raise ValueError("Unsupported input shape")
            
            feat = self.backbone(inp)  # [B, feat_dim]
        else:
            raise ValueError(f"unsupported backbone type: {self.backbone_type}")
        
        confidence = self.confidence_head(feat)
        projection = self.projector(feat)  # [B, emb_dim]
        
        projection = F.normalize(projection, p=2, dim=1)
        
        return confidence, projection
    
    def forward(self, videos):
        confidences, projections = self.forward_single(videos)
        return confidences, projections

