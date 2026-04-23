import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet, ViT, SwinUNETR
from torch_geometric.nn import GATv2Conv, TransformerConv

class HMGTNet(nn.Module):
    """
    Hybrid Multi-Scale Graph Transformer Network (HMGT-Net).
    Combines 3D ResNet50, 3D ViT, 3D Swin, and Graph Learning with Attention Fusion.
    """
    def __init__(self, in_channels=4, num_classes=4, img_size=(64, 64, 64)):
        super(HMGTNet, self).__init__()
        self.img_size = img_size
        
        # Branch 1: 3D ResNet50
        self.resnet_branch = ResNet(
            block='bottleneck',
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=256
        )
        
        # Branch 2: 3D Vision Transformer (ViT)
        self.vit_branch = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=(16, 16, 16),
            hidden_size=256,
            mlp_dim=512,
            num_layers=4,
            num_heads=8,
            classification=True,
            num_classes=256,
            dropout_rate=0.3
        )
        
        # Branch 3: 3D Swin Transformer
        self.swin_model = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=24,
            spatial_dims=3
        )
        self.swin_branch = self.swin_model.swinViT
        self.swin_pool = nn.AdaptiveAvgPool3d(1)
        self.swin_fc = nn.Linear(384, 256) # 384 is output of stage 4 for feature_size=24
        
        # Attention-based Feature Fusion
        self.fusion_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.fusion_norm = nn.LayerNorm(256)
        
        # Graph Construction and Relational Learning
        self.graph_conv1 = TransformerConv(in_channels=256, out_channels=256, heads=4, concat=False)
        self.graph_conv2 = GATv2Conv(in_channels=256, out_channels=256, heads=4, concat=False)
        
        # Classification Head (with Dropout & LayerNorm)
        self.clf_head = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Extract Features from Branches
        f_resnet = self.resnet_branch(x)
        f_vit = self.vit_branch(x)[0]
        
        f_swin_stages = self.swin_branch(x)
        f_swin = self.swin_pool(f_swin_stages[-1]).view(batch_size, -1)
        f_swin = self.swin_fc(f_swin)
        
        # 2. Attention Fusion
        nodes = torch.stack([f_resnet, f_vit, f_swin], dim=1) # (Batch, 3, 256)
        attn_out, _ = self.fusion_attn(nodes, nodes, nodes)
        nodes = self.fusion_norm(nodes + attn_out) # Residual + LayerNorm
        
        # 3. Graph Construction
        f_combined = nodes.view(batch_size * 3, 256)
        
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                                   [1, 2, 0, 2, 0, 1]], device=x.device)
        
        f_graph = self.graph_conv1(f_combined, edge_index.repeat(1, batch_size))
        f_graph = self.graph_conv2(f_graph, edge_index.repeat(1, batch_size))
        
        # 4. Final Fusion and Classification
        f_final = f_graph.view(batch_size, -1) # (Batch, 256 * 3)
        out = self.clf_head(f_final)
        
        return out

if __name__ == "__main__":
    # Test the model
    model = HMGTNet(in_channels=4, num_classes=4, img_size=(64, 64, 64)).cuda()
    dummy_input = torch.randn(1, 4, 64, 64, 64).cuda()
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

