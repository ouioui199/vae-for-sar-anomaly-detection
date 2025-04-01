from types import NoneType

import torch
from torch import nn
from torchvision.models import vgg16
from torch.nn import functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


class l1_loss(nn.Module):
    def __init__(self, m, M, device='cuda') -> None:
        super().__init__()

        denorm = 2*(M-m)
        denorm = torch.tensor([denorm], device=device)

        self.denorm = torch.reshape(denorm, (1, denorm.shape[0], 1, 1))
        self.loss = nn.L1Loss()
        
    def forward(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            
        inp_log = torch.mul(inp, self.denorm)
        out_log = torch.mul(out, self.denorm)
        return self.loss(inp_log, out_log)
    

class disc_loss(nn.Module):
    def __init__(self, eps: float = 10 ** -10) -> None:
        super().__init__()
        self.eps = eps
            
    def forward(self, disc_real: torch.Tensor, disc_fake: torch.Tensor) -> torch.Tensor:
        return - torch.mean(torch.log(disc_real + self.eps) + torch.log(1 - disc_fake + self.eps))
    

class gen_loss(nn.Module):
    def __init__(self, eps: float = 10 ** -10) -> None:
        super().__init__()
        self.eps = eps
        
    def forward(self, disc_fake: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.log(1 - disc_fake + self.eps))


def kullback_leibler_divergence_loss(mu: torch.Tensor, log_var: torch.Tensor, weight: float | NoneType = None) -> torch.Tensor:
    loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    if weight is not None:
        return weight * loss
    return loss


class SSIMLoss(nn.Module):
    def __init__(self, data_range=(0., 1.)):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        
    def forward(self, x, y):
        # SSIM returns similarity, so we convert to loss (1 - similarity)
        return 1.0 - self.ssim(x, y)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature maps for SAR image reconstruction.
    
    This implementation extracts features from multiple layers and computes
    the weighted mean squared error between the features of real and generated images.
    """
    def __init__(self, 
                 resize=True, 
                 normalize=True,
                 layer_weights=None,
                 device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        
        # Use a pretrained VGG network
        vgg = vgg16(pretrained=True).features#.to(device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.model = vgg
        self.device = device
        self.resize = resize
        self.normalize = normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)#.to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)#.to(device)
        
        # Define which layers to extract features from
        self.layers = {
            '3': 'relu1_2',   # Low-level features (edges)
            '8': 'relu2_2',   # Mid-level features (textures)
            '15': 'relu3_3',  # Higher-level features (patterns)
            '22': 'relu4_3',  # More abstract features
        }
        
        # Layer weights (can be adjusted based on SAR image characteristics)
        self.layer_weights = layer_weights or {
            'relu1_2': 0.1,
            'relu2_2': 0.2,
            'relu3_3': 0.4,
            'relu4_3': 0.3,
        }
    
    def _adapt_input_for_vgg(self, x):
        if x.shape[1] == 1:
            # If single channel, repeat to make 3 channels
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            # If 2 channels (e.g., real/imaginary), convert to 3 channels
            # Use first channel as R, second as G, and average as B
            zeros = torch.zeros_like(x[:, :1])
            x = torch.cat([x, zeros], dim=1)
        
        # Resize if needed
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        # Normalize with ImageNet stats if needed
        if self.normalize:
            x = (x - self.mean) / self.std
            
        return x
        
    def get_features(self, x):
        """Extract features from specified VGG layers"""
        features = {}
        x = self._adapt_input_for_vgg(x)
        
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
                
        return features
        
    def forward(self, y_true, y_pred):
        """
        Compute perceptual loss between real and generated SAR images
        
        Args:
            y_true: Ground truth SAR image
            y_pred: Generated/reconstructed SAR image
            
        Returns:
            Weighted perceptual loss
        """
        # Ensure inputs are on the correct device
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        
        # Get features
        true_features = self.get_features(y_true)
        pred_features = self.get_features(y_pred)
        
        # Compute weighted loss across different feature layers
        loss = 0.0
        for layer_name, weight in self.layer_weights.items():
            true_feat = true_features[layer_name]
            pred_feat = pred_features[layer_name]
            loss += weight * F.mse_loss(pred_feat, true_feat)
            
        return loss
