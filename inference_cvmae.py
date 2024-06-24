import torch
import torchvision
from torchvision import transforms
from torch import nn
from lightly.transforms.mae_transform import MAETransform
from inference_dataset import InferDataset
from tqdm import tqdm
import os
import click
import numpy as np
from segmentation import ConvMAE, FPN, SegformerHead, F

class SegmentationModel(nn.Module):
    def __init__(self, backbone, neck, decode_head):
        super(SegmentationModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_head = decode_head

    def forward(self, x):
        input_size = x.size()[2:]  # Store the input spatial dimensions
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        out = self.decode_head(neck_features)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)  # Upsample to match input size
        return out

def inference_vit(dataset: str,
                  output_path: str,
                  transform_kwargs: dict = {'min_scale': 0.2, 'normalize': False},
                  vit_model: str = 'ViT_L_16', 
                  starting_weights: str = "ViT_L_16_Weights.DEFAULT", 
                  batch_size: int = 64,
                  n_workers: int = 4) -> None:
    # Define model configurations
    img_size = [224, 56, 28]
    patch_size = [4, 2, 2]
    in_chans = 3
    num_classes = 4
    embed_dim = [256, 384, 768]
    depth = [2, 2, 11]
    num_heads = 12
    mlp_ratio = [4, 4, 4]
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.0
    attn_drop_rate = 0.0
    drop_path_rate = 0.1
    init_values = 1
    use_checkpoint = False
    use_abs_pos_emb = True
    use_rel_pos_bias = True
    use_shared_rel_pos_bias = False
    out_indices = [3, 5, 7, 11]
    fpn1_norm = 'BN'
    
    # Initialize backbone
    backbone = ConvMAE(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        init_values=init_values,
        use_checkpoint=use_checkpoint,
        use_abs_pos_emb=use_abs_pos_emb,
        use_rel_pos_bias=use_rel_pos_bias,
        use_shared_rel_pos_bias=use_shared_rel_pos_bias,
        out_indices=out_indices,
        fpn1_norm=fpn1_norm
    )
    
    # Load weights into the model      
    state_dict = backbone.state_dict()
    checkpoint = torch.load(starting_weights, map_location=torch.device('cpu'))
    for name, param in checkpoint.items():
        if name in state_dict and state_dict[name].shape == param.shape:
            state_dict[name].copy_(param)
    
    # Initialize neck and head
    neck = FPN(in_channels=[256, 384, 768, 768], out_channels=256, num_outs=4)
    head = SegformerHead(in_channels=[256, 256, 256, 256], channels=256, num_classes=num_classes, in_index=[0, 1, 2, 3], dropout_ratio=0.1, norm_cfg=dict(type='BN', requires_grad=True), align_corners=False)
    
    # Combine to create the segmentation model
    model = SegmentationModel(backbone, neck, head)

    # Load weights into the model
    state_dict = model.state_dict()
    checkpoint = torch.load(starting_weights, map_location=torch.device('cpu'))
    for name, param in checkpoint.items():
        if name in state_dict and state_dict[name].shape == param.shape:
            state_dict[name].copy_(param)
    
    transform = MAETransform(**transform_kwargs)
    
    image_folder = os.path.join(dataset, 'image')
    image_filenames = [file for file in os.listdir(image_folder) if file.lower().endswith(('.jpg', '.jpeg'))]
    
    dataset = InferDataset(image_filenames, None, dataset)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    
    model.eval()
    with torch.no_grad():
        for images, image_names in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
            outputs = F.interpolate(outputs, size=(1024, 1360), mode='bilinear', align_corners=False)
            print(f"output shape is:{outputs.shape}")
            outputs = outputs.cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                save_path = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}_pred")
                np.save(save_path, output)

@click.command(context_settings={'show_default': True})
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--output_path', '-o', type=click.Path(), required=True, help='Directory to save the output .npy files')
@click.option('--transform-min-scale', type=float, default=0.2, help='Minimum scale for data transformation')
@click.option('--transform-normalize', type=bool, is_flag=True, default=False, help='Normalize the data during transformation')
@click.option('--vit-model', type=str, default='ViT_L_16', help='ViT model type')
@click.option('--starting-weights', type=str, default='ViT_L_16_Weights.DEFAULT', help='ViT starting weights or path to local checkpoint')
@click.option('--batch-size', type=int, default=64, help='Batch size')
@click.option('--n-workers', type=int, default=4, help='Number of workers for data loader')
def main(dataset, output, transform_min_scale, transform_normalize, vit_model,
         starting_weights, batch_size, n_workers):

    transform_kwargs = {'min_scale': transform_min_scale, 'normalize': transform_normalize}
       
    inference_vit(dataset, 
                  output, 
                  transform_kwargs=transform_kwargs,
                  vit_model=vit_model,
                  starting_weights=starting_weights,
                  batch_size=batch_size, 
                  n_workers=n_workers)
              
if __name__ == '__main__':
    main()
