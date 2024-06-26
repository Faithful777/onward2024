import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.optim.lr_scheduler import CyclicLR, ConstantLR, SequentialLR, LinearLR
from lightly.models import utils
from lightly.models.modules import masked_autoencoder
from lightly.transforms.mae_transform import MAETransform
from dataset import ImageDataset
from custom_dataset import MyDataset
from timm.models.vision_transformer import Block
from torchvision.transforms import v2
from tqdm import tqdm
import sys
import os
import click
import json
import matplotlib.pyplot as plt
from segmentation_upernet import ConvMAE, FPN, SegformerHead, F
from sklearn.model_selection import train_test_split

class SegmentationModel(nn.Module):
    '''
    Segmentation head for segmentation task
    
    Args:
        backbone: model, model of the backbone with pretrained weights
        neck: model, model of the neck that connects the backbone to the head
        decode_head: model, model of the segmentation head
    
    Returns:
        x: tensor, output tensor of the segmentation head
    '''
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

def transfer_vit(dataset: str,
                 output: str = 'checkpoints/ViT_L_16_pretrained.pth',
                 transform_kwargs: dict = {'min_scale': 0.2, 'normalize': False},
                 vit_model: str = 'ViT_L_16', 
                 starting_weights: str = "ViT_L_16_Weights.DEFAULT", 
                 local_checkpoint: bool = False,
                 batch_size: int = 64,
                 n_workers: int = 4,
                 optimizer: str = 'SGD',
                 optimizer_kwargs: dict = {'lr': 5e-5, 'momentum': 0.0},
                 warmup_epochs: int = 10,
                 start_factor: float = 0.2,
                 linear_schedule: bool = True,
                 end_factor: float = 0.5,
                 n_epochs: int = 50,
                 cyclic_schedule: bool = False,
                 cyclic_step_size: int = 1000,
                 freeze_backbone: bool = True,
                 shell_call: bool = False) -> dict:
    """
    Transfer a Vision Transformer (ViT) model on a custom image dataset using a Masked Autoencoder (MAE) backbone.

    Args:
        dataset (str): Path to the seismic dataset directory.

    Options:
        output (str): Path to output checkpoint file. .pth'.
        transform_kwargs (dict): Dictionary containing transformation arguments.
            - min_scale (float): Minimum scale for data transformation. Default is 0.2.
            - normalize (bool): Normalize the data during transformation. Default is False.
        vit_model (str): ViT model type. Default is 'ViT_L_16'.
        starting_weights (str): ViT starting weights. Default is 'ViT_L_16_Weights.DEFAULT'.
        local_checkpoint (bool): load weights from local checkpoint.
        batch_size (int): Batch size. Default is 64.
        n_workers (int): Number of workers for data loader. Default is 4.
        optimizer (str): Optimizer type. Default is 'SGD'.
        optimizer_kwargs (dict): Dictionary containing additional keyword arguments passed to optimizer init.
            Default is {'lr': 5e-5, 'momentum': 0.0}.
        warmup_epochs (int): Number of warmup epochs. Default is 10.
        start_factor (float): Initial LR decrease for warmup. Default is 0.2.
        linear_schedule (bool): Use linear LR schedule after warmup period. Default is True.
        end_factor (float): Final decay factor for linear LR schedule. Default is 0.5.
        n_epochs (int): Number of training epochs. Default is 50.
        cyclic_schedule (bool): Use pre-optimized cyclic LR schedule. Default is False.
        cyclic_step_size (int): Number for iterations for half of LR cycle. Default is 1000.
       
    Returns:
        Dictionary containing loss for each epoch of training
    """
    
    def get_schedulers(optimizer):
        base_lr = optimizer_kwargs.get('lr')
        warmup = warmup_epochs > 0  
        if linear_schedule:
            linear_scheduler = LinearLR(optimizer, start_factor=1.0, total_iters=n_epochs, end_factor=end_factor)

        else:
            # placeholder
            linear_scheduler = ConstantLR(optimizer, factor=1.0)
        
        cyclic_scheduler = None
        if cyclic_schedule:
            cyclic_scheduler = CyclicLR(optimizer, base_lr=base_lr / 4.0, max_lr=base_lr,
                                        step_size_up=cyclic_step_size, mode='exp_range', gamma=0.99994)

        if warmup:
            warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
            sequential_scheduler = SequentialLR(optimizer, [warmup_scheduler, linear_scheduler], milestones=[warmup_epochs])
            return sequential_scheduler, cyclic_scheduler
        
        return linear_scheduler, cyclic_scheduler
    
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
    
    # Get the state dictionary
    state_dict = backbone.state_dict()
    
    checkpoint = torch.load(starting_weights, map_location=torch.device('cpu'))
    
    # Load the weights into the model, matching parameter names
    for name, param in checkpoint.items():
        if name in state_dict:
            if state_dict[name].shape == param.shape:
                state_dict[name].copy_(param)
                print(f"Loaded {name} from checkpoint")
            else:
                print(f"Shape mismatch for {name}: expected {state_dict[name].shape}, got {param.shape}")
        else:
            print(f"Skipping {name} as it is not in the segmentation model")
    
    for param in backbone.parameters():
        param.requires_grad = not freeze_backbone
    
    # Initialize neck
    neck = FPN(
        in_channels=[256, 384, 768, 768],
        out_channels=256,
        num_outs=4
    )
    
    # Initialize head
    head = SegformerHead(
        in_channels=[256, 256, 256, 256],
        channels=256,
        num_classes=num_classes,
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False
    )
    
    # Combine to create the segmentation model
    model = SegmentationModel(backbone, neck, head)
    
    transform = MAETransform(**transform_kwargs)

    image_folder = os.path.join(dataset, 'image')
    label_folder = os.path.join(dataset, 'label')
    image_filenames = [file for file in os.listdir(image_folder) if file.endswith(('.JPG','.jpg','.jpeg', '.JPEG'))]
    label_filenames = [file for file in os.listdir(label_folder) if file.endswith(('.npy'))]
    
    dataset = MyDataset(image_filenames, label_filenames, dataset)

    # Split the dataset into training and validation sets (80:20)
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
    elif optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    total_epochs = n_epochs if warmup_epochs < 1 else n_epochs + warmup_epochs
    main_scheduler, cyclic_scheduler = get_schedulers(optimizer)
    loss_history = dict()
    print("Entering Training Loop")
    
    for epoch in range(total_epochs):
        total_loss = .0
        model.train()
        for images, labels in tqdm(train_dataloader):
            images = images.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            out = model(images)
        
            criterion = nn.CrossEntropyLoss()
            label = labels.squeeze(1)  # Squeeze the channel dimension
            loss = criterion(out, label)
            
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            if cyclic_schedule:
                cyclic_scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        current_lr = main_scheduler.get_last_lr()[-1]
        print(f"epoch: {epoch:>03}, train_loss: {avg_loss:.5f}, base_lr: {current_lr:.7f}")
        main_scheduler.step()
        loss_history.update({epoch: {'train_loss': avg_loss.item(), 'base_lr': current_lr}})

        # Validation run
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader):
                images = images.to(device)
                labels = labels.to(device).long()
                out = model(images)
                label = labels.squeeze(1)
                loss = criterion(out, label)
                val_loss += loss.detach()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"epoch: {epoch:>03}, val_loss: {avg_val_loss:.5f}")
        loss_history[epoch]['val_loss'] = avg_val_loss.item()

    print("Training Completed")
    torch.save(model.state_dict(), output)
    if shell_call:
        checkpoint_dir, checkpoint_name = os.path.split(output)
        report_path = checkpoint_dir + '/' + checkpoint_name.split('.')[0] + '_report.json'
        with open(report_path, 'w+') as f:
            json.dump(loss_history, f)
        return
    return loss_history

@click.command(context_settings={'show_default': True})
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='ViT_L_16_SEISMIC.pth', help='Path to output checkpoint')
@click.option('--transform-min-scale', type=float, default=0.2, help='Minimum scale for data transformation')
@click.option('--transform-normalize', type=bool, is_flag=True, default=False, help='Normalize the data during transformation')
@click.option('--vit-model', type=str, default='ViT_L_16', help='ViT model type')
@click.option('--local-checkpoint', type=bool, default=False, is_flag=True, help='Use local checkpoint of ViT')
@click.option('--starting-weights', type=str, default='ViT_L_16_Weights.DEFAULT', help='ViT starting weights or path to local checpoint')
@click.option('--batch-size', type=int, default=64, help='Batch size')
@click.option('--n-workers', type=int, default=4, help='Number of workers for data loader')
@click.option('--optimizer', type=str, default='SGD', help='Optimizer type')
@click.option('--lr', type=float, default=5e-5, help='Learning rate for the optimizer')
@click.option('--optimizer-params', type=(str, float), multiple=True, default=[['momentum', 0.0]], help='Additional keyword arguments passed to optimizer init')
@click.option('--n-epochs', type=int, default=50, help='Number of training epochs')
@click.option('--warmup-epochs', type=int, default=10, help='Number of warmup epochs')
@click.option('--start-factor', type=float, default=0.2, help='Initial LR decrease for warmup')
@click.option('--linear-schedule', type=bool, default=False, is_flag=True, help='Use linear LR schedule after warmup period')
@click.option('--end-factor', type=float, default=0.5, help='Final decay factor for linear LR schedule')
@click.option('--cyclic-schedule', type=bool, default=False, is_flag=True, help='Use pre-optimized cyclic LR schedule')
@click.option('--cyclic-step-size', type=int, default=1000, help='Number for iterations for half of LR cycle')   
@click.option('--freeze-backbone', type=bool, default=False, is_flag=True, help='Freeze segmentation backbone')
def main(dataset, output, transform_min_scale, transform_normalize, vit_model,
         local_checkpoint, starting_weights, batch_size, n_workers, optimizer, 
         lr, optimizer_params, n_epochs, warmup_epochs, start_factor, linear_schedule,
         end_factor, cyclic_schedule, cyclic_step_size,
         freeze_backbone):

    transform_kwargs = {'min_scale': transform_min_scale, 'normalize': transform_normalize}
    optimizer_kwargs = {'lr': lr, **dict(optimizer_params)}
       
    transfer_vit(dataset, 
                 output, 
                 transform_kwargs=transform_kwargs,
                 vit_model=vit_model,
                 starting_weights=starting_weights,
                 local_checkpoint=local_checkpoint,
                 batch_size=batch_size, 
                 n_workers=n_workers, 
                 optimizer=optimizer, 
                 optimizer_kwargs=optimizer_kwargs, 
                 n_epochs=n_epochs, 
                 warmup_epochs=warmup_epochs,
                 start_factor=start_factor,
                 linear_schedule=linear_schedule,
                 end_factor=end_factor,
                 cyclic_schedule=cyclic_schedule,
                 cyclic_step_size=cyclic_step_size,
                 freeze_backbone=freeze_backbone,
                 shell_call=True)

if __name__ == '__main__':
    main()
