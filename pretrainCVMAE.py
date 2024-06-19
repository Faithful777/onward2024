import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch import nn
from torch.optim.lr_scheduler import CyclicLR, ConstantLR, SequentialLR, LinearLR
from lightly.models import utils
from lightly.models.modules import masked_autoencoder
from lightly.transforms.mae_transform import MAETransform
import sys
import os
from dataset import ImageDataset
import click
import json
import matplotlib.pyplot as plt
from functools import partial
from vision_transformer import PatchEmbed, Block, CBlock
from util.pos_embed import get_2d_sincos_pos_embed

class MaskedAutoencoderConvViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # ConvMAE encoder specifics
        self.patch_embed1 = PatchEmbed(
                img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
                img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
                img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False)
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[2])])
        self.norm = norm_layer(embed_dim[-1])

        # --------------------------------------------------------------------------
        # ConvMAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size[0] * patch_size[1] * patch_size[2])**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches
#        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
#        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_for_patch1 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 14, 14, 4, 4).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1)
        mask_for_patch2 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1)
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        stage1_embed = torch.gather(stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))


        # apply Transformer blocks
        for blk in self.blocks3:
            x = blk(x)
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]  - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def convmae_convvit_base_patch16_dec512d8b(starting_weights,**kwargs):
    model = MaskedAutoencoderConvViT(
        img_size=[1024, 256, 128], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint = torch.load(starting_weights,
                              map_location=torch.device('cpu'))
    #print(checkpoint)
    #model.load_state_dict(checkpoint, strict=False)
    # Load the weights into the model, matching parameter names
    state_dict = model.state_dict()
    for name, param in checkpoint.items():
        for name,param2 in param.items():
            print(f'this is the{name}')
            if name in state_dict:
                if state_dict[name].shape == param2.shape:
                    state_dict[name].copy_(param2)
                    print(f"Loaded {name} from checkpoint")
                else:
                    print(f"Shape mismatch for {name}: expected {state_dict[name].shape}, got {param2.shape}")
            else:
                print(f"Skipping {name} as it is not in the model")
    
        return model
    
def pretrain_mae(dataset: str,
                 output: str = 'checkpoints/ViT_L_16_SEISMIC.pth',
                 transform_kwargs: dict = {'min_scale': 0.2, 'normalize': False},
                 vit_model: str = 'ViT_L_16', 
                 starting_weights: str = "ViT_L_16_Weights.DEFAULT", 
                 local_checkpoint: bool = False,
                 batch_size: int=64,
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
                 #masking_rate: float = 0.75,
                 #decoder_dim: int = 1024,
                 #freeze_embeddings: bool = True,
                 #freeze_projection: bool = True,
                 shell_call: bool = False) -> dict:
    """
    Pre-train-tune a Vision Transformer (ViT) model on a seismic dataset.

    Args:
        dataset (str): Path to the seismic dataset directory.

    Options:
        output (str): Path to output checkpoint file. Default is 'checkpoints/ViT_L_16_SEISMIC_PRETRAINED.pth'.
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
        Xmasking_rate (float): Masking rate for pretraining. Default is 0.75.
        Xdecoder_dim (int): Dimension of the decoder tokens. Default is 1024.
        Xfreeze_projection (bool): Freeze convolutional projection layer of ViT. Default is True.
        Xfreeze_embeddings (bool): Freeze embedding layer of ViT. Default is True.
       
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
            cyclic_scheduler = CyclicLR(optimizer, base_lr=base_lr/4.0, max_lr=base_lr,
                                        step_size_up=cyclic_step_size, mode='exp_range', gamma=0.99994)

        if warmup:
            warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
            sequential_scheduler = SequentialLR(optimizer, [warmup_scheduler, linear_scheduler], milestones=[warmup_epochs])
            return sequential_scheduler, cyclic_scheduler
        
        return linear_scheduler, cyclic_scheduler
    
    transform = MAETransform(input_size=1024,**transform_kwargs)
    
    # Loading unlabeled image dataset from folder
    dataset = torchvision.datasets.ImageFolder(root=dataset, transform=transform)
    
    
    model = convmae_convvit_base_patch16_dec512d8b(starting_weights)  # decoder: 512 dim, 8 blocks


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
    )

    criterion = nn.MSELoss()
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
        for batch in dataloader:
            views = batch[0]
            images = views[0].to(device)  # views contains only a single view
            #_, predictions, targets = model(images)
            #loss = criterion(predictions, targets)
            loss, _, _ = model(images)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if cyclic_schedule:
                cyclic_scheduler.step()
        avg_loss = total_loss / len(dataloader)
        current_lr = main_scheduler.get_last_lr()[-1]
        print(f"epoch: {epoch:>03}, loss: {avg_loss:.5f}, base_lr: {current_lr:.7f}")
        main_scheduler.step()
        loss_history.update({epoch: {'loss': avg_loss.item(), 'base_lr': current_lr}})
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
#@click.option('--masking-rate', type=float, default=0.75, help='Masking rate for pretraining')
#@click.option('--decoder-dim', type=int, default=1024, help='Dimension of the decoder tokens')       
#@click.option('--freeze-projection', type=bool, default=False, is_flag=True, help='Freeze class token and convolutional projection layer of ViT') 
#@click.option('--freeze-embeddings', type=bool, default=False, is_flag=True, help='Freeze positional embedding layer of ViT')  
def main(dataset, output, transform_min_scale, transform_normalize, vit_model,
         local_checkpoint, starting_weights, batch_size, n_workers, optimizer, 
         lr, optimizer_params, n_epochs, warmup_epochs, start_factor, linear_schedule,
         end_factor, cyclic_schedule, cyclic_step_size):

    transform_kwargs = {'min_scale': transform_min_scale, 'normalize': transform_normalize}
    optimizer_kwargs = {'lr': lr, **dict(optimizer_params)}
       
    pretrain_mae(dataset, 
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
                 shell_call=True)
              
if __name__ == '__main__':
    main()
