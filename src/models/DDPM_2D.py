from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
import torch
from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from typing import Any, List

import torchio as tio
from src.utils.generate_noise import gen_noise
class DDPM_2D(LightningModule):
    def __init__(self,cfg,prefix=None):
        super().__init__()
        
        self.cfg = cfg
        
        # Model 
        model = OpenAI_UNet(
                            image_size =  (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)),
                            in_channels = 1,
                            model_channels = cfg.get('unet_dim',64),
                            out_channels = 1,
                            num_res_blocks = cfg.get('num_res_blocks',3),
                            attention_resolutions = tuple(cfg.get('att_res',[3,6,12])), # 32, 16, 8
                            dropout=cfg.get('dropout_unet',0), 
                            channel_mult=cfg.get('dim_mults',[1, 2, 4, 8]),
                            conv_resample=True,
                            dims=2,
                            num_classes=None,
                            use_checkpoint=False,
                            use_fp16=True,
                            num_heads=1,
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=False,
                            resblock_updown=True,
                            use_new_attention_order=True,
                            use_spatial_transformer=False,    
                            transformer_depth=1,              
                            context_dim=None,                    
                            )
        model.convert_to_fp16()
      

        timesteps = cfg.get('timesteps',1000)
        sampling_timesteps = cfg.get('sampling_timesteps',timesteps)
        self.test_timesteps = cfg.get('test_timesteps',150) 

        self.diffusion = GaussianDiffusion(
        model,
        image_size = (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)), # only important when sampling
        timesteps = timesteps,   # number of steps
        sampling_timesteps = sampling_timesteps,
        objective = cfg.get('objective','pred_x0'), # pred_noise or pred_x0
        channels = 1,
        loss_type = cfg.get('loss','l1'),    # L1 or L2
        p2_loss_weight_gamma = cfg.get('p2_gamma',0),
        cfg=cfg
        )
        
        self.prefix = prefix
        
        self.save_hyperparameters()

    def forward(self):
        return None


    def training_step(self, batch, batch_idx: int):
        # process batch
        input = batch['vol'][tio.DATA].squeeze(-1)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None
        # reconstruct
        loss, reco = self.diffusion(input, noise = noise)

        self.log(f'{self.prefix}train/Loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}
    
    def validation_step(self, batch: Any, batch_idx: int):
        input = batch['vol'][tio.DATA].squeeze(-1) 


        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None
        # reconstruct
        loss, reco = self.diffusion(input,noise=noise)


        self.log(f'{self.prefix}val/Loss_comb', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.histo_input = []
        self.histo_reco = []
        self.latentSpace_slice = []
        self.grad_feat = []
        self.new_size = [160,190,160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self,'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):
        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        self.stage = batch['stage']
        label = batch['label']
        AnomalyScoreComb = []
        AnomalyScoreReg = []
        AnomalyScoreReco = []
        latentSpace = []

        if self.cfg.get('num_eval_slices', input.size(4)) != input.size(4): # if not all slices are evaluated. Default is to evaluate all slices
            num_slices = self.cfg.get('num_eval_slices', input.size(4))  # number of center slices to evaluate. If not set, the whole Volume is evaluated
            start_slice = int((input.size(4) - num_slices) / 2)
            input = input[...,start_slice:start_slice+num_slices]
            data_orig = data_orig[...,start_slice:start_slice+num_slices] 
            data_seg = data_seg[...,start_slice:start_slice+num_slices]
            data_mask = data_mask[...,start_slice:start_slice+num_slices]
            ind_offset = start_slice
        else: 
            ind_offset = 0 

        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device = self.device)

        # reorder depth to batch dimension
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]


        latentSpace.append(torch.tensor([0],dtype=float).repeat(input.shape[0])) # dummy latent space 

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None

        # reconstruct
        loss_diff, reco = self.diffusion(input,t=self.test_timesteps-1,noise=noise)
        

        # calculate loss and Anomalyscores
        AnomalyScoreComb.append(loss_diff.cpu())
        AnomalyScoreReg.append(loss_diff.cpu()) # dummy
        AnomalyScoreReco.append(loss_diff.cpu()) # dummy

        # reassamble the reconstruction volume
        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1,2,0) # to HxWxD

        # average across slices to get volume-based scores
        self.latentSpace_slice.extend(latentSpace)
        self.eval_dict['latentSpace'].append(torch.mean(torch.stack(latentSpace),0))
        AnomalyScoreComb_vol = np.mean(AnomalyScoreComb) 
        AnomalyScoreReg_vol = AnomalyScoreComb_vol # dummy
        AnomalyScoreReco_vol = AnomalyScoreComb_vol # dummy

        self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)


        if not self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)


        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)

        # calculate metrics
        _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label) # everything that is independent of the model choice

           
    def on_test_end(self) :
        _test_end(self) # everything that is independent of the model choice 


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    def update_prefix(self, prefix): # prefix for logging
        self.prefix = prefix 