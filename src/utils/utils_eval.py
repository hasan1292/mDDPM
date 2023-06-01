
from torch import nn
import torch
from skimage.measure import regionprops, label
from torchvision.transforms import ToTensor, ToPILImage
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score
import wandb 
import monai
from torch.nn import functional as F
from PIL import Image

import matplotlib.colors as colors

def _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label_vol) :
        self.healthy_sets = ['IXI']
        # Resize the images if desired
        if not self.cfg.resizedEvaluation: # in case of full resolution evaluation 
            final_volume = F.interpolate(final_volume, size=self.new_size, mode="trilinear",align_corners=True).squeeze() # resize
        else: 
            final_volume = final_volume.squeeze()
        
        # calculate the residual image
        if self.cfg.get('residualmode','l1'): # l1 or l2 residual
            diff_volume = torch.abs((data_orig-final_volume))
        else:
            diff_volume = (data_orig-final_volume)**2

       # Calculate Reconstruction errors with respect to anomal/normal regions
        l1err = nn.functional.l1_loss(final_volume.squeeze(),data_orig.squeeze())
        l2err = nn.functional.mse_loss(final_volume.squeeze(),data_orig.squeeze())
        l1err_anomal = nn.functional.l1_loss(final_volume.squeeze()[data_seg.squeeze() > 0],data_orig[data_seg > 0]) 
        l1err_healthy = nn.functional.l1_loss(final_volume.squeeze()[data_seg.squeeze() == 0],data_orig[data_seg == 0]) 
        l2err_anomal = nn.functional.mse_loss(final_volume.squeeze()[data_seg.squeeze() > 0],data_orig[data_seg > 0]) 
        l2err_healthy = nn.functional.mse_loss(final_volume.squeeze()[data_seg.squeeze() == 0],data_orig[data_seg == 0])

        # store in eval dict
        self.eval_dict['l1recoErrorAll'].append(l1err.item())
        self.eval_dict['l1recoErrorUnhealthy'].append(l1err_anomal.item())
        self.eval_dict['l1recoErrorHealthy'].append(l1err_healthy.item())
        self.eval_dict['l2recoErrorAll'].append(l2err.item())
        self.eval_dict['l2recoErrorUnhealthy'].append(l2err_anomal.item())
        self.eval_dict['l2recoErrorHealthy'].append(l2err_healthy.item())

        # move data to CPU
        data_seg = data_seg.cpu() 
        data_mask = data_mask.cpu()
        diff_volume = diff_volume.cpu()
        data_orig = data_orig.cpu()
        final_volume = final_volume.cpu()
        # binarize the segmentation
        data_seg[data_seg > 0] = 1
        data_mask[data_mask > 0] = 1

        # Erode the Brainmask 
        if self.cfg['erodeBrainmask']:
            diff_volume = apply_brainmask_volume(diff_volume.cpu(), data_mask.squeeze().cpu())   

        # Filter the DifferenceImage
        if self.cfg['medianFiltering']:
            diff_volume = torch.from_numpy(apply_3d_median_filter(diff_volume.numpy().squeeze(),kernelsize=self.cfg.get('kernelsize_median',5))).unsqueeze(0) # bring back to tensor

        # save image grid
        if self.cfg['saveOutputImages'] :
            log_images(self,diff_volume, data_orig, data_seg, data_mask, final_volume, ID)
            
        ### Compute Metrics per Volume / Step ###
        if self.cfg.evalSeg and self.dataset[0] not in self.healthy_sets: # only compute metrics if segmentation is available

            # Pixel-Wise Segmentation Error Metrics based on Differenceimage
            AUC, _fpr, _tpr, _threshs = compute_roc(diff_volume.squeeze().flatten(), np.array(data_seg.squeeze().flatten()).astype(bool))
            AUPRC, _precisions, _recalls, _threshs = compute_prc(diff_volume.squeeze().flatten(),np.array(data_seg.squeeze().flatten()).astype(bool))

            # gready search for threshold
            bestDice, bestThresh = find_best_val(np.array(diff_volume.squeeze()).flatten(),  # threshold search with a subset of EvaluationSet
                                                np.array(data_seg.squeeze()).flatten().astype(bool), 
                                                val_range=(0, np.max(np.array(diff_volume))),
                                                max_steps=10, 
                                                step=0, 
                                                max_val=0, 
                                                max_point=0)

            if 'test' in self.stage:
                bestThresh = self.threshold['total']

            if self.cfg["threshold"] == 'auto':
                diffs_thresholded = diff_volume > bestThresh
            else: # never used
                diffs_thresholded = diff_volume > self.cfg["threshold"]    
            
            # Connected Components
            if not 'node' in self.dataset[0].lower(): # no 3D data
                diffs_thresholded = filter_3d_connected_components(np.squeeze(diffs_thresholded)) # this is only done for patient-wise evaluation atm
            
            # Calculate Dice Score with thresholded volumes
            diceScore = dice(np.array(diffs_thresholded.squeeze()),np.array(data_seg.squeeze().flatten()).astype(bool))
            
            # Other Metrics
            TP, FP, TN, FN = confusion_matrix(np.array(diffs_thresholded.squeeze().flatten()), np.array(data_seg.squeeze().flatten()).astype(bool),labels=[0, 1]).ravel()
            TPR = tpr(np.array(diffs_thresholded.squeeze()), np.array(data_seg.squeeze().flatten()).astype(bool))
            FPR = fpr(np.array(diffs_thresholded.squeeze()), np.array(data_seg.squeeze().flatten()).astype(bool))
            self.eval_dict['lesionSizePerVol'].append(np.count_nonzero(np.array(data_seg.squeeze().flatten()).astype(bool)))
            self.eval_dict['DiceScorePerVol'].append(diceScore)
            self.eval_dict['BestDicePerVol'].append(bestDice)
            self.eval_dict['BestThresholdPerVol'].append(bestThresh)
            self.eval_dict['AUCPerVol'].append(AUC)
            self.eval_dict['AUPRCPerVol'].append(AUPRC)
            self.eval_dict['TPPerVol'].append(TP)
            self.eval_dict['FPPerVol'].append(FP)
            self.eval_dict['TNPerVol'].append(TN)
            self.eval_dict['FNPerVol'].append(FN) 
            self.eval_dict['TPRPerVol'].append(TPR)
            self.eval_dict['FPRPerVol'].append(FPR)
            self.eval_dict['IDs'].append(ID[0])

            PrecRecF1PerVol = precision_recall_fscore_support(np.array(data_seg.squeeze().flatten()).astype(bool),np.array(diffs_thresholded.squeeze()).flatten(),labels=[0,1])
            self.eval_dict['AccuracyPerVol'].append(accuracy_score(np.array(data_seg.squeeze().flatten()).astype(bool),np.array(diffs_thresholded.squeeze()).flatten()))
            self.eval_dict['PrecisionPerVol'].append(PrecRecF1PerVol[0][1])
            self.eval_dict['RecallPerVol'].append(PrecRecF1PerVol[1][1])
            self.eval_dict['SpecificityPerVol'].append(TN / (TN+FP+0.0000001))

            # other metrics from monai:
            if len(data_seg.shape) == 4:
                data_seg = data_seg.unsqueeze(0)
            Haus = monai.metrics.compute_hausdorff_distance(diffs_thresholded.unsqueeze(0).unsqueeze(0),data_seg, include_background=False, distance_metric='euclidean', percentile=None, directed=False)
            self.eval_dict['HausPerVol'].append(Haus.item())

            # compute slice-wise metrics
            for slice in range(data_seg.squeeze().shape[0]): 
                    if np.array(data_seg.squeeze()[slice].flatten()).astype(bool).any():
                        self.eval_dict['DiceScorePerSlice'].append(dice(np.array(diff_volume.squeeze()[slice] > bestThresh),np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))
                        PrecRecF1PerSlice = precision_recall_fscore_support(np.array(data_seg.squeeze()[slice].flatten()).astype(bool),np.array(diff_volume.squeeze()[slice] > bestThresh).flatten(),warn_for=tuple(),labels=[0,1])
                        self.eval_dict['PrecisionPerSlice'].append(PrecRecF1PerSlice[0][1])
                        self.eval_dict['RecallPerSlice'].append(PrecRecF1PerSlice[1][1])
                        self.eval_dict['lesionSizePerSlice'].append(np.count_nonzero(np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))

        if 'val' in self.stage  :
            if batch_idx == 0:
                self.diffs_list = np.array(diff_volume.squeeze().flatten())
                self.seg_list = np.array(data_seg.squeeze().flatten()).astype(np.int8)
            else: 
                self.diffs_list = np.append(self.diffs_list,np.array(diff_volume.squeeze().flatten()),axis=0)
                self.seg_list = np.append(self.seg_list,np.array(data_seg.squeeze().flatten()),axis=0).astype(np.int8)

        # Reconstruction based Anomaly score for Slice-Wise evaluation  
        if self.cfg.get('use_postprocessed_score', True):
            AnomalyScoreReco_vol = diff_volume.squeeze()[data_mask.squeeze()>0].mean() # for sample-wise detection 

        AnomalyScoreReco = [] # Reconstruction based Anomaly score
        if len(diff_volume.squeeze().shape) !=2:
            for slice in range(diff_volume.squeeze().shape[0]): 
                score = diff_volume.squeeze()[slice][data_mask.squeeze()[slice]>0].mean()
                if score.isnan() : # if no brain exists in that slice
                    AnomalyScoreReco.append(0.0) 
                else: 
                    AnomalyScoreReco.append(score) 

            # create slice-wise labels 
            data_seg_downsampled = np.array(data_seg.squeeze())
            label = [] # store labels here
            for slice in range(data_seg_downsampled.shape[0]) :  #iterate through volume
                if np.array(data_seg_downsampled[slice]).astype(bool).any(): # if there is an anomaly segmentation
                    label.append(1) # label = 1
                else :
                    label.append(0) # label = 0 if there is no Anomaly in the slice
                    
            if self.dataset[0] not in self.healthy_sets:
                AUC, _fpr, _tpr, _threshs = compute_roc(np.array(AnomalyScoreReco),np.array(label))
                AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(AnomalyScoreReco),np.array(label))
                self.eval_dict['AUCAnomalyRecoPerSlice'].append(AUC)
                self.eval_dict['AUPRCAnomalyRecoPerSlice'].append(AUPRC)
                self.eval_dict['labelPerSlice'].extend(label)
                # store Slice-wise Anomalyscore (reconstruction based)
                self.eval_dict['AnomalyScoreRecoPerSlice'].extend(AnomalyScoreReco)

        # sample-Wise Anomalyscores
        if self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreReco_vol )
            self.eval_dict['AnomalyScoreCombPriorPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(AnomalyScoreReco_vol )
        

        self.eval_dict['labelPerVol'].append(label_vol.item())

def _test_end(self) :
    # average over all test samples
        self.eval_dict['l1recoErrorAllMean'] = np.nanmean(self.eval_dict['l1recoErrorAll'])
        self.eval_dict['l1recoErrorAllStd'] = np.nanstd(self.eval_dict['l1recoErrorAll'])
        self.eval_dict['l2recoErrorAllMean'] = np.nanmean(self.eval_dict['l2recoErrorAll'])
        self.eval_dict['l2recoErrorAllStd'] = np.nanstd(self.eval_dict['l2recoErrorAll'])

        self.eval_dict['l1recoErrorHealthyMean'] = np.nanmean(self.eval_dict['l1recoErrorHealthy'])
        self.eval_dict['l1recoErrorHealthyStd'] = np.nanstd(self.eval_dict['l1recoErrorHealthy'])
        self.eval_dict['l1recoErrorUnhealthyMean'] = np.nanmean(self.eval_dict['l1recoErrorUnhealthy'])
        self.eval_dict['l1recoErrorUnhealthyStd'] = np.nanstd(self.eval_dict['l1recoErrorUnhealthy'])

        self.eval_dict['l2recoErrorHealthyMean'] = np.nanmean(self.eval_dict['l2recoErrorHealthy'])
        self.eval_dict['l2recoErrorHealthyStd'] = np.nanstd(self.eval_dict['l2recoErrorHealthy'])
        self.eval_dict['l2recoErrorUnhealthyMean'] = np.nanmean(self.eval_dict['l2recoErrorUnhealthy'])
        self.eval_dict['l2recoErrorUnhealthyStd'] = np.nanstd(self.eval_dict['l2recoErrorUnhealthy'])

        self.eval_dict['AUPRCPerVolMean'] = np.nanmean(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUPRCPerVolStd'] = np.nanstd(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUCPerVolMean'] = np.nanmean(self.eval_dict['AUCPerVol'])
        self.eval_dict['AUCPerVolStd'] = np.nanstd(self.eval_dict['AUCPerVol'])

        self.eval_dict['DicePerVolMean'] = np.nanmean(self.eval_dict['DiceScorePerVol'])
        self.eval_dict['DicePerVolStd'] = np.nanstd(self.eval_dict['DiceScorePerVol'])
        self.eval_dict['BestDicePerVolMean'] = np.mean(self.eval_dict['BestDicePerVol'])
        self.eval_dict['BestDicePerVolStd'] = np.std(self.eval_dict['BestDicePerVol'])
        self.eval_dict['BestThresholdPerVolMean'] = np.mean(self.eval_dict['BestThresholdPerVol'])
        self.eval_dict['BestThresholdPerVolStd'] = np.std(self.eval_dict['BestThresholdPerVol'])


        self.eval_dict['TPPerVolMean'] = np.nanmean(self.eval_dict['TPPerVol'])
        self.eval_dict['TPPerVolStd'] = np.nanstd(self.eval_dict['TPPerVol'])
        self.eval_dict['FPPerVolMean'] = np.nanmean(self.eval_dict['FPPerVol'])
        self.eval_dict['FPPerVolStd'] = np.nanstd(self.eval_dict['FPPerVol'])
        self.eval_dict['TNPerVolMean'] = np.nanmean(self.eval_dict['TNPerVol'])
        self.eval_dict['TNPerVolStd'] = np.nanstd(self.eval_dict['TNPerVol'])
        self.eval_dict['FNPerVolMean'] = np.nanmean(self.eval_dict['FNPerVol'])
        self.eval_dict['FNPerVolStd'] = np.nanstd(self.eval_dict['FNPerVol'])
        self.eval_dict['TPRPerVolMean'] = np.nanmean(self.eval_dict['TPRPerVol'])
        self.eval_dict['TPRPerVolStd'] = np.nanstd(self.eval_dict['TPRPerVol'])
        self.eval_dict['FPRPerVolMean'] = np.nanmean(self.eval_dict['FPRPerVol'])
        self.eval_dict['FPRPerVolStd'] = np.nanstd(self.eval_dict['FPRPerVol'])
        self.eval_dict['HausPerVolMean'] = np.nanmean(np.array(self.eval_dict['HausPerVol'])[np.isfinite(self.eval_dict['HausPerVol'])])
        self.eval_dict['HausPerVolStd'] = np.nanstd(np.array(self.eval_dict['HausPerVol'])[np.isfinite(self.eval_dict['HausPerVol'])])
        


        self.eval_dict['PrecisionPerVolMean'] = np.mean(self.eval_dict['PrecisionPerVol'])
        self.eval_dict['PrecisionPerVolStd'] =np.std(self.eval_dict['PrecisionPerVol'])
        self.eval_dict['RecallPerVolMean'] = np.mean(self.eval_dict['RecallPerVol'])
        self.eval_dict['RecallPerVolStd'] = np.std(self.eval_dict['RecallPerVol'])
        self.eval_dict['PrecisionPerSliceMean'] = np.mean(self.eval_dict['PrecisionPerSlice'])
        self.eval_dict['PrecisionPerSliceStd'] = np.std(self.eval_dict['PrecisionPerSlice'])
        self.eval_dict['RecallPerSliceMean'] = np.mean(self.eval_dict['RecallPerSlice'])
        self.eval_dict['RecallPerSliceStd'] = np.std(self.eval_dict['RecallPerSlice'])
        self.eval_dict['AccuracyPerVolMean'] = np.mean(self.eval_dict['AccuracyPerVol'])
        self.eval_dict['AccuracyPerVolStd'] = np.std(self.eval_dict['AccuracyPerVol'])
        self.eval_dict['SpecificityPerVolMean'] = np.mean(self.eval_dict['SpecificityPerVol'])
        self.eval_dict['SpecificityPerVolStd'] = np.std(self.eval_dict['SpecificityPerVol'])


        if 'test' in self.stage :
            del self.threshold
                
        if 'val' in self.stage: 
            if self.dataset[0] not in self.healthy_sets:
                bestdiceScore, bestThresh = find_best_val((self.diffs_list).flatten(), (self.seg_list).flatten().astype(bool), 
                                        val_range=(0, np.max((self.diffs_list))), 
                                        max_steps=10, 
                                        step=0, 
                                        max_val=0, 
                                        max_point=0)

                self.threshold['total'] = bestThresh 
                if self.cfg.get('KLDBackprop',False): 
                    bestdiceScoreKLComb, bestThreshKLComb = find_best_val((self.diffs_listKLComb).flatten(), (self.seg_list).flatten().astype(bool), 
                        val_range=(0, np.max((self.diffs_listKLComb))), 
                        max_steps=10, 
                        step=0, 
                        max_val=0, 
                        max_point=0)

                    self.threshold['totalKLComb'] = bestThreshKLComb 
                    bestdiceScoreKL, bestThreshKL = find_best_val((self.diffs_listKL).flatten(), (self.seg_list).flatten().astype(bool), 
                        val_range=(0, np.max((self.diffs_listKL))), 
                        max_steps=10, 
                        step=0, 
                        max_val=0, 
                        max_point=0)

                    self.threshold['totalKL'] = bestThreshKL 
            else: # define thresholds based on the healthy validation set
                _, fpr_healthy, _, threshs = compute_roc((self.diffs_list).flatten(), np.zeros_like(self.diffs_list).flatten().astype(int))
                self.threshholds_healthy= {
                        'thresh_1p' : threshs[np.argmax(fpr_healthy>0.01)], # 1%
                        'thresh_5p' : threshs[np.argmax(fpr_healthy>0.05)], # 5%
                        'thresh_10p' : threshs[np.argmax(fpr_healthy>0.10)]} # 10%}
                self.eval_dict['t_1p'] = self.threshholds_healthy['thresh_1p']
                self.eval_dict['t_5p'] = self.threshholds_healthy['thresh_5p']
                self.eval_dict['t_10p'] = self.threshholds_healthy['thresh_10p']
def calc_thresh(dataset):
    data = dataset['Datamodules_train.Chexpert']
    _, fpr_healthy_comb, _, threshs_healthy_comb = compute_roc(np.array(data['AnomalyScoreCombPerVol']),np.array(data['labelPerVol'])) 
    _, fpr_healthy_combPrior, _, threshs_healthy_combPrior = compute_roc(np.array(data['AnomalyScoreCombPriorPerVol']),np.array(data['labelPerVol']))
    _, fpr_healthy_reg, _, threshs_healthy_reg = compute_roc(np.array(data['AnomalyScoreRegPerVol']),np.array(data['labelPerVol']))
    _, fpr_healthy_reco, _, threshs_healthy_reco = compute_roc(np.array(data['AnomalyScoreRecoPerVol']),np.array(data['labelPerVol']))
    _, fpr_healthy_prior_kld, _, threshs_healthy_prior_kld = compute_roc(np.array(data['KLD_to_learned_prior']),np.array(data['labelPerVol']))
    threshholds_healthy= {
                'thresh_1p_comb' : threshs_healthy_comb[np.argmax(fpr_healthy_comb>0.01)], 
                'thresh_1p_combPrior' : threshs_healthy_combPrior[np.argmax(fpr_healthy_combPrior>0.01)], 
                'thresh_1p_reg' : threshs_healthy_reg[np.argmax(fpr_healthy_reg>0.01)], 
                'thresh_1p_reco' : threshs_healthy_reco[np.argmax(fpr_healthy_reco>0.01)], 
                'thresh_1p_prior_kld' : threshs_healthy_prior_kld[np.argmax(fpr_healthy_prior_kld>0.01)], 
                'thresh_5p_comb' : threshs_healthy_comb[np.argmax(fpr_healthy_comb>0.05)], 
                'thresh_5p_combPrior' : threshs_healthy_combPrior[np.argmax(fpr_healthy_combPrior>0.05)], 
                'thresh_5p_reg' : threshs_healthy_reg[np.argmax(fpr_healthy_reg>0.05)], 
                'thresh_5p_reco' : threshs_healthy_reco[np.argmax(fpr_healthy_reco>0.05)], 
                'thresh_5p_prior_kld' : threshs_healthy_prior_kld[np.argmax(fpr_healthy_prior_kld>0.05)], 
                'thresh_10p_comb' : threshs_healthy_comb[np.argmax(fpr_healthy_comb>0.1)], 
                'thresh_10p_combPrior' : threshs_healthy_combPrior[np.argmax(fpr_healthy_combPrior>0.1)],
                'thresh_10p_reg' : threshs_healthy_reg[np.argmax(fpr_healthy_reg>0.1)], 
                'thresh_10p_reco' : threshs_healthy_reco[np.argmax(fpr_healthy_reco>0.1)],
                'thresh_10p_prior_kld' : threshs_healthy_prior_kld[np.argmax(fpr_healthy_prior_kld>0.1)], } 
    return threshholds_healthy

def get_eval_dictionary():
    _eval = {
        'IDs': [],
        'x': [],
        'reconstructions': [],
        'diffs': [],
        'diffs_volume': [],
        'Segmentation': [],
        'reconstructionTimes': [],
        'latentSpace': [],
        'Age': [],
        'AgeGroup': [],
        'l1reconstructionErrors': [],
        'l1recoErrorAll': [],
        'l1recoErrorUnhealthy': [],
        'l1recoErrorHealthy': [],
        'l2recoErrorAll': [],
        'l2recoErrorUnhealthy': [],
        'l2recoErrorHealthy': [],
        'l1reconstructionErrorMean': 0.0,
        'l1reconstructionErrorStd': 0.0,
        'l2reconstructionErrors': [],
        'l2reconstructionErrorMean': 0.0,
        'l2reconstructionErrorStd': 0.0,
        'HausPerVol': [],
        'TPPerVol': [],
        'FPPerVol': [],
        'FNPerVol': [],
        'TNPerVol': [],
        'TPRPerVol': [],
        'FPRPerVol': [],
        'TPTotal': [],
        'FPTotal': [],
        'FNTotal': [],
        'TNTotal': [],
        'TPRTotal': [],
        'FPRTotal': [],

        'PrecisionPerVol': [],
        'RecallPerVol': [],
        'PrecisionPerSlice': [],
        'RecallPerSlice': [],
        'lesionSizePerSlice': [],
        'lesionSizePerVol': [],
        'Dice': [],
        'DiceScorePerSlice': [],
        'DiceScorePerVol': [],
        'BestDicePerVol': [],
        'BestThresholdPerVol': [],
        'AUCPerVol': [],
        'AUPRCPerVol': [],
        'SpecificityPerVol': [],
        'AccuracyPerVol': [],
        'TPgradELBO': [],
        'FPgradELBO': [],
        'FNgradELBO': [],
        'TNgradELBO': [],
        'TPRgradELBO': [],
        'FPRgradELBO': [],
        'DicegradELBO': [],
        'DiceScorePerVolgradELBO': [],
        'BestDicePerVolgradELBO': [],
        'BestThresholdPerVolgradELBO': [],
        'AUCPerVolgradELBO': [],
        'AUPRCPerVolgradELBO': [],
        'KLD_to_learned_prior':[],

        'AUCAnomalyCombPerSlice': [], # PerVol!!! + Confusionmatrix.
        'AUPRCAnomalyCombPerSlice': [],
        'AnomalyScoreCombPerSlice': [],


        'AUCAnomalyKLDPerSlice': [],
        'AUPRCAnomalyKLDPerSlice': [],
        'AnomalyScoreKLDPerSlice': [],


        'AUCAnomalyRecoPerSlice': [],
        'AUPRCAnomalyRecoPerSlice': [],
        'AnomalyScoreRecoPerSlice': [],
        'AnomalyScoreRecoBinPerSlice': [],
        'AnomalyScoreAgePerSlice': [],
        'AUCAnomalyAgePerSlice': [],
        'AUPRCAnomalyAgePerSlice': [],

        'labelPerSlice' : [],
        'labelPerVol' : [],
        'AnomalyScoreCombPerVol' : [],
        'AnomalyScoreCombiPerVol' : [],
        'AnomalyScoreCombMeanPerVol' : [],
        'AnomalyScoreRegPerVol' : [],
        'AnomalyScoreRegMeanPerVol' : [],
        'AnomalyScoreRecoPerVol' : [],
        'AnomalyScoreCombPriorPerVol': [],
        'AnomalyScoreCombiPriorPerVol': [],
        'AnomalyScoreAgePerVol' : [],
        'AnomalyScoreRecoMeanPerVol' : [],
        'DiceScoreKLPerVol': [],
        'DiceScoreKLCombPerVol': [],
        'BestDiceKLCombPerVol': [],
        'BestDiceKLPerVol': [],
        'AUCKLCombPerVol': [],
        'AUPRCKLCombPerVol': [],
        'AUCKLPerVol': [],
        'AUPRCKLPerVol': [],
        'TPKLCombPerVol': [],
        'FPKLCombPerVol': [],
        'TNKLCombPerVol': [],
        'FNKLCombPerVol': [],
        'TPRKLCombPerVol': [],
        'FPRKLCombPerVol': [],
        'TPKLPerVol': [],
        'FPKLPerVol': [],
        'TNKLPerVol': [],
        'FNKLPerVol': [],
        'TPRKLPerVol': [],
        'FPRKLPerVol': [],



    }
    return _eval

def apply_brainmask(x, brainmask, erode , iterations):
    strel = scipy.ndimage.generate_binary_structure(2, 1)
    brainmask = np.expand_dims(brainmask, 2)
    if erode:
        brainmask = scipy.ndimage.morphology.binary_erosion(np.squeeze(brainmask), structure=strel, iterations=iterations)
    return np.multiply(np.squeeze(brainmask), np.squeeze(x))

def apply_brainmask_volume(vol,mask_vol,erode=True, iterations=10) : 
    for s in range(vol.squeeze().shape[2]): 
        slice = vol.squeeze()[:,:,s]
        mask_slice = mask_vol.squeeze()[:,:,s]
        eroded_vol_slice = apply_brainmask(slice, mask_slice, erode = True, iterations=vol.squeeze().shape[1]//25)
        vol.squeeze()[:,:,s] = eroded_vol_slice
    return vol

def apply_3d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume
def apply_2d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    img = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize))
    return img
    
def squash_intensities(img):
    # logistic function intended to squash reconstruction errors from [0;0.2] to [0;1] (just an example)
    k = 100
    offset = 0.5
    return 2.0 * ((1.0 / (1.0 + np.exp(-k * img))) - offset)


def apply_colormap(img, colormap_handle):
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = Image.fromarray(np.uint8(colormap_handle(img) * 255))
    return img

def add_colorbar(img):
    for i in range(img.squeeze().shape[0]):
        img[i, -1] = float(i) / img.squeeze().shape[0]

    return img

def filter_3d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=3)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 7:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume



# From Zimmerer iterative algorithm for threshold search
def find_best_val(x, y, val_range=(0, 1), max_steps=4, step=0, max_val=0, max_point=0):  #x: Image , y: Label
    if step == max_steps:
        return max_val, max_point

    if val_range[0] == val_range[1]:
        val_range = (val_range[0], 1)

    bottom = val_range[0]
    top = val_range[1]
    center = bottom + (top - bottom) * 0.5

    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75
    val_bottom = dice(x > q_bottom, y)
    #print(str(np.mean(x>q_bottom)) + str(np.mean(y)))
    val_top = dice(x > q_top, y)
    #print(str(np.mean(x>q_top)) + str(np.mean(y)))
    #val_bottom = val_fn(x, y, q_bottom) # val_fn is the dice calculation dice(p, g)
    #val_top = val_fn(x, y, q_top)

    if val_bottom >= val_top:
        if val_bottom >= max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_range=(bottom, center), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)
    else:
        if val_top >= max_val:
            max_val = val_top
            max_point = q_top
        return find_best_val(x, y, val_range=(center, top), step=step + 1, max_steps=max_steps,
                             max_val=max_val,max_point=max_point)
def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score

    
def compute_roc(predictions, labels):
    _fpr, _tpr, _ = roc_curve(labels.astype(int), predictions,pos_label=1)
    roc_auc = auc(_fpr, _tpr)
    return roc_auc, _fpr, _tpr, _


def compute_prc(predictions, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels.astype(int), predictions)
    auprc = average_precision_score(labels.astype(int), predictions)
    return auprc, precisions, recalls, thresholds   

# Dice Score 
def xfrange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1

def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tp)


def normalize(tensor): # THanks DZimmerer
    tens_deta = tensor.detach().cpu()
    tens_deta -= float(np.min(tens_deta.numpy()))
    tens_deta /= float(np.max(tens_deta.numpy()))

    return tens_deta

def log_images(self, diff_volume, data_orig, data_seg, data_mask, final_volume, ID, diff_volume_KL=None,  flow=None ):
    ImagePathList = {
                    'imagesGrid': os.path.join(os.getcwd(),'grid')}
    for key in ImagePathList :
        if not os.path.isdir(ImagePathList[key]):
            os.mkdir(ImagePathList[key])

    for j in range(0,diff_volume.squeeze().shape[2],10) : 

        # create a figure of images with 1 row and 4 columns for subplots
        fig, ax = plt.subplots(1,4,figsize=(16,4))
        # change spacing between subplots
        fig.subplots_adjust(wspace=0.0)
        # orig
        ax[0].imshow(data_orig.squeeze()[...,j].rot90(3),'gray')
        # reconstructed
        ax[1].imshow(final_volume[...,j].rot90(3).squeeze(),'gray')
        # difference
        ax[2].imshow(diff_volume.squeeze()[:,...,j].rot90(3),'inferno',norm=colors.Normalize(vmin=0, vmax=diff_volume.max()+.01))
        # mask
        ax[3].imshow(data_seg.squeeze()[...,j].rot90(3),'gray')
        
        # remove all the ticks (both axes), and tick labels
        for axes in ax:
            axes.set_xticks([])
            axes.set_yticks([])
        # remove the frame of the chart
        for axes in ax:
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
        # remove the white space around the chart
        plt.tight_layout()
        
        if self.cfg.get('save_to_disc',True):
            plt.savefig(os.path.join(ImagePathList['imagesGrid'], '{}_{}_Grid.png'.format(ID[0],j)),bbox_inches='tight')
        self.logger.experiment[0].log({'images/{}/{}_Grid.png'.format(self.dataset[0],j) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()
