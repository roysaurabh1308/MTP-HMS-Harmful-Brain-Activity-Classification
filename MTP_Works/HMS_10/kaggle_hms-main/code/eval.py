import torch
import numpy as np
import time
import torch.nn.functional as F
from sklearn import metrics
import os
import shutil
# from plot_confusion_matrix import cm_analysis
import mlflow
import pandas as pd
from losses.kldiv_loss import KLDivLossWithLogitsForVal


counting = 0
def eval(val_loader, criterion, model, device, epoch, is_ema, default_configs):
    if is_ema:
        print("EMA EVAL")
    else:
        print("NORMAL EVAL")

    k = 0
    model.eval()

    loss_predictions = []
    loss_1_predictions = []
    loss_2_predictions = []
    loss_3_predictions = []
    loss_4_predictions = []

    prob_1_predictions = []
    prob_2_predictions = []
    prob_3_predictions = []
    prob_4_predictions = []
    ground_truths_torch = []
    ground_truths = []
    eeg_id_list = []
    number_votes = []

    loss_predictions_10 = []
    ground_truths_10 = []

    loss_predictions_3 = []
    ground_truths_3 = []

    n_images = 0

    val_metric = {"loss": 0, "loss_10": 0, "loss_3": 0}

    with torch.no_grad():
        for batch_idx, (spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs, labels, eeg_ids, num_votes) in enumerate(val_loader):   
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw_50s_imgs = raw_50s_imgs.to(device).float()
            raw_10s_imgs = raw_10s_imgs.to(device).float()
            logits_kl, logits_1, logits_2, logits_3, logits_4 = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs)
            preds = logits_kl.softmax(dim=1).detach().cpu().numpy()
            preds_1 = logits_1.softmax(dim=1).detach().cpu().numpy()
            preds_2 = logits_2.softmax(dim=1).detach().cpu().numpy()
            preds_3 = logits_3.softmax(dim=1).detach().cpu().numpy()
            preds_4 = logits_4.softmax(dim=1).detach().cpu().numpy()

            logits_kl = logits_kl.detach().cpu()
            logits_1 = logits_1.detach().cpu()
            logits_2 = logits_2.detach().cpu()
            logits_3 = logits_3.detach().cpu()
            logits_4 = logits_4.detach().cpu()
            criterion(logits_kl, labels)
            

            k = k + spec_imgs.size(0)

            n_images += len(labels)
            for j in range(len(labels)):
                ground_truths.append(labels[j].detach().cpu().numpy())
                ground_truths_torch.append(labels[j].detach().cpu())
                loss_predictions.append(preds[j])
                loss_1_predictions.append(logits_1[j])
                loss_2_predictions.append(logits_2[j])
                loss_3_predictions.append(logits_3[j])
                loss_4_predictions.append(logits_4[j])

                prob_1_predictions.append(preds_1[j])
                prob_2_predictions.append(preds_2[j])
                prob_3_predictions.append(preds_3[j])
                prob_4_predictions.append(preds_4[j])
                eeg_id_list.append(eeg_ids[j])
                number_votes.append(num_votes[j])
                if num_votes[j] >= 3:
                    loss_predictions_3.append(logits_kl[j])
                    ground_truths_3.append(labels[j].detach().cpu())
                if num_votes[j] >= 10:
                    loss_predictions_10.append(logits_kl[j])
                    ground_truths_10.append(labels[j].detach().cpu())
 
        val_loss = criterion.compute()
        criterion_10 = KLDivLossWithLogitsForVal()
        criterion_3 = KLDivLossWithLogitsForVal()
        criterion_spec = KLDivLossWithLogitsForVal()
        criterion_eeg = KLDivLossWithLogitsForVal()
        criterion_50 = KLDivLossWithLogitsForVal()
        criterion_10 = KLDivLossWithLogitsForVal()
        for i in range(len(ground_truths_torch)):
            criterion_spec(loss_1_predictions[i].unsqueeze(0), ground_truths_torch[i].unsqueeze(0))
            criterion_eeg(loss_2_predictions[i].unsqueeze(0), ground_truths_torch[i].unsqueeze(0))
            criterion_50(loss_3_predictions[i].unsqueeze(0), ground_truths_torch[i].unsqueeze(0))
            criterion_10(loss_4_predictions[i].unsqueeze(0), ground_truths_torch[i].unsqueeze(0))

        val_loss_spec = criterion_spec.compute()
        val_loss_eeg = criterion_eeg.compute()
        val_loss_raw50 = criterion_50.compute()
        val_loss_raw10 = criterion_10.compute()
        
        for i in range(len(ground_truths_10)):
            criterion_10(loss_predictions_10[i].unsqueeze(0), ground_truths_10[i].unsqueeze(0))
        if len(ground_truths_10) > 0:
            val_loss_10 = criterion_10.compute()
        else:
            val_loss_10 = 0
        for i in range(len(ground_truths_3)):
            criterion_3(loss_predictions_3[i].unsqueeze(0), ground_truths_3[i].unsqueeze(0))
        
        if len(ground_truths_3) > 0:
            val_loss_3 = criterion_3.compute()
        else:
            val_loss_3 = 0

        print("val loss: ", val_loss)
        print("val loss 3: ", val_loss_3)
        print("val loss 10: ", val_loss_10)
        print("val loss spec: ", val_loss_spec)
        print("val loss eeg: ", val_loss_eeg)
        print("val loss raw 50: ", val_loss_raw50)
        print("val loss raw 10: ", val_loss_raw10)

    
    val_metric['loss'] = val_loss
    val_metric['loss_3'] = val_loss_3
    val_metric['loss_10'] = val_loss_10
    total_loss = {'all': loss_predictions, 'spec': prob_1_predictions, 'eeg': prob_2_predictions, '50': prob_3_predictions, '10': prob_4_predictions}
         
    return val_metric, ground_truths, total_loss, eeg_id_list, number_votes
