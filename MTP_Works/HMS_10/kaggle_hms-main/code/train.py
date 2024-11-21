import os

import mlflow
from losses.kldiv_loss import KLDivLossWithLogits, KLDivLossWithLogitsForVal
from kl_div.kl_div import score
from torch.optim import lr_scheduler
import shutil


from timm.data.random_erasing import RandomErasing
from data_augmentations.mixup import mixup, cutmix
from optimizer.sam import SAM
from optimizer.adan import Adan
from optimizer.ranger21.ranger21 import Ranger21
from optimizer.lion_pytorch.lion_pytorch import Lion


import numpy as np
import time
import torch

from timm.utils import ModelEma
from models.model import VitNet
from tqdm import tqdm

import torch

from eval import eval
import gc

from timm.utils import get_state_dict


from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.distributed as dist

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def remove_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def make_weight_folder(default_configs, RESULT_PATH, fold):
    weight_path = os.path.join(RESULT_PATH, "weights")
    os.makedirs(weight_path, exist_ok=True)
    weight_path = os.path.join(weight_path, default_configs["a_name"])
    os.makedirs(weight_path, exist_ok=True)
    weight_path = os.path.join(weight_path, str(fold))
    os.makedirs(weight_path, exist_ok=True)
    return weight_path

def load_old_weight(model, weight_path):
    if weight_path is not None:
        pretrained_dict = torch.load(weight_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model.load_state_dict(pretrained_dict, strict=True)
    return model


def build_criterion(default_configs, device_id):
    criterion_train_kl, criterion_val = KLDivLossWithLogits().to(device_id), KLDivLossWithLogitsForVal()
    criterion_train_1 = KLDivLossWithLogits().to(device_id)
    criterion_train_2 = KLDivLossWithLogits().to(device_id)
    criterion_train_3 = KLDivLossWithLogits().to(device_id)
    criterion_train_4 = KLDivLossWithLogits().to(device_id)

    return criterion_train_kl, criterion_train_1, criterion_train_2, criterion_train_3, criterion_train_4, criterion_val

def build_net(default_configs, device_id):
    model = VitNet(default_configs, device_id).to(device_id)
    
    return model

def build_optimizer(default_configs, model_without_ddp, device_id):
    num_tasks = dist.get_world_size()
    # lr = default_configs["lr"]*num_tasks
    lr = default_configs["lr"]
    if default_configs["optimizer"] == "SAM":
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer_model = SAM(model_without_ddp.parameters(), base_optimizer, lr=lr, momentum=0.9, weight_decay=default_configs["weight_decay"], adaptive=True)
    # optimizer_model = torch.optim.AdamW(model.parameters(), lr=default_configs["lr"], weight_decay=default_configs["weight_decay"])
    elif default_configs["optimizer"] == "Ranger21":
        optimizer_model = Ranger21(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"], 
        num_epochs=default_configs["num_epoch"], num_batches_per_epoch=len(train_loader))
    elif default_configs["optimizer"] == "SGD":
        optimizer_model = torch.optim.SGD(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"], momentum=0.9)
    elif default_configs["optimizer"] == "Lion":
        optimizer_model = Lion(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"])
    elif default_configs["optimizer"] == "Adan":
        optimizer_model = Adan(model_without_ddp.parameters(), lr=lr, weight_decay=default_configs["weight_decay"])
    
    return optimizer_model

def log_metric_mlflow(rank, metric_name, metric_value, step):
    if rank == 0:
        mlflow.log_metric(metric_name, metric_value, step=step)

def train_one_fold(RESULT_PATH, default_configs, fold, train_loader, test_loader, rank, old_weight_path):
    # random_erase = RandomErasing(mode='const', probability=0.5)
    if rank == 0:
        print("FOLD: ", fold)
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    
    DATA_PATH = "train"
    start_epoch = 0

    scaler = torch.cuda.amp.GradScaler()
    weight_path = make_weight_folder(default_configs, RESULT_PATH, fold)
    criterion_train_kl, criterion_train_1, criterion_train_2, criterion_train_3, criterion_train_4, criterion_val = build_criterion(default_configs, device_id)
    model = build_net(default_configs, device_id)
    if old_weight_path is not None and len(old_weight_path) > 0:
        print("Load weights: ", old_weight_path)
        model.load_state_dict(torch.load("{}/{}/checkpoint_loss_best_ema.pt".format(old_weight_path, fold), map_location=torch.device('cpu')))
    
    if rank == 0:
        model_ema = ModelEma(
            model,
            decay=default_configs["model_ema_decay"],
            device=device_id, resume='')
    model_without_ddp = model    
    model = torch.compile(model)
    # model.model.set_grad_checkpointing()
    ddp_model = NativeDDP(model, device_ids=[device_id])
    
    optimizer_model = build_optimizer(default_configs, ddp_model, device_id)

    scheduler = lr_scheduler.OneCycleLR(optimizer_model, default_configs["lr"], steps_per_epoch=len(train_loader), epochs=default_configs["num_epoch"])
    
    best_metric = {"loss": 1000}
    best_metric_ema = {"loss": {"score": 1000, "list": []}}
    best_model_path = ""


    for epoch in range(start_epoch, default_configs["num_epoch"]):
        if epoch == default_configs["epoch_end"]:
            break
        if rank == 0:
            print("\n-----------------Epoch: " + str(epoch) + " -----------------")
        train_loader.sampler.set_epoch(epoch)
        # grid.set_prob(epoch, default_configs["num_epoch"])
        # scheduler_lr(optimizer_model, epoch)
        for param_group in optimizer_model.param_groups:
            log_metric_mlflow(rank, "lr", param_group['lr'], step=epoch)
            
        start = time.time()
        optimizer_model.zero_grad()

        batch_idx = 0
        n_iters = len(train_loader)
        for spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs, labels, eeg_ids in tqdm(train_loader):
            spec_imgs = spec_imgs.to(device_id).float()
            eeg_imgs = eeg_imgs.to(device_id).float()
            raw_50s_imgs = raw_50s_imgs.to(device_id).float()
            raw_10s_imgs = raw_10s_imgs.to(device_id).float()
            labels = labels.to(device_id).float()
            
            if torch.rand(1)[0] < 0.5 and default_configs["use_mixup"]:
                mix_spec_images, mix_eeg_images, mix_raw_50s_images, mix_raw_10s_images, target_a, target_b, lam = mixup(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs, labels, alpha=default_configs["mixup_alpha"])
                
                with torch.cuda.amp.autocast():
                    logits_kl, logits_1, logits_2, logits_3, logits_4 = ddp_model(mix_spec_images, mix_eeg_images, mix_raw_50s_images, mix_raw_10s_images)
                    loss_kl = criterion_train_kl(logits_kl, target_a) * lam + \
                            (1 - lam) * criterion_train_kl(logits_kl, target_b) 
                    loss_1 = criterion_train_1(logits_1, target_a) * lam + \
                            (1 - lam) * criterion_train_kl(logits_1, target_b) 
                    loss_2 = criterion_train_2(logits_2, target_a) * lam + \
                            (1 - lam) * criterion_train_kl(logits_2, target_b) 
                    loss_3 = criterion_train_3(logits_3, target_a) * lam + \
                            (1 - lam) * criterion_train_kl(logits_3, target_b) 
                    loss_4 = criterion_train_4(logits_4, target_a) * lam + \
                            (1 - lam) * criterion_train_kl(logits_4, target_b) 
                    loss = (loss_kl + loss_1 + loss_2 + loss_3 + loss_4)/4
                    loss /= default_configs["accumulation_steps"]
                scaler.scale(loss).backward()
                
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    # scaler.unscale_(optimizer_model)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), default_configs["max_norm"])
                    scaler.step(optimizer_model)
                    scaler.update()
                    if rank == 0:
                        model_ema.update(model_without_ddp)
                    optimizer_model.zero_grad()
            else:
                with torch.cuda.amp.autocast():
                    # imgs = random_erase(imgs)
                    logits_kl, logits_1, logits_2, logits_3, logits_4 = ddp_model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs )
                    loss_kl =  criterion_train_kl(logits_kl, labels) 
                    loss_1 =  criterion_train_1(logits_1, labels) 
                    loss_2 =  criterion_train_2(logits_2, labels) 
                    loss_3 =  criterion_train_3(logits_3, labels) 
                    loss_4 =  criterion_train_4(logits_4, labels) 
                    loss = (loss_kl + loss_1 + loss_2 + loss_3 + loss_4)/4
                    loss /= default_configs["accumulation_steps"]
                scaler.scale(loss).backward()

                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    # scaler.unscale_(optimizer_model)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), default_configs["max_norm"])
                    scaler.step(optimizer_model)
                    scaler.update()
                    if rank == 0:
                        model_ema.update(model_without_ddp)
                    optimizer_model.zero_grad()
                
            scheduler.step()
            batch_idx += 1
            
            dist.barrier()

            if rank == 0 and (batch_idx == n_iters//2 or batch_idx == n_iters) and epoch >= 2:
                val_metric_type_list = ["loss"]
                start = time.time()
                val_metric, ground_truths, predictions, eeg_ids, number_votes = eval(test_loader, criterion_val, model_ema.ema, device_id, epoch, True, default_configs)
                end = time.time()
                print("val elapsed time", end - start)
                for val_metric_type in val_metric_type_list:
                    print("Val ema {}: {}".format(val_metric_type, val_metric[val_metric_type]))
                    mlflow.log_metric("val_{}_ema".format(val_metric_type), val_metric[val_metric_type], step=epoch)
                    flag = False
                    if val_metric_type in ["loss"]:
                        if(val_metric[val_metric_type] < best_metric_ema[val_metric_type]["score"]):
                            flag = True
                    else:
                        if(val_metric[val_metric_type] > best_metric_ema[val_metric_type]["score"]):
                            flag = True 
                    # model_path = os.path.join(weight_path, 'checkpoint_{}.pt'.format(epoch))
                    # exported_model = get_state_dict(model_ema)
                    # torch.save(exported_model, model_path)
                    if flag == True:
                        best_model_path = os.path.join(weight_path, 'checkpoint_{}_best_ema.pt'.format(val_metric_type))
                        try:
                            os.remove(best_model_path)
                        except Exception as e:
                            print(e)
                        # exported_model = torch._dynamo.export(get_state_dict(model_ema), input)
                        exported_model = get_state_dict(model_ema)
                        torch.save(exported_model, best_model_path)
                        # mlflow.log_artifact(best_model_path)
                        best_metric_ema[val_metric_type] = {"score": val_metric[val_metric_type], "list": [ground_truths, predictions, eeg_ids, number_votes]}
                        print("Save best model ema: ", best_model_path, val_metric[val_metric_type])
            dist.barrier()

    del model
    del ddp_model
    torch.cuda.empty_cache()
    gc.collect()

    return best_metric_ema
