import os
from utils import seed_torch
import mlflow
from train import train_one_fold
from kl_div.kl_div import score
import numpy as np

import json
import pandas as pd
import torch
from datasets.dataset import ImageFolder
from torch.utils.data import DataLoader
import torch
import argparse

default_configs = {}


import torch.distributed as dist
import random

def seed_worker(worker_id):
    worker_seed = 2021
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--exp", type=str, default="exp_0")
    args = parser.parse_args()
    dist.init_process_group("nccl")
    RESULT_PATH = "../results"
    os.makedirs(RESULT_PATH, exist_ok=True)
    MLRUNS_PATH = os.path.join(RESULT_PATH, "mlruns")
    os.makedirs(RESULT_PATH, exist_ok=True)
    OOF_PATH = os.path.join(RESULT_PATH, "oof")
    os.makedirs(OOF_PATH, exist_ok=True)
    OOF_PATH = os.path.join(OOF_PATH, args.exp)
    os.makedirs(OOF_PATH, exist_ok=True)
    # os.makedirs(MLRUNS_PATH, exist_ok=True)
    # RESULT_PATH = os.path.join(RESULT_PATH, args.exp)

    DATA_TYPE = args.exp.split("_")[1]
 
    # experiment_id = mlflow.create_experiment(args.exp)
 
    rank = dist.get_rank()
    num_tasks = dist.get_world_size()
    print(f"Start running basic DDP example on rank {rank}.")

    mlflow.set_tracking_uri(MLRUNS_PATH)
    existing_exp = mlflow.get_experiment_by_name(args.exp)
    
    if rank == 0:
        if not existing_exp:
            mlflow.create_experiment(args.exp, artifact_location=MLRUNS_PATH)
 
    dist.barrier()
    experiment = mlflow.set_experiment(args.exp)
    experiment_id = experiment.experiment_id
 
    f = open(os.path.join('configs', "{}.json".format(args.exp)))
    default_configs = json.load(f)
    f.close()
    if rank == 0:
        print(default_configs)
 
    seed_torch()
    
    avg_score = {"loss": 0}
    n_fold = 5
 
    train_loader_list = {}
    test_loader_list = {}
    g = torch.Generator()
    g.manual_seed(0)
    for fold in range(n_fold):
        train_df = pd.read_csv("data/train_fold{}_{}.csv".format(fold, DATA_TYPE))
        val_df =  pd.read_csv("data/val_fold{}_{}.csv".format(fold, DATA_TYPE))
        train_data = ImageFolder(train_df, default_configs, None, "train")
        test_data = ImageFolder(val_df, default_configs, None, "test")
        sampler_train = torch.utils.data.DistributedSampler(
            train_data, num_replicas=num_tasks, rank=rank, shuffle=True, seed=2023,
        )
        
        print("Sampler_train = %s" % str(sampler_train))
        train_loader = DataLoader(train_data, sampler=sampler_train, batch_size=default_configs["batch_size"], pin_memory=False, 
            num_workers=default_configs["num_workers"], drop_last=True, worker_init_fn=seed_worker, generator=g)
 
        test_loader = DataLoader(test_data, batch_size=int(default_configs["batch_size"]*2), 
                pin_memory=False, num_workers=default_configs["num_workers"], drop_last=False)
             
        train_loader_list[fold] = train_loader
        test_loader_list[fold] = test_loader
 
    if rank == 0:
        with mlflow.start_run(
            experiment_id=experiment_id,
        ) as parent_run:
            mlflow.set_tag("mlflow.runName", "hms_{}".format(DATA_TYPE))
            mlflow.log_params(default_configs)
            # mlflow.log_artifacts("code") 
            avg_ground_truths = [] 
            avg_predictions = []
            avg_predictions_spec = []
            avg_predictions_eeg = []
            avg_predictions_raw50 = []
            avg_predictions_raw10 = []
            avg_number_votes = []
            avg_eeg_ids = []
            for fold in [0, 1, 2, 3, 4]:
                with mlflow.start_run(experiment_id=experiment_id,
                    description="fold_{}".format(fold),
                    tags={
                        mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: parent_run.info.run_id
                    }, nested=True):
                    mlflow.set_tag("mlflow.runName", "fold_{}".format(fold))
                    val_score = train_one_fold(RESULT_PATH, default_configs, fold, train_loader_list[fold], test_loader_list[fold], rank, default_configs["pretrained"]) 
                    for k, v in avg_score.items():
                        avg_score[k] += val_score[k]["score"]
                        mlflow.log_metric("{}".format(k), val_score[k]["score"])
                        # mlflow.log_metric("{}_onnx".format(k), onnx_metric[k])
                        print("{}: ".format(k), val_score[k]["score"])
                    ground_truths, predictions, eeg_ids, number_votes = val_score["loss"]["list"]
                    avg_ground_truths = avg_ground_truths + ground_truths
                    avg_predictions = avg_predictions + predictions['all']
                    avg_predictions_spec = avg_predictions_spec + predictions['spec']
                    avg_predictions_eeg = avg_predictions_eeg + predictions['eeg']
                    avg_predictions_raw50 = avg_predictions_raw50 + predictions['50']
                    avg_predictions_raw10 = avg_predictions_raw10 + predictions['10']
                    avg_eeg_ids = avg_eeg_ids + eeg_ids
                    avg_number_votes = avg_number_votes + number_votes

            CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
            avg_predictions_10 = []
            avg_ground_truths_10 = []
            avg_eeg_ids_10 = []

            avg_predictions_3 = []
            avg_ground_truths_3 = []
            avg_eeg_ids_3 = []

            for (eeg_id, gt, pred, n_vote) in zip(avg_eeg_ids, avg_ground_truths, avg_predictions, avg_number_votes):
                if n_vote >= 3:
                    avg_predictions_3.append(pred)
                    avg_ground_truths_3.append(gt)
                    avg_eeg_ids_3.append(eeg_id)
                if n_vote >= 10:
                    avg_predictions_10.append(pred)
                    avg_ground_truths_10.append(gt)
                    avg_eeg_ids_10.append(eeg_id)
            true = pd.DataFrame(avg_ground_truths, columns=CLASSES)
            true.insert(0, "eeg_id", avg_eeg_ids)
            print(true.head())
            true.to_csv(os.path.join(OOF_PATH, 'hms_{}_gt.csv'.format(args.exp)), index=False)

            oof = pd.DataFrame(avg_predictions, columns=CLASSES)
            oof.insert(0, "eeg_id", avg_eeg_ids)
            oof.to_csv(os.path.join(OOF_PATH, 'hms_{}_oof.csv'.format(args.exp)), index=False)

            
            print(oof.head())

            cv_score = score(solution=true, submission=oof, row_id_column_name='eeg_id')
            print(true.head())

            print("Average loss: ", avg_score["loss"]/n_fold)
            print("OOF loss: ", cv_score)
            mlflow.log_metric("CV_kl", cv_score)

            true_3 = pd.DataFrame(avg_ground_truths_3, columns=CLASSES)
            true_3.insert(0, "eeg_id", avg_eeg_ids_3)
            true_3.to_csv(os.path.join(OOF_PATH, 'hms_{}_gt_3.csv'.format(args.exp)), index=False)
            oof_3 = pd.DataFrame(avg_predictions_3, columns=CLASSES)
            oof_3.to_csv(os.path.join(OOF_PATH, 'hms_{}_oof_3.csv'.format(args.exp)), index=False)
            oof_3.insert(0, "eeg_id", avg_eeg_ids_3)
            cv_score_3 = score(solution=true_3, submission=oof_3, row_id_column_name='eeg_id')
            print("OOF loss 3: ", cv_score_3)
            mlflow.log_metric("CV_kl_3", cv_score_3)  

            true_10 = pd.DataFrame(avg_ground_truths_10, columns=CLASSES)
            true_10.insert(0, "eeg_id", avg_eeg_ids_10)
            true_10.to_csv(os.path.join(OOF_PATH, 'hms_{}_gt_10.csv'.format(args.exp)), index=False)
            oof_10 = pd.DataFrame(avg_predictions_10, columns=CLASSES)
            oof_10.to_csv(os.path.join(OOF_PATH, 'hms_{}_oof_10.csv'.format(args.exp)), index=False)
            oof_10.insert(0, "eeg_id", avg_eeg_ids_10)
            cv_score_10 = score(solution=true_10, submission=oof_10, row_id_column_name='eeg_id')
            print("OOF loss 10: ", cv_score_10)
            mlflow.log_metric("CV_kl_10", cv_score_10)     

            true = pd.DataFrame(avg_ground_truths, columns=CLASSES)
            true.insert(0, "eeg_id", avg_eeg_ids)
            oof_spec = pd.DataFrame(avg_predictions_spec, columns=CLASSES)
            oof_spec.insert(0, "eeg_id", avg_eeg_ids)
            oof_spec.to_csv(os.path.join(OOF_PATH, 'hms_{}_oof_spec.csv'.format(args.exp)), index=False)
            cv_score_spec = score(solution=true, submission=oof_spec, row_id_column_name='eeg_id')
            print("OOF loss spec: ", cv_score_spec)
            mlflow.log_metric("CV_kl_spec", cv_score_spec) 

            true = pd.DataFrame(avg_ground_truths, columns=CLASSES)
            true.insert(0, "eeg_id", avg_eeg_ids)
            oof_eeg = pd.DataFrame(avg_predictions_eeg, columns=CLASSES)
            oof_eeg.insert(0, "eeg_id", avg_eeg_ids)
            oof_eeg.to_csv(os.path.join(OOF_PATH, 'hms_{}_oof_eeg.csv'.format(args.exp)), index=False)
            cv_score_eeg = score(solution=true, submission=oof_eeg, row_id_column_name='eeg_id')
            print("OOF loss eeg: ", cv_score_eeg)
            mlflow.log_metric("CV_kl_eeg", cv_score_eeg) 

            true = pd.DataFrame(avg_ground_truths, columns=CLASSES)
            true.insert(0, "eeg_id", avg_eeg_ids)
            oof_50 = pd.DataFrame(avg_predictions_raw50, columns=CLASSES)
            oof_50.insert(0, "eeg_id", avg_eeg_ids)
            oof_50.to_csv(os.path.join(OOF_PATH, 'hms_{}_oof_50.csv'.format(args.exp)), index=False)
            cv_score_50 = score(solution=true, submission=oof_50, row_id_column_name='eeg_id')
            print("OOF loss raw 50: ", cv_score_50)
            mlflow.log_metric("CV_kl_raw_50", cv_score_50) 

            true = pd.DataFrame(avg_ground_truths, columns=CLASSES)
            true.insert(0, "eeg_id", avg_eeg_ids)
            oof_10 = pd.DataFrame(avg_predictions_raw10, columns=CLASSES)
            oof_10.insert(0, "eeg_id", avg_eeg_ids)
            oof_10.to_csv(os.path.join(OOF_PATH, 'hms_{}_oof_10.csv'.format(args.exp)), index=False)
            cv_score_10 = score(solution=true, submission=oof_10, row_id_column_name='eeg_id')
            print("OOF loss raw 10: ", cv_score_10)
            mlflow.log_metric("CV_kl_raw_10", cv_score_10) 
            mlflow.end_run()
    else:
        for fold in [0, 1, 2, 3, 4]:
            val_score = train_one_fold(RESULT_PATH, default_configs, fold, train_loader_list[fold], test_loader_list[fold], rank, default_configs["pretrained"]) 

