import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from sklearn.model_selection import GroupKFold

RANDOM_SEED = 1086

CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)

train = pd.read_csv("/home/m1/23CS60R76/MTP_Works/HMS_Brain_Activity/Harmful_Brain_Activity/train.csv")

train['total_evaluators'] = train[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)
train[CLASSES] /= train[CLASSES].sum(axis=1).values[:, None]
# train = train.groupby(["eeg_id", "seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"])


nan_list = set([228018333, 2609800009, 1033402741, 3360776627, 1936370864, 3447241762, 1799086501, 2294110417, 1774977261, 837428467,
           1511903313, 3254156300, 3030710864, 2393305422, 191408832, 1944688715, 1092242633, 2565199369, 1750767188, 352100210,
           227051891, 3980685149, 3197027554, 689412237, 84103002, 788653799, 952997900, 1410292431, 1339041688, 725185416,
           1267818343, 1907765019, 1293001777, 523936200, 3326234321, 1309137409, 2190373347, 2538961182, 2406368236,
           3418300291, 3780781224, 3707844385, 975631111, 1459425125, 3932380488, 420827282, 23656323, 1301620928,
           1916955481, 3042198969, 2525259601, 2763615057, 2641635192, 2276240743, 2728561398, 4018623581, 21746311,
           524927151, 2185189413, 116770645, 3931449367, 4045810693, 1828657109, 1889311261, 3386127802, 2800397565,
           1379952459, 3948834462, 1747732327, 1366044762, 4115113596, 1593385762, 1119914885, 402182162, 3595195521,
           2572900449, 2906001188, 1782988316, 2690795118, 579740230, 2272873515, 1926819138, 1534551817, 82511342, 2081405553,
           3168916926, 2551548463, 3939775387])
train = train[~train['eeg_id'].isin(nan_list)].reset_index(drop=False)
print(train.shape)

sgkf = GroupKFold(n_splits=N_FOLDS)

train["fold"] = -1

for fold_id, (_, val_idx) in enumerate(
    sgkf.split(train, y=train["expert_consensus"], groups=train["patient_id"])
):
    train.loc[val_idx, "fold"] = fold_id

for fold in FOLDS:
    train_idx = train[train["fold"] != fold].index.values
    val_idx   = train[train["fold"] == fold].index.values
    eeg_ids = []
    raw_labels = train[CLASSES].values
    labels = []
    spec_ids = []
    eeg_label_offset_seconds = []
    spectrogram_label_offset_seconds = []
    total_votes = []
    for raw_label in raw_labels:
        label = ' '.join(map(str, raw_label))
        labels.append(label)
    
    for eeg_id in train["eeg_id"].values:
        eeg_ids.append(eeg_id)
        
    for spec_id in train["spectrogram_id"].values:
        spec_ids.append(spec_id)
        
    for spectrogram_label_offset_second in train["spectrogram_label_offset_seconds"].values:
        spectrogram_label_offset_seconds.append(spectrogram_label_offset_second)
        
    for eeg_label_offset_second in train["eeg_label_offset_seconds"].values:
        eeg_label_offset_seconds.append(eeg_label_offset_second)
        
    for total_vote in train["total_evaluators"].values:
        total_votes.append(int(total_vote))

    train_eeg_ids = [eeg_ids[idx] for idx in train_idx]
    train_spec_ids = [spec_ids[idx] for idx in train_idx]
    train_labels = [labels[idx] for idx in train_idx]
    train_spectrogram_label_offset_seconds = [spectrogram_label_offset_seconds[idx] for idx in train_idx]
    train_eeg_label_offset_seconds = [eeg_label_offset_seconds[idx] for idx in train_idx]
    train_total_votes = [total_votes[idx] for idx in train_idx]

    val_eeg_ids = [eeg_ids[idx] for idx in val_idx]
    val_spec_ids = [spec_ids[idx] for idx in val_idx]
    val_labels = [labels[idx] for idx in val_idx]
    val_spectrogram_label_offset_seconds = [spectrogram_label_offset_seconds[idx] for idx in val_idx]
    val_eeg_label_offset_seconds = [eeg_label_offset_seconds[idx] for idx in val_idx]
    val_total_votes = [total_votes[idx] for idx in val_idx]
    
    train_eeg_ids = np.array(train_eeg_ids)
    train_spec_ids = np.array(train_spec_ids)
    train_labels = np.array(train_labels)
    train_spectrogram_label_offset_seconds = np.array(train_spectrogram_label_offset_seconds)
    train_eeg_label_offset_seconds = np.array(train_eeg_label_offset_seconds)
    train_total_votes = np.array(train_total_votes)
    val_eeg_ids = np.array(val_eeg_ids)
    val_spec_ids = np.array(val_spec_ids)
    val_labels = np.array(val_labels)
    val_spectrogram_label_offset_seconds = np.array(val_spectrogram_label_offset_seconds)
    val_eeg_label_offset_seconds = np.array(val_eeg_label_offset_seconds)
    val_total_votes = np.array(val_total_votes)
    train_eeg_ids = np.expand_dims(train_eeg_ids, 1)
    train_spec_ids = np.expand_dims(train_spec_ids, 1)
    train_labels = np.expand_dims(train_labels, 1)
    train_spectrogram_label_offset_seconds = np.expand_dims(train_spectrogram_label_offset_seconds, 1)
    train_eeg_label_offset_seconds = np.expand_dims(train_eeg_label_offset_seconds, 1)
    train_total_votes = np.expand_dims(train_total_votes, 1)
    val_eeg_ids = np.expand_dims(val_eeg_ids, 1)
    val_spec_ids = np.expand_dims(val_spec_ids, 1)
    val_labels = np.expand_dims(val_labels, 1)
    val_spectrogram_label_offset_seconds = np.expand_dims(val_spectrogram_label_offset_seconds, 1)
    val_eeg_label_offset_seconds = np.expand_dims(val_eeg_label_offset_seconds, 1)
    val_total_votes = np.expand_dims(val_total_votes, 1)
    
    train_df = pd.DataFrame(np.concatenate((train_eeg_ids, train_spec_ids, train_labels, train_eeg_label_offset_seconds, train_spectrogram_label_offset_seconds, train_total_votes), axis=1), columns=["eeg_id", "spec_id", "label", "eeg_label_offset_seconds", "spectrogram_label_offset_seconds", "total_votes"])
    train_df.to_csv('train_fold{}.csv'.format(fold), index=False)
    
    val_df = pd.DataFrame(np.concatenate((val_eeg_ids, val_spec_ids, val_labels, val_eeg_label_offset_seconds, val_spectrogram_label_offset_seconds, val_total_votes), axis=1), columns=["eeg_id", "spec_id", "label", "eeg_label_offset_seconds", "spectrogram_label_offset_seconds", "total_votes"])
    val_df = val_df.groupby(["eeg_id"]).head(1)
    val_df.to_csv('val_fold{}.csv'.format(fold), index=False)
    train_df_stage1 = train_df.loc[train_df['total_votes'].astype(np.int64) < 10]
    train_df_stage1.to_csv('train_fold{}_stage1.csv'.format(fold), index=False)
    val_df_stage1 = val_df.loc[val_df['total_votes'].astype(np.int64) < 10]
    val_df_stage1.to_csv('val_fold{}_stage1.csv'.format(fold), index=False)
    train_df_stage2 = train_df.loc[train_df['total_votes'].astype(np.int64) >= 10]
    train_df_stage2.to_csv('train_fold{}_stage2.csv'.format(fold), index=False)
    val_df_stage2 = val_df.loc[val_df['total_votes'].astype(np.int64) >= 10]
    val_df_stage2.to_csv('val_fold{}_stage2.csv'.format(fold), index=False)