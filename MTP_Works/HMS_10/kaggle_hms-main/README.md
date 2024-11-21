# kaggle_hms
https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification
# Install
torch==2.2.2
```
pip install mlflow
pip install --upgrade timm
```
# Split data
Find Nan notebook: https://www.kaggle.com/code/quan0095/find-nan/ \
Split folds notebook: https://www.kaggle.com/code/quan0095/split-kfold-totalvote-hms/ \
Copy *.csv files to data directory.
# Training
```
export PYTHONWARNINGS="ignore"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 code/run_train.py --exp exp_stage1_6
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 code/run_train.py --exp exp_stage2_6
```
# Inference
Inference notebook: https://www.kaggle.com/code/quan0095/hms-best-public-final/
# Solution writeup
https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492619
