@echo off
pip install -r requirements.txt
python train.py --data_root DATASET_PATH --epochs 1 --batch_size 8 --max_train_samples 1000
pause
