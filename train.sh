#!/bin/bash

# Initialize conda and activate the ducosygan environment
source /opt/conda/etc/profile.d/conda.sh
conda activate ducosygan
# python train.py를 실행 (tmux를 활용하여 백그라운드에서 실행, 로그는 train_$(date +%Y%m%d%H%M%S).log에 저장)
# Check if the session already exists
if tmux has-session -t ducosygan_session 2>/dev/null; then
    read -p "Session 'ducosygan_session' is already running. Do you want to kill it and start a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t ducosygan_session
    else
        echo "Aborted."
        exit 0
    fi
fi
tmux new-session -d -s ducosygan_session "python train.py > train_$(date +%Y%m%d%H%M%S).log 2>&1"
echo "Training started in tmux session 'ducosygan_session'. Logs are being saved to train_$(date +%Y%m%d%H%M%S).log"
