#!/bin/bash

# Initialize conda and activate the ducosygan environment
source /opt/conda/etc/profile.d/conda.sh
conda activate ducosygan

read -p "Enter the target model name (soft_tissue/lung): " TARGET_MODEL
TARGET_MODEL=${TARGET_MODEL:-soft_tissue}
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
tmux new-session -d -s ducosygan_train_${TARGET_MODEL}_session "python train.py --target_model ${TARGET_MODEL} > train_$(date +%Y%m%d%H%M%S).log 2>&1"
echo "- Training started in tmux session 'ducosygan_train_${TARGET_MODEL}_session'. Logs are being saved to train_$(date +%Y%m%d%H%M%S).log"
# 유용한 명령어들
echo "- To attach to the session: tmux attach -t ducosygan_train_${TARGET_MODEL}_session"
echo "- To list sessions: tmux ls"
echo "- To kill the session: tmux kill-session -t ducosygan_train_${TARGET_MODEL}_session"
# To monitor training progress, you can use the following command:
echo "- To monitor training progress: tail -f train_$(date +%Y%m%d%H%M%S).log"