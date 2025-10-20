#!/usr/bin/env bash

for i in {1..30}; do
    python get_all_pdf.py
    tmux capture-pane -J -pS - > /Users/xujingjun/SUSTech/Research/LLM_copy/Graph_LLM/Code/formal-code/tmux_full_log.txt
done