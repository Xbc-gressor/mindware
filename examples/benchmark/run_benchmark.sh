#!/bin/bash

# 最大并发数
MAX_JOBS=3

# 当前运行的作业数
CURRENT_JOBS=0

for i in {0..5}; do
    # 启动后台作业
    python cls_benchmark.py --Opt CASHFE --job_idx $i &

    # 增加当前作业计数
    ((CURRENT_JOBS++))

    # 如果当前作业数达到最大限制，等待任何作业完成
    if [[ $CURRENT_JOBS -ge $MAX_JOBS ]]; then
        wait -n  # 等待至少有一个作业完成
        ((CURRENT_JOBS--))
    fi
done

# 等待所有剩余作业完成
wait