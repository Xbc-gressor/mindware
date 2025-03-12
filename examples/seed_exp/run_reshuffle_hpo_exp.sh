#!/bin/bash
MAX_JOBS=2
# 初始化任务列表

# 当前运行的作业数
CURRENT_JOBS=0


declare -a TASKS

# 生成任务列表
for i in 0 1 2 3 5 6; do
    TASKS+=("python reshuffle_holdout.py --time_limit 3600 --Opt hpo --output_file results.txt --reshuffle --late_reshuffle --job_idx $i")
    TASKS+=("python reshuffle_holdout.py --time_limit 3600 --Opt hpo --output_file results.txt --reshuffle --job_idx $i")
    TASKS+=("python reshuffle_holdout.py --time_limit 3600 --Opt hpo --output_file results.txt --job_idx $i")
done

# for TASK in "${TASKS[@]}"; do
#     # 启动后台
#     eval "$TASK "
# done


# 遍历任务列表并执行
for TASK in "${TASKS[@]}"; do
    # 启动后台作业
    eval "$TASK &"

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