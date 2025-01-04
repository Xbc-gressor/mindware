#!/bin/bash

# 最大并发数
MAX_JOBS=2

# 当前运行的作业数
CURRENT_JOBS=0

# 初始化任务列表
declare -a TASKS

# 生成任务列表
for i in 5; do
    TASKS+=("python cls_neural_network.py --time_limit 600 --job_idx $i  --algorithm Autogluon_neural_network")
    TASKS+=("python cls_neural_network.py --time_limit 600 --job_idx $i  --algorithm Autogluon_neural_network --encoder ord")
    TASKS+=("python cls_neural_network.py --time_limit 600 --job_idx $i  --algorithm neural_network")
done

# for i in 7; do
#     TASKS+=("python cls_neural_network.py --time_limit 3600 --job_idx $i --algorithm Autogluon_neural_network")
#     TASKS+=("python cls_neural_network.py --time_limit 3600 --job_idx $i --algorithm Autogluon_neural_network --encoder ord")
#     TASKS+=("python cls_neural_network.py --time_limit 3600 --job_idx $i --algorithm neural_network")
# done

#生成任务列表
for i in 0 1 2; do
    TASKS+=("python cls_neural_network.py --time_limit 600 --job_idx $i  --algorithm Autogluon_neural_network")
    TASKS+=("python cls_neural_network.py --time_limit 600 --job_idx $i  --algorithm neural_network")
done

# for i in 3 4; do
#     TASKS+=("python cls_neural_network.py --time_limit 3600 --job_idx $i --algorithm Autogluon_neural_network")
#     TASKS+=("python cls_neural_network.py --time_limit 3600 --job_idx $i --algorithm neural_network")
# done

# 遍历任务列表并执行
for TASK in "${TASKS[@]}"; do
    # 启动后台
    eval "$TASK "
done


# # 遍历任务列表并执行
# for TASK in "${TASKS[@]}"; do
#     # 启动后台作业
#     eval "$TASK &"

#     # 增加当前作业计数
#     ((CURRENT_JOBS++))

#     # 如果当前作业数达到最大限制，等待任何作业完成
#     if [[ $CURRENT_JOBS -ge $MAX_JOBS ]]; then
#         wait -n  # 等待至少有一个作业完成
#         ((CURRENT_JOBS--))
#     fi
# done

# # 等待所有剩余作业完成
# wait
