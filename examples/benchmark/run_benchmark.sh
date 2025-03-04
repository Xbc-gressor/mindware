#!/bin/bash

# 最大并发数
MAX_JOBS=2

# 当前运行的作业数
CURRENT_JOBS=0

# 初始化任务列表
declare -a TASKS

# # 生成任务列表
# for i in 4 7; do
#     TASKS+=("python cls_benchmark.py --Opt cashfe --time_limit 7200 --job_idx $i")
#     TASKS+=("python cls_benchmark.py --Opt cash --time_limit 3600 --job_idx $i")
# done

# # 生成任务列表
# for i in 0 1 2 3 5 6; do
#     TASKS+=("python cls_benchmark.py --Opt cashfe --time_limit 3600 --job_idx $i")
#     TASKS+=("python cls_benchmark.py --Opt cash --time_limit 1800 --job_idx $i")
# done

# # 生成任务列表
# for i in 5; do
#     TASKS+=("python rgs_benchmark.py --Opt cashfe --time_limit 7200 --job_idx $i")
#     TASKS+=("python rgs_benchmark.py --Opt cash --time_limit 3600 --job_idx $i")
# done

# # 生成任务列表
# for i in 0 1 2 3 4; do
#     TASKS+=("python rgs_benchmark.py --Opt cashfe --time_limit 3600 --job_idx $i")
#     TASKS+=("python rgs_benchmark.py --Opt cash --time_limit 1800 --job_idx $i")
# done





# # 生成任务列表
# for i in 5; do
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_2 --n_algorithm 6 --n_preprocessor 6 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_2 --n_algorithm 6 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_1 --n_algorithm 6 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_2 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_1 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     # TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_0 --time_limit 14400 --job_idx $i")
# done

# # 生成任务列表
# for i in 0 1 2 3 4; do
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_2 --n_algorithm 6 --n_preprocessor 6 --time_limit 3600 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --time_limit 3600 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_2 --n_algorithm 6 --time_limit 3600 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_1 --n_algorithm 6 --time_limit 3600 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_2 --time_limit 3600 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_1 --time_limit 3600 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
#     # TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_0 --time_limit 3600 --job_idx $i")
# done

# 生成任务列表
for i in 5; do
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_2 --n_algorithm 6 --n_preprocessor 6 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_2 --n_algorithm 6 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_1 --n_algorithm 6 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_2 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_1 --time_limit 14400 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    # TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_0 --time_limit 7200 --job_idx $i")
done

# 生成任务列表
for i in 0 1 2 3 4; do
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_2 --n_algorithm 6 --n_preprocessor 6 --time_limit 7200 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --time_limit 7200 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_2 --n_algorithm 6 --time_limit 7200 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_1 --n_algorithm 6 --time_limit 7200 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_2 --time_limit 7200 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_1 --time_limit 7200 --job_idx $i --output_dir ./compress_data --output_file compress_data.txt")
    # TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_0 --time_limit 3600 --job_idx $i")
done


# 生成任务列表，测试为什么block1变差了
# for i in 7; do
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_1 --time_limit 7200 --job_idx $i --output_dir ./newfe_icafix_data --output_file ./newfe_icafix_data.txt")
# done

# # # 生成任务列表
# for i in 1; do
#     TASKS+=("python cls_benchmark.py --Opt cashfe --optimizer block_1 --time_limit 3600 --job_idx $i --output_dir ./newfe_icafix_data --output_file ./newfe_icafix_data.txt")
# done


# # 生成任务列表
# for i in 1 2 3; do
#     TASKS+=("python rgs_benchmark.py --Opt cashfe --optimizer block_1 --time_limit 3600 --job_idx $i --output_dir ./newfe_icafix_data --output_file ./newfe_icafix_data.txt")
# done


# 最大并发数
MAX_JOBS=4
# 当前运行的作业数
CURRENT_JOBS=0
# 每个任务使用的核心数量
CORES_PER_TASK=24
# 总核心数量
TOTAL_CORES=96


# 初始化互斥锁文件
LOCK_FILE="/tmp/free_cores_lock"
FREE_CORES_FILE="/tmp/free_cores_list"
echo $(seq 0 $((TOTAL_CORES - 1))) > "$FREE_CORES_FILE"

# 分配核心函数
allocate_cores() {
    local num_cores=$1
    local allocated=()

    # 加锁
    exec 200>"$LOCK_FILE"
    flock -x 200
    
    # 读取当前空闲核心池
    FREE_CORES=($(cat "$FREE_CORES_FILE"))

    for ((i = 0; i < num_cores; i++)); do
        if [[ ${#FREE_CORES[@]} -gt 0 ]]; then
            allocated+=("${FREE_CORES[0]}")
            FREE_CORES=("${FREE_CORES[@]:1}")  # 移除分配的核心
        else
            echo "Error: Not enough free cores available!" >&2
            flock -u 200  # 解锁
            exec 200>&-
            exit 1
        fi
    done

    # 更新空闲核心池到文件
    echo "${FREE_CORES[@]}" > "$FREE_CORES_FILE"

    flock -u 200
    exec 200>&-

    echo "${allocated[@]}"
}

# 释放核心函数
release_cores() {
    local cores=("$@")

    # 加锁
    exec 200>"$LOCK_FILE"
    flock -x 200

    # 读取当前空闲核心池
    FREE_CORES=($(cat "$FREE_CORES_FILE"))

    # 释放核心
    FREE_CORES+=("${cores[@]}")
    FREE_CORES=($(echo "${FREE_CORES[@]}" | tr ' ' '\n' | sort -n | tr '\n' ' '))  # 保持核心池有序

    # 更新空闲核心池到文件
    echo "${FREE_CORES[@]}" > "$FREE_CORES_FILE"
    # 解锁
    flock -u 200
    exec 200>&-
}


# 遍历任务列表并执行
for ((i = 0; i < ${#TASKS[@]}; i++)); do

    # 分配核心
    CORES=($(allocate_cores $CORES_PER_TASK))
    CORE_RANGE=$(IFS=,; echo "${CORES[*]}")  # 转换为逗号分隔的核心范围


    # 获取当前任务
    TASK=${TASKS[$i]}
    # 生成一个 1 到 3 的随机数
    # RANDOM_NUMBER=$((RANDOM % 10 + 1))
    # TASK="sleep $RANDOM_NUMBER"
    # 启动任务并绑定到指定核心
    (
        taskset -c $CORE_RANGE $TASK  # 运行任务
        release_cores "${CORES[@]}"    # 任务完成后释放核心
    ) &

    # 启动任务并绑定到指定核心
    echo "Running task on cores $CORE_RANGE: $TASK"
    # taskset -c $CORE_RANGE $TASK &

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


# # 遍历任务列表并执行
# for TASK in "${TASKS[@]}"; do
#     # 启动后台
#     eval "$TASK "
# done


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
