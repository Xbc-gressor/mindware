#!/bin/bash

# """
# RGS
# """
# cls_fil=( \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_kc1_2025-03-12-06-45-15-460493/2025-03-12-06-45-15-460493_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_sick_2025-03-12-07-45-55-313896/2025-03-12-07-45-55-313896_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-12-08-46-33-150779/2025-03-12-08-46-33-150779_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_ailerons_2025-03-12-09-09-57-213528/2025-03-12-09-09-57-213528_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_mv_2025-03-12-09-47-22-363231/2025-03-12-09-47-22-363231_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_covertype_2025-03-12-05-00-58-004513/2025-03-12-05-00-58-004513_topk_config.pkl" \
# )
# cls_ori=( \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_kc1_2025-03-12-06-59-15-555134/2025-03-12-06-59-15-555134_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_sick_2025-03-12-07-59-52-980092/2025-03-12-07-59-52-980092_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-12-09-00-33-301429/2025-03-12-09-00-33-301429_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_ailerons_2025-03-12-09-10-33-297642/2025-03-12-09-10-33-297642_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_mv_2025-03-12-10-01-29-630397/2025-03-12-10-01-29-630397_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_covertype_2025-03-12-05-03-40-625369/2025-03-12-05-03-40-625369_topk_config.pkl" \
# )
# rgs_fil=( \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_Moneyball_2025-03-11-22-54-38-072888/2025-03-11-22-54-38-072888_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_debutanizer_2025-03-12-00-41-06-575912/2025-03-12-00-41-06-575912_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_puma8NH_2025-03-12-02-41-50-522801/2025-03-12-02-41-50-522801_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-12-02-59-49-725734/2025-03-12-02-59-49-725734_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_bank32nh_2025-03-12-04-44-47-445772/2025-03-12-04-44-47-445772_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_black_friday_2025-03-11-22-54-40-162585/2025-03-11-22-54-40-162585_topk_config.pkl" \
# )
# rgs_ori=( \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_Moneyball_2025-03-11-22-54-38-029448/2025-03-11-22-54-38-029448_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_debutanizer_2025-03-12-00-55-07-596446/2025-03-12-00-55-07-596446_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_puma8NH_2025-03-12-02-56-04-170021/2025-03-12-02-56-04-170021_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-12-03-02-39-989552/2025-03-12-03-02-39-989552_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_bank32nh_2025-03-12-04-57-10-412143/2025-03-12-04-57-10-412143_topk_config.pkl" \
#     "/root/mindware/examples/benchmark/norefit_data/CASHFE-block_1(1)-holdout_black_friday_2025-03-11-22-54-40-253658/2025-03-11-22-54-40-253658_topk_config.pkl" \
# )

cls_fil=( \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_kc1_2025-03-20-23-23-19-744510/2025-03-20-23-23-19-744510_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_sick_2025-03-21-00-24-47-596213/2025-03-21-00-24-47-596213_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-21-01-28-09-571649/2025-03-21-01-28-09-571649_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_ailerons_2025-03-21-01-37-44-168182/2025-03-21-01-37-44-168182_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_mv_2025-03-20-21-10-30-636620/2025-03-20-21-10-30-636620_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_covertype_2025-03-20-21-10-31-084786/2025-03-20-21-10-31-084786_topk_config.pkl" \
)
cls_ori=( \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_kc1_2025-03-20-23-26-41-876197/2025-03-20-23-26-41-876197_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_sick_2025-03-21-00-32-03-951978/2025-03-21-00-32-03-951978_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-21-01-29-52-252459/2025-03-21-01-29-52-252459_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_ailerons_2025-03-21-01-50-20-259979/2025-03-21-01-50-20-259979_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_mv_2025-03-20-21-10-30-650461/2025-03-20-21-10-30-650461_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_covertype_2025-03-20-21-10-31-122521/2025-03-20-21-10-31-122521_topk_config.pkl" \
)
rgs_fil=( \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_Moneyball_2025-03-21-02-42-26-608732/2025-03-21-02-42-26-608732_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_debutanizer_2025-03-21-04-42-42-438506/2025-03-21-04-42-42-438506_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_puma8NH_2025-03-21-06-35-56-018447/2025-03-21-06-35-56-018447_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-21-06-44-02-209187/2025-03-21-06-44-02-209187_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_bank32nh_2025-03-21-08-37-03-949491/2025-03-21-08-37-03-949491_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_black_friday_2025-03-21-02-30-43-420097/2025-03-21-02-30-43-420097_topk_config.pkl" \
)
rgs_ori=( \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_Moneyball_2025-03-21-02-55-52-221729/2025-03-21-02-55-52-221729_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_debutanizer_2025-03-21-04-56-38-664945/2025-03-21-04-56-38-664945_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_puma8NH_2025-03-21-06-38-34-392736/2025-03-21-06-38-34-392736_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-21-06-57-51-407974/2025-03-21-06-57-51-407974_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_bank32nh_2025-03-21-08-40-04-089264/2025-03-21-08-40-04-089264_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_black_friday_2025-03-21-02-33-21-385823/2025-03-21-02-33-21-385823_topk_config.pkl" \
)


# 初始化任务列表
declare -a TASKS

# ensemble_sizes=(5)

# # # 生成任务列表
# for ensemble_size in "${ensemble_sizes[@]}"; do
#     for i in 0 1 2 3 4 5; do
#         TASKS+=("python cls_benchmark.py --refit full --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size $ensemble_size --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#         TASKS+=("python cls_benchmark.py --refit full --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size $ensemble_size --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
        # TASKS+=("python rgs_benchmark.py --refit full --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size $ensemble_size --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
        # TASKS+=("python rgs_benchmark.py --refit full --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size $ensemble_size --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
#     done
# done

# ensemble_sizes=(5 10 20 30 40 50)
# ratios=(0 0.1 0.2 0.3 0.4 0.49)

# for ensemble_size in "${ensemble_sizes[@]}"; do
#     for ratio in "${ratios[@]}"; do
#         # # 生成任务列表
#         for i in 0 1 2 3 4 5; do
#             TASKS+=("python cls_benchmark.py --refit full --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size $ensemble_size --ratio $ratio --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#             TASKS+=("python cls_benchmark.py --refit full --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size $ensemble_size --ratio $ratio --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
#             TASKS+=("python rgs_benchmark.py --refit full --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size $ensemble_size --ratio $ratio --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
#             TASKS+=("python rgs_benchmark.py --refit full --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size $ensemble_size --ratio $ratio --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
#         done
#     done
# done

# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit cv --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit cv --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${cls_ori[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit cv --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit cv --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${rgs_ori[$i]}")
# done
# for i in 0 1 2 3 4 5; do
    # TASKS+=("python cls_benchmark.py --refit full --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_stackskip_data --output_file res_refitcv_data.txt --stats_path ${cls_fil[$i]}")
    # TASKS+=("python cls_benchmark.py --refit full --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_stack_data --output_file res_refitcv_data.txt --stats_path ${cls_ori[$i]}")
    # TASKS+=("python rgs_benchmark.py --refit full --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_stackskip_data --output_file res_refitcv_data.txt --stats_path ${rgs_fil[$i]}")
    # TASKS+=("python rgs_benchmark.py --refit full --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_stack_data --output_file res_refitcv_data.txt --stats_path ${rgs_ori[$i]}")
# done
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit partial --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit partial --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${cls_ori[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit partial --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit partial --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${rgs_ori[$i]}")
# done

# # # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit cv --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method stacking --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit cv --job_idx $i --optimizer block_1                                    --ensemble_method stacking --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${cls_ori[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit cv --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method stacking --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit cv --job_idx $i --optimizer block_1                                    --ensemble_method stacking --ensemble_size 10 --Opt cashfe --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${rgs_ori[$i]}")
# done
# for layer in 4; do
#     for i in 0 1 2 3 4 5; do
#         TASKS+=("python cls_benchmark.py --refit full --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method stacking --ensemble_size 10 --layer $layer --ens_thr 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_stackskip_data --output_file res_refitcv_data.txt --stats_path ${cls_fil[$i]}")
#     #     TASKS+=("python cls_benchmark.py --refit f079016ull --job_idx $i --optimizer block_1                                    --ensemble_method stacking --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${cls_ori[$i]}")
#         TASKS+=("python rgs_benchmark.py --refit full --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method stacking --ensemble_size 10 --layer $layer --ens_thr 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_stackskip_data --output_file res_refitcv_data.txt --stats_path ${rgs_fil[$i]}")
#     #     TASKS+=("python rgs_benchmark.py --refit full --job_idx $i --optimizer block_1                                    --ensemble_method stacking --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refitcv_data.txt --stats_path ${rgs_ori[$i]}")
#     done
# done

for i in 0 1 2 3 4; do
    TASKS+=("python cls_benchmark.py --time_limit 3600 --refit full --job_idx $i --Opt ens --optimizer block_1 --output_dir ./res_ensopt_data --output_file res_ensopt_data.txt --stats_path ${cls_fil[$i]}")
    TASKS+=("python rgs_benchmark.py --time_limit 7200 --refit full --job_idx $i --Opt ens --optimizer block_1 --output_dir ./res_ensopt_data --output_file res_ensopt_data.txt --stats_path ${rgs_fil[$i]}")
done


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
