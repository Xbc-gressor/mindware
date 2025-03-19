#!/bin/bash

"""
RGS
"""
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
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_kc1_2025-03-14-15-14-54-312757/2025-03-14-15-14-54-312757_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_sick_2025-03-14-16-16-03-034868/2025-03-14-16-16-03-034868_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-14-17-17-47-387869/2025-03-14-17-17-47-387869_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_ailerons_2025-03-14-18-23-09-891326/2025-03-14-18-23-09-891326_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_mv_2025-03-12-09-47-22-363231/2025-03-12-09-47-22-363231_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_covertype_2025-03-14-15-14-54-961805/2025-03-14-15-14-54-961805_topk_config.pkl" \
)
cls_ori=( \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_kc1_2025-03-14-15-14-54-115484/2025-03-14-15-14-54-115484_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_sick_2025-03-14-16-20-04-212727/2025-03-14-16-20-04-212727_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-14-17-27-33-154048/2025-03-14-17-27-33-154048_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_ailerons_2025-03-14-18-33-42-506572/2025-03-14-18-33-42-506572_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_mv_2025-03-12-10-01-29-630397/2025-03-12-10-01-29-630397_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_covertype_2025-03-14-15-14-54-911685/2025-03-14-15-14-54-911685_topk_config.pkl" \
)
rgs_fil=( \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_Moneyball_2025-03-14-20-38-59-167158/2025-03-14-20-38-59-167158_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_debutanizer_2025-03-14-22-38-19-728928/2025-03-14-22-38-19-728928_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_puma8NH_2025-03-14-23-48-15-880091/2025-03-14-23-48-15-880091_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-15-00-44-26-734738/2025-03-15-00-44-26-734738_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_bank32nh_2025-03-12-04-44-47-445772/2025-03-12-04-44-47-445772_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_black_friday_2025-03-14-19-39-28-583324/2025-03-14-19-39-28-583324_topk_config.pkl" \
)
rgs_ori=( \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_Moneyball_2025-03-14-20-41-15-041167/2025-03-14-20-41-15-041167_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_debutanizer_2025-03-14-22-42-55-505962/2025-03-14-22-42-55-505962_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_puma8NH_2025-03-15-00-02-58-090871/2025-03-15-00-02-58-090871_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_cpu_act_2025-03-15-00-49-38-463648/2025-03-15-00-49-38-463648_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_bank32nh_2025-03-15-02-14-25-062709/2025-03-15-02-14-25-062709_topk_config.pkl" \
    "/root/mindware/examples/benchmark/refit_data/CASHFE-block_1(1)-holdout_black_friday_2025-03-14-19-53-45-445057/2025-03-14-19-53-45-445057_topk_config.pkl" \
)


# 初始化任务列表
declare -a TASKS



# # # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 50 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 50 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
# done

# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 50 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 50 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
# done


# # # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 40 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 40 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
# done

# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 40 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 40 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
# done

# # # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 30 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 30 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
# done

# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 30 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 30 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
# done


# # # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
# done

# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
# done

# # # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
# done

# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method ensemble_selection --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method ensemble_selection --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
# done


# # # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
# done

# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 20 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
# done


# # # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
# done

# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
# done


# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 5 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_fil[$i]}")
#     TASKS+=("python cls_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 5 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${cls_ori[$i]}")
# done

# # 生成任务列表
# for i in 0 1 2 3 4 5; do
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 5 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_fil[$i]}")
#     TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 5 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_refit_data.txt --stats_path ${rgs_ori[$i]}")
# done

# 生成任务列表
for i in 0 1 2 3 4 5; do
    TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1 --n_algorithm 6 --n_preprocessor 6 --ensemble_method blending --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_blending_data.txt --stats_path ${rgs_fil[$i]}")
    TASKS+=("python rgs_benchmark.py --refit --job_idx $i --optimizer block_1                                    --ensemble_method blending --ensemble_size 10 --Opt cashfe --optimizer block_1 --output_dir ./ens_data --output_file res_blending_data.txt --stats_path ${rgs_ori[$i]}")
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
