cls_datasets=( \
    "hypothyroid(2)" "spectf" "mfeat-morphological(2)" "fri_c3_1000_50" "fri_c3_1000_25" "splice" "puma32H" "covertype" "mnist_784" "fri_c2_1000_25" \
    "adult" "kr-vs-kp" "glass" "adult-census" "letter(2)" "delta_ailerons" "balloon" "fri_c2_1000_50" "car(2)" "satimage" "car(1)" "semeion" \
    "messidor_features" "cmc" "pol" "socmob" "credit-g" "fri_c3_1000_10" "kropt" "quake" "fri_c1_1000_10" "sick" "elevators" "bank32nh" "dna" \
    "space_ga" "hypothyroid(1)" "waveform-5000(2)" "baseball" "pc4" "ionosphere" "mfeat-karhunen(1)" "poker" "mushroom" "pc1" "kc1" "cal_housing" \
    "nursery" "lymphography" "diabetes" "credit" "eeg" "mv" "fri_c4_1000_25" "rmftsa_sleepdata(1)" "rmftsa_sleepdata(2)" "vehicle" "analcatdata_supreme" \
    "fri_c4_1000_50" "vehicle_sensIT" "page-blocks(1)" "optdigits" "delta_elevators" "fri_c0_1000_5" "house_16H" "fri_c0_1000_25" "fri_c1_1000_50" \
    "gina_prior2" "cpu_act" "fri_c1_1000_5" "mfeat-zernike(1)" "wind" "analcatdata_halloffame" "mfeat-fourier(2)" "mfeat-fourier(1)" "jm1" "fri_c4_1000_10" \
    "fri_c0_1000_10" "mfeat-morphological(1)" "madelon" "pc3" "fri_c1_1000_25" "mfeat-karhunen(2)" "musk" "letter(1)" "amazon_employee" "mammography" \
    "cpu_small" "isolet" "houses" "abalone" "winequality_red" "waveform-5000(1)" "fri_c2_1000_5" "mfeat-factors(2)" "fri_c3_1000_5" "usps" "puma8NH" \
    "yeast" "fri_c0_1000_50" "colleges_usnews" "pollen" "magic_telescope" "mfeat-factors(1)" "colleges_aaup" "pendigits" "mfeat-zernike(2)" "sylva_prior" \
    "ailerons")

echo "${#cls_datasets[@]}"

# 初始化任务列表
declare -a TASKS

start_idx=0
end_idx=108
length=$((end_idx - start_idx + 1))
# 提取子数组
subset=("${cls_datasets[@]:$start_idx:$length}")
# 打印提取的子数组
echo "Subset (index $start_idx to $end_idx):"
for item in "${subset[@]}"; do
    echo "python create_algorithm_meta_info_new.py --task cls --datasets $item --time_limit 1200 --amount_of_resource 100 --rep 3 --start_id 0"
    TASKS+=("python create_algorithm_meta_info_new.py --task cls --datasets $item --time_limit 1200 --amount_of_resource 100 --rep 3 --start_id 0")
done


# rgs_datasets=( \
#     "NewFuelCar" "mbagrade" "meta" "puma32H" "plasma_retinol" "chscase_census2" "house_8L" "fruitfly" "auto_price" "sulfur" "insurance" "visualizing_soil" \
#     "stock" "bolts" "lowbwt" "Moneyball" "AutoHorse_fixed" "us_crime" "socmob" "2dplanes" "echoMonths" "disclosure_x_noise" "kin8nm" "cloud" "bank32nh" \
#     "elusage" "cleveland" "electricity_prices_ICON" "space_ga" "fishcatch" "bank8FM" "rainfall_bangladesh" "pwLinear" "pyrim" "mtp" "mv" "weather_izmir" \
#     "strikes" "fri_c3_100_25" "boston" "arsenic-female-lung" "kdd_coil_7" "debutanizer" "cpu_act" "wind" "cholesterol" "triazines" "chscase_foot" \
#     "sleuth_case1201" "FacultySalaries" "pharynx" "carprice" "disclosure_z" "Crash" "cpu_small" "analcatdata_negotiation" "kc_house_data" "black_friday" \
#     "OnlineNewsPopularity" "liver-disorders" "wisconsin" "SMRA" "puma8NH" "analcatdata_gsssexsurvey" "pollen" "pbc")

# echo "${#rgs_datasets[@]}"

# # 初始化任务列表
# declare -a TASKS

# start_idx=0
# end_idx=65
# length=$((end_idx - start_idx + 1))
# # 提取子数组
# subset=("${rgs_datasets[@]:$start_idx:$length}")
# # 打印提取的子数组
# echo "Subset (index $start_idx to $end_idx):"
# for item in "${subset[@]}"; do
#     echo "python create_algorithm_meta_info_new.py --task reg --datasets $item --time_limit 1200 --amount_of_resource 100 --rep 3 --start_id 0"
#     TASKS+=("python create_algorithm_meta_info_new.py --task reg --datasets $item --time_limit 1200 --amount_of_resource 100 --rep 3 --start_id 0")
# done


# 最大并发数
MAX_JOBS=10
# 当前运行的作业数
CURRENT_JOBS=0
# 每个任务使用的核心数量
CORES_PER_TASK=8
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