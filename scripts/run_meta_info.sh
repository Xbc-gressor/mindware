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
end_idx=30
length=$((end_idx - start_idx + 1))
# 提取子数组
subset=("${cls_datasets[@]:$start_idx:$length}")
# 打印提取的子数组
echo "Subset (index $start_idx to $end_idx):"
for item in "${subset[@]}"; do
    echo "python create_algorithm_meta_info_new.py --task cls --datasets \"$item\" --time_limit 1200 --amount_of_resource 100"
    TASKS+=("python create_algorithm_meta_info_new.py --task cls --datasets \"$item\" --time_limit 1200 --amount_of_resource 100")
done


rgs_datasets=( \
    "NewFuelCar" "mbagrade" "meta" "puma32H" "plasma_retinol" "chscase_census2" "house_8L" "fruitfly" "auto_price" "sulfur" "insurance" "visualizing_soil" \
    "stock" "bolts" "lowbwt" "Moneyball" "AutoHorse_fixed" "us_crime" "socmob" "2dplanes" "echoMonths" "disclosure_x_noise" "kin8nm" "cloud" "bank32nh" \
    "elusage" "cleveland" "electricity_prices_ICON" "space_ga" "fishcatch" "bank8FM" "rainfall_bangladesh" "pwLinear" "pyrim" "mtp" "mv" "weather_izmir" \
    "strikes" "fri_c3_100_25" "boston" "arsenic-female-lung" "kdd_coil_7" "debutanizer" "cpu_act" "wind" "cholesterol" "triazines" "chscase_foot" \
    "sleuth_case1201" "FacultySalaries" "pharynx" "carprice" "disclosure_z" "Crash" "cpu_small" "analcatdata_negotiation" "kc_house_data" "black_friday" \
    "OnlineNewsPopularity" "liver-disorders" "wisconsin" "SMRA" "puma8NH" "analcatdata_gsssexsurvey" "pollen" "pbc")

echo "${#rgs_datasets[@]}"




# 最大并发数
MAX_JOBS=10

# 当前运行的作业数
CURRENT_JOBS=0


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