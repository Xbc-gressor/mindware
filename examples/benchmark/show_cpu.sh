#!/bin/bash

# 获取所有包含 "create_algorithm_meta_info_new.py" 的进程，获取 PID 和父进程 PID
processes=$(ps -ef | grep "benchmark_" | grep -v "grep" | awk '{print $2":"$3}')

# 检查是否找到相关进程
if [[ -z "$processes" ]]; then
    echo "No processes containing 'benchmark_' found."
    exit 0
fi

# 创建一个数组保存所有的 PID
all_pids=()
for entry in $processes; do
    pid=$(echo "$entry" | cut -d: -f1)
    all_pids+=("$pid")
done

# 创建一个数组保存合法的 PID
valid_pids=()

# 遍历每个进程，筛选出合法的 PID
for entry in $processes; do
    pid=$(echo "$entry" | cut -d: -f1)
    ppid=$(echo "$entry" | cut -d: -f2)

    # 检查父进程是否也在 PID 列表中
    if [[ " ${all_pids[@]} " =~ " $ppid " ]]; then
        echo "Skipping PID $pid: Parent process ($ppid) is in the PID list."
        continue
    fi

    # 如果父进程不在 PID 列表中，将该 PID 添加到合法的 PID 列表中
    valid_pids+=("$pid")
done

# 输出合法的 PID 列表
echo "Valid PIDs: ${valid_pids[@]}"

# 遍历每个进程，获取其 PID 和绑定的 CPU 核
echo -e "PID\tCPU Affinity"
for pid in "${valid_pids[@]}"; do

    # 使用 taskset -cp 获取进程的 CPU 核绑定信息
    cpu_affinity=$(taskset -cp $pid 2>/dev/null | awk -F: '{print $2}' | xargs)

    # 检查是否成功获取 CPU 绑定信息
    if [[ -n "$cpu_affinity" ]]; then
        echo -e "$pid\t$cpu_affinity"
    else
        echo -e "$pid\tError retrieving CPU affinity"
    fi
done
