import pickle as pkl
import pandas as pd
import numpy as np
import os

exp_itr = 0
task = 'cls'
metric = 'f1'

slices = [[0, 42], [42, 85], [85, 109], [0, 109]]
if task == 'reg':
    slices = [[0, 38], [38, 56], [56, 66], [0, 66]]
subtitles = ['small dataset', 'medium dataset', 'large dataset', 'all dataset']
algorithms = ['adaboost','extra_trees','gradient_boosting','k_nearest_neighbors','lda','liblinear_svc','libsvm_svc','lightgbm','logistic_regression','qda','random_forest','xgboost']
sim_algorithms = ['AdaBoost','ExtraTrees','GradBoost','KNN','LDA','LinearSvc','SvmSvc','Lightgbm','LogisReg','QDA','RF','Xgboost']
if task == 'reg':
    algorithms = ['adaboost','extra_trees','gradient_boosting','k_nearest_neighbors','lasso_regression','liblinear_svr','libsvm_svr','lightgbm','random_forest','ridge_regression','xgboost']
    sim_algorithms = ['AdaBoost','ExtraTrees','GradBoost','KNN','LassoReg','LinearSvr','SvmSvr','Lightgbm','RF','RidgeReg','Xgboost']
res_dir = './data_cls/meta_res'
if task == 'reg':
    res_dir = './data_rgs/meta_res'
embed_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_embedding.pkl'
if task == 'reg':
    embed_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_embedding.pkl'
dataset_info_path = '/root/automl_data/automl_data/数据集.xlsx'

with open(embed_path, 'rb') as f:
    embed = pkl.load(f)
datasets_info = pd.read_excel(dataset_info_path, sheet_name='CLS').set_index('Datasets')
if task == 'reg':
    datasets_info = pd.read_excel(dataset_info_path, sheet_name='REG_SORT').set_index('Datasets')
datasets = np.array([t[5:] for t in embed['task_ids']])
order_dict = {dataset: index for index, dataset in enumerate(datasets_info.index)}
# 获取排序后的索引
sorted_indices = np.argsort([order_dict.get(item, float('inf')) for item in datasets])
sorted_datasets = datasets[sorted_indices]

for i in range(len(slices)):
    _s = slices[i]
    _datasets = sorted_datasets[_s[0]:_s[1]]
    _tmp = datasets_info.loc[_datasets]['Instances']
    subtitles[i] += ' %d~%d' % (np.min(_tmp), np.max(_tmp))

res_scores = np.full((len(sorted_datasets), len(algorithms)), np.nan)
res_times = np.full((len(sorted_datasets), len(algorithms)), np.nan)
for i, dataset in enumerate(sorted_datasets):
    for j, algo in enumerate(algorithms):
        # r_path = os.path.join(res_dir, '%s-%s-%s-%d-1200.pkl'%(dataset, algo, metric, exp_itr))
        # with open(r_path, 'rb') as f:
        #     r = pkl.load(f)
        # res_scores[i, j] = r[3]
        # res_times[i, j] = r[2]
        scores = []
        times = []
        for k in range(3):
            r_path = os.path.join(res_dir, '%s-%s-%s-%d-1200.pkl'%(dataset, algo, metric, k))
            with open(r_path, 'rb') as f:
                r = pkl.load(f)
                scores.append(r[3])
                times.append(r[2])
        res_scores[i, j] = np.median(scores)
        res_times[i, j] = np.mean(times)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=((16, 8)))
for row in range(2):
    for col in range(2):
        _s = slices[row * 2 + col]
        axs[row,col].set_title(subtitles[row * 2 + col])
        tmp_res_times = res_times[_s[0]:_s[1]]
        # 绘制直方图,统计运行次数
        counts, bins, patches = axs[row,col].hist(tmp_res_times.reshape(-1), bins=10, alpha=0.7, color='blue', edgecolor='black')
        # 标注数据
        for count, bin, patch in zip(counts, bins, patches):
            height = patch.get_height()
            axs[row,col].annotate(f'{int(count)}', xy=(bin + (bins[1] - bins[0])/2, height), 
                        xytext=(0, 2), textcoords='offset points', ha='center', va='bottom')
        # 添加标题和标签
        if row == 1:
            axs[row,col].set_xlabel('Value')
        if col == 0:
            axs[row,col].set_ylabel('Frequency')
        

# 显示图形
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
fig.suptitle('Histogram')
plt.savefig('./images/%s_try_times_%d.png' % (task, exp_itr))
plt.show()

fig, axs = plt.subplots(3, 4, figsize=((20, 10)))
for row in range(3):
    for col in range(4):
        if row * 4 + col >= len(sim_algorithms):
            continue
        axs[row,col].set_title(sim_algorithms[row * 4 + col])
        tmp_res_times = res_times[:, row * 4 + col]
        # 绘制直方图,统计运行次数
        counts, bins, patches = axs[row,col].hist(tmp_res_times.reshape(-1), bins=10, alpha=0.7, color='blue', edgecolor='black')
        # 标注数据
        for count, bin, patch in zip(counts, bins, patches):
            height = patch.get_height()
            axs[row,col].annotate(f'{int(count)}', xy=(bin + (bins[1] - bins[0])/2, height), 
                        xytext=(0, 2), textcoords='offset points', ha='center', va='bottom')
        # 添加标题和标签
        if row == 2:
            axs[row,col].set_xlabel('Value')
        if col == 0:
            axs[row,col].set_ylabel('Frequency')


# 显示图形
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
fig.suptitle('Histogram')
plt.savefig('./images/%s_try_times_by_algo_%d.png' % (task, exp_itr))
plt.show()

# 最优的
# 创建一个 2x2 的子图结构
fig, axs = plt.subplots(2, 2, figsize=((16, 8)))

for row in range(2):
    for col in range(2):
        _s = slices[row * 2 + col]
        axs[row,col].set_title(subtitles[row * 2 + col])
        tmp_res_scores = res_scores[_s[0]:_s[1]]

        counts = np.zeros(len(algorithms), dtype=int)
        max_scores = np.max(tmp_res_scores, axis=1)
        for i in range(len(max_scores)):
            max_idx = np.where(tmp_res_scores[i] == max_scores[i])[0]
            for t in max_idx:
                counts[t] += 1
        # 绘制柱状图
        axs[row,col].bar(sim_algorithms, counts, color='blue', alpha=0.7, edgecolor='black')
        # 在每个柱子上方标记数字
        for i, count in enumerate(counts):
            axs[row,col].text(i, count, str(count), ha='center', va='bottom')
        if row == 1:
            # 添加标题和标签
            axs[row,col].set_xticklabels(sim_algorithms, rotation=30)
        else:
            axs[row,col].set_xticks([])
        if col == 0:
            axs[row,col].set_ylabel('Frequency')

# 添加标题和标签
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
plt.subplots_adjust(hspace=0.15)
fig.suptitle('Frequency of Best Algorithm')
plt.savefig('./images/%s_winners_%d.png' % (task, exp_itr))
plt.show()



# 前六
# 创建一个 2x2 的子图结构
fig, axs = plt.subplots(2, 2, figsize=((16, 8)))

for row in range(2):
    for col in range(2):
        _s = slices[row * 2 + col]
        axs[row,col].set_title(subtitles[row * 2 + col])
        tmp_res_scores = res_scores[_s[0]:_s[1]]

        counts = np.zeros(len(algorithms), dtype=int)
        for i in range(len(tmp_res_scores)):
            top_idx = np.argsort(-tmp_res_scores[i])
            for t in top_idx[:6]:
                counts[t] += 1
        # 绘制柱状图
        axs[row,col].bar(sim_algorithms, counts, color='blue', alpha=0.7, edgecolor='black')
        # 在每个柱子上方标记数字
        for i, count in enumerate(counts):
            axs[row,col].text(i, count, str(count), ha='center', va='bottom')
        if row == 1:
            # 添加标题和标签
            axs[row,col].set_xticklabels(sim_algorithms, rotation=30)
        else:
            axs[row,col].set_xticks([])
        if col == 0:
            axs[row,col].set_ylabel('Frequency')

plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
plt.subplots_adjust(hspace=0.15)
fig.suptitle('Frequency of Algorithm Chosen as Top-6')
plt.savefig('./images/%s_winners_top6_%d.png' % (task, exp_itr))
plt.show()

breakpoint()