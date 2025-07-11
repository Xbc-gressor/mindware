import os
import json
import pandas as pd
import numpy as np
import pickle as pkl

with open('./images/bingo.pkl', 'rb') as f:
    valid_datasets = pkl.load(f)

# with open('./images/perf_rank.pkl', 'rb') as f:
#     perf_ranks = pkl.load(f)
new_valid_datasets = {
    'CLS': [],
    'RGS': []
}
# valid_datasets = None
perf_ranks = None
bingo_dataframe = pd.DataFrame(columns=['Task Type', 'Datasets', 'OpenML ID', 'Classes', 'Samples', 'Features'])

size_lists = {
    'CLS': [],
    'RGS': []
}
fea_lists = {
    'CLS': [],
    'RGS': []
}
rank_lists = {
    'CLS': [],
    'RGS': []
}
datasets_dir = '/root/automl_data/automl_data/'
for task_type in ['CLS', 'RGS']:
    if task_type == 'CLS':
        datasets_info = pd.read_excel(os.path.join(datasets_dir, '数据集.xlsx'), sheet_name='CLS')
        with open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_ori/cls_meta_dataset_embedding.pkl', 'rb') as f:
            res = pkl.load(f)['task_ids']
    else:
        datasets_info = pd.read_excel(os.path.join(datasets_dir, '数据集.xlsx'), sheet_name='REG_SORT')
        with open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_ori/rgs_meta_dataset_embedding.pkl', 'rb') as f:
            res = pkl.load(f)['task_ids']
    datasets = [tmp[5:] for tmp in res]

    for dataset in datasets:
        if valid_datasets is not None:
            if dataset not in valid_datasets[task_type]:
                continue
        if perf_ranks is not None:
            if dataset not in perf_ranks[task_type]:
                continue
        idx = np.where(datasets_info['Datasets'].values == dataset)[0][0]
        instance = datasets_info['Instances'][idx]
        feature = datasets_info['Continuous'][idx] + datasets_info['Nominal'][idx]
        ID = int(datasets_info['ID'][idx])

        if task_type == 'CLS':
            tt = 'Classification'
            classes = datasets_info['Classes'][idx]
        else:
            tt = 'Regression'
            classes = '/'
        bingo_dataframe.loc[len(bingo_dataframe)] = [task_type, dataset, ID, classes, instance, feature]

        if instance >= 500 and ~np.isnan(ID):
            new_valid_datasets[task_type].append(dataset)
            size_lists[task_type].append(instance)
            fea_lists[task_type].append(feature)
            if perf_ranks is not None:
                rank_lists[task_type].append(15 - perf_ranks[task_type][dataset]['mine'])  # all_L2_weighted

bingo_dataframe = bingo_dataframe.sort_values(by=['Task Type', 'Samples'])
bingo_dataframe.to_csv('./images/bingo.csv', index=False)
from prettytable import PrettyTable
table = PrettyTable()
# 设置字段名
table.field_names = bingo_dataframe.columns
# 添加行
for row in bingo_dataframe.itertuples(index=False):
    table.add_row(row)
# 打印 PrettyTable
print(table)

# with open('./images/valid_datasets_500.pkl', 'wb') as f:
#     pkl.dump(new_valid_datasets, f)

import matplotlib.pyplot as plt

# 示例评述打分数据（例如问卷评分结果）

# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(size_lists, bins=30, edgecolor='black', rwidth=0.8)

# 添加标题和坐标轴标签
plt.title("dataset size")
plt.xlabel("size")
plt.ylabel("number of dataset")

plt.savefig('./images/dataset_size.png')
# 显示图形
plt.show()


# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(fea_lists, bins=30, edgecolor='black', rwidth=0.8)

# 添加标题和坐标轴标签
plt.title("feature number")
plt.xlabel("feature")
plt.ylabel("number of dataset")

plt.tight_layout()
plt.savefig('./images/feature_number.png')
# 显示图形
plt.show()


plt.figure(figsize=(8, 6))
if perf_ranks is not None:
    plt.scatter(size_lists['CLS'], fea_lists['CLS'], s=[t * 2 for t in rank_lists['CLS']], label='Classification')
    plt.scatter(size_lists['RGS'], fea_lists['RGS'], s=[t * 2 for t in rank_lists['RGS']], label='Regression')
else:
    plt.scatter(size_lists['CLS'], fea_lists['CLS'], label='Classification')
    plt.scatter(size_lists['RGS'], fea_lists['RGS'], label='Regression')
plt.legend()

plt.title('Datasets by Data Dimensions')
plt.xlabel('Number of Instances')
plt.ylabel('Number of Features')

plt.xscale('log')
plt.yscale('log')
plt.savefig('./images/size_feature.png')
plt.show()

print(f"CLS: {len(size_lists['CLS'])}, RGS: {len(size_lists['RGS'])}")

if perf_ranks is not None:

    plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))  # 2行1列
    axes[0].scatter(size_lists['CLS'], rank_lists['CLS'], label='Classification')
    axes[0].set_title('Classification')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('Ranks')

    axes[1].scatter(size_lists['RGS'], rank_lists['RGS'], label='Regression')
    axes[1].set_title('Regression')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('Ranks')


    plt.savefig('./images/rank_with_size.png')
    plt.show()

    print(f"CLS: {len(size_lists['CLS'])}, RGS: {len(size_lists['RGS'])}")



    plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))  # 2行1列
    axes[0].scatter(fea_lists['CLS'], rank_lists['CLS'], label='Classification')
    axes[0].set_title('Classification')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('Ranks')

    axes[1].scatter(fea_lists['RGS'], rank_lists['RGS'], label='Regression')
    axes[1].set_title('Regression')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('Ranks')


    plt.savefig('./images/rank_with_feature.png')
    plt.show()

    print(f"CLS: {len(size_lists['CLS'])}, RGS: {len(size_lists['RGS'])}")