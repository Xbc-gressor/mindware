import numpy as np
import time
import concurrent.futures as cfutures

def f(i, X):
    dropout = 0.3
    n_base_model = 40
    n_dim = 1
    dropout_num = int(n_base_model * dropout)

    rng = np.random.default_rng(i)
    if dropout_num > 0:
        train_num, feature_dim = X.shape
        n_dim = feature_dim // n_base_model

        dropout_mask = np.zeros((train_num, n_base_model), dtype=int)
        for i in range(train_num):
            dropout_mask[i, rng.choice(n_base_model, dropout_num, replace=False)] = 1

    return dropout_mask

# data_node = object()
# train_node = object()
# n_base_model = 10
# dropout_num = 3


# train_num, all_dim = train_node.data[0]
# predict_dim = data_node.data[0].shape[1]
# ori_dim = all_dim - predict_dim
# n_dim = predict_dim // n_base_model
# dropout_mask = np.zeros((train_num, all_dim), dtype=int)
# for i in range(train_num):
#     dropout_mask[np.random.choice(n_base_model, dropout_num, replace=False)+ori_dim] = 1

# for dim in range(n_dim):
#     col =[ori_dim + dim + idx * n_dim for idx in range(n_base_model)]
#     data_pure_mask = 1 - dropout_mask.copy()
#     data_pure_mask[:, :ori_dim] = 0
#     data_pure = train_node.data[0] * data_pure_mask
#     train_node.data[0][:, col] = (1-dropout_mask) * train_node.data[0] + dropout_mask * np.sum(data_pure, axis=1, keepdims=True) / (n_base_model - dropout_num)

# # # 为每行随机选择 k 个位置设置为 1
# # for row in array:
# #     row[np.random.choice(b, k, replace=False)] = 1

thread = 16
start = time.time()
X = np.random.rand(3000, 40)
res = []
with cfutures.ProcessPoolExecutor(max_workers=thread) as executor:
    fs_wait = set()
    valid_indexes = []
    for i in range(10):

        if len(fs_wait) < thread:
            fs_wait.add(executor.submit(f, *[i, X]))
        else:
            fs_done, fs_wait = cfutures.wait(fs_wait, return_when=cfutures.FIRST_COMPLETED)
            fs_wait.add(executor.submit(f, *[i, X]))
            for fs in fs_done:
                res.append(fs.result())


    while len(fs_wait) > 0:
        fs_done, fs_wait = cfutures.wait(fs_wait, return_when=cfutures.FIRST_COMPLETED)
        for fs in fs_done:
            res.append(fs.result())

breakpoint()
print(time.time() - start)

