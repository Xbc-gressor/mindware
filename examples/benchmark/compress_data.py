# 把过去的data压缩一下

import os
import sys
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import joblib
import pickle


"""
include_algorithms = [
    'adaboost', 'extra_trees', 'gradient_boosting',
    'random_forest', 'lightgbm', 'xgboost'
]
block012_data: 169G (extra_trees)-> 75G (random forest)-> 54G  (all)-> 45G
cash_vs_cashfe_data: 108G -> 29G

"""

# 获取原来topk文件里面的root_dir
def get_path_before_second_slash(path):
    parts = path.split('/')
    if len(parts) > 2:
        return '/'.join(parts[:2])
    else:
        raise Exception()  # 如果没有两个斜杠，返回原路径
    
def process_directory(root_dir):
    # 遍历 root_dir 下的所有子目录
    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            filepath = os.path.join(subdir, filename)
            if filename.endswith('topk_config.pkl'):
                process_topk_file(root_dir, filepath)

def process_topk_file(root_dir, filepath):
    # 读取 topk_config.pkl 文件
    with open(filepath, 'rb') as file:
        topk = pickle.load(file)
    
    # 处理 topk['extra_trees'] 中的每个元素
    for key in topk:
        for i in range(len(topk[key])):
            t = topk[key][i]
            pkl_path = t[2]
            ori_root = get_path_before_second_slash(pkl_path)
            pkl_path = pkl_path.replace(ori_root, root_dir)
            topk[key][i] = (t[0], t[1], pkl_path)  # 可能根目录变了，重新赋值一下
            
            # if not os.path.exists(pkl_path):
            #     print(pkl_path)
            #     continue
            
            if pkl_path.endswith('.pkl'):  # and key in ['extra_trees', 'random_forest']:
                content = pickle.load(open(pkl_path, 'rb'))
                new_path = pkl_path.replace('.pkl', '.joblib')
                joblib.dump(content, new_path, compress=True)
                print("save file: ", new_path)
                # 删除原来的文件
                os.remove(pkl_path)
                print("remove file: ", pkl_path)
                
                # 更新路径
                topk[key][i] = (t[0], t[1], new_path)
    
    # 将更新后的 topk 重新保存到 topk_config.pkl
    with open(filepath, 'wb') as file:
        pickle.dump(topk, file)

if __name__ == "__main__":
    root_dir = './block_vs_ori_data'  # 设置你的目录路径
    process_directory(root_dir)
