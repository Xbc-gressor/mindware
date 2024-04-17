
import os
import pickle as pkl



if __name__ == '__main__':
    dir_path = '../data/CASH-mab(1)_2024-04-17-22-33-35-391024'

    for path in os.listdir(dir_path):
        if 'topk' in path or 'log' in path:
            continue
        _path = os.path.join(dir_path, path)
        try:
            with open(_path, 'rb') as f:
                op_list, model, _ = pkl.load(f)
                print(model)
        except:
            print(_path)

