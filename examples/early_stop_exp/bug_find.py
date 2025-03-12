import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
from openbox.utils.limit import run_with_time_limit, run_without_time_limit
import numpy as np
import pickle as pkl
import xgboost as xgb
# 在 run 函数结束时添加

X = np.random.rand(100, 4)
Y = np.array([0]*30 + [1]*70)
def run():
    model = xgb.XGBClassifier()
    model.n_jobs = 1
    print("fit")
    model.fit(X, Y)
    print("end")
    
if __name__ == "__main__":
    run()
    res1 = run_with_time_limit(run, {}, {}, 300)
    print(res1)