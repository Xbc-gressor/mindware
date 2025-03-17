from openbox.utils.limit import run_with_time_limit, run_without_time_limit
import numpy as np
import pickle as pkl
import xgboost as xgb


def run():
    X = np.random.rand(100, 4)
    Y = np.array([0] * 30 + [1] * 70)

    model = xgb.XGBClassifier()
    print("fit")
    model.fit(X, Y)
    print("end")


if __name__ == '__main__':

    res2 = run_without_time_limit(run, {}, {})
    print(res2)

    res1 = run_with_time_limit(run, {}, {}, 300)
    print(res1)
