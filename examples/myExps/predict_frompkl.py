import os
import pickle as pkl
import pandas as pd
import numpy as np
from mindware.utils.data_manager import DataManager
from mindware.components.utils.constants import CLASSIFICATION

def load_model_from_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'rb') as f:
                    op_list, model, _ = pkl.load(f)
                print(f"Loaded model from {filepath}")
                print(f"Operations: {op_list}")
                print(f"Model Type: {type(model).__name__}")
                return model
            except Exception as e:
                print(f"Failed to load {filename}: {str(e)}")
                continue
    print("No model found in the directory.")
    return None

directory_path = './data/test_fe' # pkl文件的位置，代码是拿测试交叉验证的改的所以从是文件夹里批量读，这里文件夹里放训练出来的那一个模型就可以

model = load_model_from_directory(directory_path)

if model is None:
    print("No model to predict.")
    exit()

data_dir = 'D:\\Code\\MindWare\\Data\\santander-customer-transaction-prediction'

dm = DataManager()
train_data_node = dm.load_train_csv(
    os.path.join(data_dir, 'train.csv'),
    ignore_columns=['ID_code'],
    label_name='target'
)

train_data_node = dm.preprocess_fit(train_data_node, task_type=CLASSIFICATION)
test_data_node = dm.load_test_csv(
    os.path.join(data_dir, 'test.csv'),
    ignore_columns=['ID_code']
)
test_data_node = dm.preprocess_transform(test_data_node)

X_test = test_data_node.data

print(f"type of X_test: {type(X_test)}")
print(f"X_test: {X_test}")

predictions = model.predict_proba(X_test[0])[:, 1]
print(predictions)

test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
result_df = pd.DataFrame({
    'ID_code': test_df['ID_code'],
    'target': predictions
})
result_df['target'] = result_df['target'].rank() / len(test_df)
result_df.to_csv(os.path.join(data_dir, 'predictions_with_fe.csv'), index=False)
print('Predictions have been saved to predictions.csv.')
