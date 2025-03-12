import numpy as np
import sys
import logging

# 设置基本的日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 直接导入需要的类，避免导入整个openbox包
from openbox.utils.early_stop import EarlyStopAlgorithm
from openbox.utils.history import History, Observation
from openbox.core.generic_advisor import Advisor
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

def create_mock_advisor():
    """创建一个模拟的Advisor对象"""
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('x1', -5, 10))
    cs.add_hyperparameter(UniformFloatHyperparameter('x2', 0, 15))
    
    advisor = Advisor(config_space=cs, 
                     num_objectives=1,
                     num_constraints=0,
                     initial_trials=3,
                     surrogate_type='gp',
                     acq_type='ei',
                     acq_optimizer_type='random_scipy')
    return advisor

def create_mock_history(num_points=10, converging=True):
    """创建一个模拟的优化历史，可以选择是否创建收敛的数据"""
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('x1', -5, 10))
    cs.add_hyperparameter(UniformFloatHyperparameter('x2', 0, 15))
    
    history = History(task_id='test_task')
    
    # 基准目标值（用于生成收敛序列）
    base_objective = 10.0
    
    for i in range(num_points):
        config = cs.sample_configuration()
        
        if converging:
            # 生成逐渐收敛的目标值
            noise = np.random.normal(0, 0.1)  # 小噪声
            decay = np.exp(-i/5)  # 指数衰减
            objective = base_objective * decay + noise
        else:
            objective = np.random.uniform(0, 10)
            
        # 生成相似的validation metrics来触发early stopping
        validation_metrics = [objective + np.random.normal(0, 0.01) for _ in range(10)]
        
        observation = Observation(
            config=config,
            objectives=[objective],
            constraints=None,
            extra_info={
                'validation_metrics': validation_metrics
            }
        )
        history.update_observation(observation)
    
    return history

def test_check_regret_stop():
    """测试check_regret_stop功能"""
    print("\nStarting early stopping test...")
    
    # 创建EarlyStopAlgorithm实例，使用较小的参数来更容易触发停止
    early_stop = EarlyStopAlgorithm(
        min_iter=5,
        min_improvement_percentage=0.001,
        max_no_improvement_rounds=5,
        decay_rates=[0.21, 0.5, 1.0]
    )
    
    # 创建模拟的advisor和收敛的历史
    advisor = create_mock_advisor()
    history = create_mock_history(num_points=20, converging=True)
    
    # 多轮训练和检查
    for round_num in range(100):
        print(f"\nRound {round_num + 1}:")
        
        # 更新advisor的历史记录并训练模型
        X = []
        y = []
        for obs in history.observations:
            X.append(obs.config.get_array())
            y.append(obs.objectives[0])
        
        # 将数据转换为numpy数组
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        print(f"Training model with {len(X)} samples...")
        advisor.surrogate_model.train(X, y)
        print("Model trained successfully.")
        
        # 测试early stopping
        should_stop = early_stop.check_regret_stop(history, advisor)
        print(f"Should stop: {should_stop}")
        
        if should_stop:
            print("\nEarly stopping triggered!")
            break
            
        # 如果没有停止，添加更多的观察结果
        new_history = create_mock_history(num_points=5, converging=True)
        for obs in new_history.observations:
            history.update_observation(obs)
        
        # 打印当前最优值
        best_objective = min([obs.objectives[0] for obs in history.observations])
        print(f"Current best objective: {best_objective:.6f}")
        
        # 打印最后一个observation的extra_info
        if len(history.observations) > 0:
            last_obs = history.observations[-1]
            print("\nLast observation extra info:")
            print(f"Regret bound: {last_obs.extra_info.get('regret_bound', 'Not found')}")
            print(f"Incumbent std: {last_obs.extra_info.get('incumbent_std', 'Not found')}")

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    try:
        test_check_regret_stop()
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 