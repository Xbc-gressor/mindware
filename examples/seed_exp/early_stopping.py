def apply_ours(path, results):
    for i in range(10):
        csv_path = str(path) + f"/{i:03}.csv"
        algo = extract_algo(csv_path)
        bb_name = path.parts[-1]
        dataset = bb_name[:bb_name.find("_dpp")]
        df = pd.read_csv(csv_path)

        for decay in [0.21, 0.5, 1.0]:
            method_name, stop_iter, rel_iter, rel_val, rel_test, rel_time, total_iter, total_time = get_stop_info_ours(df, decay)
            d = {'algo': algo, 'dataset': dataset, 'seed': i, 'method': method_name, 'stop_iter': stop_iter,
                 'rel_iter': rel_iter, 'rel_val': rel_val, 'rel_test': rel_test, 'rel_time': rel_time,
                 'total_iter': total_iter, 'total_time': total_time}
            results.append(d)
def get_stop_info_ours(df, decay=0.21, std=None, use_true_regret=False, nas=False):
    method_name = f"Ours_{decay}" if std is None else f"Ours_{std}"
    final_iter = df.iloc[-1]['iteration']
    final_val_metric = df.iloc[-1]['incumbent_val_metric']
    final_test_metric = df.iloc[-1]['incumbent_test_metric'] if not nas else df.iloc[-1]['incumbent_val_metric']
    final_total_time = df.iloc[-1]['total_train_time']
    stop_iter = 0
    for _, row in df.iterrows():
        it = row['iteration']
        regret = row['regret_bound'] if not use_true_regret else row['true_regret']
        if std is None:
            std = row['incumbent_std'] / 0.21 * decay
        val = row['incumbent_val_metric']
        test = row['incumbent_test_metric'] if not nas else row['incumbent_val_metric']
        time = row['total_train_time']
        stop_iter = it
        if regret < std:
            break
    rel_iter = (final_iter - it) / final_iter
    rel_val = (final_val_metric - val) / max(final_val_metric, val)
    rel_test = (final_test_metric - test) / max(final_test_metric, test)
    rel_time = (final_total_time - time) / final_total_time
    return method_name, stop_iter, rel_iter, rel_val, rel_test, rel_time, final_iter, final_total_time

