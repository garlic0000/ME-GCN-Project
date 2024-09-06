import glob, os
import pandas as pd
import multiprocessing


if __name__ == '__main__':
    import opts

    # /opt/conda/lib/python3.10/multiprocessing/popen_fork.py: 66: RuntimeWarning: os.fork()
    # was called.os.fork() is incompatible
    # with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
    # self.pid = os.fork()

    # 使用spawn而不是fork: Python的multiprocessing库默认使用fork来创建子进程
    # 但在多线程应用中可能会出现问题。你可以通过设置multiprocessing的启动方式为spawn来避免这种问题。
    multiprocessing.set_start_method("spawn")
    args = opts.parse_args()
    
    def _cal_metrics(tp, n, m):
        recall = float(tp) / m if m > 0 else 0
        precision = float(tp) / n if n > 0 else 0
        f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
    
        print('Precision = ', round(precision, 4))
        print('Recall = ', round(recall, 4))
        print('F1-Score = ', round(f1_score, 4)) 
        
        return recall, precision, f1_score
    
    res_list = glob.glob(
        os.path.join(args.output, '*/best_res.csv')
    )
    df_list = []
    for res_file in res_list:
        df = pd.read_csv(res_file)
        df_list.append(df)
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    full_df = list(full_df.sum(axis=0).values)
    micro_tp, micro_n, micro_m, macro_tp, \
    macro_n, macro_m, all_tp, all_n, all_m  = full_df
    
    
    print(f'Micro result: TP:{micro_tp}, FP:{micro_n - micro_tp}, FN:{micro_m - micro_tp}')
    mic_rec, mic_pr, mic_f1 = _cal_metrics(micro_tp, micro_n, micro_m)
    
    print(f'Macro result: TP:{macro_tp}, FP:{macro_n - macro_tp}, FN:{macro_m - macro_tp}')
    mac_rec, mac_pr, mac_f1 = _cal_metrics(macro_tp, macro_n, macro_m)
    
    print(f'Total result: TP:{all_tp}, FP:{all_n - all_tp}, FN:{all_m - all_tp}')
    all_rec, all_pr, all_f1 = _cal_metrics(all_tp, all_n, all_m)