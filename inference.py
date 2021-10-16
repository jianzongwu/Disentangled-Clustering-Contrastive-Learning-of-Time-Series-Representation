import numpy as np
import pandas as pd
import argparse
import os
import time
import datetime
# from model import OursModel
from model_swa import OursModel
from model_km import OursModel
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('--run_name', type=str, default='', help='The run name')
    parser.add_argument('--eval_name', type=str, default='', help='The eval name')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--K', type=int, default=3, help='The number of classes')
    parser.add_argument('--sim_fun', type=str, default='cosine', help='The cluster similarity function')
    parser.add_argument('--cate_fun', type=str, default='softmax', help='The cluster function')
    parser.add_argument('--seed', type=int, default=0, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=8, help='The maximum allowed number of threads used by this process')
    args = parser.parse_args()

    print("Arguments:", str(args))
    
    device = utils.init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    config = dict(
        classes=args.K,
        sim_fun=args.sim_fun,
        cate_fun=args.cate_fun,
    )
            
    # find path
    run_dir = "./training"
    eval_dir = "./eval"
    run_dic_list = []
    runs = os.listdir(run_dir)
    for run in runs:
        dic = {}
        run_split = run.split("__")
        dic["data"] = run_split[0]
        dic["datetime"] =  run_split[1][-15:]
        run_name = run_split[1][:-16]
        dic["name"] = run_name
        run_dic_list.append(dic)

    target_run = [run for run in run_dic_list if run["data"]==args.dataset and run["name"]==args.run_name]
    time_sorted_target_run = sorted(target_run, key=lambda target_run: target_run["datetime"], reverse=True)
    datetime = time_sorted_target_run[0]["datetime"]
    model_dir = os.path.join(run_dir, args.dataset + "__" + args.run_name + '_' + datetime)
    model_path_list = os.listdir(model_dir)

    if len(args.val_name):
        eval_dir = os.path.join(eval_dir, args.dataset + "__" + args.run_name + '_' + args.eval_name + '_' + datetime)
    else:
        eval_dir = os.path.join(eval_dir, args.dataset + "__" + args.run_name + '_' + datetime)
    os.makedirs(eval_dir, exist_ok=True)
    
    while True:
        for model_path in model_path_list:
            if not model_path.startswith('model_i'):
                continue
            model_iter = model_path[7:-4]
            if os.path.exists(f'{eval_dir}/classification_i{model_iter}.pkl'):
                continue
            # load model
            model = OursModel(
                device=device,
                save_path=run_dir,
                **config
            )
            model.load(f'{model_dir}/{model_path}')

            t = time.time()
            data_path = './data/UCRArchive_2018'
            datasets = [x[0][len(data_path) + 1:] for x in os.walk(data_path)][1:]
            result = pd.DataFrame(columns=['Test', 'Accuracy'])
            for dataset in datasets:
                if utils.is_nan_dataset(dataset):
                    continue
                train, train_labels, test, test_labels = utils.load_UCR_dataset(
                    data_path, dataset
                )
                features = model.encode(train).cpu().numpy()
                svm = utils.fit_svm(features, train_labels)
                test_features = model.encode(test).cpu().numpy()
                score = svm.score(test_features, test_labels)
                score = np.round(score, 4)
                result = result.append({'Test': dataset, 'Accuracy': score}, ignore_index=True)

            print(f"Model iter {model_iter} avg acc: {result['Accuracy'].mean().round(4)}")
            result.to_csv(f'{eval_dir}/classification_i{model_iter}.pkl', index=False)
            t = time.time() - t
            print(f"\nEvaluation iter {model_iter} time: {datetime.timedelta(seconds=t)}\n")
        
        if len(os.listdir(model_dir)) <= len(model_path_list):
            break
        else:
            model_path_list = os.listdir(model_dir)
    
    print("Finished.")
