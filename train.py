import numpy as np
import pandas as pd
import argparse
import os
import time
import datetime
from model import OursModel
# from model_swa import OursModel
# from model_km import OursModel
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('--run_name', type=str, default='', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--datapath', type=str, required=True, help='The dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=10, help='The batch size (defaults to 10)')
    parser.add_argument('--lr', type=int, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--K', type=int, default=3, help='The number of classes')
    parser.add_argument('--sim_fun', type=str, default='cosine', help='The cluster similarity function')
    parser.add_argument('--cate_fun', type=str, default='softmax', help='The cluster function')
    parser.add_argument('--iters', type=int, default=2000, help='The number of iterations')
    parser.add_argument('--save-every', type=int, default=100, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--valid', action="store_true", help='Whether to save and valid model every save-every iters')
    parser.add_argument('--latest', action="store_true", help='Whether to save model in a latest model folder')
    parser.add_argument('--seed', type=int, default=0, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=8, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = utils.init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    train_data, train_labels, test_data, test_labels = utils.load_UCR_dataset(args.datapath, args.dataset)
            
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        classes=args.K,
        sim_fun=args.sim_fun,
        cate_fun=args.cate_fun,
    )

    if args.latest:
        # find latest path
        run_dir = "./training"
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
        if len(target_run) == 0:
            run_dir = 'training/' + args.dataset + '__' + utils.name_with_datetime(args.run_name)
            os.makedirs(run_dir, exist_ok=True)
        else:
            time_sorted_target_run = sorted(target_run, key=lambda target_run: target_run["datetime"], reverse=True)
            _datetime = time_sorted_target_run[0]["datetime"]
            run_dir = os.path.join(run_dir, args.dataset + "__" + args.run_name + '_' + _datetime)
    else:
        run_dir = 'training/' + args.dataset + '__' + utils.name_with_datetime(args.run_name)
        os.makedirs(run_dir, exist_ok=True)
    
    print(f'run_dir: {run_dir}')
    
    t = time.time()
    
    model = OursModel(
        device=device,
        save_path=run_dir,
        **config
    )
    loss_log = model.fit(
        train_data,
        n_iters=args.iters,
        save_every=args.save_every,
        verbose=True,
        valid=args.valid
    )
    model.save(f'{run_dir}/model.pth')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        eval_dir = 'eval/' + run_dir.split('/')[-1]
        print(f'eval_dir: {eval_dir}')
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
            print(dataset, "Test accuracy: " + str(score))
            result = result.append({'Test': dataset, 'Accuracy': score}, ignore_index=True)

        result.to_csv(f'{eval_dir}/classification.pkl', index=False)
        t = time.time() - t
        print(f"\nEvaluation time: {datetime.timedelta(seconds=t)}\n")
    
    print("Finished.")
