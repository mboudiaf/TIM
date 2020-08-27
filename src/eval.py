
import numpy as np
from sacred import Ingredient
from src.utils import warp_tqdm, compute_confidence_interval, load_checkpoint
from src.utils import load_pickle, save_pickle
from src.datasets.ingredient import get_dataloader
import os
import math
import torch
import collections
import pickle
from src.tim import TIM, TIM_ADM, TIM_GD
from src.utils import get_logs_path
import matplotlib
import torch.nn.functional as F
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
# plt.rcParams.update({'font.size': 13})
plt.style.use('ggplot')
font = {'family': 'serif',
        'size'  : 14}

matplotlib.rc('font', **font)

eval_ingredient = Ingredient('eval')
@eval_ingredient.config
def config():
    meta_test_iter = 10000
    meta_val_way = 5
    meta_val_query = 15
    method = 'baseline'
    save_plt_info = False
    model_tag = 'best'
    norms_types = ["L2N"]
    target_data_path = None  # Only for cross-domain scenario
    target_split_dir = None  # Only for cross-domain scenario
    plt_metrics = ['accs']
    shots = [1, 5]
    used_set = 'test'


class Evaluator:
    @eval_ingredient.capture
    def __init__(self, device, ex):
        self.device = device
        self.ex = ex

    @eval_ingredient.capture
    def run_full_evaluation(self, model, model_path, model_tag, norms_types, save_plt_info,
                            shots, method, callback):
        print("=> Runnning full evaluation with method: {}".format(method))

        # Load pre-trained model
        load_checkpoint(model=model, model_path=model_path, type=model_tag)

        # Get loaders
        loaders_dic = self.get_loaders()

        # Extract features (just load them if already in memory)
        extracted_features_dic = self.extract_features(model=model,
                                                       model_path=model_path,
                                                       loaders_dic=loaders_dic)
        results = []
        train_mean = extracted_features_dic['train_mean']
        for shot in shots:
            un_list = []
            l2n_list = []
            cl2n_list = []
            timestamps = []
            accs = []
            mutual_infos = []
            entropies = []
            cond_entropies = []
            losses = []
            for data, _, indexes in warp_tqdm(loaders_dic['episodic_test'][shot], False):
                if self.ex.current_run.config['tim']['finetune_encoder']:
                    load_checkpoint(model=model, model_path=model_path, type=model_tag)
                task_dic = self.get_task(data=data, indexes=indexes, shot=shot,
                                         extracted_features_dic=extracted_features_dic)

                if 'CL2N' in norms_types:
                    acc, logs = self.run_task(task_dic=task_dic,
                                              model=model,
                                              train_mean=train_mean,
                                              norm_type='CL2N',
                                              callback=callback)
                    cl2n_list.append(acc)

                if 'L2N' in norms_types:
                    acc, logs = self.run_task(task_dic=task_dic,
                                              model=model,
                                              train_mean=train_mean,
                                              norm_type='L2N',
                                              callback=callback)
                    l2n_list.append(acc)
                    timestamps.append(logs['timestamps'])
                    accs.append(logs['acc'])
                    mutual_infos.append(logs['mutual_info'])
                    entropies.append(logs['entropy'])
                    cond_entropies.append(logs['cond_entropy'])
                    losses.append(logs['losses'])

                if 'UN' in norms_types:
                    acc, logs = self.run_task(task_dic=task_dic,
                                              model=model,
                                              train_mean=train_mean,
                                              norm_type='UN',
                                              callback=callback)
                    un_list.append(acc)

            if save_plt_info:
                if self.ex.current_run.config['tim']['finetune_encoder']:
                    method += '_all'
                with open(get_logs_path(model_path=model_path, method=method, shot=shot), 'wb') as f:
                    logs = {'timestamps': np.array(timestamps),
                            'accs': np.array(accs),
                            'mutual_info': np.array(mutual_infos),
                            'entropy': np.array(entropies),
                            'cond_entropy': np.array(cond_entropies),
                            'losses': np.array(losses)}
                    pickle.dump(logs, f)
            un_mean, un_conf = compute_confidence_interval(un_list)
            l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
            cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)

            print('==> Meta Test: {} \nfeature\tUN\tL2N\tCL2N\n{}-shot \t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
                  model_tag.upper(), shot, un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf))
            results.append(l2n_mean)
        return results

    def run_task(self, task_dic, train_mean, norm_type, model, callback):

        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model=model)

        z_s, z_q = task_dic['z_s'], task_dic['z_q']
        support = z_s.unsqueeze(0).to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.unsqueeze(0).to(self.device)  # [ N * (K_s + K_q), d]

        # Transfer tensors to GPU if needed
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        y_s = y_s.unsqueeze(0).long().to(self.device)
        y_q = y_q.unsqueeze(0).long().to(self.device)

        # Perform normalizations required
        if norm_type == 'CL2N':
            train_mean = train_mean.to(self.device)
            support = support - train_mean
            support = F.normalize(support, dim=2)
            query = query - train_mean
            query = F.normalize(query, dim=2)
        elif norm_type == 'L2N':
            support = F.normalize(support, dim=2)
            query = F.normalize(query, dim=2)

        if tim_builder.finetune_encoder:
            support, query = task_dic['x_s'], task_dic['x_q']
            support, query = support.float().to(self.device), query.float().to(self.device)

        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s)
        tim_builder.init_weights(support=support, y_s=y_s, query=query, y_q=y_q)

        # If all parameters are being finetuned, then set support and query to be input features
        if tim_builder.finetune_encoder:
            support, query = task_dic['x_s'], task_dic['x_q']
            support, query = support.float().to(self.device), query.float().to(self.device)

        # Run adaptation
        tim_builder.run_adaptation(support=support, query=query, y_s=y_s, y_q=y_q, callback=callback)

        # Get accuracy
        preds_q = tim_builder.get_preds(samples=query)
        acc = (preds_q == y_q).float().mean().item()

        # Extract adaptation logs
        logs = tim_builder.get_logs()
        return acc, logs

    @eval_ingredient.capture
    def get_tim_builder(self, model, save_plt_info, method):
        # Initialize TIM classifier builder
        tim_info = {'model': model, 'save_plt_info': save_plt_info}
        if method == 'tim_adm':
            tim_builder = TIM_ADM(**tim_info)
        elif method == 'tim_gd':
            tim_builder = TIM_GD(**tim_info)
        elif method == 'baseline':
            tim_builder = TIM(**tim_info)
        else:
            raise ValueError("Method must be in ['tim_gd', 'tim_adm', 'baseline']")
        return tim_builder

    @eval_ingredient.capture
    def get_loaders(self, used_set, target_data_path, target_split_dir, meta_test_iter, meta_val_way,
                    meta_val_query, shots):
        # First, get loaders
        episodic_test_loaders = {}
        loaders_dic = {}
        loader_info = {'aug': False, 'shuffle': False,
                       'out_name': False}
        if target_data_path is not None:  # This mean we are in the cross-domain scenario
            loader_info.update({'path': target_data_path,
                                'split_dir': target_split_dir})
        train_loader = get_dataloader('train', **loader_info)
        loaders_dic['train_loader'] = train_loader
        try:  # In case there are predefined support-query files (for iNat only)
            support_loader = get_dataloader('support', **loader_info)
            query_loader = get_dataloader('query', **loader_info)
            loaders_dic.update({'query': query_loader, 'support': support_loader})
        except:  # For all other datasets
            test_loader = get_dataloader(used_set, **loader_info)
            loaders_dic.update({'test': test_loader})

        for shot in shots:
            sample_info = [meta_test_iter, meta_val_way, shot, meta_val_query]
            loader_info.update({'sample': sample_info})
            episodic_test_loaders[shot] = get_dataloader(used_set, **loader_info)
        loaders_dic.update({'episodic_test': episodic_test_loaders})
        return loaders_dic

    @eval_ingredient.capture
    def extract_features(self, model, model_path, model_tag, used_set, loaders_dic):

        # Load features from memory if previously saved ...
        save_dir = os.path.join(model_path, model_tag, used_set)
        filepath = os.path.join(save_dir, 'output.plk')
        if os.path.isfile(filepath):
            extracted_features_dic = load_pickle(filepath)
            print(" ==> Features loaded from {}".format(filepath))
            return extracted_features_dic

        # ... otherwise just extract them
        else:
            print(" ==> Beginning feature extraction")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

        model.eval()
        with torch.no_grad():
            out_mean = []
            for i, (inputs, _, _) in enumerate(warp_tqdm(loaders_dic['train_loader'], False)):
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs, True)
                out_mean.append(outputs.cpu().view(outputs.size(0), -1))
            out_mean = torch.cat(out_mean, axis=0).mean(0)

            if 'test' in loaders_dic:
                test_dict = collections.defaultdict(list)
                all_features = []
                all_labels = []
                for i, (inputs, labels, _) in enumerate(warp_tqdm(loaders_dic['test'], False)):
                    inputs = inputs.to(self.device)
                    outputs, _ = model(inputs, True)
                    all_features.append(outputs.cpu())
                    all_labels.append(labels)
                all_features = torch.cat(all_features, 0)
                all_labels = torch.cat(all_labels, 0)
                extracted_features_dic = {'train_mean': out_mean,
                                          'test': test_dict,
                                          'concat_features': all_features,
                                          'concat_labels': all_labels
                                          }
            else:
                support_dict = collections.defaultdict(list)
                for i, (inputs, labels, _) in enumerate(warp_tqdm(loaders_dic['support'], False)):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = model(inputs, True)
                    outputs = outputs.cpu()
                    for out, label in zip(outputs, labels):
                        support_dict[label.item()].append(out.view(1, -1))

                query_dict = collections.defaultdict(list)
                for i, (inputs, labels, _) in enumerate(warp_tqdm(loaders_dic['query'], False)):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = model(inputs, True)
                    outputs = outputs.cpu()
                    for out, label in zip(outputs, labels):
                        query_dict[label.item()].append(out.view(1, -1))

                for label in support_dict:
                    support_dict[label] = torch.cat(support_dict[label], dim=0)
                    query_dict[label] = torch.cat(query_dict[label], dim=0)
                extracted_features_dic = {'train_mean': out_mean,
                                          'support': support_dict,
                                          'query': query_dict
                                          }
        print(" ==> Saving features to {}".format(filepath))
        save_pickle(filepath, extracted_features_dic)
        return extracted_features_dic

    @eval_ingredient.capture
    def get_task(self, data, indexes, shot, meta_val_way, meta_val_query, extracted_features_dic):

        y_support = torch.arange(meta_val_way)[:, None].repeat(1, shot).reshape(-1)
        y_query = torch.arange(meta_val_way)[:, None].repeat(1, meta_val_query).reshape(-1)

        x_support = data[:meta_val_way * shot]
        x_query = data[meta_val_way * shot:]

        all_features = extracted_features_dic['concat_features']
        support_indexes = indexes[:meta_val_way*shot]
        query_indexes = indexes[meta_val_way*shot:]
        z_support = all_features[support_indexes]
        z_query = all_features[query_indexes]

        task = {'x_s': x_support, 'z_s': z_support, 'y_s': y_support,
                'x_q': x_query, 'z_q': z_query, 'y_q': y_query}
        return task

    @eval_ingredient.capture
    def print_runtimes(self, model_path, shots, method):
        print("=> Only printing runtimes")
        for algo in method:
            for shot in shots:
                path = get_logs_path(model_path=model_path, method=algo, shot=shot)
                if os.path.isfile(path):
                    with open(path, 'rb') as f:
                        dic = pickle.load(f)
                        timestamps = dic['timestamps']
                    timestamps = np.cumsum(timestamps, axis=1)
                    mean_time, interv = compute_confidence_interval(timestamps, axis=0)
                    print("{} : {} Shot : {} ({}) ".format(algo, shot, mean_time[-1], interv[-1]))

    @eval_ingredient.capture
    def make_plots(self, model_path, method, plt_metrics, shots):
        print("=> Only doing plots")
        method2name = {'tim_adm': "TIM-ADM", 'tim_gd': "TIM-GD",
                       'hybrid': "HYBRID", 'tim_gd_all': r"TIM-GD $\{\phi, W\}$"}
        method2color = {'tim_gd_all': "#2b8cbe", 'tim_adm': '#2ca25f', 'tim_gd': '#8856a7', 'hybrid': 'c'}
        method2linestyle = {'tim_gd_all': 'dashdot', 'tim_adm': 'dashed', 'tim_gd': 'dotted'}
        ylabels = {'mutual_info': r'\textbf{Mutual information (nats)}', 'accs': r'\textbf{Test accuracy}',
                   'entropy': 'Marginal entropy', 'cond_entropy': 'Conditional Entropy',
                   'losses': 'Training loss'}

        for metric in plt_metrics:
            latest_time = 0
            earliest_time = 1000
            for algo in method:
                for shot in shots:
                    shot_path = get_logs_path(model_path=model_path, method=algo, shot=shot)
                    assert os.path.isfile(shot_path) or os.path.isfile(shot_path), shot_path
                    if os.path.isfile(shot_path):
                        with open(shot_path, 'rb') as f:
                            dic = pickle.load(f)
                            timestamps = dic['timestamps']
                            timestamps = np.cumsum(timestamps.mean(axis=0))
                            if timestamps[-1] > latest_time:
                                latest_time = timestamps[-1]
                            if timestamps[0] < earliest_time:
                                earliest_time = timestamps[0]
            break
        i = 0
        n_rows = math.ceil(len(plt_metrics) / 2)
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5))
        # if len(plt_metrics) != 1:
        #     plt.subplots(n_rows, 1, figsize=(10, 5))
        # fig, (ax1, ax2, ax3) = plt.subplots(n_rows, 2, figsize=(10, 5))

        for metric in plt_metrics:
            for algo in method:
                for shot in shots:
                    shot_path = get_logs_path(model_path=model_path, method=algo, shot=shot)
                    assert os.path.isfile(shot_path) or os.path.isfile(shot_path), shot_path
                    if os.path.isfile(shot_path):
                        with open(shot_path, 'rb') as f:
                            dic = pickle.load(f)
                            timestamps = dic['timestamps']
                            if metric in dic:
                                values = dic[metric].squeeze()
                                timestamps = np.cumsum(timestamps.mean(axis=0))
                                # print(values.shape, timestamps.shape)
                                mean_acc, interv = compute_confidence_interval(values, axis=0)
                                if timestamps[-1] < latest_time:
                                    mean_acc = np.append(mean_acc, mean_acc[-1])
                                    interv = np.append(interv, interv[-1])
                                    timestamps = np.append(timestamps, latest_time)
                                timestamps[0] = earliest_time
                                if len(shots) > 1:
                                    label = f'{method2name[algo]} {shot}-shot'
                                else:
                                    label = f'{method2name[algo]}'
                                axes[i].plot(timestamps, mean_acc,
                                             label=label,
                                             color=method2color[algo],
                                             linestyle=method2linestyle[algo],
                                             linewidth=3)
                                axes[i].fill_between(timestamps, mean_acc - interv, mean_acc + interv,
                                                     alpha=0.2, color=method2color[algo])
            axes[i].grid(True)
            axes[i].set_xlabel(r'\textbf{Time (s)}')
            axes[i].set_xscale('log')
            axes[i].set_ylabel(ylabels[metric])
            # if i == 1:
            #     axes[i].legend(loc='upper center', bbox_to_anchor=(1.0, 1.14),
            #                    ncol=3, fancybox=True, shadow=True)

            i += 1
        handles, labels = axes[-1].get_legend_handles_labels()
        lgd = fig.legend(handles, labels, loc='upper center', ncol=3, fancybox=True, shadow=True,
                         bbox_to_anchor=(0.52, 1.05))
        plt.tight_layout()
        exp = '_'.join(model_path.split('/')[1:])
        root = os.path.join('plots', str(method))
        os.makedirs(root, exist_ok=True)
        plt.savefig(os.path.join(root, f'{exp}.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()