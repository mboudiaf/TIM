import numpy as np
from sacred import Ingredient
from src.utils import warp_tqdm, compute_confidence_interval, load_checkpoint
from src.utils import load_pickle, save_pickle
from src.datasets.ingredient import get_dataloader
import os
import torch
import collections
import torch.nn.functional as F
from src.tim import TIM, TIM_ADM, TIM_GD

eval_ingredient = Ingredient('eval')
@eval_ingredient.config
def config():
    meta_test_iter = 10000
    meta_val_way = 5
    meta_val_query = 15
    method = 'baseline'
    model_tag = 'best'
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

    def merge_tasks(self, tasks_dics):
        merged_tasks = {}
        n_tasks = len(tasks_dics)
        for key in tasks_dics[0].keys():
            batch_size = tasks_dics[0][key].size(0)
            merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks, batch_size, -1)
        return merged_tasks

    @eval_ingredient.capture
    def generate_tasks(self, loaders_dic, extracted_features_dic, model_path, shot, model_tag, used_set, meta_test_iter):

        print(f" ==> Generating {meta_test_iter} tasks ...")
        tasks_dics = []
        for _ in warp_tqdm(range(meta_test_iter), False):
            task_dic = self.get_task(shot=shot, extracted_features_dic=extracted_features_dic)
            tasks_dics.append(task_dic)
        tasks = self.merge_tasks(tasks_dics)
        return tasks

    @eval_ingredient.capture
    def run_full_evaluation(self, model, model_path, model_tag, shots, method, callback):
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

            tasks = self.generate_tasks(loaders_dic=loaders_dic, extracted_features_dic=extracted_features_dic,
                                        shot=shot, model_path=model_path)
            logs = self.run_task(task_dic=tasks,
                                 model=model,
                                 train_mean=train_mean,
                                 callback=callback)

            l2n_mean, l2n_conf = compute_confidence_interval(logs['acc'][:, -1])

            print('==> Meta Test: {} \nfeature\tL2N\n{}-shot \t{:.4f}({:.4f})'.format(
                  model_tag.upper(), shot, l2n_mean, l2n_conf))
            results.append(l2n_mean)
        return results

    def run_task(self, task_dic, train_mean, model, callback):

        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model=model)

        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        z_s, z_q = task_dic['z_s'], task_dic['z_q']

        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        train_mean = train_mean.to(self.device)

        # Perform normalizations required
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)

        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s)
        tim_builder.init_weights(support=support, y_s=y_s, query=query, y_q=y_q)

        # Run adaptation
        tim_builder.run_adaptation(support=support, query=query, y_s=y_s, y_q=y_q, callback=callback)

        # Extract adaptation logs
        logs = tim_builder.get_logs()
        return logs

    @eval_ingredient.capture
    def get_tim_builder(self, model, method):
        # Initialize TIM classifier builder
        tim_info = {'model': model}
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
    def get_task(self, shot, meta_val_way, meta_val_query, extracted_features_dic):
        all_features = extracted_features_dic['concat_features']
        all_labels = extracted_features_dic['concat_labels']
        all_classes = torch.unique(all_labels)
        samples_classes = np.random.choice(a=all_classes, size=meta_val_way, replace=False)
        support_samples = []
        query_samples = []
        for each_class in samples_classes:
            class_indexes = torch.where(all_labels == each_class)[0]
            indexes = np.random.choice(a=class_indexes, size=shot + meta_val_query, replace=False)
            support_samples.append(all_features[indexes[:shot]])
            query_samples.append(all_features[indexes[shot:]])

        y_support = torch.arange(meta_val_way)[:, None].repeat(1, shot).reshape(-1)
        y_query = torch.arange(meta_val_way)[:, None].repeat(1, meta_val_query).reshape(-1)

        z_support = torch.cat(support_samples, 0)
        z_query = torch.cat(query_samples, 0)

        task = {'z_s': z_support, 'y_s': y_support,
                'z_q': z_query, 'y_q': y_query}
        return task

