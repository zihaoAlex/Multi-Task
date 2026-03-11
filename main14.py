import torch
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
import os
import numpy as np
import pandas as pd
import argparse
import random
from scipy.linalg import eigh

# ===================== 集中管理所有随机种子（核心修改区） =====================
BASE_SEED = 42  ##42
KFOLD_SEED = BASE_SEED
DATALOADER_SEED = BASE_SEED
MODEL_INIT_SEED_OFFSET = 0
CUDA_DETERMINISTIC = True
CUDA_BENCHMARK = False
PYTHON_HASH_SEED = str(BASE_SEED)

# 自定义模块（需确保存在）
from aliexpress import AliExpressDataset
from sharedbottom import SharedBottomModel
from singletask import SingleTaskModel
from DNNomoe import OMoEModel
from DNNmmoe import MMoEModel
from DNNple import PLEModel
from layer import *


# ===================== 全局随机种子固定函数 =====================
def set_seed(seed=BASE_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = CUDA_DETERMINISTIC
    torch.backends.cudnn.benchmark = CUDA_BENCHMARK
    os.environ['PYTHONHASHSEED'] = PYTHON_HASH_SEED
    print(f"Random seed fixed to: {seed} (base seed: {BASE_SEED})")


# ===================== JDA 迁移学习核心实现（完全修复数据泄露） =====================
class JDA:
    def __init__(self, dim=20, kernel_type='linear', gamma=1.0, degree=2, coef0=1, mu=1.0):
        self.dim = dim
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.mu = mu
        self.eig_vectors = None
        self.scaler = None  # 保存JDA适配后的scaler（仅基于训练集）

    def kernel(self, X, Y):
        if self.kernel_type == 'linear':
            return np.dot(X, Y.T)
        elif self.kernel_type == 'rbf':
            sqdist = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
            return np.exp(-self.gamma * sqdist)
        elif self.kernel_type == 'poly':
            return (np.dot(X, Y.T) + self.coef0) ** self.degree
        else:
            raise ValueError('Unknown kernel type: ' + self.kernel_type)

    def fit_transform(self, Xs, Xt, ys=None):
        """
        仅基于源域+目标域训练集拟合JDA，且标准化仅拟合训练集
        Xs: 源域训练集数据
        Xt: 目标域训练集数据
        ys: 源域训练集标签
        """
        # 合并源域和目标域训练数据（仅训练集！）
        X = np.vstack((Xs, Xt))
        n, m = Xs.shape[0], Xt.shape[0]

        # 计算核矩阵（仅训练集）
        K = self.kernel(X, X)

        # 构造中心化矩阵
        one_n = np.ones((n, n)) / n
        one_m = np.ones((m, m)) / m
        one_nm = np.ones((n, m)) / (n * m)

        # 计算类内散度矩阵
        if ys is not None:
            C = len(np.unique(ys))
            Sw = np.zeros((n + m, n + m))
            for c in np.unique(ys):
                idx = np.where(ys == c)[0]
                nc = len(idx)
                if nc == 0:
                    continue
                one_c = np.ones((nc, nc)) / nc
                Sw[idx[:, None], idx] += K[idx[:, None], idx] @ (np.eye(nc) - one_c) @ K[idx[:, None], idx].T
        else:
            Sw = np.eye(n + m)

        # 计算域间散度矩阵
        Sb = np.zeros((n + m, n + m))
        Sb[:n, :n] = K[:n, :n] @ (one_n) @ K[:n, :n].T
        Sb[:n, n:] = -K[:n, n:] @ (one_nm).T @ K[n:, n:].T
        Sb[n:, :n] = -K[n:, :n] @ one_nm @ K[:n, :n].T
        Sb[n:, n:] = K[n:, n:] @ one_m @ K[n:, n:].T

        # 求解广义特征值问题
        eig_values, eig_vectors = eigh(
            (Sw + self.mu * np.eye(n + m)).dot(Sb),
            Sw + self.mu * np.eye(n + m)
        )

        # 选择前dim个特征向量
        idx = np.argsort(eig_values)[::-1][:self.dim]
        self.eig_vectors = eig_vectors[:, idx]
        A = self.eig_vectors

        # 投影到新空间
        X_jda = K.dot(A)
        Xs_jda = X_jda[:n, :]
        Xt_jda = X_jda[n:, :]

        # 仅用训练集拟合Scaler，避免验证集/测试集信息泄露
        self.scaler = StandardScaler()
        Xs_jda = self.scaler.fit_transform(Xs_jda)  # fit仅基于源域训练集
        Xt_jda = self.scaler.transform(Xt_jda)  # 目标域训练集仅transform

        return Xs_jda, Xt_jda

    def transform(self, X, X_train):
        """
        【核心修复】仅用训练集的核矩阵转换验证/测试集，不拼接数据
        X: 验证集/测试集数据
        X_train: 源域+目标域训练集合并数据
        """
        # 仅计算X与X_train的核矩阵（不拼接），避免泄露验证/测试集分布
        K = self.kernel(X, X_train)
        X_jda = K.dot(self.eig_vectors)
        # 使用训练集拟合的scaler转换（不重新fit）
        return self.scaler.transform(X_jda)


# ===================== 工具函数 =====================
def get_data_file_path(dataset_path, dataset_name, filename):
    full_path = os.path.join(dataset_path, dataset_name, filename)
    full_path = os.path.normpath(full_path)
    return full_path


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nError: Data file not found at path: {file_path}\n"
            f"Please check:\n"
            f"1. The dataset path is correct\n"
            f"2. The {os.path.basename(file_path)} file exists in the {os.path.dirname(file_path)} directory\n"
            f"3. The dataset name '{os.path.basename(os.path.dirname(file_path))}' is correct"
        )


# ===================== 数据集包装类（简化逻辑，消除潜在泄露） =====================
class NumericalDataWrapper(Dataset):
    def __init__(self, base_dataset_subset, numerical_data):
        self.base_dataset_subset = base_dataset_subset
        self.numerical_data = numerical_data
        assert len(self.numerical_data) == len(base_dataset_subset), \
            f"Numerical data length ({len(self.numerical_data)}) doesn't match base dataset subset ({len(base_dataset_subset)})"

    def __len__(self):
        return len(self.base_dataset_subset)

    def __getitem__(self, index):
        # 修正：从subset中获取数据，避免直接访问全量数据集
        _, labels = self.base_dataset_subset[index]
        return torch.tensor(self.numerical_data[index], dtype=torch.float32), labels


def get_dataset(name, path, filename):
    full_path = get_data_file_path(path, name, filename)
    check_file_exists(full_path)
    if 'AliExpress' in name:
        return AliExpressDataset(full_path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, numerical_num, task_num, expert_num, embed_dim):
    if name == 'sharedbottom':
        return SharedBottomModel(numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                                 tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'singletask':
        return SingleTaskModel(numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64),
                               task_num=task_num, dropout=0.2)
    elif name == 'omoe':
        return OMoEModel(numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64),
                         task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mmoe':
        return MMoEModel(numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64),
                         task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'ple':
        return PLEModel(numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64),
                        task_num=task_num, shared_expert_num=int(expert_num / 2),
                        specific_expert_num=int(expert_num / 2), dropout=0.5)
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_loss = float('inf')
        self.save_path = save_path

    def is_continuable(self, model, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"New best model saved with loss {loss} at {self.save_path}")
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, task_weights, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)

    for i, (numerical_fields, labels) in enumerate(loader):
        numerical_fields, labels = numerical_fields.to(device), labels.to(device)
        y = model(numerical_fields)

        if isinstance(y, list):
            y = torch.stack(y, dim=1)

        loss_list = []
        for i in range(labels.size(1)):
            task_loss = criterion(y[:, i], labels[:, i].float())
            loss_list.append(task_loss * task_weights[i])

        loss = sum(loss_list) / len(loss_list)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, task_num, device, save_csv_path=None):
    model.eval()
    all_labels = [[] for _ in range(task_num)]
    all_predictions = [[] for _ in range(task_num)]

    with torch.no_grad():
        for numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            numerical_fields, labels = numerical_fields.to(device), labels.to(device)
            y = model(numerical_fields)

            for i in range(task_num):
                all_labels[i].extend(labels[:, i].tolist())
                all_predictions[i].extend(y[i].tolist())

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        data_to_save = {}
        for i in range(task_num):
            data_to_save[f'Task_{i + 1}_True'] = all_labels[i]
            data_to_save[f'Task_{i + 1}_Predicted'] = all_predictions[i]
        df = pd.DataFrame(data_to_save)
        df.to_csv(save_csv_path, index=False)
        print(f"Predictions saved to: {save_csv_path}")

    mse_results = []
    for i in range(task_num):
        mse_results.append(np.mean((np.array(all_labels[i]) - np.array(all_predictions[i])) ** 2))

    return mse_results


def save_results_to_csv(val_mse, test_mse, args, save_dir):
    # 保存单折结果
    val_mse_df = pd.DataFrame([val_mse], columns=[f'Task_{i + 1}_MSE' for i in range(args.task_num)])
    val_mse_df.index.name = 'Fold'
    val_mse_df.index = [1]
    val_mse_df['Average_MSE'] = val_mse_df.mean(axis=1)
    val_mse_path = f'{save_dir}/validation_mse_results.csv'
    val_mse_df.to_csv(val_mse_path)

    test_mse_df = pd.DataFrame([test_mse], columns=[f'Task_{i + 1}_MSE' for i in range(args.task_num)])
    test_mse_df.index.name = 'Fold'
    test_mse_df.index = [1]
    test_mse_df['Average_MSE'] = test_mse_df.mean(axis=1)
    test_mse_path = f'{save_dir}/test_mse_results.csv'
    test_mse_df.to_csv(test_mse_path)

    summary_df = pd.DataFrame({
        'Metric': ['Validation MSE (Fold 1)', 'Test MSE (Fold 1)'],
        'Value': [np.mean(val_mse), np.mean(test_mse)]
    })
    summary_path = f'{save_dir}/summary_results.csv'
    summary_df.to_csv(summary_path, index=False)

    params_df = pd.DataFrame({
        'Parameter': list(vars(args).keys()),
        'Value': list(vars(args).values())
    })
    params_path = f'{save_dir}/parameters.csv'
    params_df.to_csv(params_path, index=False)

    print(f"\nResults saved to:\n"
          f" 1. Validation MSE: {val_mse_path}\n"
          f" 2. Test MSE: {test_mse_path}\n"
          f" 3. Summary: {summary_path}\n"
          f" 4. Parameters: {params_path}")


def main(dataset_name, dataset_path, task_num, expert_num, model_name, epoch, learning_rate, batch_size, embed_dim,
         weight_decay, device, save_dir, use_jda=True, source_dataset_name='AliExpress_US', jda_dim=50,
         jda_kernel='linear', task_weights=None, seed=None):
    final_seed = seed if seed is not None else BASE_SEED
    set_seed(final_seed)
    device = torch.device(device)

    if task_weights is None:
        task_weights = [0.2 if i == 0 else 0.4 for i in range(task_num)]
    print(f"Using task loss weights: {task_weights}")

    # 数据文件名定义
    train_val_filename = 'covariance_RefVIs1.csv'
    test_filename = 'RefVIs1.csv'

    # 加载目标域数据集
    print(f"\nLoading target domain dataset: {dataset_name}")
    train_val_dataset = get_dataset(dataset_name, dataset_path, train_val_filename)
    test_dataset = get_dataset(dataset_name, dataset_path, test_filename)
    print(f"Target Train/Val dataset size: {len(train_val_dataset)}")
    print(f"Target Test dataset size: {len(test_dataset)}")

    # 加载源域数据集（仅JDA模式，且仅用源域训练子集）
    jda = None
    source_train_numerical = None
    source_train_labels = None
    numerical_num = train_val_dataset.numerical_num

    if use_jda and source_dataset_name:
        print(f"\n=== Enabling JDA Transfer Learning ===")
        print(f"Source domain: {source_dataset_name}, Target domain: {dataset_name}")
        try:
            # 加载源域全量数据集
            source_full_dataset = get_dataset(source_dataset_name, dataset_path, train_val_filename)
            print(f"Source full dataset size: {len(source_full_dataset)}")

            # 源域仅使用训练子集（80%）
            source_train_size = int(0.8 * len(source_full_dataset))
            source_train_indices = np.arange(len(source_full_dataset))[:source_train_size]
            source_full_numerical = np.array(source_full_dataset.numerical_data)
            source_train_numerical = source_full_numerical[source_train_indices]
            source_full_labels = np.array(source_full_dataset.labels)
            source_train_labels = source_full_labels[source_train_indices, 0]  # 取第一个任务标签

            print(f"Source TRAIN subset size: {len(source_train_numerical)} (80% of full source dataset)")
            jda = JDA(dim=jda_dim, kernel_type=jda_kernel, mu=1.0)
            numerical_num = jda_dim
        except FileNotFoundError as e:
            print(f"\nWarning: Source domain dataset not found! {e}")
            print(f"Falling back to non-JDA mode")
            use_jda = False


    print(f"\n======== 9:1数据划分 Preparing ========")
    kfold = KFold(n_splits=10, shuffle=True, random_state=KFOLD_SEED)
    fold_1_indices = next(kfold.split(train_val_dataset))
    train_indices, val_indices = fold_1_indices
    print(f"Train indices: {len(train_indices)}, Val indices: {len(val_indices)}")

    original_numerical_data = np.array(train_val_dataset.numerical_data)
    original_test_numerical_data = np.array(test_dataset.numerical_data)

    train_numerical = original_numerical_data[train_indices]
    val_numerical = original_numerical_data[val_indices]
    test_numerical = original_test_numerical_data

    # --------------------------
    # 步骤2：标准化（仅拟合训练集）
    # --------------------------
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_numerical)  # fit仅训练集
    val_scaled = scaler.transform(val_numerical)  # 验证集仅transform
    test_scaled = scaler.transform(test_numerical)  # 测试集仅transform

    # --------------------------
    # 步骤3：JDA转换（仅基于训练集）
    # --------------------------
    if use_jda and source_train_numerical is not None:
        # 源域训练集仅用目标域训练集的scaler转换
        source_train_scaled = scaler.transform(source_train_numerical)

        # JDA仅拟合源域训练集+目标域训练集
        print(f"Applying JDA to source TRAIN + target TRAIN data only...")
        _, train_scaled = jda.fit_transform(
            source_train_scaled,
            train_scaled,
            ys=source_train_labels
        )

        # 验证集/测试集转换（仅用训练集参数）
        train_combined = np.vstack((source_train_scaled, train_scaled))  # 仅训练集合并
        val_scaled = jda.transform(val_scaled, train_combined)
        test_scaled = jda.transform(test_scaled, train_combined)

    # --------------------------
    # 步骤4：数据集包装
    # --------------------------
    # 构建训练/验证子集
    train_base_subset = Subset(train_val_dataset, train_indices)
    val_base_subset = Subset(train_val_dataset, val_indices)
    test_base_subset = Subset(test_dataset, np.arange(len(test_dataset)))

    # 包装数值数据
    train_dataset = NumericalDataWrapper(train_base_subset, train_scaled)
    val_dataset = NumericalDataWrapper(val_base_subset, val_scaled)
    test_dataset_wrapper = NumericalDataWrapper(test_base_subset, test_scaled)

    # 数据加载器
    train_generator = torch.Generator().manual_seed(DATALOADER_SEED)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True,
                              generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    test_loader = DataLoader(test_dataset_wrapper, batch_size=batch_size, num_workers=0, shuffle=False)

    # --------------------------
    # 步骤5：模型初始化与训练
    # --------------------------
    model_seed = final_seed + MODEL_INIT_SEED_OFFSET
    set_seed(model_seed)

    model = get_model(model_name, numerical_num, task_num, expert_num, embed_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 模型保存路径
    save_path = f'{save_dir}/{dataset_name}_{model_name}.pt'
    if use_jda:
        save_path = save_path.replace('.pt', f'_jda_{source_dataset_name}.pt')

    # 早停（基于验证集MSE）
    early_stopper = EarlyStopper(num_trials=3, save_path=save_path)
    epoch_mse_results = []

    for epoch_i in range(epoch):
        print(f"\n  {epoch_i + 1}/{epoch}")
        # 训练
        train(model, optimizer, train_loader, criterion, device, task_weights)

        # 验证集评估（用于早停）
        val_mse = test(model, val_loader, task_num, device,
                       save_csv_path=f'{save_dir}/{dataset_name}_{model_name}_fold1_val_predictions_epoch{epoch_i + 1}.csv')
        print(f'Epoch {epoch_i + 1} - Validation MSE: {val_mse}')

        avg_val_mse = np.mean(val_mse)
        epoch_mse_results.append({
            'epoch': epoch_i + 1,
            'val_mse': val_mse,
            'average_val_mse': avg_val_mse
        })

        # 早停判断
        if not early_stopper.is_continuable(model, avg_val_mse):
            print(f' Early stopping triggered at epoch {epoch_i + 1}. Best Validation MSE: {early_stopper.best_loss}')
            break

    # 保存epoch MSE结果
    if epoch_mse_results:
        epoch_df = pd.DataFrame()
        epoch_df['Epoch'] = [r['epoch'] for r in epoch_mse_results]
        for i in range(task_num):
            epoch_df[f'Task_{i + 1}_Val_MSE'] = [r['val_mse'][i] for r in epoch_mse_results]
        epoch_df['Average_Val_MSE'] = [r['average_val_mse'] for r in epoch_mse_results]

        epoch_csv_path = f'{save_dir}/{dataset_name}_{model_name}_epoch_mse.csv'
        if use_jda:
            epoch_csv_path = epoch_csv_path.replace('.csv', f'_jda_{source_dataset_name}.csv')
        epoch_df.to_csv(epoch_csv_path, index=False)
        print(f"\nFold 1 epoch MSE saved to: {epoch_csv_path}")

    # 加载最优模型，评估最终结果
    model.load_state_dict(torch.load(save_path))
    final_val_mse = test(model, val_loader, task_num, device,
                         save_csv_path=f'{save_dir}/{dataset_name}_{model_name}_final_val_predictions.csv')
    final_test_mse = test(model, test_loader, task_num, device,
                          save_csv_path=f'{save_dir}/{dataset_name}_{model_name}_final_test_predictions.csv')

    print(f"\n======== Final Results ========")
    print(f"Final Validation MSE: {final_val_mse} (Average: {np.mean(final_val_mse):.6f})")
    print(f"Final Test MSE: {final_test_mse} (Average: {np.mean(final_test_mse):.6f})")

    # 保存结果
    save_results_to_csv(final_val_mse, final_test_mse, args, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='AliExpress_NL',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US'])
    parser.add_argument('--dataset_path',
                        default=r'C:\Users\monster\PycharmProjects\pythonProject\Multitask-Recommendation-Library-main')
    parser.add_argument('--model_name', default='ple',
                        choices=['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm', 'metaheac'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--task_num', type=int, default=3)
    parser.add_argument('--expert_num', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--use_jda', action='store_false', default=True)
    parser.add_argument('--source_dataset_name', default='AliExpress_US',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US'])
    parser.add_argument('--jda_dim', type=int, default=50)
    parser.add_argument('--jda_kernel', default='linear', choices=['linear', 'rbf', 'poly'])
    parser.add_argument('--task_weights', nargs='+', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args.dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.use_jda,
         args.source_dataset_name,
         args.jda_dim,
         args.jda_kernel,
         args.task_weights,
         args.seed)