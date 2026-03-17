import os
import random
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.linalg import eigh
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset

from aliexpress import AliExpressDataset
from DNNple import PLEModel
from layer import *


BASE_SEED = 42
CUDA_DETERMINISTIC = True
CUDA_BENCHMARK = False


def set_seed(seed: int = BASE_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = CUDA_DETERMINISTIC
    torch.backends.cudnn.benchmark = CUDA_BENCHMARK
    print(f"Random seed fixed to: {seed}")


@dataclass
class Config:
    dataset_name: str
    dataset_path: str
    model_name: str
    epoch: int
    task_num: int
    expert_num: int
    learning_rate: float
    batch_size: int
    embed_dim: int
    weight_decay: float
    device: str
    save_dir: str

    use_jda: bool
    source_dataset_name: str
    jda_dim: int
    jda_kernel: str
    jda_fit_samples: int

    task_weights: list
    seed: int
    patience: int

    train_val_filename: str
    test_filename: str

    @staticmethod
    def from_args(args):
        if args.task_weights is None:
            task_weights = [0.35, 0.325, 0.325]
        else:
            task_weights = args.task_weights

        return Config(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            model_name=args.model_name,
            epoch=args.epoch,
            task_num=args.task_num,
            expert_num=args.expert_num,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            embed_dim=args.embed_dim,
            weight_decay=args.weight_decay,
            device=args.device,
            save_dir=args.save_dir,
            use_jda=args.use_jda,
            source_dataset_name=args.source_dataset_name,
            jda_dim=args.jda_dim,
            jda_kernel=args.jda_kernel,
            jda_fit_samples=args.jda_fit_samples,
            task_weights=task_weights,
            seed=args.seed if args.seed is not None else BASE_SEED,
            patience=args.patience,
            train_val_filename="train.csv",
            test_filename="test.csv",
        )


class JDA:
    def __init__(self, dim=20, kernel_type="linear", gamma=1.0, degree=2, coef0=1, mu=1.0):
        self.dim = dim
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.mu = mu
        self.eig_vectors = None
        self.scaler = None
        self.X_train_base = None

    def kernel(self, X, Y):
        if self.kernel_type == "linear":
            return np.dot(X, Y.T)
        elif self.kernel_type == "rbf":
            sqdist = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
            return np.exp(-self.gamma * sqdist)
        elif self.kernel_type == "poly":
            return (np.dot(X, Y.T) + self.coef0) ** self.degree
        else:
            raise ValueError("Unknown kernel type: " + self.kernel_type)

    def fit_transform(self, Xs, Xt, ys=None):
        X = np.vstack((Xs, Xt))
        n, m = Xs.shape[0], Xt.shape[0]
        self.X_train_base = X

        K = self.kernel(X, X)

        one_n = np.ones((n, n)) / n
        one_m = np.ones((m, m)) / m
        one_nm = np.ones((n, m)) / (n * m)

        if ys is not None:
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

        Sb = np.zeros((n + m, n + m))
        Sb[:n, :n] = K[:n, :n] @ one_n @ K[:n, :n].T
        Sb[:n, n:] = -K[:n, n:] @ one_nm.T @ K[n:, n:].T
        Sb[n:, :n] = -K[n:, :n] @ one_nm @ K[:n, :n].T
        Sb[n:, n:] = K[n:, n:] @ one_m @ K[n:, n:].T

        reg_Sw = Sw + self.mu * np.eye(n + m)
        try:
            eig_values, eig_vectors = eigh(Sb, reg_Sw)
        except np.linalg.LinAlgError:
            reg_Sw += 1e-6 * np.eye(n + m)
            eig_values, eig_vectors = eigh(Sb, reg_Sw)

        idx = np.argsort(eig_values)[::-1][:self.dim]
        self.eig_vectors = eig_vectors[:, idx]

        X_jda = K.dot(self.eig_vectors)
        Xs_jda = X_jda[:n, :]
        Xt_jda = X_jda[n:, :]

        self.scaler = StandardScaler()
        X_combined_jda = np.vstack((Xs_jda, Xt_jda))
        self.scaler.fit(X_combined_jda)

        Xs_jda = self.scaler.transform(Xs_jda)
        Xt_jda = self.scaler.transform(Xt_jda)
        return Xs_jda, Xt_jda

    def transform(self, X):
        if self.X_train_base is None:
            raise RuntimeError("JDA must be fit before transform().")
        K = self.kernel(X, self.X_train_base)
        X_jda = K.dot(self.eig_vectors)
        return self.scaler.transform(X_jda)


class NumericalDataWrapper(Dataset):
    def __init__(self, base_subset, numerical_data):
        self.base_subset = base_subset
        self.numerical_data = numerical_data
        assert len(self.base_subset) == len(self.numerical_data)

    def __len__(self):
        return len(self.base_subset)

    def __getitem__(self, index):
        _, labels = self.base_subset[index]
        return torch.tensor(self.numerical_data[index], dtype=torch.float32), labels


class EarlyStopper:
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_loss = float("inf")
        self.save_path = save_path

    def is_continuable(self, model, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"New best model saved with loss {loss:.6f}")
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def get_data_file_path(dataset_path, dataset_name, filename):
    return os.path.normpath(os.path.join(dataset_path, dataset_name, filename))


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")


def get_dataset(name, path, filename):
    full_path = get_data_file_path(path, name, filename)
    check_file_exists(full_path)
    if "AliExpress" in name:
        return AliExpressDataset(full_path)
    raise ValueError("unknown dataset name: " + name)


def get_model(name, numerical_num, task_num, expert_num, embed_dim):
    if name == "ple":
        return PLEModel(
            numerical_num,
            embed_dim=embed_dim,
            bottom_mlp_dims=(512, 256),
            tower_mlp_dims=(128, 64),
            task_num=task_num,
            shared_expert_num=int(expert_num / 2),
            specific_expert_num=int(expert_num / 2),
            dropout=0.5,
        )
    raise ValueError("unknown model name: " + name)


def split_train_val(n_samples, seed):
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_idx = int(0.9 * len(indices))
    return indices[:split_idx], indices[split_idx:]


def sample_source_train(source_dataset, seed):
    source_train_size = int(0.8 * len(source_dataset))
    source_indices = np.arange(len(source_dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(source_indices)
    source_train_indices = source_indices[:source_train_size]

    source_full_numerical = np.array(source_dataset.numerical_data)
    source_full_labels = np.array(source_dataset.labels)

    source_train_numerical = source_full_numerical[source_train_indices]
    source_train_labels = source_full_labels[source_train_indices, 0]
    return source_train_numerical, source_train_labels


def apply_jda_if_needed(cfg, scaler, train_scaled, val_scaled, test_scaled, source_train_numerical, source_train_labels):
    if not cfg.use_jda or source_train_numerical is None:
        return train_scaled, val_scaled, test_scaled, train_scaled.shape[1]

    source_train_scaled = scaler.transform(source_train_numerical)

    rng_jda = np.random.default_rng(cfg.seed)
    half_n = max(1, cfg.jda_fit_samples // 2)

    src_n = min(len(source_train_scaled), half_n)
    tgt_n = min(len(train_scaled), half_n)

    src_idx = rng_jda.choice(len(source_train_scaled), src_n, replace=False)
    tgt_idx = rng_jda.choice(len(train_scaled), tgt_n, replace=False)

    source_fit = source_train_scaled[src_idx]
    target_fit = train_scaled[tgt_idx]
    source_fit_labels = source_train_labels[src_idx]

    print(f"Applying JDA to sampled source TRAIN ({src_n}) + sampled target TRAIN ({tgt_n}) only...")

    jda = JDA(dim=cfg.jda_dim, kernel_type=cfg.jda_kernel, mu=1.0)
    jda.fit_transform(source_fit, target_fit, ys=source_fit_labels)

    train_scaled = jda.transform(train_scaled)
    val_scaled = jda.transform(val_scaled)
    test_scaled = jda.transform(test_scaled)

    return train_scaled, val_scaled, test_scaled, cfg.jda_dim


def build_dataloaders(cfg, train_val_dataset, test_dataset, train_indices, val_indices, train_scaled, val_scaled, test_scaled):
    train_base_subset = Subset(train_val_dataset, train_indices)
    val_base_subset = Subset(train_val_dataset, val_indices)
    test_base_subset = Subset(test_dataset, np.arange(len(test_dataset)))

    train_dataset = NumericalDataWrapper(train_base_subset, train_scaled)
    val_dataset = NumericalDataWrapper(val_base_subset, val_scaled)
    test_dataset_wrapper = NumericalDataWrapper(test_base_subset, test_scaled)

    train_generator = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=0, shuffle=True, generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=0, shuffle=False)
    test_loader = DataLoader(test_dataset_wrapper, batch_size=cfg.batch_size, num_workers=0, shuffle=False)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, optimizer, data_loader, criterion, device, task_weights, log_interval=100):
    model.train()
    total_loss = 0.0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)

    for step, (numerical_fields, labels) in enumerate(loader):
        numerical_fields = numerical_fields.to(device)
        labels = labels.to(device)

        outputs = model(numerical_fields)
        if isinstance(outputs, list):
            outputs = torch.stack(outputs, dim=1)

        loss_list = []
        for i in range(labels.size(1)):
            task_loss = criterion(outputs[:, i], labels[:, i].float())
            loss_list.append(task_loss * task_weights[i])

        loss = sum(loss_list) / len(loss_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0.0


def evaluate(model, data_loader, task_num, device, save_csv_path=None):
    model.eval()
    labels_dict = [[] for _ in range(task_num)]
    predicts_dict = [[] for _ in range(task_num)]

    with torch.no_grad():
        for numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            numerical_fields = numerical_fields.to(device)
            labels = labels.to(device)

            outputs = model(numerical_fields)
            if isinstance(outputs, list):
                outputs = torch.stack(outputs, dim=1)

            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(outputs[:, i].tolist())

    mse_results = []
    r2_results = []
    save_data = {}

    for i in range(task_num):
        y_true = np.array(labels_dict[i])
        y_pred = np.array(predicts_dict[i])

        mse = np.mean((y_true - y_pred) ** 2)
        r2 = r2_score(y_true, y_pred)

        mse_results.append(mse)
        r2_results.append(r2)

        save_data[f"task_{i + 1}_label"] = y_true
        save_data[f"task_{i + 1}_prediction"] = y_pred

    if save_csv_path is not None:
        pd.DataFrame(save_data).to_csv(save_csv_path, index=False)
        print(f"Predictions saved to: {save_csv_path}")

    return mse_results, r2_results


def save_results_to_csv(val_mse, test_mse, args, save_dir):
    val_mse_df = pd.DataFrame([val_mse], columns=[f"Task_{i + 1}_MSE" for i in range(args.task_num)])
    val_mse_df.index = [1]
    val_mse_df.index.name = "Fold"
    val_mse_df["Average_MSE"] = val_mse_df.mean(axis=1)
    val_mse_df.to_csv(f"{save_dir}/validation_mse_results.csv")

    test_mse_df = pd.DataFrame([test_mse], columns=[f"Task_{i + 1}_MSE" for i in range(args.task_num)])
    test_mse_df.index = [1]
    test_mse_df.index.name = "Fold"
    test_mse_df["Average_MSE"] = test_mse_df.mean(axis=1)
    test_mse_df.to_csv(f"{save_dir}/test_mse_results.csv")

    summary_df = pd.DataFrame({
        "Metric": ["Validation MSE (Fold 1)", "Test MSE (Fold 1)"],
        "Value": [np.mean(val_mse), np.mean(test_mse)]
    })
    summary_df.to_csv(f"{save_dir}/summary_results.csv", index=False)

    params_df = pd.DataFrame({
        "Parameter": list(vars(args).keys()),
        "Value": list(vars(args).values())
    })
    params_df.to_csv(f"{save_dir}/parameters.csv", index=False)

    print(
        f"\nResults saved to:\n"
        f" 1. Validation MSE: {save_dir}/validation_mse_results.csv\n"
        f" 2. Test MSE: {save_dir}/test_mse_results.csv\n"
        f" 3. Summary: {save_dir}/summary_results.csv\n"
        f" 4. Parameters: {save_dir}/parameters.csv"
    )


def main(cfg: Config, args):
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    print(f"Using task loss weights: {cfg.task_weights}")
    print(f"dataset={cfg.dataset_name}, model={cfg.model_name}, lr={cfg.learning_rate}, jda_dim={cfg.jda_dim}")

    train_val_dataset = get_dataset(cfg.dataset_name, cfg.dataset_path, cfg.train_val_filename)
    test_dataset = get_dataset(cfg.dataset_name, cfg.dataset_path, cfg.test_filename)

    print(f"Target Train/Val dataset size: {len(train_val_dataset)}")
    print(f"Target Test dataset size: {len(test_dataset)}")

    numerical_num = train_val_dataset.numerical_num
    source_train_numerical = None
    source_train_labels = None

    if cfg.use_jda and cfg.source_dataset_name:
        print(f"\n=== Enabling JDA Transfer Learning ===")
        print(f"Source domain: {cfg.source_dataset_name}, Target domain: {cfg.dataset_name}")
        source_dataset = get_dataset(cfg.source_dataset_name, cfg.dataset_path, cfg.train_val_filename)
        print(f"Source full dataset size: {len(source_dataset)}")
        source_train_numerical, source_train_labels = sample_source_train(source_dataset, cfg.seed)
        print(f"Source TRAIN subset size: {len(source_train_numerical)}")

    print("\n======== Preparing ========")
    train_indices, val_indices = split_train_val(len(train_val_dataset), cfg.seed)
    print(f"Train indices: {len(train_indices)}, Val indices: {len(val_indices)}")

    original_numerical_data = np.array(train_val_dataset.numerical_data)
    original_test_numerical_data = np.array(test_dataset.numerical_data)

    train_numerical = original_numerical_data[train_indices]
    val_numerical = original_numerical_data[val_indices]
    test_numerical = original_test_numerical_data

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_numerical)
    val_scaled = scaler.transform(val_numerical)
    test_scaled = scaler.transform(test_numerical)

    train_scaled, val_scaled, test_scaled, numerical_num = apply_jda_if_needed(
        cfg,
        scaler,
        train_scaled,
        val_scaled,
        test_scaled,
        source_train_numerical,
        source_train_labels,
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, train_val_dataset, test_dataset, train_indices, val_indices,
        train_scaled, val_scaled, test_scaled
    )

    set_seed(cfg.seed)
    model = get_model(cfg.model_name, numerical_num, cfg.task_num, cfg.expert_num, cfg.embed_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    save_path = f"{cfg.save_dir}/{cfg.dataset_name}_{cfg.model_name}.pt"
    if cfg.use_jda:
        save_path = save_path.replace(".pt", f"_jda_{cfg.source_dataset_name}.pt")

    early_stopper = EarlyStopper(num_trials=cfg.patience, save_path=save_path)
    epoch_mse_results = []

    for epoch_i in range(cfg.epoch):
        print(f"\n{epoch_i + 1}/{cfg.epoch}")
        train_one_epoch(model, optimizer, train_loader, criterion, device, cfg.task_weights)

        val_mse, _ = evaluate(
            model,
            val_loader,
            cfg.task_num,
            device,
            save_csv_path=f"{cfg.save_dir}/{cfg.dataset_name}_{cfg.model_name}_val_predictions_epoch{epoch_i + 1}.csv"
        )

        print(f"Epoch {epoch_i + 1} - Validation MSE: {val_mse}")
        avg_val_mse = sum(val_mse[i] * cfg.task_weights[i] for i in range(cfg.task_num))

        epoch_mse_results.append({
            "epoch": epoch_i + 1,
            "val_mse": val_mse,
            "average_val_mse": avg_val_mse
        })

        if not early_stopper.is_continuable(model, avg_val_mse):
            print(f"Early stopping triggered at epoch {epoch_i + 1}. Best Validation MSE: {early_stopper.best_loss}")
            break

    epoch_df = pd.DataFrame({
        "Epoch": [r["epoch"] for r in epoch_mse_results],
        "Average_Val_MSE": [r["average_val_mse"] for r in epoch_mse_results]
    })
    for i in range(cfg.task_num):
        epoch_df[f"Task_{i + 1}_Val_MSE"] = [r["val_mse"][i] for r in epoch_mse_results]

    epoch_csv_path = f"{cfg.save_dir}/{cfg.dataset_name}_{cfg.model_name}_epoch_mse.csv"
    if cfg.use_jda:
        epoch_csv_path = epoch_csv_path.replace(".csv", f"_jda_{cfg.source_dataset_name}.csv")
    epoch_df.to_csv(epoch_csv_path, index=False)
    print(f"\nEpoch MSE saved to: {epoch_csv_path}")

    model.load_state_dict(torch.load(save_path))
    final_val_mse, final_val_r2 = evaluate(
        model, val_loader, cfg.task_num, device,
        save_csv_path=f"{cfg.save_dir}/{cfg.dataset_name}_{cfg.model_name}_final_val_predictions.csv"
    )
    final_test_mse, final_test_r2 = evaluate(
        model, test_loader, cfg.task_num, device,
        save_csv_path=f"{cfg.save_dir}/{cfg.dataset_name}_{cfg.model_name}_final_test_predictions.csv"
    )

    print(f"\n======== Final Results ========")
    print(f"Final Validation MSE: {final_val_mse} (Average: {np.mean(final_val_mse):.6f})")
    print(f"Final Test MSE: {final_test_mse} (Average: {np.mean(final_test_mse):.6f})")
    print(f"Final Test R2: {final_test_r2} (Average: {np.mean(final_test_r2):.6f})")

    save_results_to_csv(final_val_mse, final_test_mse, args, cfg.save_dir)

    print(f"task_weights={cfg.task_weights}")
    print(
        f"task_num={cfg.task_num}, expert_num={cfg.expert_num}, epoch={cfg.epoch}, "
        f"learning_rate={cfg.learning_rate}, jda={cfg.jda_dim}, "
        f"batch_size={cfg.batch_size}, embed_dim={cfg.embed_dim}, "
        f"weight_decay={cfg.weight_decay}, device={cfg.device}, "
        f"jda_fit_samples={cfg.jda_fit_samples}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="AliExpress_NL")
    parser.add_argument("--dataset_path", default="/content/drive/MyDrive/jiajuanji")
    parser.add_argument("--model_name", default="ple")

    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--task_num", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--save_dir", default="chkpt")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_jda", action="store_false", default=True)
    parser.add_argument("--source_dataset_name", default="AliExpress_NL")
    parser.add_argument("--jda_dim", type=int, default=20)
    parser.add_argument("--jda_kernel", default="linear", choices=["linear", "rbf", "poly"])
    parser.add_argument("--jda_fit_samples", type=int, default=300)

    parser.add_argument("--task_weights", nargs="+", type=float, default=None)

    args = parser.parse_args()
    cfg = Config.from_args(args)
    main(cfg, args)


####运行命令行###
#!python /kaggle/working/m_task/main.py \
#  --dataset_name AliExpress_NL \
#  --dataset_path /kaggle/working/m_task \
#  --device cuda \
#  --learning_rate 0.001 \
#  --expert_num 2 \
#  --embed_dim 16 \
#  --jda_dim 20 \
#  --jda_fit_samples 300 \
#  --task_weights 0.35 0.325 0.325 \
#  --save_dir /kaggle/working/chkpt