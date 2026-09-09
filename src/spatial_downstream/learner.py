import os
import copy
import numpy as np
import torch
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm

from common_utils import STFT, AddChToBatch


class Learner(ABC):
    """
    Abstract base class for model training, validation, checkpointing, and evaluation.
    """
    def __init__(self, model):
        self.model = model
        self.max_score = -np.inf
        self.early_stop_counter = 0
        self.use_amp = False
        self.start_epoch = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()

    def mul_gpu(self):
        """Enable DataParallel across available GPUs."""
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def cuda(self):
        self.model.cuda()
        self.device = "cuda"

    def cpu(self):
        self.model.cpu()
        self.device = "cpu"

    def amp(self):
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @abstractmethod
    def data_preprocess(self, mic_sig_batch=None, gt_batch=None):
        pass

    @abstractmethod
    def loss(self, pred_batch, gt_batch):
        pass

    @abstractmethod
    def evaluate(self, pred_batch, gt_batch):
        pass

    def train_epoch(self, dataset, lr=0.0001, epoch=None, return_metric=False):
        """Train the model for one epoch over dataset."""
        avg_loss = 0.0
        avg_beta = 0.99
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
        loss = 0.0
        metric = 0.0 if return_metric else None

        optimizer.zero_grad()
        pbar = tqdm(enumerate(dataset), total=len(dataset), leave=False)

        for batch_idx, data in pbar:
            mic_sig_batch = data[0]
            gt_batch = data[1] if len(data) > 1 else None

            if epoch is not None:
                pbar.set_description(f'Epoch {epoch}')

            in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred_batch, _, _ = self.model(in_batch)
                loss_batch = self.loss(pred_batch=pred_batch, gt_batch=gt_batch)

            if self.use_amp:
                self.scaler.scale(loss_batch).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss_batch.backward()
                optimizer.step()

            optimizer.zero_grad()

            loss_val = loss_batch.item()
            avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss_val
            pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (batch_idx + 1)))

            loss += loss_val

            if return_metric and gt_batch is not None:
                metric_batch = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch)
                metric += metric_batch.item()

        loss /= max(1, len(pbar))
        if return_metric:
            metric /= max(1, len(pbar))
            return loss, metric
        return loss

    def test_epoch(self, dataset, return_metric=False):
        """Test/evaluate the model over dataset without gradient updates."""
        self.model.eval()
        loss = 0.0
        metric = 0.0 if return_metric else None

        with torch.no_grad():
            for data in dataset:
                mic_sig_batch = data[0]
                gt_batch = data[1] if len(data) > 1 else None

                in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred_batch, _, _ = self.model(in_batch)
                    loss_batch = self.loss(pred_batch=pred_batch, gt_batch=gt_batch)

                loss += loss_batch.item()

                if return_metric and gt_batch is not None:
                    metric_batch = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch)
                    metric += metric_batch.item()

            loss /= max(1, len(dataset))
            if return_metric:
                metric /= max(1, len(dataset))
                return loss, metric
            return loss

    def smooth_data(self, data_list, alpha=0.8):
        data_smooth_list = [copy.deepcopy(data_list[0])]
        data_smooth = copy.deepcopy(data_list[0])
        for idx in range(1, len(data_list)):
            data_smooth = alpha * data_smooth + (1 - alpha) * data_list[idx]
            data_smooth_list.append(data_smooth)
        return data_smooth_list

    def early_stopping(self, current_score, patience=5):
        if current_score >= self.max_score:
            self.max_score = current_score
            self.early_stop_counter = 0
            stop_flag = False
            is_best_epoch = True
        else:
            self.early_stop_counter += 1
            stop_flag = self.early_stop_counter >= patience
            is_best_epoch = False
        return stop_flag, is_best_epoch

    def save_checkpoint(self, epoch, checkpoints_dir, is_best_epoch=False, save_extra_hist=False):
        os.makedirs(checkpoints_dir, exist_ok=True)
        model_to_save = getattr(self.model, 'module', self.model)
        state_dict = {
            "epoch": epoch,
            "max_score": self.max_score,
            "model": model_to_save.state_dict()
        }
        if self.use_amp:
            state_dict["scaler"] = self.scaler.state_dict()

        torch.save(state_dict, os.path.join(checkpoints_dir, "latest_model.tar"))
        if save_extra_hist:
            torch.save(state_dict, os.path.join(checkpoints_dir, f"model{epoch}.tar"))
        if is_best_epoch:
            torch.save(state_dict, os.path.join(checkpoints_dir, "best_model.tar"))

    def resume_checkpoint(self, checkpoints_dir, from_latest=True, as_all_state=True):
        model_path = os.path.join(checkpoints_dir, "latest_model.tar" if from_latest else "best_model.tar")
        assert os.path.exists(model_path), f"{model_path} does not exist."
        checkpoint = torch.load(model_path, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.max_score = checkpoint.get("max_score", -np.inf)

        model_target = getattr(self.model, 'module', self.model)
        if as_all_state:
            model_target.load_state_dict(checkpoint["model"])
        print(f"Resumed model checkpoint from {model_path} at epoch {self.start_epoch}.")

    def load_checkpoint_best(self, checkpoints_dir, param_frozen=False):
        best_model_path = os.path.join(checkpoints_dir, "best_model.tar")
        assert os.path.exists(best_model_path), f"{best_model_path} does not exist."
        checkpoint = torch.load(best_model_path, map_location=self.device)
        epoch = checkpoint["epoch"]

        model_target = getattr(self.model, 'module', self.model)
        model_target.load_state_dict(checkpoint["model"], strict=False)

        if param_frozen:
            for key, value in model_target.named_parameters():
                if not key.startswith("mlp_head"):
                    value.requires_grad = False
            print("Frozen pre-trained parameters for linear evaluation.")

        print(f"Loaded best checkpoint from epoch {epoch}.")
        return epoch

    def ensembling(self, checkpoints_dir, epochs):
        nepoch_ave = len(epochs)
        state_dicts = {}
        model_target = getattr(self.model, 'module', self.model)

        for idx, epoch in enumerate(epochs):
            model_path = os.path.join(checkpoints_dir, f"model{epoch}.tar")
            if not os.path.exists(model_path):
                continue
            checkpoint = torch.load(model_path, map_location=self.device)
            sd = checkpoint["model"]
            for key in sd.keys():
                if idx == 0:
                    state_dicts[key] = sd[key] * (1.0 / nepoch_ave)
                else:
                    state_dicts[key] += sd[key] * (1.0 / nepoch_ave)

        if len(state_dicts) > 0:
            model_target.load_state_dict(state_dicts)
            torch.save({"epoch": epochs, "model": model_target.state_dict()}, os.path.join(checkpoints_dir, "ensemble_model.tar"))
            print(f"Ensemble model saved across epochs {epochs}.")

    def remove_checkpoint_epochs(self, checkpoints_dir, epochs):
        for epoch in epochs:
            model_path = os.path.join(checkpoints_dir, f"model{epoch}.tar")
            if os.path.exists(model_path):
                os.remove(model_path)


class STFTLearner(Learner):
    """
    Learner subclass for multi-channel STFT inputs.
    Handles online STFT computation, magnitude normalization, and spatial metric evaluation.
    """
    def __init__(self, model, win_len=512, win_shift_ratio=0.5, nfft=512, fre_used_ratio=1.0, fs=16000, task='TDOA', ch_mode='M'):
        super().__init__(model)
        self.ch_mode = ch_mode
        self.task = task
        self.stft = STFT(win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft)
        self.fre_range_used = range(1, int(nfft / 2 * fre_used_ratio) + 1)
        self.addbatch = AddChToBatch(ch_mode=ch_mode)

    def data_preprocess(self, mic_sig_batch=None, gt_batch=None, eps=1e-6):
        data = []
        if mic_sig_batch is not None:
            mic_sig_batch = mic_sig_batch.to(self.device) # (nb, nsample, nch)
            stft = self.stft(signal=mic_sig_batch)       # (nb, nf, nt, nch)
            stft = stft.permute(0, 3, 1, 2)              # (nb, nch, nf, nt)

            # Magnitude normalization using Left channel mean
            mag = torch.abs(stft[:, 0:1, :, :])
            mean_val = torch.mean(mag.reshape(mag.shape[0], -1), dim=1)
            mean_val = mean_val[:, None, None, None].expand_as(mag)
            stft = stft / (mean_val + eps)

            stft_rebatch = self.addbatch(stft)           # (nb, 2_ch, nf, nt)
            reim_rebatch = torch.view_as_real(stft_rebatch) # (nb, 2_ch, nf, nt, 2)
            reim_rebatch = reim_rebatch[:, :, self.fre_range_used, :, :]

            data.append(reim_rebatch)

        if gt_batch is not None:
            if isinstance(gt_batch, dict):
                target = gt_batch[self.task].to(self.device)
            else:
                target = gt_batch.to(self.device)

            target = self.get_tar_batch(target)
            data.append(target)

        return data

    def get_tar_batch(self, gt_batch):
        if gt_batch.dim() == 1:
            gt_batch = gt_batch.unsqueeze(1)

        if self.task == 'TDOA':
            # TDOA target in seconds scaled by sampling rate (16000) to get sample difference
            tar_batch = gt_batch * 16000.0
        elif self.task in ['DRR', 'C50', 'T60', 'ABS']:
            tar_batch = gt_batch
        else:
            raise ValueError(f"Task {self.task} not supported in STFTLearner")

        return tar_batch

    def loss(self, pred_batch, gt_batch):
        return torch.nn.functional.mse_loss(pred_batch.contiguous(), gt_batch.contiguous().detach())

    def evaluate(self, pred_batch, gt_batch):
        """Evaluates Mean Absolute Error (MAE)."""
        return torch.mean(torch.abs(pred_batch.contiguous().detach() - gt_batch.contiguous().detach()))
