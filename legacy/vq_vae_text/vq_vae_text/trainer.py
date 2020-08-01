import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm


class AbstractCallback:
    _callback_function_names = ['on_training_start', 'on_training_end', 'on_batch_start', 'on_batch_end']

    def on_training_start(self, trainer, epoch=None):
        pass

    def on_training_end(self, trainer, epoch=None):
        pass

    def on_batch_start(self, trainer, batch_idx, epoch=None):
        pass

    def on_batch_end(self, trainer, batch_idx, epoch=None):
        pass


class BatchIntervalCallback(AbstractCallback):
    def __init__(self, batch_interval):
        self.batch_interval = batch_interval

    def on_batch_start(self, trainer, batch_idx, epoch=None):
        if (batch_idx + 1) % self.batch_interval == 0:
            self.on_batch_interval_start(trainer, batch_idx, epoch)

    def on_batch_end(self, trainer, batch_idx, epoch=None):
        if (batch_idx + 1) % self.batch_interval == 0:
            self.on_batch_interval_start(trainer, batch_idx, epoch)

    def on_batch_interval_start(self, trainer, batch_idx, epoch=None):
        pass

    def on_batch_interval_end(self, trainer, batch_idx, epoch=None):
        pass


class CallbackList(AbstractCallback):
    def __init__(self, callbacks):
        self.callbacks = callbacks
        for callback_function_name in self._callback_function_names:
            setattr(self, callback_function_name, self._callback_proxy(callback_function_name))

    def _callback_proxy(self, target_name):
        def proxy(*args, **kwargs):
            for callback in self.callbacks:
                getattr(callback, target_name)(*args, **kwargs)
        return proxy


class SummaryWriterCallback(AbstractCallback):
    def __init__(self, summary_writer):
        self.writer = summary_writer

    def on_batch_end(self, trainer, batch_idx, epoch=None):
        loss_dict = trainer.latest_loss_dict
        if loss_dict:
            for tag, val in loss_dict.items():
                self.writer.add_scalar(tag, val, global_step=trainer.global_step)


class ModelTrainer:
    def __init__(self, model, optimizer, device, scheduler=None, callbacks=None, global_step=0):
        self.model = model

        self.module = model
        if isinstance(model, nn.DataParallel):
            self.module = model.module

        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.callback = CallbackList(callbacks if callbacks else [])
        self.latest_loss_dict = None
        self.global_step = global_step

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self, batches, epoch=None):
        self.callback.on_training_start(self, epoch)

        self.model.train()

        losses = []

        def build_description(step, loss_dict):
            components = [f'[{step}]', ', '.join(f'{name}: {val:.4f}' for name, val in loss_dict.items())]
            if epoch is not None:
                components.insert(0, f'[EPOCH {epoch}]')
            return ' '.join(components)

        total_batches = 0
        total_samples = 0

        with tqdm(batches, position=0, leave=True) as pbar:
            for batch_idx, batch in enumerate(pbar):
                self.callback.on_batch_start(self, batch_idx, epoch)

                self.optimizer.zero_grad()

                batch = batch.to(self.device)

                outputs = self.model(batch)

                loss = self.module.loss_function(outputs, batch)
                loss.backward()

                self.optimizer.step()

                losses.append(float(loss.detach()))

                total_batches += 1
                total_samples += len(batch)

                self.latest_loss_dict = dict(self.module.latest_losses())

                if self.scheduler is not None:
                    self.scheduler.step()
                    self.latest_loss_dict['lr'] = self.scheduler.get_lr()[0]

                pbar.update(1)
                pbar.set_description(build_description(total_samples, self.latest_loss_dict))

                self.global_step += 1

                self.callback.on_batch_end(self, batch_idx, epoch)

        self.callback.on_training_end(self, epoch)

        return sum(losses) / total_batches

    def evaluate(self, batches):
        self.model.eval()

        losses = []
        steps = 0

        with torch.no_grad():
            for batch in batches:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.module.loss_function(outputs, batch).detach()
                losses.append(loss.item())
                steps += 1
        return sum(losses) / steps
