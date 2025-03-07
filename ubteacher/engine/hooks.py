# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

import torch
import logging
import operator
import math
import numpy as np
from contextlib import contextmanager
from fvcore.common.checkpoint import Checkpointer


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, model_output, model_name="", ignore_burnupstep=False):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._model_output = model_output
        self._model_name = model_name
        self._ignore_burnupstep = ignore_burnupstep

    def _do_loss_eval(self):
        record_acc_dict = {}
        with inference_context(self._model), torch.no_grad():
            for _, inputs in enumerate(self._data_loader):
                record_dict = self._get_loss(inputs, self._model)
                # accumulate the losses
                for loss_type in record_dict.keys():
                    if loss_type not in record_acc_dict.keys():
                        record_acc_dict[loss_type] = record_dict[loss_type]
                    else:
                        record_acc_dict[loss_type] += record_dict[loss_type]
            # average
            for loss_type in record_acc_dict.keys():
                record_acc_dict[loss_type] = record_acc_dict[loss_type] / len(
                    self._data_loader
                )

            # divide loss and other metrics
            loss_acc_dict = {}
            for key in record_acc_dict.keys():
                if key[:4] == "loss":
                    loss_acc_dict[key] = record_acc_dict[key]

            # only output the results of major node
            if comm.is_main_process():
                total_losses_reduced = sum(loss for loss in loss_acc_dict.values())
                self.trainer.storage.put_scalar(
                    "val_total_loss_val" + self._model_name, total_losses_reduced, smoothing_hint=False
                )

                record_acc_dict = {
                    "val_" + k + self._model_name: record_acc_dict[k]
                    for k in record_acc_dict.keys()
                }

                if len(record_acc_dict) > 1:
                    self.trainer.storage.put_scalars(**record_acc_dict, smoothing_hint=False)

    def _get_loss(self, data, model):
        if self._model_output == "loss_only":
            record_dict = model(data)

        elif self._model_output == "loss_proposal":
            record_dict, _, _, _ = model(data, branch="val_loss", val_mode=True)

        elif self._model_output == "meanteacher":
            record_dict, _, _, _, _ = model(data)

        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in record_dict.items()
        }

        return metrics_dict

    def _write_losses(self, metrics_dict):
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        comm.synchronize()
        all_metrics_dict = comm.gather(metrics_dict, dst=0)

        if comm.is_main_process():
            # average the rest metrics
            metrics_dict = {
                "val_" + k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.trainer.storage.put_scalar("val_total_loss_val", total_losses_reduced, smoothing_hint=False)
            if len(metrics_dict) > 1:
                self.trainer.storage.put_scalars(**metrics_dict)

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.trainer.iter, loss_dict
                )
            )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            if self._ignore_burnupstep and (self.trainer.iter > self.trainer.cfg.SEMISUPNET.BURN_UP_STEP):
                self._do_loss_eval()

            elif self._ignore_burnupstep is False:
                self._do_loss_eval()

# save best performance model
class BestCheckpointer(HookBase):
    """
    Checkpoints best weights based off given metric.

    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """

    def __init__(
        self,
        eval_period: int,
        checkpointer: Checkpointer,
        val_metric: str,
        mode: str = "max",
        file_prefix: str = "model_best",
    ) -> None:
        """
        Args:
            eval_period (int): the period `EvalHook` is set to run.
            checkpointer: the checkpointer object used to save checkpoints.
            val_metric (str): validation metric to track for best checkpoint, e.g. "bbox/AP50"
            mode (str): one of {'max', 'min'}. controls whether the chosen val metric should be
                maximized or minimized, e.g. for "bbox/AP50" it should be "max"
            file_prefix (str): the prefix of checkpoint's filename, defaults to "model_best"
        """
        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self._val_metric = val_metric
        assert mode in [
            "max",
            "min",
        ], f'Mode "{mode}" to `BestCheckpointer` is unknown. It should be one of {"max", "min"}.'
        if mode == "max":
            self._compare = operator.gt
        else:
            self._compare = operator.lt
        self._checkpointer = checkpointer
        self._file_prefix = file_prefix
        self.best_metric = None
        self.best_iter = None

    def _update_best(self, val, iteration):
        if math.isnan(val) or math.isinf(val):
            return False
        self.best_metric = val
        self.best_iter = iteration
        return True

    def _best_checking(self):
        # 'bbox/AP : teacher model performance 
        metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        if metric_tuple is None:
            self._logger.warning(
                f"Given val metric {self._val_metric} does not seem to be computed/stored."
                "Will not be checkpointing based on it."
            )
            return
        else:
            latest_metric, metric_iter = metric_tuple

        if self.best_metric is None:
            if self._update_best(latest_metric, metric_iter):
                additional_state = {"iteration": metric_iter}
                self._checkpointer.save(f"{self._file_prefix}", **additional_state)
                self._logger.info(
                    f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps"
                )
        elif self._compare(latest_metric, self.best_metric):
            additional_state = {"iteration": metric_iter}
            self._checkpointer.save(f"{self._file_prefix}", **additional_state)
            self._logger.info(
                f"Saved best model as latest eval score for {self._val_metric} is "
                f"{latest_metric:0.5f}, better than last best score "
                f"{self.best_metric:0.5f} @ iteration {self.best_iter}."
            )
            self._update_best(latest_metric, metric_iter)
        else:
            self._logger.info(
                f"Not saving as latest eval score for {self._val_metric} is {latest_metric:0.5f}, "
                f"not better than best score {self.best_metric:0.5f} @ iteration {self.best_iter}."
            )

    def after_step(self):
        # same conditions as `EvalHook`
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self._best_checking()

    def after_train(self):
        # same conditions as `EvalHook`
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._best_checking()


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)