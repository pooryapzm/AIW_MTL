"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""
from random import randint, random

import onmt.inputters as inputters
import torch
import onmt.utils

from onmt.utils.logging import logger

from copy import deepcopy
from onmt.meta_modules import to_var

from onmt.extended_torchtext.batch import Batch
import numpy as np
import math

# poorya
from onmt.translate.translator import build_translator_in_training


def build_trainer(opt, device_id, models_list, fields_list,
                  optim_list, data_type, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    # train_loss = onmt.utils.loss.build_loss_compute(
    #     model, fields["tgt"], opt)
    # valid_loss = onmt.utils.loss.build_loss_compute(
    #     model, fields["tgt"], opt, train=False)
    num_tasks = len(models_list)
    train_loss_list = []
    valid_loss_list = []
    for task_id in range(num_tasks):
        train_loss = onmt.utils.loss.build_loss_compute(
            models_list[task_id], fields_list[task_id]["tgt"], opt)
        valid_loss = onmt.utils.loss.build_loss_compute(
            models_list[task_id], fields_list[task_id]["tgt"], opt, train=False)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
    # Poorya

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = opt.world_size
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    mtl_schedule = opt.mtl_schedule

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(opt, models_list, fields_list, train_loss_list, valid_loss_list, optim_list, trunc_size,
                           shard_size, data_type, norm_method,
                           grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver, num_tasks=num_tasks,
                           mtl_schedule=mtl_schedule, model_opt=opt)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, opt, models_list, fields_list, train_loss_list, valid_loss_list, optim_list,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None, num_tasks=1,
                 mtl_schedule=0, model_opt=None):
        # Basic attributes.
        self.models_list = models_list
        self.train_loss_list = train_loss_list
        self.valid_loss_list = valid_loss_list
        self.optim_list = optim_list
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        # poorya
        self.model_opt = opt
        self.fields_list = fields_list
        self.mtl_schedule = mtl_schedule
        self.num_tasks = num_tasks
        self.model_opt = model_opt

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert (self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        # self.model.train()
        for task_id in range(num_tasks):
            self.models_list[task_id].train()

        if self.model_opt.meta_batch_weighting:
            self.meta_losses_list = self.create_meta_models_loss()

    def create_meta_models_loss(self):
        meta_losses = []
        for task_id in range(self.num_tasks):
            meta_loss = onmt.utils.loss.build_loss_compute(
                self.models_list[task_id], self.fields_list[task_id]["tgt"], self.model_opt)
            meta_losses.append(meta_loss)
        return meta_losses

    def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps, meta_valid_iter_fct=None):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """

        logger.info('Start training...')

        is_meta_report = False
        if self.model_opt.meta_output != '' and self.model_opt.meta_batch_weighting \
                and self.model_opt.meta_mtl_concat_mode != 'none':
            is_meta_report = True
            meta_output = open(self.model_opt.meta_output, 'w')
            meta_step = 0
            meta_stats = [0 for _ in range(self.num_tasks)]

        def meta_report(meta_stats, meta_steps):
            # compute stat average
            meta_stats[:] = [stat / meta_steps for stat in meta_stats]
            meta_report_line = ""
            for i in range(len(meta_stats)):
                meta_report_line += str(meta_stats[i]) + "  "
            meta_output.write("Step: %i, Batch: %i, Stats: %s \n" % (step, int(step / valid_steps), meta_report_line))
            meta_output.flush()

        best_valid_loss = 1000000
        step = self.optim_list[0]._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        # train_iter = train_iter_fct()
        train_iter_list = [train_iter_fct(task_id) for task_id in range(self.num_tasks)]
        train_generator_list = [train_iter_list[task_id].__iter__() for task_id in range(self.num_tasks)]

        # Load meta valid datasets when it's needed!!!
        meta_generator_list = [None for _ in range(self.num_tasks)]

        def _next_meta_valid_batch(task_id=0):
            meta_id = 0 if self.model_opt.meta_batch_weighting_mode < 10 else task_id
            # create generator for the first time
            if meta_generator_list[meta_id] is None:
                meta_valid_iter = meta_valid_iter_fct(meta_id, is_log=True)
                meta_generator_list[meta_id] = meta_valid_iter.__iter__()
            # Fetch the next batch
            try:
                meta_valid_batch = next(meta_generator_list[meta_id])
            except:
                # meta_generator_list[task_id] = meta_valid_iter_fct(0)
                meta_valid_iter = meta_valid_iter_fct(meta_id, is_log=False)
                meta_generator_list[meta_id] = meta_valid_iter.__iter__()
                meta_valid_batch = next(meta_generator_list[meta_id])

            return meta_valid_batch

        total_stats_list = [onmt.utils.Statistics() for _ in range(self.num_tasks)]
        report_stats_list = [onmt.utils.Statistics() for _ in range(self.num_tasks)]

        self._start_report_manager(start_time=total_stats_list[0].start_time)
        last_epoch_loss = [1000000 for i in range(self.num_tasks)]

        def gather_valid_stats(task_id, report=True):
            valid_iter = valid_iter_fct(task_id)
            valid_stats = self.validate(valid_iter, task_id)
            valid_stats = self._maybe_gather_stats(valid_stats)
            if report:
                self._report_step(self.optim_list[task_id].learning_rate,
                                  step, valid_stats=valid_stats, task_id=task_id)
            return valid_stats

        while step <= train_steps:

            reduce_counter = 0
            task_list = self._scheduler(self.mtl_schedule, step, valid_steps)
            keep_batchs = []
            for task_id in task_list:
                try:
                    batch = next(train_generator_list[task_id])
                except:
                    logger.info("(Task %d) We completed an epoch " % task_id)
                    train_iter_list[task_id] = train_iter_fct(task_id)
                    train_generator_list[task_id] = train_iter_list[task_id].__iter__()
                    batch = next(train_generator_list[task_id])
                    if self.model_opt.decay_method == "performance":
                        valid_stats = gather_valid_stats(task_id, report=False)
                        if valid_stats.loss > last_epoch_loss[task_id]:
                            self.optim_list[task_id].set_rate(self.optim_list[task_id].learning_rate / 2)
                            logger.info(
                                "(Task %d) LR is decreased (%f)" % (task_id, self.optim_list[task_id].learning_rate))
                        else:
                            logger.info(
                                "(Task %d) LR is not changed (%f)" % (task_id, self.optim_list[task_id].learning_rate))
                        last_epoch_loss[task_id] = valid_stats.loss
                true_batchs.append(batch)
                if self.norm_method == "tokens":
                    num_tokens = batch.tgt[1:].ne(
                        self.train_loss_list[task_id].padding_idx).sum()
                    normalization += num_tokens.item()
                elif self.norm_method == "sents":
                    normalization += batch.batch_size
                elif self.norm_method == "none":
                    normalization = 1

                #
                if self.model_opt.meta_batch_weighting and self.model_opt.meta_mtl_concat_mode != 'none':
                    keep_batchs.append(batch)
                else:
                    if self.model_opt.meta_batch_weighting:
                        meta_valid_batch = _next_meta_valid_batch(
                            task_id) if self.model_opt.meta_batch_weighting else None
                        self._meta_gradient_accumulation(
                            true_batchs, total_stats_list[task_id],
                            report_stats_list[task_id], task_id, meta_valid_batch, normalization)
                    else:
                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats_list[task_id],
                            report_stats_list[task_id], task_id)

                    report_stats = self._maybe_report_training(
                        step, train_steps,
                        self.optim_list[task_id].learning_rate,
                        report_stats_list[task_id], task_id)

                true_batchs = []
                accum = 0
                normalization = 0

            if self.model_opt.meta_batch_weighting and self.model_opt.meta_mtl_concat_mode != 'none':
                is_concat = False
                if self.model_opt.meta_mtl_concat_mode == "concat":
                    is_concat = True
                meta_valid_batch = _next_meta_valid_batch()
                batch_weight_list = self._meta_gradient_accumulation_batch_list(
                    keep_batchs, task_list, total_stats_list,
                    report_stats_list, meta_valid_batch, is_concat=is_concat)
                for task_id in task_list:
                    report_stats = self._maybe_report_training(
                        step, train_steps,
                        self.optim_list[task_id].learning_rate,
                        report_stats_list[task_id], task_id)
                if is_meta_report:
                    for task_id in task_list:
                        batch_weights = batch_weight_list[task_id]
                        meta_stats[task_id] += np.sum(batch_weights)
                    meta_step += 1
                    if meta_step == self.model_opt.meta_step_report:
                        meta_report(meta_stats, meta_step)
                        meta_stats = [0 for _ in range(self.num_tasks)]
                        meta_step = 0

            if (step % int(valid_steps/3) == 0):
                for task_id in range(self.num_tasks):
                    valid_stats = gather_valid_stats(task_id, report=True)
                    # Poorya:
                    if task_id == 0:
                        if valid_stats.loss < best_valid_loss:
                            best_valid_loss = valid_stats.loss
                            self._maybe_save(step)

            step += 1
            if step > train_steps:
                break

        return total_stats_list[0]

    def _flip(self, p):
        return True if random() < p else False

    def _scheduler(self, mode, step, valid_step, alpha=.5):
        task_list = []
        if mode == 0:
            task_list.append(0)
            if self.num_tasks > 1:
                task_list.append(randint(1, self.num_tasks - 1))
        if mode == 1:
            for task_id in range(self.num_tasks):
                task_list.append(task_id)
        # schedules more 10-12 are for replicating TACL paper
        if mode == 10:
            p_q_t = alpha
            if self._flip(p_q_t):
                task_list.append(0)
            else:
                task_list.append(randint(1, self.num_tasks - 1))
        if mode == 11:
            t = step / valid_step
            tmp = -1.0 * alpha * t
            p_q_t = 1 - math.exp(tmp)
            if self._flip(p_q_t):
                task_list.append(0)
            else:
                task_list.append(randint(1, self.num_tasks - 1))
        if mode == 12:
            t = step / valid_step
            tmp = -1.0 * alpha * t
            p_q_t = 1 / (1 + math.exp(tmp))
            if self._flip(p_q_t):
                task_list.append(0)
            else:
                task_list.append(randint(1, self.num_tasks - 1))

        return task_list

    def validate(self, valid_iter, task_id=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.models_list[task_id].eval()

        stats = onmt.utils.Statistics()

        for batch in valid_iter:
            src = inputters.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            elif self.data_type == 'audio':
                src_lengths = batch.src_lengths
            else:
                src_lengths = None

            tgt = inputters.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns = self.models_list[task_id](src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss_list[task_id].monolithic_compute_loss(
                batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        if self.models_list[task_id].decoder.state is not None:
            self.models_list[task_id].decoder.detach_state()
        # Set model back to training mode.
        self.models_list[task_id].train()

        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats, task_id=0):
        if self.grad_accum_count > 1:
            self.models_list[task_id].zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            # dec_state = None
            src = inputters.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum().item()
            elif self.data_type == 'audio':
                src_lengths = batch.src_lengths
            else:
                src_lengths = None

            tgt_outer = inputters.make_features(batch, 'tgt')

            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.models_list[task_id].zero_grad()
                outputs, attns = \
                    self.models_list[task_id](src, tgt, src_lengths)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss_list[task_id].sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization)
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.models_list[task_id].parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim_list[task_id].step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.models_list[task_id].decoder.state is not None:
                    self.models_list[task_id].decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.models_list[task_id].parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim_list[task_id].step()

    def _meta_gradient_accumulation(self, true_batchs, total_stats,
                                    report_stats, task_id, meta_valid_batch, normalization):

        for batch in true_batchs:
            # initialize the meta model with the task model parameters
            meta_model = deepcopy(self.models_list[task_id])
            meta_train_loss = onmt.utils.loss.build_loss_compute(
                meta_model, deepcopy(self.fields_list[task_id])["tgt"], self.model_opt)
            # Compute meta weights and train the model with them
            weights = self._meta_gradient_weights(batch, meta_model, meta_train_loss, task_id, meta_valid_batch)
            if self.model_opt.meta_batch_weighting_mode == 1:
                weights += float(1 / normalization)
            self._meta_gradient_weighted_train(batch, weights, report_stats, total_stats, task_id)
        # Update the parameters and statistics.
        self.optim_list[task_id].step()

    def _meta_gradient_weighted_train(self, batch, weights, report_stats, total_stats, task_id, zero_gradients=True):

        src = inputters.make_features(batch, 'src', self.data_type)
        if self.data_type == 'text':
            _, src_lengths = batch.src
            report_stats.n_src_words += src_lengths.sum().item()
        elif self.data_type == 'audio':
            src_lengths = batch.src_lengths
        else:
            src_lengths = None

        tgt_outer = inputters.make_features(batch, 'tgt')
        tgt = tgt_outer
        if zero_gradients:
            self.models_list[task_id].zero_grad()
        outputs, attns = self.models_list[task_id](src, tgt, src_lengths)

        batch_stats, loss = self.train_loss_list[task_id].monolithic_compute_raw_loss(
            batch, outputs, attns)

        weighted_loss = torch.sum(loss * weights)
        weighted_loss.backward()

        total_stats.update(batch_stats)
        report_stats.update(batch_stats)

        if self.models_list[task_id].decoder.state is not None:
            self.models_list[task_id].decoder.detach_state()

    def _meta_gradient_weights(self, batch, meta_model, meta_train_loss, task_id, valid_batch, is_normalized=True):

        # First step: feed forward training_batch to the model and compute first-order gradients
        eps = to_var(torch.zeros(len(batch)))
        src = inputters.make_features(batch, 'src', self.data_type)
        if self.data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None

        tgt_outer = inputters.make_features(batch, 'tgt')
        tgt = tgt_outer

        meta_model.zero_grad()
        outputs, attns = meta_model(src, tgt, src_lengths)

        _, loss = meta_train_loss.monolithic_compute_raw_loss(
            batch, outputs, attns)

        meta_loss_train = torch.sum(loss * eps)

        meta_model.zero_grad()
        grads = torch.autograd.grad(meta_loss_train, (meta_model.params()), create_graph=True)

        if meta_model.decoder.state is not None:
            meta_model.decoder.detach_state()

        # Second step: Update the model for main task w.r.t the computed gradients
        if self.model_opt.mtl_shared_optimizer:
            lr = self.optim_list[0].learning_rate
        else:
            lr = self.optim_list[task_id].learning_rate

        if self.model_opt.mtl_fully_share or task_id == 0:
            meta_model_main = meta_model
            meta_model_main.update_params(lr, source_params=grads)
            meta_model_main_loss = meta_train_loss
        else:
            meta_model_main = deepcopy(self.models_list[0])
            # only update shared parameters
            meta_model_main.update_params(lr, source_params=grads, shared_params=self.model_opt.shared_params)
            meta_model_main_loss = onmt.utils.loss.build_loss_compute(
                meta_model_main, deepcopy(self.fields_list[0])["tgt"], self.model_opt)

        # Third step: feed forward the meta_batch to the updated main task model,
        #   and compute gradients w.r.t eps weights (requires double gradients)
        src = inputters.make_features(valid_batch, 'src', self.data_type)
        if self.data_type == 'text':
            _, src_lengths = valid_batch.src
        else:
            src_lengths = None
        tgt_outer = inputters.make_features(valid_batch, 'tgt')
        tgt = tgt_outer

        meta_model_main.zero_grad()
        outputs, attns = meta_model_main(src, tgt, src_lengths)

        _, loss = meta_model_main_loss.monolithic_compute_raw_loss(
            valid_batch, outputs, attns)
        if self.model_opt.normalize_meta_loss:
            meta_loss_valid = torch.sum(loss) / len(valid_batch)
        else:
            meta_loss_valid = torch.sum(loss)
        grad_eps = torch.autograd.grad(meta_loss_valid, eps, only_inputs=True,
                                       retain_graph=False)[0]

        if meta_model_main.decoder.state is not None:
            meta_model_main.decoder.detach_state()

        # Fourth step: normalize the weights (if required)
        if is_normalized:
            weights_tilde = torch.clamp(-grad_eps.data, min=0)
            norm_c = torch.sum(weights_tilde)
            if norm_c != 0:
                weights = weights_tilde / norm_c
            else:
                weights = weights_tilde
        else:
            return grad_eps.data

    def _meta_gradient_accumulation_batch_list(self, batch_list, task_list, total_stats,
                                               report_stats, meta_valid_batch, is_concat=False):

        raw_weights_all = None
        batch_size_list = []
        for batch in batch_list:
            batch_size_list.append(len(batch))
        if is_concat:
            # Concat mini-batches
            all_raw_data = []
            for batch in batch_list:
                all_raw_data.extend(batch.data)
            batch = Batch(all_raw_data, batch_list[0].dataset, batch_list[0].device)

            # Initialize the meta model with the task model parameters
            # (the concat mode only supports fully shared model)
            task_id = 0
            meta_model = deepcopy(self.models_list[task_id])
            meta_train_loss = onmt.utils.loss.build_loss_compute(
                meta_model, self.fields_list[task_id]["tgt"], self.model_opt)
            # Compute raw meta weights
            raw_weights_all = self._meta_gradient_weights(batch, meta_model, meta_train_loss, task_id, meta_valid_batch,
                                                          is_normalized=False)
        else:
            for i in range(len(batch_list)):
                batch = batch_list[i]
                task_id = task_list[i]
                # initialize the meta model with the task model parameters
                meta_model = deepcopy(self.models_list[task_id])
                meta_train_loss = onmt.utils.loss.build_loss_compute(
                    meta_model, self.fields_list[task_id]["tgt"], self.model_opt)
                # Compute raw meta weights
                raw_weights = self._meta_gradient_weights(batch, meta_model, meta_train_loss, task_id, meta_valid_batch,
                                                          is_normalized=False)
                if raw_weights_all is None:
                    raw_weights_all = raw_weights
                else:
                    raw_weights_all = torch.cat([raw_weights_all, raw_weights])

        # Normalize the weights
        weights_tilde = torch.clamp(-raw_weights_all.data, min=0)
        norm_c = torch.sum(weights_tilde)

        if norm_c != 0:
            weights = weights_tilde / norm_c
        else:
            weights = weights_tilde

        # Train models separately w.r.t the computed weights
        ptr = 0
        batch_weights_list = []
        for i in range(len(batch_list)):
            batch = batch_list[i]
            task_id = task_list[i]
            batch_size = batch_size_list[i]
            batch_weights = weights[ptr: ptr + batch_size]
            batch_weights_list.append(batch_weights.data.cpu().numpy())
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].ne(
                    self.train_loss_list[task_id].padding_idx).sum()
                normalization = num_tokens.item()
            elif self.norm_method == "sents":
                normalization = batch_size
            elif self.norm_method == "none":
                normalization = 1
            # Add the uniform weights to meta weights
            if self.model_opt.meta_batch_weighting_mode == 1:
                batch_weights += float(1 / normalization)
            # Add pre-defined weights to meta weights
            elif self.model_opt.meta_batch_weighting_mode == 2:
                additive = 1 if task_id == 0 else 1 / (len(task_list) - 1)
                additive *= float(1 / normalization)
                batch_weights += additive
            ptr += batch_size
            self._meta_gradient_weighted_train(batch, batch_weights, report_stats[task_id], total_stats[task_id],
                                               task_id)
            self.optim_list[task_id].step()

        return batch_weights_list

    def _meta_gradient_grads(self, batch, meta_model, meta_train_loss):

        src = inputters.make_features(batch, 'src', self.data_type)
        if self.data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None

        tgt_outer = inputters.make_features(batch, 'tgt')
        tgt = tgt_outer

        meta_model.zero_grad()
        outputs, attns = meta_model(src, tgt, src_lengths)

        _, loss = meta_train_loss.monolithic_compute_raw_loss(
            batch, outputs, attns)

        meta_loss_train = torch.sum(loss)

        meta_model.zero_grad()
        grads = torch.autograd.grad(meta_loss_train, (meta_model.params()), create_graph=True)

        if meta_model.decoder.state is not None:
            meta_model.decoder.detach_state()

        return grads

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats, task_id=0):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1, task_id=task_id)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None, task_id=0):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats, task_id=task_id)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

    def _save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver._save(step)
