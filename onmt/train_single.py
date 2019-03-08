#!/usr/bin/env python
"""
    Training on a single process
"""

import configargparse

import os
import random
import torch
import subprocess

import onmt.opts as opts

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    load_fields, _collect_report_features, dataset_stats
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim, build_optim_mtl_params
from onmt.trainer import build_trainer
from onmt.models import build_model_saver, build_mtl_model_saver
from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters

def opt_to_string(job_opt):
    message=""
    for arg in vars(job_opt):
        new_line="%s: %s"%(arg, getattr(job_opt, arg))
        message+=new_line+"\n"
    return message

def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt, device_id):
    opt.rnn_type = "shared" + opt.rnn_type


    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size

        # this check is here because audio allows the encoder and decoder to
        # be different sizes, but other model types do not yet
        same_size = opt.enc_rnn_size == opt.dec_rnn_size
        assert opt.model_type == 'audio' or same_size, \
            "The encoder and decoder rnns must be the same size for now"

    opt.brnn = opt.encoder_type == "brnn"

    assert opt.rnn_type != "SRU" or opt.gpu_ranks, \
        "Using SRU requires -gpu_ranks set."

    if torch.cuda.is_available() and not opt.gpu_ranks:
        logger.info("WARNING: You have a CUDA device, \
                    should run with -gpu_ranks")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt

def main(opt, device_id):
    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Gather information related to the training script and commit version
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(os.path.dirname(script_path))
    logger.info('Train script dir: %s' % script_dir)
    git_commit = str(subprocess.check_output(['bash', script_dir + '/cluster_scripts/git_version.sh']))
    logger.info("Git Commit: %s" % git_commit[2:-3])
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        # TODO: load MTL model
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        # Load default opts values then overwrite it with opts from
        # the checkpoint. It's usefull in order to re-train a model
        # after adding a new option (not set in checkpoint)
        dummy_parser = configargparse.ArgumentParser()
        opts.model_opts(dummy_parser)
        default_opt = dummy_parser.parse_known_args([])[0]

        model_opt = default_opt
        model_opt.__dict__.update(checkpoint['opt'].__dict__)
    else:
        checkpoint = None
        model_opt = opt


    num_tasks = len(opt.data.split(','))
    opt.num_tasks = num_tasks

    checkpoint_list=[]
    if opt.warm_model:
        base_name=opt.warm_model
        for task_id in range(num_tasks):
            chkpt_path=base_name.replace("X",str(task_id))
            if not os.path.isfile(chkpt_path):
                chkpt_path = base_name.replace("X", str(0))
            logger.info('Loading a checkpoint from %s' % chkpt_path)

            checkpoint = torch.load(chkpt_path,
                                    map_location=lambda storage, loc: storage)
            checkpoint_list.append(checkpoint)
    else:
        for task_id in range(num_tasks):
            checkpoint_list.append(None)

    fields_list = []
    data_type=None
    for task_id in range(num_tasks):
        # Peek the first dataset to determine the data_type.
        # (All datasets have the same data_type).
        first_dataset = next(lazily_load_dataset("train", opt, task_id=task_id))
        data_type = first_dataset.data_type

        # Load fields generated from preprocess phase.
        if opt.mtl_shared_vocab and task_id > 0:
            logger.info(' * vocabulary size. Same as the main task!')
            fields = fields_list[0]
        else:
            fields = load_fields(first_dataset, opt, checkpoint_list[task_id], task_id=task_id)

        # Report src/tgt features.

        src_features, tgt_features = _collect_report_features(fields)
        for j, feat in enumerate(src_features):
            logger.info(' * (Task %d) src feature %d size = %d'
                        % (task_id, j, len(fields[feat].vocab)))
        for j, feat in enumerate(tgt_features):
            logger.info(' * (Task %) tgt feature %d size = %d'
                        % (task_id, j, len(fields[feat].vocab)))
        fields_list.append(fields)

    if opt.epochs > -1:
        total_num_batch = 0
        for task_id in range(num_tasks):
            train_iter = build_dataset_iter(lazily_load_dataset("train", opt, task_id=task_id), fields_list[task_id], opt)
            for i, batch in enumerate(train_iter):
                num_batch = i
            total_num_batch+=num_batch
            if opt.mtl_schedule < 10:
                break
        num_batch = total_num_batch
        opt.train_steps = (num_batch * opt.epochs) + 1
        # Do the validation and save after each epoch
        opt.valid_steps = num_batch
        opt.save_checkpoint_steps = 1

    # logger.info(opt_to_string(opt))
    logger.info(opt)

    # Build model(s).
    models_list = []
    for task_id in range(num_tasks):
        if opt.mtl_fully_share and task_id > 0:
            # Since we only have one model, copy the pointer to the model for all
            models_list.append(models_list[0])
        else:

            main_model = models_list[0] if task_id > 0 else None
            model = build_model(model_opt, opt, fields_list[task_id], checkpoint_list[task_id], main_model=main_model, task_id=task_id)
            n_params, enc, dec = _tally_parameters(model)
            logger.info('(Task %d) encoder: %d' % (task_id, enc))
            logger.info('(Task %d) decoder: %d' % (task_id, dec))
            logger.info('* number of parameters: %d' % n_params)
            _check_save_model_path(opt)
            models_list.append(model)

    # combine parameters of different models and consider shared parameters just once.
    def combine_named_parameters(named_params_list):
        observed_params = []
        for model_named_params in named_params_list:
            for name, p in model_named_params:
                is_observed = False
                # Check whether we observed this parameter before
                for param in observed_params:
                    if p is param:
                        is_observed = True
                        break
                if not is_observed:
                    observed_params.append(p)
                    yield name, p

    # Build optimizer.
    optims_list = []
    all_models_params=[]
    for task_id in range(num_tasks):
        if not opt.mtl_shared_optimizer:
            optim = build_optim(models_list[task_id], opt, checkpoint)
            optims_list.append(optim)
        else:
            all_models_params.append(models_list[task_id].named_parameters())

    # Extract the list of shared parameters among the models of all tasks.
    observed_params = []
    shared_params = []
    for task_id in range(num_tasks):
        for name, p in models_list[task_id].named_parameters():
            is_observed = False
            # Check whether we observed this parameter before
            for param in observed_params:
                if p is param:
                    shared_params.append(name)
                    is_observed = True
                    break
            if not is_observed:
                observed_params.append(p)
    opt.shared_params = shared_params

    if opt.mtl_shared_optimizer:
        optim = build_optim_mtl_params(combine_named_parameters(all_models_params), opt, checkpoint)
        optims_list.append(optim)

    # Build model saver
    model_saver = build_mtl_model_saver(model_opt, opt, models_list, fields_list, optims_list)

    trainer = build_trainer(opt, device_id, models_list, fields_list,
                            optims_list, data_type, model_saver=model_saver)

    def train_iter_fct(task_id):
        return build_dataset_iter(
            lazily_load_dataset("train", opt, task_id=task_id), fields_list[task_id], opt)

    def valid_iter_fct(task_id):
        return build_dataset_iter(
            lazily_load_dataset("valid", opt, task_id=task_id), fields_list[task_id], opt)

    def meta_valid_iter_fct(task_id, is_log=False):
        return build_dataset_iter(
            lazily_load_dataset("meta_valid", opt, task_id=task_id, is_log=is_log), fields_list[task_id], opt)

    # Do training.
    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps,
                  opt.valid_steps, meta_valid_iter_fct=meta_valid_iter_fct)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='train.py',
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
