"""
Main HoVer-Net training script.
"""
import cv2
cv2.setNumThreads(0)
import glob
import inspect
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from .config import Config
from .dataloader.train_loader import FileLoader
from .misc.utils import rm_n_mkdir
from .run_utils.engine import RunEngine
from .run_utils.utils import (
    check_manual_seed,
    colored,
    convert_pytorch_checkpoint
)
import argparse


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return


class TrainManager(Config):
    """Either used to view the dataset or to initialise the main training loop."""


    def view_dataset(self, save_name, mode="train"):
        self.nr_gpus = 1
        check_manual_seed(self.seed)
        phase_list = self.model_config["phase_list"][0]
        target_info = phase_list["target_info"]
        prep_func, prep_kwargs = target_info["viz"]
        dataloader = self._get_datagen(2, mode, target_info["gen"])
        for i, batch_data in enumerate(dataloader):  
            batch_data = {k: v.numpy() for k, v in batch_data.items()}
            viz = prep_func(batch_data, is_batch=True, **prep_kwargs)
            plt.imshow(viz)
            plt.savefig(save_name + str(i) + '.png')
        self.nr_gpus = -1
        return


    def _get_datagen(self, batch_size, run_mode, target_gen, nr_procs=0, fold_idx=0):
        nr_procs = nr_procs if not self.debug else 0

        # ! Hard assumption on file type
        file_list = []
        if run_mode == "train":
            data_dir_list = self.train_dir_list
        else:
            data_dir_list = self.valid_dir_list
        for dir_path in data_dir_list:
            file_list.extend(glob.glob("%s/*.npy" % dir_path))
        file_list.sort()  # to always ensure same input ordering

        assert len(file_list) > 0, (
            "No .npy found for `%s`, please check `%s` in `config.py`"
            % (run_mode, "%s_dir_list" % run_mode)
        )
        print("Dataset %s: %d" % (run_mode, len(file_list)))
        input_dataset = FileLoader(
            file_list,
            mode=run_mode,
            with_type=self.type_classification,
            setup_augmentor=nr_procs == 0,
            target_gen=target_gen,
            **self.shape_info[run_mode]
        )

        dataloader = DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=batch_size * max(1, self.nr_gpus),
            shuffle=run_mode == "train",
            drop_last=run_mode == "train",
            worker_init_fn=worker_init_fn,
        )
        return dataloader


    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        """Simply run the defined run_step of the related method once."""
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            rm_n_mkdir(log_dir)

            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + "/stats.json"
            with open(json_log_file, "w") as json_file:
                json.dump({}, json_file)  # create empty file
            log_info = {
                "json_file": json_log_file,
                "tfwriter": tfwriter,
            }

        loader_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            loader_dict[runner_name] = self._get_datagen(
                opt["batch_size"][runner_name],
                runner_name,
                opt["target_info"]["gen"],
                nr_procs=runner_opt["nr_procs"],
                fold_idx=fold_idx,
            )
        
        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = prev_phase_dir + "/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = "%s/%s_epoch=%d.tar" % (
                prev_phase_dir,
                net_name,
                max(epoch_list),
            )
            return last_chkpts_path

        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(
                net_info["desc"]
            ), "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info["desc"]()
            pretrained_path = net_info["pretrained"]
            if pretrained_path is not None:
                if pretrained_path == -1:
                    # * depend on logging format so may be broken if logging format has been changed
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)["desc"]
                else:
                    chkpt_ext = os.path.basename(pretrained_path).split(".")[-1]
                    if chkpt_ext == "npz":
                        net_state_dict = dict(np.load(pretrained_path))
                        net_state_dict = {
                            k: torch.from_numpy(v) for k, v in net_state_dict.items()
                        }
                    elif chkpt_ext == "tar":  # ! assume same saving format we desire
                        net_state_dict = torch.load(pretrained_path)["desc"]
                        # TODO: make this optional
                        del net_state_dict['decoder.tp.u0.conv.weight']
                        del net_state_dict['decoder.tp.u0.conv.bias']

                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(
                    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
                )

                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * uncomment for your convenience
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])

            net_desc = DataParallel(net_desc)
            if self.nr_gpus > 0:
                net_desc = net_desc.to("cuda")
            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            scheduler = net_info["lr_scheduler"](optimizer)
            net_run_info[net_name] = {
                "use_cpu": net_info["use_cpu"],
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                # TODO: standardize API for external hooks
                "extra_info": net_info["extra_info"],
            }

        # parsing the running engine configuration
        assert (
            "train" in run_engine_opt
        ), "No engine for training detected in description file"

        # initialize runner and attach callback afterward
        # * all engine shared the same network info declaration
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_dict[runner_name] = RunEngine(
                dataloader=loader_dict[runner_name],
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for event, callback_list, in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # retrieve main runner
        main_runner = runner_dict["train"]
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # start the run loop
        main_runner.run(opt["nr_epochs"])

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")
        return


    def run(self):
        self.nr_gpus = torch.cuda.device_count()
        print('Detect #GPUS: %d' % self.nr_gpus)

        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            if len(phase_list) == 1:
                save_path = self.log_dir
            else:
                save_path = self.log_dir + "/%02d/" % (phase_idx)
            self.run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path
            )
            prev_save_path = save_path


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--view', type=str)
    parser.add_argument('--save-name', type=str)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--valid-dir', type=str, required=True)
    parser.add_argument('--shape', type=str, required=True, choices=['270', '518'])
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    return parser


def main_with_args(args):
    trainer = TrainManager(
        args.shape, args.log_dir, args.train_dir, args.valid_dir,
        args.pretrained_path, use_cpu=(args.gpu == ''), num_classes=args.num_classes
    )

    if args.view is not None:
        assert args.view == "train" or args.view == "valid", 'Use "train" or "valid" for --view.'
        assert args.save_name is not None, 'In view mode, an image name must be provided.'
        trainer.view_dataset(save_name=args.save_name, mode=args.view)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        trainer.run()

def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
    
