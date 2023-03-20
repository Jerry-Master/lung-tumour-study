import importlib
from typing import Optional

class Config(object):
    """Configuration file."""

    def __init__(self, shape: str, log_dir: str, train_dir: str, valid_dir: str, pretrained_path: str, use_cpu: bool, num_classes: Optional[int] = 2):
        self.seed = 10
        self.logging = True
        self.debug = False

        model_name = "hovernet"
        model_mode = "original" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = num_classes + 1 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        if shape == '270':
            aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
            act_shape = [270, 270] # patch shape used as input to network - central crop performed after augmentation
            out_shape = [80, 80]   # patch shape at output of network
        elif shape == '518':
            aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
            act_shape = [518, 518] # patch shape used as input to network - central crop performed after augmentation
            out_shape = [328, 328] # patch shape at output of network
        else:
            assert False, 'Wrong shape indicated in config file.'

        self.log_dir = log_dir # where checkpoints will be saved

        # paths to training and validation patches
        self.train_dir_list = [train_dir]
        self.valid_dir_list = [valid_dir]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        module = importlib.import_module(
            ".segmentation.hovernet.models.%s.opt" % model_name, package="tumourkit"
        )
        self.model_config = module.get_config(nr_type, model_mode)
        self.model_config["phase_list"][0]["run_info"]["net"]["pretrained"] = pretrained_path
        self.model_config["phase_list"][0]["run_info"]["net"]["use_cpu"] = use_cpu
        self.model_config["phase_list"][1]["run_info"]["net"]["use_cpu"] = use_cpu
