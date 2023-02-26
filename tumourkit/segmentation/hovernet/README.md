# Hovernet adaptation

This module is a fork of the [official hovernet repository](https://github.com/vqdang/hover_net). It has been modified to admit images of greater size, increasing its functionality. Previously it only admitted two resolutions for the patches: 270x270 and 256x256. Now it also has support for 518x518.

## Usage

Use `extract_patches.py` to generate input data from pngcsv GT and original images. For more details do `python extract_patches.py --help`. 

For training, modify `config.py` and `models/hovernet/opt.py` files according to the original repository instructions and then run `python run_train.py --gpu=[id]`. The script has only been tested on GPU, not on CPU nor distributed environment.

For inference, use `run_infer.py` according to original repository instructions. As an example, see the following command:

>python3 run_infer.py --gpu=0 --nr_types=3 --type_info_path=type_info.json --model_path='net_epoch=50.tar' --model_mode='original' --nr_inference_workers=0 --nr_post_proc_workers=0 --batch_size=10 --shape 518 tile --input_dir=[INPUT_DIR] --output_dir=[OUTPUT_DIR] --save_raw_map

The flag shape is included for the increased resolution execution. Use 270 or 518 when mode is 'original', or ignore it for 'fast' mode.