# VFIformer
Official PyTorch implementation of our CVPR2022 paper [Video Frame Interpolation with Transformer]()



## Dependencies
* python >= 3.8
* pytorch >= 1.8.0
* torchvision >= 0.9.0

## Prepare Dataset 
1. Download [Vimeo90K Triplet dataset](http://toflow.csail.mit.edu/)
## Get Started
1. Clone this repo
    ```
    git clone https://github.com/Jia-Research-Lab/VFIformer.git
    cd MASA-SR
    ```
1. Modify the argument `--data_root` in `test.py` and `train.py` according to your data path.
### Evaluation
1. Download the pre-trained models and place them into the `pretrained_models/` folder

    * Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Se1gIEtezTNdafeniHCSfXXwS0i4r5aA?usp=sharing)
        * *pretrained_VFIformer*: the final model in the main paper
        * *pretrained_VFIformerSmall*: the smaller version of the model mentioned in the supplementary file
1. Test on the Vimeo90K testing set:
    ```
    python test.py --testset VimeoDataset --net_name VFIformer --resume ./pretrained_models/pretrained_VFIformer/net_220.pth --save_result
    ```
    
    If you want to test with the smaller model, please change the `--net_name` and `--resume` accordingly:
    ```
    python test.py --testset VimeoDataset --net_name VFIformerSmall --resume ./pretrained_models/pretrained_VFIformerSmall/net_220.pth --save_result
    ```
    
    The testing results are saved in the `test_results/` folder. If you do not want to save the image results, you can remove the `--save_result` argument in the commands optionally.

1. Test on the MiddleBury dataset:
    Download the [MiddleBury Other dataset](https://vision.middlebury.edu/flow/data/).
    Modify the argument `--data_root` according to your data path, run:
    ```
    python test.py --data_root [your MiddleBury path] --testset MiddleburyDataset --net_name VFIformer --resume ./pretrained_models/pretrained_VFIformer/net_220.pth --save_result
    ```
    
1. Test on the UCF101 dataset:
    Download the [UCF101 dataset](https://drive.google.com/file/d/0B7EVK8r0v71pdHBNdXB6TE1wSTQ/view?resourcekey=0-r6ihCy20h3kbgZ3ZdimPiA).
    Modify the argument `--data_root` according to your data path, run:
    ```
    python test.py --data_root [your UCF101 path] --testset UFC101Dataset --net_name VFIformer --resume ./pretrained_models/pretrained_VFIformer/net_220.pth --save_result
    ```
   
1. Test on the SNU-FILM dataset:
    Download the [SNU-FILM dataset](https://myungsub.github.io/CAIN/).
    Modify the argument `--data_root` according to your data path. Choose the motion level and modify the argument `--test_level` accordingly, run:
    ```
    python FILM_test.py --data_root [your SNU-FILM path] --test_level [easy/medium/hard/extreme] --net_name VFIformer --resume ./pretrained_models/pretrained_VFIformer/net_220.pth
    ```




### Training
1. First train the flow estimator:
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4174 train.py --launcher pytorch --gpu_ids 0,1,2,3 \
            --loss_flow --use_tb_logger --batch_size 48 --net_name IFNet --name train_IFNet --max_iter 300 --crop_size 192 --save_epoch_freq 5
    ```
1. Then train the whole framework:
    ```
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=4175 train.py --launcher pytorch --gpu_ids 0,1,2,3,4,5,6,7 \
            --loss_l1 --loss_ter --loss_flow --use_tb_logger --batch_size 24 --net_name VFIformer --name train_VFIformer --max_iter 300 \
            --crop_size 192 --save_epoch_freq 5 --resume_flownet ./weights/train_IFNet/snapshot/net_final.pth
    ```
1. To train the smaller version, run:
    ```
    python -m torch.distributed.launch --nproc_per_node=8 --master_port=4175 train.py --launcher pytorch --gpu_ids 0,1,2,3,4,5,6,7 \
            --loss_l1 --loss_ter --loss_flow --use_tb_logger --batch_size 24 --net_name VFIformerSmall --name 0320_VFIformerSmall --max_iter 300 \
            --crop_size 192 --save_epoch_freq 5 --resume_flownet ./weights/train_IFNet/snapshot/net_final.pth
    ```


## Acknowledgement
We borrow some codes from [RIFE](https://github.com/hzwer/arXiv2021-RIFE) and [SwinIR](https://github.com/JingyunLiang/SwinIR). We thank the authors for their great work.

## Citation

Please consider citing our paper in your publications if it is useful for your research.
```
@inproceedings{lu2022vfiformer,
    title={Video Frame Interpolation with Transformer},
    author={Liying Lu, Ruizheng Wu, Huaijia Lin, Jiangbo Lu, and Jiaya Jia},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022},
}
```