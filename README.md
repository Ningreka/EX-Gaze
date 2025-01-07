# python dependencies
## Name                    Version                   Build  Channel
blas                      1.0                         mkl    defaults
cuda-cudart               11.7.99                       0    nvidia
cuda-cupti                11.7.101                      0    nvidia
cuda-libraries            11.7.1                        0    nvidia
cuda-nvcc                 11.7.99                       0    nvidia/label/cuda-11.7.1
cuda-nvrtc                11.7.99                       0    nvidia
cuda-nvtx                 11.7.91                       0    nvidia
cuda-runtime              11.7.1                        0    nvidia
h5py                      3.9.0           py310he06866b_0    defaults
hdf5                      1.12.1               h2b7332f_3    defaults
matplotlib                3.8.0                    pypi_0    pypi
mmcv                      2.0.1                    pypi_0    pypi
mmdeploy                  1.3.1                    pypi_0    pypi
mmdet                     3.1.0                    pypi_0    pypi
mmengine                  0.8.5                    pypi_0    pypi
mmrotate                  1.0.0rc1                  dev_0    <develop>
numpy                     1.26.0          py310h5f9d8c6_0    defaults
onnx                      1.13.1          py310h12ddb61_0    defaults
onnx-graphsurgeon         0.3.27                   pypi_0    pypi
onnxoptimizer             0.3.13                   pypi_0    pypi
onnxruntime               1.16.1                   pypi_0    pypi
onnxsim                   0.4.35                   pypi_0    pypi
opencv-python             4.8.1.78                 pypi_0    pypi
pandas                    2.1.1                    pypi_0    pypi
pillow                    10.0.1          py310ha6cbd5a_0    defaults
protobuf                  3.20.2                   pypi_0    pypi
python                    3.10.13              h955ad1f_0    defaults
pytorch                   1.13.1          py3.10_cuda11.7_cudnn8.5.0_0    pytorch
pytorch-cuda              11.7                 h778d358_5    pytorch
pytorch-mutex             1.0                        cuda    pytorch
scikit-learn              1.5.1           py310h1128e8f_0    defaults
scipy                     1.11.3                   pypi_0    pypi
six                       1.16.0                   pypi_0    pypi
tbb                       2021.8.0             hdb19cb5_0    defauy
torchaudio                0.13.1              py310_cu117    pytorch
torchsummary              1.5.1                    pypi_0    pypi
torchvision               0.14.1              py310_cu117    pytorch
tqdm                      4.65.2                   pypi_0    pypi


# preparation
1. modify the path in the following python file
    - "test/tracking_eval/end_to_end_tracking_model_cfg.py"
2. modify the model weight to absolute path in the following file
    - "misc/ev_eye_dataset_utils.py"
    - "deploy/scripts/export.py"

# before running
Export the current project path to the PYTHONPATH like this

```
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

the following cmd should running in the root of this project

# train
before training the ev-eye dataset should be download and place well
```
python train/default_train.py
```

# end to end tracking
```
python test/tracking_eval/end_to_end_tracking.py --device 0 --similarity_threshold 0.8 --pre_accum_tracking --continuous_ann_track --model_cfg_option config-eye_crop_mbv3spreX_multi_anchor_det-s1_f2_n4_trans-pre_accum10_50_blink_exp5_0.5_rand
```

# tracking result analysis
```
python test/tracking_eval/end_to_end_tracking_analysis.py
```

# model on jetson
## export models to .onnx format
```
python deploy/scripts/export.py
```

## test use trtexec on jetson

requirement
- trtexec >= 8.6

copy the exported onnx model file to jetson and run as follow
```
trtexec --onnx=ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre0.5x2.onnx --useCudaGraph --useSpinWait
trtexec --onnx=mbv3spreX_head_retina_img_pupil_det_eye_region_crop_x2.onnx --useCudaGraph --useSpinWait
```

