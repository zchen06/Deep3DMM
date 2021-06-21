# Deep3DMM
Official repository for the CVPR 2021 paper [Learning Feature Aggregation for Deep 3D Morphable Models](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Feature_Aggregation_for_Deep_3D_Morphable_Models_CVPR_2021_paper.html).

## Requirements
This code is tested on Python 3.7 and Pytorch versoin 1.4 with CUDA 10.0. Requirments can be install by running

      pip install -r requirements.txt

Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh). 

## Train
To start the training, follow these steps
1. Download the registered data from the [COMA](https://coma.is.tue.mpg.de/) and/or [DFAUST](https://dfaust.is.tue.mpg.de/).
2. Update default config file, default.cfg as needed, especially data_dir path.
3. Run the training of Deep3DMM by

     ` python main.py -m ComaAtt `

    Note that the 'sliced' dataset split is used by default.

## Evaluation
Run the evaluation by

     python main.py -m ComaAtt --eval

   Note that the checkpoint with best validation accuracy is evaluated by default.

## Acknowledgement
This implementation is built upon the Pytorch implementation of COMA ([Link](https://github.com/pixelite1201/pytorch_coma)). We also build our Deep3DMM with spiral convolution based on the implementation of [Neural3DMM](https://github.com/gbouritsas/Neural3DMM). Many thanks to the authors for releasing the source code.

## License
This code is free for non-commerical purposes only. For commercial usage, please contact the authors for more information.

## Cite
Please consider citing our work if you find it useful:
```
Zhixiang Chen and Tae-Kyun Kim, "Learning Feature Aggregation for Deep 3D Morphable Models", CVPR, 2021.
```
