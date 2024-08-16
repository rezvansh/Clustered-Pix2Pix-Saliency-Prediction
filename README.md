# Clustered Saliency Prediction
### <a href="https://proceedings.bmvc2023.org/499/">[Paper]</a> 
This is an implementation of the model in paper <a href="https://proceedings.bmvc2023.org/499/">Clustered Saliency Prediction</a> by R. Sherkati and J. Clark (British Machine Vision Conference 2023).
Here we have used the code in <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master">this repository</a> as the base and added our contributions and methods to it. We have done our experiments on the Personalized Saliency Maps dataset collected <a href="https://github.com/xuyanyu-shh/Personalized-Saliency"> here </a>. 

# Clustering the viewers
In order to put the viewers in the <a href="https://github.com/xuyanyu-shh/Personalized-Saliency">PSM dataset</a> in separate clusters, we use their personal features and training set saliency maps of subjects and run our clustering algorithm on it. To do that, first download the PSM dataset and put the dataset's folder in the main folder of the project. Then run the cells in the `Clustering_subjects.ipynb` until the cell containing `generate_5_split_files` and in this function set the paths based on your environment. The second arument in the function `generate_5_split_files` is the path to the universal saliency maps of the source images of PSM dataset, obtained using the desired method, e.g. DeepGaze IIE.

# Training Requirements
To install all the required packages, create a conda environment of the `model_environment.yml` file using the following command in the terminal: 
```
conda env create -f model_environment.yml
```
Make sure to activate this environment before training.

# Training

To train on a custom dataset, while in the main folder of the project, run the following command:
```
python3 train.py --dataroot ./datasets/my_project/my_experiment --name my_experiment_example --model pix2pix --display_id -1 --num_clusters 6 --batch_size 16 --n_epochs 100 --n_epochs_decay 100

```
Here, `--num_clusters` is the number of clusters in the dataset. Also ` --dataroot ./datasets/my_project/my_experiment ` contains the files for the current experiment. The structure of the experiments folders within the project are as below:
```
Clustered_Pix2Pix:
                  ----datasets:
                              ----my_project:
                                            -----all_images_release
                                            -----my_experiment:
                                                              -----0:
                                                                    ---A:
                                                                        ---test
                                                                        ---train
                                                                        ---val
                                                                    ---B:
                                                                        ---test
                                                                        ---train
                                                                        ---val
                                                              -----1:
                                                                      .
                                                                      .
                                                                      .
                                                              -----2:
                                                                      .
                                                                      .
                                                                      .
                                                                .
                                                                .

```

In the above structure, `all_images_release` is a folder containing all the original frames used for predicting the saliency in different clusters. These are used during training and also during inference. In `my_experiment` folder, each ofthe folders "0", "1", ... correspoond to a cluster.

The `--name my_experiment_example` is the name we want to give to this experiment, and the checkpoints and logs will be saved in `results/my_experiment_example`.

# Testing

To get prediction for the images in the test set, run the below command:
```
python3 test.py --dataroot ./datasets/my_project/my_experiment --name my_experiment_example --model pix2pix --num_clusters 6
```

Here `num_clusters` is the number of clusters (same as in the training part). `--name my_experiment_example` is the same name we gave this experiment in the training part. `--dataroot ./datasets/my_project/my_experiment` is the root to the experiment files, same as in the training.

# Evaluation 

To evaluate the inference results with some saliency metrics, we run the cells of evaluation.ipynb until at least the cell containing `def eval_average_with_all_results_general( path, pre_name="", auc = 2, mode = "pytorch", result_dir = "results", epoch_val = 0, gt_avg_cluster = False, eval_mode = "test", ext_gt = 'jpg')` function. Then we run the function as below:
```
eval_average_with_all_results_general(
    path="path/to/experiment_data",
    result_dir="results_folder", auc=3, mode="clustered_pytorch",
    gt_avg_cluster=True, eval_mode = "test", ext_gt = 'png')
```
where `results_folder` is the folder inside `path/to/experiment_data` containing the result inference images.

# Citation
If you use our methods, please cite our paper <a href="https://proceedings.bmvc2023.org/499/">Clustered Saliency Prediction</a>. 
```
@inproceedings{Sherkati_2023_BMVC,
author    = {Rezvan Sherkati and James J. Clark},
title     = {Clustered Saliency Prediction},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {BMVA},
year      = {2023},
url       = {https://papers.bmvc2023.org/0499.pdf}
}
```
