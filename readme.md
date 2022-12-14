## Dataset Description ##

Python implementation of generating datasets. Please cite the paper "Look-twice: Position Attentive Semantic Context Learning for High Percentages False Correspondences Removal", by Ruiyuan Li, Zhaolin Xiao,Meng Zhang and Haiyan Jin.

The refined Hpatch datasets contains both true positive correspondences (TPs) and false positive correspondences (FPs) of 580 image pairs with illumination and viewpoint changes. The ground truth TPs are generated using the homography matrices that provided in Hpatch datasets.

The refined GL3D datasets contains over 2000 image pairs under wide baselines and occlusions, which are captured using drones. The FPs are generated by employing the epipolar constraint. 

In the paper, the datasets have been divided as three parts: training(70\%), validation (15\%) and testing (15\%) respectively.

## Code Description ##

To generate GL3D datasets, please run the file 'generate_gl3d_20_datasets.py' and 'generate_gl3d_5_datasets' file, which have the following steps:

1. Load images and corr.bin in the Input_Data folder.

2. Parse the corr.bin file to obtain the camera parameters and the coordinate mapping table, which are mapping the relationship between a pair of images. The mapping table is generated by using SFM algorithm.

   | row\column |  0   |    1    | ...  |   998   |  999   |
   | :--------: | :--: | :-----: | :--: | :-----: | :----: |
   |     0      | 2, 3 |  3, 5   | ...  | 156, 37 |  None  |
   |     1      | 3, 6 |  None   | ...  |  None   |  None  |
   |    ...     | ...  |   ...   | ...  |   ...   |  ...   |
   |    998     | None | 990, 67 | ...  |  None   | 12, 89 |
   |    999     | None |  None   | ...  |  None   |  None  |

The above figure is a 1000 * 1000 corresponding matrix of the left image. Each element represents a position coordinates of correspondence point in the right image. 'None' means that there is no corresponding pixel in the right image. To generate Hpatch datasets, please run the file 'generate_hpatch_datasets.py' file, which is similar to the 'generate_gl3d_20_datasets.py' and and 'generate_gl3d_5_datasets'.

A Todo-list for the dataset generation:

1. Download the code to your local repository with the following command (if you are using the Windows system, please install git bash on your computer first, and then run the following command)

  > git clone  https://github.com/Livsdjo/Generate_Datasets_Code

  After running the command, a folder named Generation_Datasets_Code will be generated in the current path    

2. Use the following command to enter Generate_Datasets_Code folder

  > cd Generate_Datasets_Code

  Files  under this folder are described as follows:
  GL3D_20  (code for generate GL3D-20% datasets)
  GL3D_5             (code for generate  GL3D-5% datasets)
  Hpatch              (code for generate  Hpatch datasets)
  readme.md      (description of this repository) 

3. Use the following command to enter GL3D_5 folder

  > cd GL3D_5

  Files under this folder are described as follows:

  > GL3D-5  
       --Input_Data       (data such as input images)  
       --Sift                     (code for sift feature extraction and matching)  
       --Vision_Gemo   (code for epipolar geometry calculation)  
  config.py                                      (configuration for generate GL3D-5% datasets)  
  generate_scene1_dataset.py                       (code for generate the first scene datasets)  
  generate_scene2_dataset.py                       (code for generate the second scene datasets)  
  generate_scene3_dataset.py                       (code for generate the third scene datasets)  
  generate_gl3d_5_datasets.py                      (code for generate GL3D-5% datasets) 

4. Open the configuration file of generating GL3D-5% datasets with following command

  > vi config.py

  The config file uses the python argparse module which can parse the configuration parameters from the input command.
  The configuration items included in this config file are described as follows:

  > --datasets_output_Path                  (root path of output file)  
  --false_corr_geod                         (epipolar constraint threshold for detect false correspondences)  
  --sift_peak_thrld                         (peak threshold for sift algorithm)  
  --sift_edge_thrld                         (edge threshold for sift algorithm)

  these items can be modified if you need, otherwise default value

5. Use the following command to configure the root path of output file and run generate_gl3d_5_Datasets.py file to generate GL3D-5% datasets

  > python  generate_gl3d_5_datasets.py   --datasets_output_Path="Output file root path configured by yourself"

  Running python file will take a while. After finishing running, a folder named GL3D_5_Datasets will be generated in the output file root path. Files under GL3D_5_Datasets folder are described as follows:

  > GL3D_ 5_Datasets  
  --train  
  ----merge_imgs_data.pkl  
  ----label.pkl  
  ----others.pkl  
  ----xs_4.pkl  
  ----xs_12.pkl  
  --valid  
  ----merge_imgs_data.pkl  
  ----label.pkl  
  ----others.pkl  
  ----xs_4.pkl  
  ----xs_12.pkl  
  --test  
  ----merge_imgs_data.pkl  
  ----label.pkl  
  ----others.pkl  
  ----xs_4.pkl  
  ----xs_12.pkl

  The content and format of data for each pkl file are described as follows:
  merge_imgs_data.pkl file contains image index and image data.

  > image index(int16),image data(1000x1000 int16)  
  25   [126 127 178 148 190 ......  56 178 26 198 238 76]  
  ........

  label.pkl file contains correspondences labels(int16).

  > label(int16)  
  1 0 0 1 ........  0  0  1  
  ........

  others.pkl file contains image index and correct correspondences number.

  > left image index(int16),right image index(int16),correct correspondences number(int16)  
  12   25   67  
  ........

  xs_12.pkl file contains the correspondences 2d keypoints . Each keypoint is parameterized by a 2x3 transformation, composed of  keypoint position,   orientation and size.  Please refer to <<GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints>> in Sec.3.2 for specific details.

  > Parameterized keypoint(1x12 float32)  
  0.25712258 -0.52153296 -0.58181703 -0.38229898 -0.38166887 -0.53448558 0.02235542 0.56109281 0.36525526 -0.84163922  0.13353313 -0.16430511  
  ........

  xs_4.pkl file contains normalized correspondences 2d keypoints

  > Normalize keypoint(1x4 float32)  
  -0.125  0.625    0.125   0.250  
  ........




## Requirements ##

Please use Python 3.6, opencv-contrib-python (3.4.0.12). Other dependencies should be installed through pip or conda.


## For loading training, validation and testing datasets ##

Here we provide python scripts for generating matches on the refined GL3D and Hpatch datasets.

> cd gl3d_5  
python generate_gl3d_5_datasets.py  
cd gl3d_20  
python generate_gl3d_20_datasets.py  
cd hpatch  
python generate_hpatch_datasets.py  
