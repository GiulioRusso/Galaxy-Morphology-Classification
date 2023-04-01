README Galaxy Morphology Classification

Galaxy Morphology Classification project on the Galaxy DECals dataset developed with
- python 3.9.5
- opencv-python 4.5.5.62
- torchvision 0.14.1


The dataset is divided in 10 classes:

Galaxy10 dataset (17733 images)
├── Class 0 (1081 images): Disturbed Galaxies
├── Class 1 (1853 images): Merging Galaxies
├── Class 2 (2645 images): Round Smooth Galaxies
├── Class 3 (2027 images): In-between Round Smooth Galaxies
├── Class 4 ( 334 images): Cigar Shaped Smooth Galaxies
├── Class 5 (2043 images): Barred Spiral Galaxies
├── Class 6 (1828 images): Unbarred Tight Spiral Galaxies
├── Class 7 (2627 images): Unbarred Loose Spiral Galaxies
├── Class 8 (1422 images): Edge-on Galaxies without Bulge
└── Class 9 (1873 images): Edge-on Galaxies with Bulge

The 10 classes were merged into the following 4:

Galaxy10 dataset (17733 images)
├── Class 0 (2934 images): Disturbed Galaxies + Merging Galaxies
├── Class 1 (5006 images): Round Smooth Galaxies + In-between Round Smooth Galaxies + Cigar Shaped Smooth Galaxies
├── Class 2 (6498 images): Barred Spiral Galaxies + Unbarred Tight Spiral Galaxies + Unbarred Loose Spiral Galaxies
└── Class 3 (3295 images): Edge-on Galaxies without Bulge + Edge-on Galaxies with Bulge

Due to neural network confusion, Class 0 on 4 classes is deleted:

Galaxy10 dataset (14799 images)
├── Class 0 (5006 images): Round Smooth Galaxies + In-between Round Smooth Galaxies + Cigar Shaped Smooth Galaxies
├── Class 1 (6498 images): Barred Spiral Galaxies + Unbarred Tight Spiral Galaxies + Unbarred Loose Spiral Galaxies
└── Class 2 (3295 images): Edge-on Galaxies without Bulge + Edge-on Galaxies with Bulge

The images have been extracted from an .h5 file following the tutorial https://astronn.readthedocs.io/en/latest/galaxy10.html. The images (named with progressive index Galaxy10_DECals-dataset-*index padded to 5 digist*.png) and the labels where saved on a .csv file



- DATA CLEANING

Images from the .h5 file where 17736, but three of that are black, with index:
11492
12659
15422

They have been removed from the images folder and from the .csv files of the splits and features.

Since the dataset accesses the images based on the index. For example if in the name of the images I have:
Galaxy10_DECals-dataset-11491
Galaxy10_DECals-dataset-11492
Galaxy10_DECals-dataset-11493

And the 11492 I remove it, I have:

Galaxy10_DECals-dataset-11491
Galaxy10_DECals-dataset-11493

But accessing via index, the image accessed with index 11492 will be the image:

Galaxy10_DECals-dataset-11493

But it will be identified as:

Galaxy10_DECals-dataset-11492

Therefore all images were renamed via the rename-images.sh script

Each .csv file was updated in the FILENAME column with Excel:
- split the columns based on the ‘,’ of the csv
- I replace the column with the contents of filename-list.txt (the .sh script create it)
- save with name in .csv format
- I open the file in textedit and replace the ';' with the ',' (Excel saves the csv with the ';' as separator instead of ',')



- PROJECT DESCRIPTION

AIA
|
|- dataset_extraction.py						            <- extract the images and labels from the .h5 file downloaded from https://astronn.readthedocs.io/en/latest/galaxy10.html
|- handcrafted_feature_extraction.py	                    <- extract features from image analysis techinques

The starting point is the file handcrafted_feature_extraction.py that save the features extracted from the images in a .csv file. The dataset is extracted using dataset_extraction.py using the .h5 file downloaded. It is recommended to specify the paths of the .h5 file, the path where the images will be saved, and the path where the .csv file of the features will be saved

ML
|
|- features
|     |
|     |- cnn-features-merged.csv				           <- features extracted from ResNet50 model on 4 classes (DELETED)
|     |- cnn-features.csv              			           <- features extracted from ResNet50 model on 10 classes (DELETED)
|     |- features_merged.csv					           <- all handcrafted features on 4 classes
|     |- features.csv 						               <- all handcrafted features on 10 classes
|
|- main.ipynb									           <- starting point of machine learning project
|- scores										           <- folder that contain the scores of the classification on test set
|- results										           <- folder that contain all the results of the experiments done

The starting point of the project is the python notebook main.ipynb. All the classification task is performed using the features specified in the relative folder
N.B. cnn-features-merged.csv and cnn-features.csv are deleted due to the file weight. They can be reobtained using the file in DL/net/model/feature_extraction.py

DL
|
|- dataset
|	  |
|	  |- annotations 							             <- contains all the annotations needed
|	  |		  |
|	  |		  |- all-3-augmented.csv 			             <- 3 classes with data augmentation
|	  |		  |- all-3.csv 					                 <- 3 classes
|	  |		  |- all-4-augmented.csv 			             <- 4 classes with data augmentation
|	  |		  |- all-4.csv 					                 <- 4 classes
|	  |		  |- all-10-augmented.csv 			             <- 10 classes with data augmentation
|	  |		  |- all-10.csv 					             <- 10 classes
|	  |		  |- debug-all-4.csv 				             <- debug annotation with 4 classes
|	  |		  |- debug-all-10.csv 				             <- debug annotation with 10 classes
|	  |		  |- filename-list.txt				             <- list of the file names of the images
|	  |		  |- statistics.csv                             <- contain the mean and std for each channel of the images
|	  |
|	  |- augmented 						                     <- folder that contain all the data augmentation images (DELETED)
|	  |
|	  |- data_augmentation.py 					             <- data augmentation code
|	  |
|	  |- Galaxy10_DECals.h5						             <- original file downloaded from https://astronn.readthedocs.io/en/latest/galaxy10.html (DELETED)
|	  |
|	  |- image-removed							             <- just for debug purpose
|	  |		  |
|	  |		  |- all-merged-no-rename-after-removing.csv    <- split file on 4 classes with all the data without rename after the delete of the 3 black images
|	  |		  |- all-no-rename-after-removing.csv           <- split file on 10 classes with all the data without rename after the delete of the 3 black images
|	  |		  |- Galaxy10_DECals-dataset-11492.png          <- image removed
|	  |		  |- Galaxy10_DECals-dataset-12659.png          <- image removed
|	  |		  |- Galaxy10_DECals-dataset-15422.png          <- image removed
|	  |
|	  |- images-3								             <- folder with images for 3 class problem (DELETED)
|	  |- images-3-augmented						             <- folder with images for 3 class problem with data augmentation (DELETED)
|	  |- images-10-4							             <- folder with images for 10 and 4 class problem (DELETED)
|	  |- images-10-4-augmented					             <- folder with images for 10 and 4 class problem with data augmentation (DELETED)
|	  |
|	  |- rename-images.sh						             <- script bash to rename the images if someone is removed
|
|
|- experiments									             <- folder containing all the folders of the different experiments
|	  |
|	  |- *folder named with the experiment ID*              <- every experiment create a folder identified with it’s own experiment identifier
|	   		  |
|	   		  |- debug-training					             <- folder containing all the .csv debug files of the training
|	   		  |		 |
|	  		  |		 |- e=i|b=j.csv				             <- file containing the j-batch infos at the i-epoch
|	  		  |		 |- e=i|metrics.csv			             <- metrics average on the i-epoch
|	  		  |
|	  		  |- debug-validation				             <- folder containing all the .csv debug files of the validation
|	  		  |		 |
|	  		  |		 |- e=i|validation-metrics.csv          <- file containing the metrics infos at the i-epoch
|	  		  |		 |- e=i|validation-scores.csv           <- scores on the i-epoch
|	  		  |
|	  		  |- errors.csv						             <- file of the wrong classified images
|	  		  |
|	  		  |- plot					                     <- folder containing all the graphs
|	  		  |	   |
|	  		  |	   |- f1.png					             <- f1 graph
|	  		  |	   |- loss.png					             <- loss graph
|	  		  |	   |- train-validation-accuracy.png         <- accuracy train/validation graph
|	  		  |
|	  		  |
|	 		  |- *experiment ID*.csv	                     <- file with all the experiment parameters
|	  		  |
|	  		  |- *experiment ID*.tar                        <- best model
|	  		  |
|	  		  |- resume					                     <- folder containing all the graphs
|	  		  |	   |
|	  		  |	   |- *experiment ID*.tar                   <- resume model
|	  		  |
|	 		  |- test-metrics.csv		                     <- test metrics
|	  		  |- test-scores.csv	                         <- test scores
|	  
|- main.py									                <- main file where the code start
|	  
|- net	  
 	|
 	|- dataset	 								            <- all the files that work on the dataset
 	|	  |
 	|	  |- dataset_split.py				                <- split the dataset in train, validation and test
 	|	  |- dataset_statistics.py				            <- extract the statistics values of the dataset
 	|	  |- dataset_trasform.py				            <- contain the application of all the transforms
 	|	  |- dataset.py							            <- dataset overload class
 	|	  |- transforms							            <- folder containing all the transforms
 	|	  		 |
 	|	  		 |- MinMaxNormalization.py		            <- min max normalization
 	|	  		 |- StandardNormalization.py	            <- standardization
 	|	  		 |- ToTensor.py	  				            <- tensor transform of the images
 	|	  
 	|- evaluation								            <- all the files for the metrics evaluations
    |      |
    |      |- metrics.py				                    <- all metrics to evaluate
    |      |- plot.py							            <- all plot functions
    |      |- save.py							            <- all functions that save every .csv file
    |
 	|- model
    |    |
    |    |- feature_extraction.py				            <- extract the features from the CNN model specified
    |    |- model_selector.py                              <- choose the model to train
    |
	|- reproducibility
    |        |
    |        |- reproducibility.py				            <- set the reproducibility of the experiments
	|
	|- test.py								                <- test function
	|
	|- train.py 								            <- train function


The starting point of the project is the main.py. The code can be launched by argument parsing like:
python3 main.py
--file (all-10.csv, all-4.csv ecc...)
--model (severa models are implemented: resnet18, resnet34, resnet50, resnet101, resnet152, vgg11, vgg13, vgg16, vgg19, densenet121, densenet161, densenet169, densenet201)
--bs (batch size)
--ep (number of epochs)
--lr (learning rate)
--ss (step size)
--gamma (gamma)
--m (momentum)
--opt (different optimizer are implemented: sgd, rms, adam)
--loss (ce for cross entropy, f for focal loss)
--test (include it to perform test, otherwise if not specified, the code perform training)
--statistics (to evaluate statistics and save it in statistics.csv)
--dataset (select the name of the dataset chosen according to the annotation file: images-10-4/images-10-4-augmented or images-3/images-3-augmented

e.g. python3 main.py --file all-10-augmented.csv --model resnet50 --bs 20 --ep 2 --lr 1e-4 --ss 1 --gamma 0.1 --m 0.9 --opt adam --dataset images-10-4-augmented --loss f --test --statistics

After launching the code, it will create a folder, named with the experiment identifier, in the experiments folder. Here all the debug files and results are stored, as shown in the project description. So, the experiment identifier is the key used to refer to a specific experiment

N.B. the .h5 file, the dataset images folder and the augmented images folder are deleted due to the file weight. To obtain the images is it necessary to download the .h5 file from the specified link, extract the images using the AIA/dataset_extraction.py (pay attention that the code save annotations and images, but all the annotations file are already created and saved in DL/dataset/annotations/...). From this all the procedures to create the dataset folder for 10/4/3 classes are mentioned down here using the rename-images.sh


The split is made handcrafted dividing the data in 50% training, 20% validation and 30% test.

On 10 classes:
----------------------------------
Dataset shape: 17733 elements     |
----------------------------------
Training set length: 8876         |
Validation set length: 3552       |
Test set length: 5305             |
----------------------------------
Class | Train | Validation | Test |
----------------------------------
  0   |  539  |    216     | 326  |
  1   |  927  |    372     | 554  |
  2   |  1323 |    530     | 792  |
  3   |  1015 |    405     | 607  |
  4   |  168  |    67      | 99   |
  5   |  1023 |    409     | 611  |
  6   |  916  |    367     | 545  |
  7   |  1314 |    526     | 787  |
  8   |  713  |    285     | 424  |
  9   |  938  |    375     | 560  |
----------------------------------

On 4 classes:
----------------------------------
Dataset shape: 17733 elements     |
----------------------------------
Training set length: 8869         |
Validation set length: 3547       |
Test set length: 5317             |
----------------------------------
Class | Train | Validation | Test |
----------------------------------
  0   |  1466 |    587     | 881  |
  1   |  2504 |    1001    | 1501 |
  2   |  3250 |    1300    | 1948 |
  3   |  1649 |    659     | 987  |
----------------------------------

On 3 classes: on the 4 class file (all-4.csv) we decide to remove the class 0. It is composed of 2934 images. So, a new dataset with only images of class 1, 2, 3 is created, where images are named from 0 to 14798 (17733 - 2934 = 14799 images), and the same is done for the annotation file all-3.csv, that contain only images with labels 1, 2, 3 named from 0 to 14798. But the code need a class numeration starting from 0. So, the .csv updated contained classes 1, 2, 3, but they were shifted to 0, 1, 2.
So:
Class 0: Round Smooth Galaxies + In-between Round Smooth Galaxies + Cigar Shaped Smooth Galaxies
Class 1: Barred Spiral Galaxies + Unbarred Tight Spiral Galaxies + Unbarred Loose Spiral Galaxies
Class 2: Edge-on Galaxies without Bulge + Edge-on Galaxies with Bulge
The ex Class 0: Disturbed Galaxies + Merging Galaxies is deleted

----------------------------------
Dataset shape: 14799 elements     |
----------------------------------
Training set length: 7403         |
Validation set length: 2960       |
Test set length: 4436             |
----------------------------------
Class | Train | Validation | Test |
----------------------------------
  0   |  2504 |    1001    | 1501 |
  1   |  3250 |    1300    | 1948 |
  2   |  1649 |    659     | 987  |
----------------------------------



- DATA AUGMENTATION

Also. A data augmentation process is done on the 10 classes:
class 4: + 1001 images from 17733 to 18733
class 0: + 406 images from 18734 to 19139
All the indexes to reproduce the data augmentation from the original dataset are in the comments of DL/dataset/data_augmentation.py

----------------------------------
Dataset shape: 19140 elements     |
----------------------------------
Training set length: 9680         |
Validation set length: 3753       |
Test set length: 5707             |
----------------------------------
Class | Train | Validation | Test |
----------------------------------
  0   |  771  |    274     | 442  |
  1   |  927  |    372     | 554  |
  2   |  1323 |    530     | 792  |
  3   |  1015 |    405     | 607  |
  4   |  740  |    210     | 385  |
  5   |  1023 |    409     | 611  |
  6   |  916  |    367     | 545  |
  7   |  1314 |    526     | 787  |
  8   |  713  |    285     | 424  |
  9   |  938  |    375     | 560  |
----------------------------------


So, we get the all-10-augmented.csv. From this, to get the all-4-augmented.csv:
Class 1 -> 0 (here fall the class 0 on 10 classes augmented)
Class 2, 3, 4 -> 1 (here fall the class 4 on 10 classes augmented)
Class 5, 6, 7 -> 2
Class 8, 9 -> 3

----------------------------------
Dataset shape: 19140 elements     |
----------------------------------
Training set length: 9680         |
Validation set length: 3753       |
Test set length: 5707             |
----------------------------------
Class | Train | Validation | Test |
----------------------------------
  0   |  1698 |    646     | 996  |
  1   |  3078 |    1145    | 1784 |
  2   |  3253 |    1302    | 1943 |
  3   |  1651 |    660     | 984  |
----------------------------------


So, we get the all-4-augmented.csv. From this, to get the all-3-augmented.csv:
Remove all the rows where Class is 0
Substitute the FILENAME column with names from 00000 to 15799 (effectively, the class 0 in 10 classes is included in class 0 of 4 classes. So, if it is removed, respect of the all-3 NOT augmented, we get the 1001 samples more of class 4, which falls into class 1 on 4 classes, which became class 0 on 3 classes. Finally, the new all-3-augmented has 14799 + 1001 = 15800)
Images of class 0 in 4 class have to be removed, so a new folder is created in order to do not lose the original dataset. We remove images from 00000 to 02933 AND also the images of class 0 on 10 classes added with the augmentation: images from 18734 to 19139
In the new folder for 3 class, images have to be renamed with the script rename-images.sh

----------------------------------
Dataset shape: 15800 elements     |
----------------------------------
Training set length: 7982         |
Validation set length: 3107       |
Test set length: 4711             |
----------------------------------
Class | Train | Validation | Test |
----------------------------------
  0   |  3078 |    1145    | 1784 |
  1   |  3253 |    1302    | 1943 |
  2   |  1651 |    660     | 984  |
----------------------------------


