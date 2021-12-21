# Mitigating the Imacts of Spurious Data Correlations on In-Distribution Classificaton and Out-of-Distribution Detection

## Data

The datasets are identical to a subset of the ones used in [Spurious_OOD](https://github.com/deeplearning-wisc/Spurious_OOD). 
So, this setup process is only necessary if never having run Spurious_OOD before. Otherwise, it is possible to
simply copy the data files over. Also make sure to update data paths in datasets/generate_placebg.py and 
datasets/generate_waterbird.py

### ID and SPOOD

Download [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and update the path to this dataset in generate_waterbird.py.

For [Places](http://places2.csail.mit.edu), ensure the file structure is as follows: the overall dataset is in a directory places365_standard 
(modify the path to this directory in datasets/generate_placebg.py and datasets/generate_waterbird.py). 
Inside are categories_places365.txt, places365_train_standard.txt, 
places365_val.txt, places365_test.txt and a directory data_large. You will download 
train_large_places365standard.tar, val_large.tar and test_large.tar, rename their expanded directories to
train, val and test respectively, and place train, val and test inside data_large. In train, 
move all images in l/lake/natural into l/lake and delete the natural directory.
In categories_places365.txt, replace the line "/l/lake/natural 205" with "/l/lake 205".

Next run datasets/generate_placebg.py to generate the subset of Places used in the SPOOD dataset. Also
run datasets/generate_waterbird.py to create the ID dataset.

### NSPOOD

Download [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz), [LSUN_resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz) and [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat), placing them in directory datasets/ood_datasets. The .mat file for SVHN
should be named selected_test_32x32.mat and placed in a directory called SVHN. The iSUN directory
should be called iSUN and LSUN directory called LSUN_resize.

## Run experiments

The code was developed using Python 3.8.11 and PyTorch 1.9.0.

Simply run `./runexp.sh waterbird 0.9 7` to do an experiment. The first argument is the dataset,
second argument is the amount of correlation between class and environment and the third
argument is the GPU to use.

The three trained checkpoints used for the paper's results are in 
experiments/waterbird/pres*. To use them, change the `name` variable in runexp.sh 
to pres1, pres2 or pres3 and comment out the training command (line 16). Note that
some variance across runs is expected (occurs regardless of sparsification).

When the experiment is done running, the shell script prints out the experiment_name. 
Files generated from the experiment will be in experiments/waterbird/experiment_name.
