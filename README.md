Study of perceptual cues of median plane sound localization
============================

Implementation of [Learning from Between-class Examples for Deep Sound Recognition](https://arxiv.org/abs/1711.10282) by Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada (ICLR 2018).


## News

## Contents
- Training, Test and Recording
	- train the model with training dataset convolved with certain type of HRTF
   	- test the model with test dataset convolved with certain type of HRTF


## Setup
- Install [Chainer](https://chainer.org/) v7.8.1 on a machine with CUDA GPU.


## Training
- Template:

		python main.py --dataset [samrai_original, samrai_arma, samrai_averaged or samrai_N1N2P1P2] --netType [envnetstereov2] --data path/to/dataset/directory/ (--save path/to/save/directory) (--testOnly)
 
- Recipes:
	- Traning of EnvNetstereo_v2 on samrai_original:

			python main.py --dataset samrai_original --netType envnetstereov2 --data path/to/dataset/directory/
   	- Test and recording of EnvNetstereo_v2 on samrai_test_original:
   	  		
			python main.py --dataset samrai_test_original --netType envnetstereov2 --data path/to/dataset/directory/ --testOnly --save path/to/save/directory
- Notes:
	- Validation accuracy is calculated using 10-crop testing.
	- By default, it performs K-fold cross validation using the original fold settings. You can run on a particular split by using --split command.
	- Please check [opts.py](https://github.com/mil-tokyo/bc_learning_sound/blob/master/opts.py) for other command line arguments.


## Results



## See also
[Between-class Learning for Image Clasification](https://arxiv.org/abs/1711.10284) ([github](https://github.com/mil-tokyo/bc_learning_image))

---
<i id=1></i><sup>1</sup> Training/testing schemes are simplified from those in the ICASSP paper.

<i id=2></i><sup>2</sup> It is higher than that reported in the ICASSP paper (36% error), mainly because here we use 4 out of 5 folds for training, whereas we used only 3 folds in the ICASSP paper.

#### Reference
<i id=1></i>[1] Karol J Piczak. Esc: Dataset for environmental sound classification. In *ACM Multimedia*, 2015.

<i id=2></i>[2] Justin Salamon, Christopher Jacoby, and Juan Pablo Bello. A dataset and taxonomy for urban sound research. In *ACM Multimedia*, 2014.
