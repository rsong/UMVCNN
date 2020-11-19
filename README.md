# UMVCNN
Unsupervised Multi-View CNN for Salient View Selection of 3D Objects and Scenes

Make sure that the dependencies are correctly installed and compiled. If you already have them on your compouter, please make sure that the files listed below (maybe more than them) are added or replaced with the corresponding ones provided by us.

.\dependencies\matconvnet\vl_nnloss.m

.\dependencies\matconvnet\matlab\+dagnn\Loss.m

.\dependencies\matconvnet\matlab\layers\SP.m


The pretrained network can be downloaded from https://drive.google.com/file/d/1ock5q7MYtFdaSEVe0YTJ6zUeLPeXh5ie/view?usp=sharing. Please save the downloaded network named 'net-deployed-bestv.mat' into the .\pretrained\ directory.

For a demo, simply implement the following lines in MATLAB command window

setup;

bv_view_p = bestview('airplane_0001.off');

If you want to train it from the baseline VGG net, please follow the steps below:

Download the dataset of the rendered images subject to 30 object categories from https://drive.google.com/drive/folders/1torDio1Of7dMWCvPrUyaxaP6OxoAl3JK. Unzip the dataset (including the imdb.mat file that removes the categorical information of the objects) to the .\data\model30\ directory.

Download the baseline VGG19 net from the official website of matconvnet or
https://drive.google.com/open?id=1jezzXXTUnySn0tTtpt41lP61o8r1c71N

Save it in .\data\models\

Then, call the cnn_shape file using the pieces of codes below:

setup; % Ignore this line if it has been implemented.

expDir = 'data';

cnn_shape('model30', 'expDir', fullfile(expDir,'models'), 'numFetchThreads', 8, 'pad', 32, 'border', 32, 'batchSize', 5, 'maxIterPerEpoch', Inf, 'numEpochs', 50, 'learningRate', 0.00001);

Please cite our paper if you use the codes:
Ran Song, Wei Zhang, Yitian Zhao, Yonghuai Liu. Unsupervised Multi-View CNN for Salient View Selection of 3D Objects and Scenes. In Proc. ECCV, 2020.


