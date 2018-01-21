# Deformable-3D-ConvNets

Modified according to resnet2d

##Prerequisite
Pytorch 0.2.0\
Cuda 8.0\
TensorboardX\
PIL\
setproctitle tqdm colorama pandas numpy

The pretrained models are from https://github.com/kenshohara/3D-ResNets/releases/tag/1.0

First, you need to compile the deformable convolution operation in model/deform3d dir, change the path in make.sh to your own path, then execute bash make.sh.

For Ego-Gesture, you need to modify the data_root, for jester and something-something, you need to additionally provide csv root.