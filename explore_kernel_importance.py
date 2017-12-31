'''Explore CNN Kernel Importance

Dataset and model: 1. Mnist+LeNet, 2. Cifar10+ConvNet, 3. ImageNet2012+(AlexNet, ResNet)

Method:
1. Disable each kernel of each layer, and observe the loss change
2. Find 5 most important kernels and 5 useless kernels
3. Find the image patch that most activate the selected kernels
'''
