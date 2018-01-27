#!/bin/bash

mkdir -p /home/ninad/ASU_DATA/datasets/mnist

if ! [ -e /home/ninad/ASU_DATA/datasets/mnist/train-images-idx3-ubyte.gz ]
	then
		wget -P /home/ninad/ASU_DATA/datasets/mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d /home/ninad/ASU_DATA/datasets/mnist/train-images-idx3-ubyte.gz

if ! [ -e /home/ninad/ASU_DATA/datasets/mnist/train-labels-idx1-ubyte.gz ]
	then
		wget -P /home/ninad/ASU_DATA/datasets/mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d /home/ninad/ASU_DATA/datasets/mnist/train-labels-idx1-ubyte.gz

if ! [ -e /home/ninad/ASU_DATAdatasets/mnist/t10k-images-idx3-ubyte.gz ]
	then
		wget -P /home/ninad/ASU_DATAdatasets/mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d /home/ninad/ASU_DATA/datasets/mnist/t10k-images-idx3-ubyte.gz

if ! [ -e /home/ninad/ASU_DATA/datasets/mnist/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P /home/ninad/ASU_DATA/datasets/mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d /home/ninad/ASU_DATA/datasets/mnist/t10k-labels-idx1-ubyte.gz

