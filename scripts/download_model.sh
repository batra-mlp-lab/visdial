#!/bin/sh

print_usage () {
    echo "Usage: download_model.sh [vgg|resnet] [layers]"
    echo "For vgg, 'layers' can be one of {16, 19}"
    echo "For resnet, 'layers' can be one of {18, 34, 50, 101, 152, 200}"
}

if [ $1 = "vgg" ] 
then
    if [ $2 = "16" ] 
    then
        mkdir -p data/models/vgg16
        cd data/models/vgg16
        wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt
        wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
    elif [ $2 = "19" ] 
    then
        mkdir -p data/models/vgg19
        cd data/models/vgg19
        wget https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt
        wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
    else
        print_usage
    fi
elif [ $1 = "resnet" ] 
then
    if echo "18 34 50 101 152 200" | grep -w $2 > /dev/null
    then
        mkdir -p data/models/resnet
        cd data/models/resnet
        wget https://d2j0dndfm35trm.cloudfront.net/resnet-$2.t7
    else
        print_usage
    fi
else
    print_usage
fi

cd ../../..
