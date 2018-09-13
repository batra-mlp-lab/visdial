# VisDial

Code for the paper

**[Visual Dialog][1]**  
Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José M. F. Moura, Devi Parikh, Dhruv Batra  
[arxiv.org/abs/1611.08669][1]  
[CVPR 2017][10] (Spotlight)

**Visual Dialog** requires an AI agent to hold a meaningful dialog with humans in natural, conversational language about visual content. Given an image, dialog history, and a follow-up question about the image, the AI agent has to answer the question.

Demo: [demo.visualdialog.org][11]

<!-- [![Vimeo](http://i.imgur.com/18aMyaj.png)][12] -->

This repository contains code for **training**, **evaluating** and **visualizing results** for all combinations of encoder-decoder architectures described in the paper. Specifically, we have 3 encoders: **Late Fusion** (LF), **Hierarchical Recurrent Encoder** (HRE), **Memory Network** (MN), and 2 kinds of decoding: **Generative** (G) and **Discriminative** (D).

[![models](http://i.imgur.com/mdSOZPj.jpg)][1]

If you find this code useful, consider citing our work:

```
@inproceedings{visdial,
  title={{V}isual {D}ialog},
  author={Abhishek Das and Satwik Kottur and Khushi Gupta and Avi Singh
    and Deshraj Yadav and Jos\'e M.F. Moura and Devi Parikh and Dhruv Batra},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

## Setup

All our code is implemented in [Torch][13] (Lua). Installation instructions are as follows:

```sh
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
TORCH_LUA_VERSION=LUA51 ./install.sh
```

Additionally, our code uses the following packages: [torch/torch7][14], [torch/nn][15], [torch/nngraph][16], [Element-Research/rnn][17], [torch/image][18], [lua-cjson][19], [loadcaffe][20], [torch-hdf5][25]. After Torch is installed, these can be installed/updated using:

```sh
luarocks install torch
luarocks install nn
luarocks install nngraph
luarocks install image
luarocks install lua-cjson
luarocks install loadcaffe
luarocks install luabitop
luarocks install totem
```

**NOTE**: `luarocks install rnn` defaults to [torch/rnn][33], follow these steps to install [Element-Research/rnn][17].

```sh
git clone https://github.com/Element-Research/rnn.git
cd rnn
luarocks make rocks/rnn-scm-1.rockspec
```

Installation instructions for torch-hdf5 are given [here][26].

**NOTE**: torch-hdf5 does not work with few versions of gcc. It is recommended that you use gcc 4.8 / gcc 4.9 with Lua 5.1 for proper installation of torch-hdf5. 

### Running on GPUs

Although our code should work on CPUs, it is *highly* recommended to use GPU acceleration with [CUDA][21]. You'll also need [torch/cutorch][22], [torch/cudnn][31] and [torch/cunn][23].

```sh
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```

## Training your own network

### Preprocessing VisDial

The preprocessing script is in Python, and you'll need to install [NLTK][24].

```sh
pip install nltk
pip install numpy
pip install h5py
python -c "import nltk; nltk.download('all')"
```

[VisDial v1.0][27] dataset can be downloaded and preprocessed as specified below. The path provided as `-image_root` must have four subdirectories - [`train2014`][34] and [`val2014`][35] as per COCO dataset, `VisualDialog_val2018` and `VisualDialog_test2018` which can be downloaded from [here][27].

```sh
cd data
python prepro.py -download -image_root /path/to/images
cd ..
```

To download and preprocess [Visdial v0.9][27] dataset, provide an extra `-version 0.9` argument while execution.

This script will generate the files `data/visdial_data.h5` (contains tokenized captions, questions, answers, image indices) and `data/visdial_params.json` (contains vocabulary mappings and COCO image ids).


### Extracting image features

Since we don't finetune the CNN, training is significantly faster if image features are pre-extracted. Currently this repository provides support for extraction from VGG-16 and ResNets. We use image features from [VGG-16][28]. The VGG-16 model can be downloaded and features extracted using:

```sh
sh scripts/download_model.sh vgg 16  # works for 19 as well
cd data
# For all models except mn-att-ques-im-hist
th prepro_img_vgg16.lua -imageRoot /path/to/images -gpuid 0
# For mn-att-ques-im-hist
th prepro_img_vgg16.lua -imageRoot /path/to/images -imgSize 448 -layerName pool5 -gpuid 0
```

Similarly, [ResNet models][32] released by Facebook can be used for feature extraction. Feature extraction can be carried out in a similar manner as VGG-16:

```sh
sh scripts/download_model.sh resnet 200  # works for 18, 34, 50, 101, 152 as well
cd data
th prepro_img_resnet.lua -imageRoot /path/to/images -cnnModel /path/to/t7/model -gpuid 0
```

Running either of these should generate `data/data_img.h5` containing features for `train`, `val` and `test` splits corresponding to VisDial v1.0.


### Training

Finally, we can get to training models! All supported encoders are in the `encoders/` folder (`lf-ques`, `lf-ques-im`, `lf-ques-hist`, `lf-ques-im-hist`, `hre-ques-hist`, `hre-ques-im-hist`, `hrea-ques-im-hist`, `mn-ques-hist`, `mn-ques-im-hist`, `mn-att-ques-im-hist`), and decoders in the `decoders/` folder (`gen` and `disc`).

**Generative** (`gen`) decoding tries to maximize likelihood of ground-truth response and only has access to single input-output pairs of dialog, while **discriminative** (`disc`) decoding makes use of 100 candidate option responses provided for every round of dialog, and maximizes likelihood of correct option.

Encoders and decoders can be arbitrarily plugged together. For example, to train an HRE model with question and history information only (no images), and generative decoding:

```sh
th train.lua -encoder hre-ques-hist -decoder gen -gpuid 0
```

Similarly, to train a Memory Network model with question, image and history information, and discriminative decoding:

```sh
th train.lua -encoder mn-ques-im-hist -decoder disc -gpuid 0
```

**Note:** For attention based encoders, set both `imgSpatialSize` and `imgFeatureSize` command line params, feature dimensions are interpreted as `(batch X spatial X spatial X feature)`. For other encoders, `imgSpatialSize` is redundant.

The training script saves model snapshots at regular intervals in the `checkpoints/` folder.

It takes about 15-20 epochs to train models with generative decoding to convergence, and 4-8 epochs for discriminative decoding.

## Evaluation

We evaluate model performance by where it ranks human response given 100 response options for every round of dialog, based on retrieval metrics — mean reciprocal rank, R@1, R@5, R@10, mean rank.

Model evaluation can be run using:

```sh
th evaluate.lua -loadPath checkpoints/model.t7 -gpuid 0
```

Note that evaluation requires image features `data/data_img.h5`, tokenized dialogs `data/visdial_data.h5` and vocabulary mappings `data/visdial_params.json`.

## Running Beam Search & Visualizing Results

We also include code for running beam search on your model snapshots. This gives significantly nicer results than argmax decoding, and can be run as follows:

```sh
th generate.lua -loadPath checkpoints/model.t7 -maxThreads 50
```

This would compute predictions for 50 threads from the `val` split and save results in `vis/results/results.json`.

```sh
cd vis
# python 3.6
python -m http.server
# python 2.7
# python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser to see generated results.

Sample results from HRE-QIH-G available [here](https://computing.ece.vt.edu/~abhshkdz/visdial/browse_results/).

![](http://i.imgur.com/R3HJ2E5.gif)

## Download Extracted Features & Pretrained Models

### v0.9

Extracted features for v0.9 train and val are available for download.

* [`visdial_data.h5`](https://s3.amazonaws.com/visual-dialog/data/v0.9/visdial_data.h5): Tokenized captions, questions, answers, image indices
* [`visdial_params.json`](https://s3.amazonaws.com/visual-dialog/data/v0.9/visdial_params.json): Vocabulary mappings and COCO image ids
* [`data_img_vgg16_relu7.h5`](https://s3.amazonaws.com/visual-dialog/data/v0.9/data_img_vgg16_relu7.h5): VGG16 `relu7` image features
* [`data_img_vgg16_pool5.h5`](https://s3.amazonaws.com/visual-dialog/data/v0.9/data_img_vgg16_pool5.h5): VGG16 `pool5` image features

#### Pretrained models

Trained on v0.9 `train`, results on v0.9 `val`.

<table>
    <thead>
        <tr>
            <th><sup><sub>Encoder</sub></sup></th><th><sup><sub>Decoder</sub></sup></th><th><sup><sub>CNN</sub></sup></th><th><sup><sub>MRR</sub></sup></th><th><sup><sub>R@1</sub></sup></th><th><sup><sub>R@5</sub></sup></th><th><sup><sub>R@10</sub></sup></th><th><sup><sub>MR</sub></sup></th><th><sup><sub>Download</sub></sup></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><sup><sub>lf-ques</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5048</sub></sup></td><td><sup><sub>0.3974</sub></sup></td><td><sup><sub>0.6067</sub></sup></td><td><sup><sub>0.6649</sub></sup></td><td><sup><sub>17.8003</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-ques-gen-vgg16-18.t7"><sup><sub>lf-ques-gen-vgg16-18</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>lf-ques-hist</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5099</sub></sup></td><td><sup><sub>0.4012</sub></sup></td><td><sup><sub>0.6155</sub></sup></td><td><sup><sub>0.6740</sub></sup></td><td><sup><sub>17.3974</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-ques-hist-gen-vgg16-18.t7"><sup><sub>lf-ques-hist-gen-vgg16-18</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>lf-ques-im</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5206</sub></sup></td><td><sup><sub>0.4206</sub></sup></td><td><sup><sub>0.6165</sub></sup></td><td><sup><sub>0.6760</sub></sup></td><td><sup><sub>17.0578</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-ques-im-gen-vgg16-22.t7"><sup><sub>lf-ques-im-gen-vgg16-22</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>lf-ques-im-hist</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5146</sub></sup></td><td><sup><sub>0.4086</sub></sup></td><td><sup><sub>0.6205</sub></sup></td><td><sup><sub>0.6828</sub></sup></td><td><sup><sub>16.7553</sub></sup></td><td><sup><sub><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-ques-im-hist-gen-vgg16-26.t7">lf-ques-im-hist-gen-vgg16-26</a></sub></sup></td>
        </tr>
        <tr>
            <td><sup><sub>lf-att-ques-im-hist</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5354</sub></sup></td><td><sup><sub>0.4354</sub></sup></td><td><sup><sub>0.6355</sub></sup></td><td><sup><sub>0.6941</sub></sup></td><td><sup><sub>16.7663</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-att-ques-im-hist-gen-vgg16-80.t7"><sup><sub>lf-att-ques-im-hist-gen-vgg16-80</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>hre-ques-hist</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5089</sub></sup></td><td><sup><sub>0.4000</sub></sup></td><td><sup><sub>0.6154</sub></sup></td><td><sup><sub>0.6739</sub></sup></td><td><sup><sub>17.3618</sub></sup></td><td><sup><sub><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/hre-ques-hist-gen-vgg16-18.t7">hre-ques-hist-gen-vgg16-18</a></sub></sup></td>
        </tr>
        <tr>
            <td><sup><sub>hre-ques-im-hist</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5237</sub></sup></td><td><sup><sub>0.4223</sub></sup></td><td><sup><sub>0.6228</sub></sup></td><td><sup><sub>0.6811</sub></sup></td><td><sup><sub>16.9669</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/hre-ques-im-hist-gen-vgg16-14.t7"><sup><sub>hre-ques-im-hist-gen-vgg16-14</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>hrea-ques-im-hist</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5238</sub></sup></td><td><sup><sub>0.4213</sub></sup></td><td><sup><sub>0.6244</sub></sup></td><td><sup><sub>0.6842</sub></sup></td><td><sup><sub>16.6044</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/hrea-ques-im-hist-gen-vgg16-24.t7"><sup><sub>hrea-ques-im-hist-gen-vgg16-24</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>mn-ques-hist</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5131</sub></sup></td><td><sup><sub>0.4057</sub></sup></td><td><sup><sub>0.6176</sub></sup></td><td><sup><sub>0.6770</sub></sup></td><td><sup><sub>17.6253</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/mn-ques-hist-gen-vgg16-102.t7"><sup><sub>mn-ques-hist-gen-vgg16-102</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>mn-ques-im-hist</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5258</sub></sup></td><td><sup><sub>0.4229</sub></sup></td><td><sup><sub>0.6274</sub></sup></td><td><sup><sub>0.6874</sub></sup></td><td><sup><sub>16.9871</sub></sup></td><td><sup><sub><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/mn-ques-im-hist-gen-vgg16-78.t7">mn-ques-im-hist-gen-vgg16-78</a></sub></sup></td>
        </tr>
        <tr>
            <td><sup><sub>mn-att-ques-im-hist</sub></sup></td><td><sup><sub>gen</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5341</sub></sup></td><td><sup><sub>0.4354</sub></sup></td><td><sup><sub>0.6318</sub></sup></td><td><sup><sub>0.6903</sub></sup></td><td><sup><sub>17.0726</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/mn-att-ques-im-hist-gen-vgg16-100.t7"><sup><sub>mn-att-ques-im-hist-gen-vgg16-100</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>lf-ques</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5491</sub></sup></td><td><sup><sub>0.4113</sub></sup></td><td><sup><sub>0.7020</sub></sup></td><td><sup><sub>0.7964</sub></sup></td><td><sup><sub>7.1519</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-ques-disc-vgg16-10.t7"><sup><sub>lf-ques-disc-vgg16-10</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>lf-ques-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5724</sub></sup></td><td><sup><sub>0.4319</sub></sup></td><td><sup><sub>0.7308</sub></sup></td><td><sup><sub>0.8251</sub></sup></td><td><sup><sub>6.2847</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-ques-hist-disc-vgg16-8.t7"><sup><sub>lf-ques-hist-disc-vgg16-8</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>lf-ques-im</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5745</sub></sup></td><td><sup><sub>0.4331</sub></sup></td><td><sup><sub>0.7398</sub></sup></td><td><sup><sub>0.8340</sub></sup></td><td><sup><sub>5.9801</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-ques-im-disc-vgg16-12.t7"><sup><sub>lf-ques-im-disc-vgg16-12</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>lf-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5911</sub></sup></td><td><sup><sub>0.4490</sub></sup></td><td><sup><sub>0.7563</sub></sup></td><td><sup><sub>0.8493</sub></sup></td><td><sup><sub>5.5493</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-ques-im-hist-disc-vgg16-8.t7"><sup><sub>lf-ques-im-hist-disc-vgg16-8</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>lf-att-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.6079</sub></sup></td><td><sup><sub>0.4692</sub></sup></td><td><sup><sub>0.7731</sub></sup></td><td><sup><sub>0.8635</sub></sup></td><td><sup><sub>5.1965</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/lf-att-ques-im-hist-disc-vgg16-20.t7"><sup><sub>lf-att-ques-im-hist-disc-vgg16-20</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>hre-ques-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5668</sub></sup></td><td><sup><sub>0.4265</sub></sup></td><td><sup><sub>0.7245</sub></sup></td><td><sup><sub>0.8207</sub></sup></td><td><sup><sub>6.3701</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/hre-ques-hist-disc-vgg16-4.t7"><sup><sub>hre-ques-hist-disc-vgg16-4</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>hre-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5818</sub></sup></td><td><sup><sub>0.4461</sub></sup></td><td><sup><sub>0.7373</sub></sup></td><td><sup><sub>0.8342</sub></sup></td><td><sup><sub>5.9647</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/hre-ques-im-hist-disc-vgg16-4.t7"><sup><sub>hre-ques-im-hist-disc-vgg16-4</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>hrea-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5821</sub></sup></td><td><sup><sub>0.4456</sub></sup></td><td><sup><sub>0.7378</sub></sup></td><td><sup><sub>0.8341</sub></sup></td><td><sup><sub>5.9646</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/hrea-ques-im-hist-disc-vgg16-4.t7"><sup><sub>hrea-ques-im-hist-disc-vgg16-4</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>mn-ques-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5831</sub></sup></td><td><sup><sub>0.4388</sub></sup></td><td><sup><sub>0.7507</sub></sup></td><td><sup><sub>0.8434</sub></sup></td><td><sup><sub>5.8090</sub></sup></td><td><sup><sub><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/mn-ques-hist-disc-vgg16-20.t7">mn-ques-hist-disc-vgg16-20</a></sub></sup></td>
        </tr>
        <tr>
            <td><sup><sub>mn-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.5971</sub></sup></td><td><sup><sub>0.4562</sub></sup></td><td><sup><sub>0.7627</sub></sup></td><td><sup><sub>0.8539</sub></sup></td><td><sup><sub>5.4218</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/mn-ques-im-hist-disc-vgg16-12.t7"><sup><sub>mn-ques-im-hist-disc-vgg16-12</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>mn-att-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.6082</sub></sup></td><td><sup><sub>0.4700</sub></sup></td><td><sup><sub>0.7724</sub></sup></td><td><sup><sub>0.8623</sub></sup></td><td><sup><sub>5.2930</sub></sup></td><td><sup><sub><a href="https://s3.amazonaws.com/visual-dialog/models/v0.9/mn-att-ques-im-hist-disc-vgg16-28.t7">mn-att-ques-im-hist-disc-vgg16-28</a></sub></sup></td>
        </tr>
    </tbody>
</table>

### v1.0

Extracted features for v1.0 train, val and test are available for download.

* [`visdial_data_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/visdial_data_train.h5): Tokenized captions, questions, answers, image indices, for training on `train`
* [`visdial_params_train.json`](https://s3.amazonaws.com/visual-dialog/data/v1.0/visdial_params_train.json): Vocabulary mappings and COCO image ids for training on `train`
* [`data_img_vgg16_relu7_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/data_img_vgg16_relu7_train.h5): VGG16 `relu7` image features for training on `train`
* [`data_img_vgg16_pool5_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/data_img_vgg16_pool5_train.h5): VGG16 `pool5` image features for training on `train`
* [`visdial_data_trainval.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/visdial_data_trainval.h5): Tokenized captions, questions, answers, image indices, for training on `train`+`val`
* [`visdial_params_trainval.json`](https://s3.amazonaws.com/visual-dialog/data/v1.0/visdial_params_trainval.json): Vocabulary mappings and COCO image ids for training on `train`+`val`
* [`data_img_vgg16_relu7_trainval.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/data_img_vgg16_relu7_trainval.h5): VGG16 `relu7` image features for training on `train`+`val`
* [`data_img_vgg16_pool5_trainval.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/data_img_vgg16_pool5_trainval.h5): VGG16 `pool5` image features for training on `train`+`val`

#### Pretrained models

Trained on v1.0 `train` + v1.0 `val`, results on v1.0 `test-std`. Leaderboard [here][evalai-leaderboard].

<table>
    <thead>
        <tr>
            <th><sup><sub>Encoder</sub></sup></th><th><sup><sub>Decoder</sub></sup></th><th><sup><sub>CNN</sub></sup></th><th><sup><sub>NDCG</sub></sup></th><th><sup><sub>MRR</sub></sup></th><th><sup><sub>R@1</sub></sup></th><th><sup><sub>R@5</sub></sup></th><th><sup><sub>R@10</sub></sup></th><th><sup><sub>MR</sub></sup></th><th><sup><sub>Download</sub></sup></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><sup><sub>lf-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.4531</sub></sup></td><td><sup><sub>0.5542</sub></sup></td><td><sup><sub>40.95</sub></sup></td><td><sup><sub>72.45</sub></sup></td><td><sup><sub>82.83</sub></sup></td><td><sup><sub>5.9532</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v1.0/lf-ques-im-hist-disc-vgg16-8.t7"><sup><sub>lf-ques-im-hist-disc-vgg16-8</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>hre-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.4546</sub></sup></td><td><sup><sub>0.5416</sub></sup></td><td><sup><sub>39.93</sub></sup></td><td><sup><sub>70.45</sub></sup></td><td><sup><sub>81.50</sub></sup></td><td><sup><sub>6.4082</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v1.0/hre-ques-im-hist-disc-vgg16-4.t7"><sup><sub>hre-ques-im-hist-disc-vgg16-4</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>mn-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.4750</sub></sup></td><td><sup><sub>0.5549</sub></sup></td><td><sup><sub>40.98</sub></sup></td><td><sup><sub>72.30</sub></sup></td><td><sup><sub>83.30</sub></sup></td><td><sup><sub>5.9245</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v1.0/mn-ques-im-hist-disc-vgg16-12.t7"><sup><sub>mn-ques-im-hist-disc-vgg16-12</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>lf-att-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.4976</sub></sup></td><td><sup><sub>0.5707</sub></sup></td><td><sup><sub>42.08</sub></sup></td><td><sup><sub>74.82</sub></sup></td><td><sup><sub>85.05</sub></sup></td><td><sup><sub>5.4092</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v1.0/lf-att-ques-im-hist-disc-vgg16-24.t7"><sup><sub>lf-att-ques-im-hist-disc-vgg16-24</sub></sup></a></td>
        </tr>
        <tr>
            <td><sup><sub>mn-att-ques-im-hist</sub></sup></td><td><sup><sub>disc</sub></sup></td><td><sup><sub>VGG-16</sub></sup></td><td><sup><sub>0.4958</sub></sup></td><td><sup><sub>0.5690</sub></sup></td><td><sup><sub>42.42</sub></sup></td><td><sup><sub>74.00</sub></sup></td><td><sup><sub>84.35</sub></sup></td><td><sup><sub>5.5852</sub></sup></td><td><a href="https://s3.amazonaws.com/visual-dialog/models/v1.0/mn-att-ques-im-hist-disc-vgg16-24.t7"><sup><sub>mn-att-ques-im-hist-disc-vgg16-24</sub></sup></a></td>
        </tr>
    </tbody>
</table>

## License

BSD


[1]: https://arxiv.org/abs/1611.08669
[2]: https://abhishekdas.com
[3]: https://satwikkottur.github.io
[4]: http://www.linkedin.com/in/khushi-gupta-9a678448
[5]: http://people.eecs.berkeley.edu/~avisingh/
[6]: http://deshraj.github.io
[7]: http://users.ece.cmu.edu/~moura/
[8]: http://www.cc.gatech.edu/~parikh/
[9]: http://www.cc.gatech.edu/~dbatra
[10]: http://cvpr2017.thecvf.com/
[11]: http://demo.visualdialog.org
[12]: https://vimeo.com/193092429
[13]: http://torch.ch/
[14]: https://github.com/torch/torch7
[15]: https://github.com/torch/nn
[16]: https://github.com/torch/nngraph
[17]: https://github.com/Element-Research/rnn/
[18]: https://github.com/torch/image
[19]: https://luarocks.org/modules/luarocks/lua-cjson
[20]: https://github.com/szagoruyko/loadcaffe
[21]: https://developer.nvidia.com/cuda-toolkit
[22]: https://github.com/torch/cutorch
[23]: https://github.com/torch/cunn
[24]: http://www.nltk.org/
[25]: https://github.com/deepmind/torch-hdf5
[26]: https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md
[27]: https://visualdialog.org/data
[28]: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
[31]: https://www.github.com/soumith/cudnn.torch
[32]: https://github.com/facebook/fb.resnet.torch/tree/master/pretrained
[33]: https://github.com/torch/rnn
[34]: http://images.cocodataset.org/zips/train2014.zip
[35]: http://images.cocodataset.org/zips/val2014.zip
[evalai-leaderboard]: https://evalai.cloudcv.org/web/challenges/challenge-page/103/leaderboard/298
