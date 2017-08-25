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

```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
TORCH_LUA_VERSION=LUA51 ./install.sh
```

Additionally, our code uses the following packages: [torch/torch7][14], [torch/nn][15], [torch/nngraph][16], [Element-Research/rnn][17], [torch/image][18], [lua-cjson][19], [loadcaffe][20], [torch-hdf5][25]. After Torch is installed, these can be installed/updated using:

```
luarocks install torch
luarocks install nn
luarocks install nngraph
luarocks install rnn
luarocks install image
luarocks install lua-cjson
luarocks install loadcaffe
luarocks install luabitop
luarocks install totem
```

Installation instructions for torch-hdf5 are given [here][26].

### Running on GPUs

Although our code should work on CPUs, it is *highly* recommended to use GPU acceleration with [CUDA][21]. You'll also need [torch/cutorch][22] and [torch/cunn][23].

```
luarocks install cutorch
luarocks install cunn
```

## Training your own network

### Preprocessing VisDial

The preprocessing script is in Python, and you'll need to install [NLTK][24].

```
pip install nltk
pip install numpy
pip install h5py
python -c "import nltk; nltk.download('all')"
```

[VisDial v0.9][27] dataset can be downloaded and preprocessed as follows:

```
cd data
python prepro.py -download 1
cd ..
```

This will generate the files `data/visdial_data.h5` (contains tokenized captions, questions, answers, image indices) and `data/visdial_params.json` (contains vocabulary mappings and COCO image ids).

### Extracting image features

Since we don't finetune the CNN, training is significantly faster if image features are pre-extracted. We use image features from [VGG-16][28]. The model can be downloaded and features extracted using:

```
sh scripts/download_vgg16.sh
cd data
# For all models except mn-att-ques-im-hist
th prepro_img.lua -imageRoot /path/to/coco/images/ -gpuid 0
# For mn-att-ques-im-hist
th prepro_img_pool5.lua -imageRoot /path/to/coco/images -gpuid 0
```

This should generate `data/data_img.h5` containing features for COCO `train` and `val` splits corresponding to VisDial v0.9.

### Training

Finally, we can get to training models! All supported encoders are in the `encoders/` folder (`lf-ques`, `lf-ques-im`, `lf-ques-hist`, `lf-ques-im-hist`, `hre-ques-hist`, `hre-ques-im-hist`, `hrea-ques-im-hist`, `mn-ques-hist`, `mn-ques-im-hist`, `mn-att-ques-im-hist`), and decoders in the `decoders/` folder (`gen` and `disc`).

**Generative** (`gen`) decoding tries to maximize likelihood of ground-truth response and only has access to single input-output pairs of dialog, while **discriminative** (`disc`) decoding makes use of 100 candidate option responses provided for every round of dialog, and maximizes likelihood of correct option.

Encoders and decoders can be arbitrarily plugged together. For example, to train an HRE model with question and history information only (no images), and generative decoding:

```
th train.lua -encoder hre-ques-hist -decoder gen -gpuid 0
```

Similarly, to train a Memory Network model with question, image and history information, and discriminative decoding:

```
th train.lua -encoder mn-ques-im-hist -decoder disc -gpuid 0
```

The training script saves model snapshots at regular intervals in the `checkpoints/` folder.

It takes about 15-20 epochs to train models with generative decoding to convergence, and 4-8 epochs for discriminative decoding.

## Evaluation

We evaluate model performance by where it ranks human response given 100 response options for every round of dialog, based on retrieval metrics — mean reciprocal rank, R@1, R@5, R@10, mean rank.

Model evaluation can be run using:

```
th evaluate.lua -loadPath checkpoints/model.t7 -gpuid 0
```

Note that evaluation requires image features `data/data_img.h5`, tokenized dialogs `data/visdial_data.h5` and vocabulary mappings `data/visdial_params.json`.

## Running Beam Search & Visualizing Results

We also include code for running beam search on your model snapshots. This gives significantly nicer results than argmax decoding, and can be run as follows:

```
th generate.lua -loadPath checkpoints/model.t7 -maxThreads 50
```

This would compute predictions for 50 threads from the `val` split and save results in `vis/results/results.json`.

```
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

All files available for download [here][29].

* `visdial_data.h5`: Tokenized captions, questions, answers, image indices
* `visdial_params.json`: Vocabulary mappings and COCO image ids
* `data_img.h5`: VGG16 image features for COCO `train` and `val`

### Pretrained models

Model checkpoints available [here][30].

#### Discriminative decoding

* `hre-qih-d.t7`: **H**ierarchical **R**ecurrent **E**ncoder
* `hrea-qih-d.t7`: **H**ierarchical **R**ecurrent **E**ncoder with **A**ttention
* `mn-qih-d.t7`: **M**emory **N**etwork
* `lf-qih-d.t7`: **L**ate **F**usion

#### Generative decoding

* `hre-qih-g.t7`: **H**ierarchical **R**ecurrent **E**ncoder
* `hrea-qih-g.t7`: **H**ierarchical **R**ecurrent **E**ncoder with **A**ttention
* `mn-qih-g.t7`: **M**emory **N**etwork
* `lf-qih-g.t7`: **L**ate **F**usion

## Contributors

* [Abhishek Das][2] (abhshkdz@gatech.edu)
* [Satwik Kottur][3]
* [Avi Singh][5]

## License

BSD


[1]: https://arxiv.org/abs/1611.08669
[2]: https://abhishekdas.com
[3]: https://satwikkottur.github.io
[4]: http://www.linkedin.com/in/khushi-gupta-9a678448
[5]: http://people.eecs.berkeley.edu/~avisingh/
[6]: http://deshraj.github.io
[7]: http://users.ece.cmu.edu/~moura/
[8]: https://computing.ece.vt.edu/~parikh/
[9]: https://computing.ece.vt.edu/~batra/
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
[29]: https://computing.ece.vt.edu/~abhshkdz/visdial/
[30]: https://computing.ece.vt.edu/~abhshkdz/visdial/models/


