cmd = torch.CmdLine()
cmd:text('Train the Visual Dialog model')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-inputImg', 'data/data_img.h5', 'HDF5 file with image features')
cmd:option('-inputQues', 'data/visdial_data.h5', 'HDF5 file with preprocessed questions')
cmd:option('-inputJson', 'data/visdial_params.json', 'JSON file with info and vocab')
cmd:option('-savePath', 'checkpoints/', 'Path to save checkpoints')

-- specify encoder/decoder
cmd:option('-encoder', 'lf-ques-hist', 'Name of the encoder to use')
cmd:option('-decoder', 'gen', 'Name of the decoder to use (gen/disc)')
cmd:option('-imgNorm', 1, 'normalize the image feature. 1=yes, 0=no')

-- model params
cmd:option('-imgEmbedSize', 300, 'Size of the multimodal embedding')
cmd:option('-imgFeatureSize', 4096, 'Size of the image feature');
cmd:option('-embedSize', 300, 'Size of input word embeddings')
cmd:option('-rnnHiddenSize', 512, 'Size of the LSTM state')
cmd:option('-maxHistoryLen', 60, 'Maximum history to consider when using concatenated QA pairs');
cmd:option('-numLayers', 2, 'Number of layers in LSTM')

-- optimization params
cmd:option('-batchSize', 40, 'Batch size (number of threads) (Adjust base on GPU memory)')
cmd:option('-learningRate', 1e-3, 'Learning rate')
cmd:option('-dropout', 0.5, 'Dropout')
cmd:option('-numEpochs', 400, 'Epochs')
cmd:option('-LRateDecay', 10, 'After lr_decay epochs lr reduces to 0.1*lr')
cmd:option('-lrDecayRate', 0.9997592083, 'Decay for learning rate')
cmd:option('-minLRate', 5e-5, 'Minimum learning rate')
cmd:option('-gpuid', 0, 'GPU id to use')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

local opts = cmd:parse(arg);

-- if save path is not given, use default — time
-- get the current time
local curTime = os.date('*t', os.time());
-- create another folder to avoid clutter
local modelPath = string.format('checkpoints/model-%d-%d-%d-%d:%d:%d-%s-%s/',
                                curTime.month, curTime.day, curTime.year,
                                curTime.hour, curTime.min, curTime.sec,
                                opts.encoder, opts.decoder)
if opts.savePath == 'checkpoints/' then opts.savePath = modelPath end;

-- check for inputs required
if string.match(opts.encoder, 'hist') then opts.useHistory = true end
if string.match(opts.encoder, 'im') then opts.useIm = true end

-- check if history is to be concatenated (only for late fusion encoder)
if string.match(opts.encoder, 'lf') then opts.concatHistory = true end

-- attention is always on conv features, not fc7
if string.match(opts.encoder, 'att') then
    opts.inputImg = 'data/data_img_pool5.h5'
    opts.imgNorm = 0
end

return opts;
