require 'nn'
require 'rnn'
require 'nngraph'
utils = dofile('utils.lua');

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test the VisDial model for retrieval')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-inputImg','data/data_img.h5','h5file path with image feature')
cmd:option('-inputQues','data/visdial_data.h5','h5file file with preprocessed questions')
cmd:option('-inputJson','data/visdial_params.json','json path with info and vocab')

cmd:option('-loadPath', 'checkpoints/model.t7', 'path to saved model')
cmd:option('-split', 'val', 'split to evaluate on')

-- Inference params
cmd:option('-batchSize', 30, 'Batch size (number of threads) (Adjust base on GRAM)')
cmd:option('-gpuid', 0, 'GPU id to use')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

cmd:option('-saveRanks', false, 'Whether to save ranks or not');
cmd:option('-saveRankPath', 'logs/ranks.json');

local opt = cmd:parse(arg);
print(opt)

-- seed for reproducibility
torch.manualSeed(1234);

-- set default tensor based on gpu usage
if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.backend == 'cudnn' then require 'cudnn' end
    cutorch.setDevice(opt.gpuid+1)
    cutorch.manualSeed(1234)
    torch.setdefaulttensortype('torch.CudaTensor');
else
    torch.setdefaulttensortype('torch.FloatTensor');
end

------------------------------------------------------------------------
-- Read saved model and parameters
------------------------------------------------------------------------
local savedModel = torch.load(opt.loadPath)

-- transfer all options to model
local modelParams = savedModel.modelParams

-- default values of these opts to load models saved with older version of code
if not modelParams.weightInit then modelParams.weightInit = 'xavier'; end
if not modelParams.saveIter then modelParams.saveIter = 2; end

opt.imgNorm = modelParams.imgNorm
opt.encoder = modelParams.encoder
opt.decoder = modelParams.decoder
modelParams.gpuid = opt.gpuid
modelParams.batchSize = opt.batchSize

-- add flags for various configurations
-- additionally check if its imitation of discriminative model
if string.match(opt.encoder, 'hist') then opt.useHistory = true; end
if string.match(opt.encoder, 'im') then opt.useIm = true; end
-- check if history is to be concatenated (only for late fusion encoder)
if string.match(opt.encoder, 'lf') then opt.concatHistory = true end

------------------------------------------------------------------------
-- Loading dataset
------------------------------------------------------------------------
local dataloader = dofile('dataloader.lua')
dataloader:initialize(opt, {opt.split});
collectgarbage();

------------------------------------------------------------------------
-- Setup the model
------------------------------------------------------------------------
require 'model'
local model = Model(modelParams)

-- copy the weights from loaded model
model.wrapperW:copy(savedModel.modelW);

------------------------------------------------------------------------
-- Evaluation
------------------------------------------------------------------------
print('Evaluating..')
local retrieval = model:retrieve(dataloader, opt.split);

if opt.saveRanks == true then
    print(string.format('Writing ranks to %s', opt.saveRankPath));
    utils.writeJSON(opt.saveRankPath, retrieval);
end
