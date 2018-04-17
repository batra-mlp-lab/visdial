require 'nn'
require 'nngraph'
require 'rnn'

------------------------------------------------------------------------
-- Input arguments and options
------------------------------------------------------------------------
local opt = require 'opts';
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

-- transfer all options to model
local modelParams = opt;

------------------------------------------------------------------------
-- Read saved model and parameters
------------------------------------------------------------------------
local savedModel = false;
if opt.loadPath ~= '' then
    savedModel  = torch.load(opt.loadPath);
    modelParams = savedModel.modelParams;

    opt.imgNorm = modelParams.imgNorm;
    opt.encoder = modelParams.encoder;
    opt.decoder = modelParams.decoder;
    modelParams.gpudid = opt.gpuid;
    modelParams.batchSize = opt.batchSize;
end

------------------------------------------------------------------------
-- Loading dataset
------------------------------------------------------------------------
local dataloader = dofile('dataloader.lua');
dataloader:initialize(opt, {'train'});
collectgarbage();

------------------------------------------------------------------------
-- Setting model parameters
------------------------------------------------------------------------
-- transfer parameters from dataloader to model
paramNames = {'numTrainThreads', 'numTestThreads', 'numValThreads',
                'vocabSize', 'maxQuesCount', 'maxQuesLen', 'maxAnsLen'};
for _, value in pairs(paramNames) do
    modelParams[value] = dataloader[value];
end

-- path to save the model
local modelPath = opt.savePath

-- creating the directory to save the model
paths.mkdir(modelPath);

-- Iterations per epoch
modelParams.numIterPerEpoch = math.ceil(modelParams.numTrainThreads /
                                                modelParams.batchSize);
print(string.format('\n%d iter per epoch.', modelParams.numIterPerEpoch));

------------------------------------------------------------------------
-- Setup the model
------------------------------------------------------------------------
require 'model'
local model = Model(modelParams);

if opt.loadPath ~= '' then
    model.wrapperW:copy(savedModel.modelW);
    model.optims.learningRate = savedModel.optims.learningRate;
end

------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------
print('Training..')
collectgarbage()

runningLoss = 0;
for iter = 1, modelParams.numEpochs * modelParams.numIterPerEpoch do
    -- forward and backward propagation
    model:trainIteration(dataloader);

    -- evaluate on val and save model
    if iter % (modelParams.saveIter * modelParams.numIterPerEpoch) == 0 then
        local currentEpoch = iter / modelParams.numIterPerEpoch

        -- save model and optimization parameters
        torch.save(string.format(modelPath .. 'model_epoch_%d.t7', currentEpoch),
                                                    {modelW = model.wrapperW,
                                                    optims = model.optims,
                                                    modelParams = modelParams})
        -- validation accuracy
        -- model:retrieve(dataloader, 'val');
    end

    -- print after every few iterations
    if iter % 100 == 0 then
        local currentEpoch = iter / modelParams.numIterPerEpoch;

        -- print current time, running average, learning rate, iteration, epoch
        print(string.format('[%s][Epoch:%.02f][Iter:%d][Loss:%.05f][lr:%f]',
                                os.date(), currentEpoch, iter, runningLoss,
                                            model.optims.learningRate))
    end
    if iter % 10 == 0 then collectgarbage(); end
end

-- Saving the final model
torch.save(modelPath .. 'model_final.t7', {modelW = model.wrapperW:float(),
                                            modelParams = modelParams});
