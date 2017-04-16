require 'model_utils.MaskTime'
require 'model_utils.MaskFuture'
require 'model_utils.ReplaceZero'

local encoderNet = {}

function encoderNet.model(params)
    local dropout = params.dropout or 0.5

    -- Use `nngraph`
    nn.FastLSTM.usenngraph = true;

    -- encoder network
    local enc = nn.Sequential();

    -- create the two branches
    local concat = nn.ConcatTable();

    -- word branch, along with embedding layer
    enc.wordEmbed = nn.LookupTableMaskZero(params.vocabSize, params.embedSize);
    local wordBranch = nn.Sequential():add(nn.SelectTable(1)):add(enc.wordEmbed);

    -- make clones for embed layer
    local qEmbedNet = enc.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias');
    local hEmbedNet = enc.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias');

    -- create two branches
    local histBranch = nn.Sequential()
                            :add(nn.SelectTable(3))
                            :add(hEmbedNet);
    enc.histLayers = {};
    -- number of layers to read the history
    for layer = 1, params.numLayers do
        local inputSize = (layer == 1) and params.embedSize 
                                    or params.rnnHiddenSize;
        enc.histLayers[layer] = nn.SeqLSTM(inputSize, params.rnnHiddenSize); 
        enc.histLayers[layer]:maskZero();

        histBranch:add(enc.histLayers[layer]);
    end
    histBranch:add(nn.Select(1, -1));

    -- image branch
    -- embedding for images
    local imgPar = nn.ParallelTable()
                        :add(nn.Identity())
                        :add(nn.Sequential()
                                :add(nn.Dropout(0.5))
                                :add(nn.Linear(params.imgFeatureSize,
                                                params.imgEmbedSize)));
    -- select words and image only
    local imageBranch = nn.Sequential()
                            :add(nn.NarrowTable(1, 2)) 
                            :add(imgPar)
                            :add(nn.MaskTime(params.imgEmbedSize));

    -- add concatTable and join
    concat:add(wordBranch)
    concat:add(imageBranch)
    concat:add(histBranch)
    enc:add(concat);

    -- another concat table
    local concat2 = nn.ConcatTable();

    -- select words + image, and history
    local wordImageBranch = nn.Sequential()
                                :add(nn.NarrowTable(1, 2))
                                :add(nn.JoinTable(-1))

    -- language model
    enc.rnnLayers = {};
    for layer = 1, params.numLayers do
        local inputSize = (layer==1) and (params.imgEmbedSize+params.embedSize)
                                    or params.rnnHiddenSize;
        enc.rnnLayers[layer] = nn.SeqLSTM(inputSize, params.rnnHiddenSize);
        enc.rnnLayers[layer]:maskZero();

        wordImageBranch:add(enc.rnnLayers[layer]);
    end
    wordImageBranch:add(nn.Select(1, -1));

    -- add both the branches (wordImage, select history) to concat2
    concat2:add(wordImageBranch):add(nn.SelectTable(3));
    enc:add(concat2);

    -- single layer neural network
    -- create attention based history
    local prepare = nn.Sequential()
                        :add(nn.Linear(params.rnnHiddenSize, 1))
                        :add(nn.View(-1, params.maxQuesCount))
    local wordHistBranch = nn.Sequential()
                            :add(nn.ParallelTable()
                                    :add(prepare:clone())
                                    :add(prepare:clone()))
                            :add(nn.ParallelTable()
                                    :add(nn.Replicate(10, 3))
                                    :add(nn.Replicate(10, 2)))
                            :add(nn.CAddTable())
                            --:add(nn.Tanh())
                            :add(nn.MaskFuture(params.maxQuesCount))
                            :add(nn.View(-1, params.maxQuesCount))
                            :add(nn.ReplaceZero(-1*math.huge))
                            :add(nn.SoftMax())
                            :add(nn.View(-1, params.maxQuesCount, params.maxQuesCount))
                            :add(nn.Replicate(params.rnnHiddenSize, 4));

    local histOnlyBranch = nn.Sequential()
                            :add(nn.SelectTable(2))
                            :add(nn.View(-1, params.maxQuesCount, params.rnnHiddenSize))
                            :add(nn.Replicate(params.maxQuesCount, 2))

    -- add another concatTable to create attention over history
    local concat3 = nn.ConcatTable()
                        :add(wordHistBranch)
                        :add(histOnlyBranch)
                        :add(nn.SelectTable(1)) -- append attended history with question
    enc:add(concat3);

    -- parallel table to multiply first two tables, and leave the third one untouched
    local multiplier = nn.Sequential()
                        :add(nn.NarrowTable(1, 2))
                        :add(nn.CMulTable())
                        :add(nn.Sum(3))
                        :add(nn.View(-1, params.rnnHiddenSize));
    local concat4 = nn.ConcatTable()
                        :add(multiplier)
                        :add(nn.SelectTable(3));
    enc:add(concat4);

    -- join both the tensors (att over history and encoded question)
    enc:add(nn.JoinTable(-1));
    enc:add(nn.View(-1, params.maxQuesCount, 2*params.rnnHiddenSize))
    enc:add(nn.Transpose({1, 2}))
    enc:add(nn.SeqLSTM(2 * params.rnnHiddenSize, params.rnnHiddenSize))
    enc:add(nn.Transpose({1, 2}))
    enc:add(nn.View(-1, params.rnnHiddenSize))

    return enc;
end

return encoderNet
