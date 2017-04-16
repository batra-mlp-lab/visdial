require 'model_utils.MaskTime'

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
                                -- :add(nn.Dropout(0.5))
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

    -- join both the tensors
    enc:add(nn.JoinTable(-1));

    -- change the view of the data
    -- always split it back wrt batch size and then do transpose
    enc:add(nn.View(-1, params.maxQuesCount, 2*params.rnnHiddenSize));
    enc:add(nn.Transpose({1, 2}));
    enc:add(nn.SeqLSTM(2*params.rnnHiddenSize, params.rnnHiddenSize))
    enc:add(nn.Transpose({1, 2}));
    enc:add(nn.View(-1, params.rnnHiddenSize))

    return enc;
end

return encoderNet
