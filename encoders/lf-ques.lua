local encoderNet = {};

function encoderNet.model(params)
    local dropout = params.dropout or 0.5;
    -- Use `nngraph`
    nn.FastLSTM.usenngraph = true;

    -- encoder network
    local enc = nn.Sequential();

    -- word branch, along with embedding layer
    enc.wordEmbed = nn.LookupTableMaskZero(params.vocabSize, params.embedSize);
    local wordBranch = nn.Sequential():add(nn.SelectTable(1)):add(enc.wordEmbed);

    -- language model
    enc.rnnLayers = {};
    for layer = 1, params.numLayers do
        local inputSize = (layer==1) and (params.embedSize)
                                    or params.rnnHiddenSize;
        enc.rnnLayers[layer] = nn.SeqLSTM(inputSize, params.rnnHiddenSize);
        enc.rnnLayers[layer]:maskZero();

        wordBranch:add(enc.rnnLayers[layer]);
    end
    wordBranch:add(nn.Select(1, -1));

    enc:add(wordBranch);

    if dropout > 0 then
        enc:add(nn.Dropout(dropout))
    end
    enc:add(nn.Linear(params.rnnHiddenSize, 512))
    enc:add(nn.Tanh())

    return enc;
end

return encoderNet;
