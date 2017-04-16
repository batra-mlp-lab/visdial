local decoderNet = {}

function decoderNet.model(params, enc)
    -- use `nngraph`
    nn.FastLSTM.usenngraph = true

    -- decoder network
    local dec = nn.Sequential()
    -- use the same embedding for both encoder and decoder lstm
    local embedNet = enc.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias');
    dec:add(embedNet);

    dec.rnnLayers = {};
    -- check if decoder has different hidden size
    local hiddenSize = (params.ansHiddenSize ~= 0) and params.ansHiddenSize
                                            or params.rnnHiddenSize;
    for layer = 1, params.numLayers do
        local inputSize = (layer == 1) and params.embedSize or hiddenSize;
        dec.rnnLayers[layer] = nn.SeqLSTM(inputSize, hiddenSize);
        dec.rnnLayers[layer]:maskZero();
        dec:add(dec.rnnLayers[layer]);
    end
    dec:add(nn.Sequencer(nn.MaskZero(nn.Linear(hiddenSize, params.vocabSize), 1)))
    dec:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))
    
    return dec;
end

-- transfer the hidden state from encoder to decoder
function decoderNet.forwardConnect(enc, dec, encOut, seqLen)
    if enc.rnnLayers ~= nil then
        for ii = 1, #enc.rnnLayers do
            dec.rnnLayers[ii].userPrevOutput = enc.rnnLayers[ii].output[seqLen];
            dec.rnnLayers[ii].userPrevCell = enc.rnnLayers[ii].cell[seqLen];
        end

        -- last layer gets output gradients
        dec.rnnLayers[#enc.rnnLayers].userPrevOutput = encOut;
    else
        dec.rnnLayers[#dec.rnnLayers].userPrevOutput = encOut
    end
end

-- transfer gradients from decoder to encoder
function decoderNet.backwardConnect(enc, dec)
    if enc.rnnLayers ~= nil then
        -- borrow gradients from decoder
        for ii = 1, #dec.rnnLayers do
            enc.rnnLayers[ii].userNextGradCell = dec.rnnLayers[ii].userGradPrevCell;
            enc.rnnLayers[ii].gradPrevOutput = dec.rnnLayers[ii].userGradPrevOutput;
        end

        -- return the gradients for the last layer
        return dec.rnnLayers[#enc.rnnLayers].userGradPrevOutput;
    else
        return dec.rnnLayers[#dec.rnnLayers].userGradPrevOutput
    end
end

-- connecting decoder to itself; useful while sampling
function decoderNet.decoderConnect(dec)
    for ii = 1, #dec.rnnLayers do
        dec.rnnLayers[ii].userPrevCell = dec.rnnLayers[ii].cell[1]
        dec.rnnLayers[ii].userPrevOutput = dec.rnnLayers[ii].output[1]
    end
end

return decoderNet;
