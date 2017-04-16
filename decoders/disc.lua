local decoderNet = {}

function decoderNet.model(params, enc)
    local optionLSTM = nn.SeqLSTM(params.embedSize, params.rnnHiddenSize)
    optionLSTM.batchfirst = true

    local optionEnc = {}
    local numOptions = 100
    for i = 1, numOptions do
        optionEnc[i] = nn.Sequential()
        optionEnc[i]:add(nn.Select(2,i))
        optionEnc[i]:add(enc.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias'));
        optionEnc[i]:add(optionLSTM:clone('weight', 'bias', 'gradWeight', 'gradBias'))
        optionEnc[i]:add(nn.Select(2,-1))
        optionEnc[i]:add(nn.Reshape(1,params.rnnHiddenSize, true)) -- True ensures that the first dimension remains the batch size
    end
    optionEncConcat = nn.Concat(2)
    for i = 1, numOptions do
        optionEncConcat:add(optionEnc[i])
    end

    local jointModel = nn.ParallelTable()
    jointModel:add(optionEncConcat)
    jointModel:add(nn.Reshape(params.rnnHiddenSize, 1, true))

    local dec = nn.Sequential()
    dec:add(jointModel)
    dec:add(nn.MM())
    dec:add(nn.Squeeze())

    return dec;
end

-- dummy forwardConnect
function decoderNet.forwardConnect(enc, dec, encOut, seqLen) end

-- dummy backwardConnect
function decoderNet.backwardConnect(enc, dec) end

return decoderNet;
