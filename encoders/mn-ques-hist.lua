require 'model_utils.MaskSoftMax'

local encoderNet = {}

function encoderNet.model(params)

    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) -- question
    table.insert(inputs, nn.Identity()()) -- history
    table.insert(inputs, nn.Identity()()) -- 10x10 mask

    local ques = inputs[1]
    local hist = inputs[2]
    local mask = inputs[3]

    -- word embed layer
    wordEmbed = nn.LookupTableMaskZero(params.vocabSize, params.embedSize);

    -- make clones for embed layer
    local qEmbed = nn.Dropout(0.5)(wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias')(ques));
    local hEmbed = nn.Dropout(0.5)(wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias')(hist));

    local lst1 = nn.SeqLSTM(params.embedSize, params.rnnHiddenSize)
    lst1:maskZero()

    local lst2 = nn.SeqLSTM(params.rnnHiddenSize, params.rnnHiddenSize)
    lst2:maskZero()

    local h1 = lst1(hEmbed)
    local h2 = lst2(h1)
    local h3 = nn.Select(1, -1)(h2)

    local lst3 = nn.SeqLSTM(params.embedSize, params.rnnHiddenSize)
    lst3:maskZero()

    local lst4 = nn.SeqLSTM(params.rnnHiddenSize, params.rnnHiddenSize)
    lst4:maskZero()

    local q1 = lst3(qEmbed)
    local q2 = lst4(q1)
    local q3 = nn.Select(1, -1)(q2)

    -- View as batch x rounds
    local qEmbedView = nn.View(-1, params.maxQuesCount, params.rnnHiddenSize)(q3)
    local hEmbedView = nn.View(-1, params.maxQuesCount, params.rnnHiddenSize)(h3)

    -- Inner product
    -- q is Bx10xE, h is Bx10xE
    -- qh is Bx10x10, rows correspond to questions, columns to facts
    local qh = nn.MM(false, true)({qEmbedView, hEmbedView})
    local qhView = nn.View(-1, params.maxQuesCount)(qh)
    local qhprobs = nn.MaskSoftMax(){qhView, mask}
    local qhView2 = nn.View(-1, params.maxQuesCount, params.maxQuesCount)(qhprobs)

    -- Weighted sum of h features
    -- h is Bx10xE, qhView2 is Bx10x10
    local hAtt = nn.MM(){qhView2, hEmbedView}
    local hAttView = nn.View(-1, params.rnnHiddenSize)(hAtt)

    local hAttTr = nn.Tanh()(nn.Linear(params.rnnHiddenSize, params.rnnHiddenSize)(nn.Dropout(0.5)(hAttView)))
    local qh2 = nn.Tanh()(nn.Linear(params.rnnHiddenSize, params.rnnHiddenSize)(nn.CAddTable(){hAttTr, nn.View(-1, params.rnnHiddenSize)(qEmbedView)}))

    table.insert(outputs, qh2)

    local enc = nn.gModule(inputs, outputs)
    enc.wordEmbed = wordEmbed

    return enc;
end

return encoderNet
