-- abstract class for models
require 'model_utils.optim_updates'
require 'xlua'
require 'hdf5'

local utils = require 'utils'

local Model = torch.class('Model');

-- initialize
function Model:__init(params)
    print('Setting up model..')
    self.params = params

    print('Encoder: ', params.encoder)
    print('Decoder: ', params.decoder)

    -- build the model - encoder, decoder
    local encFile = string.format('encoders/%s.lua', params.encoder);
    local encoder = dofile(encFile);

    local decFile = string.format('decoders/%s.lua', params.decoder);
    local decoder = dofile(decFile);

    enc = encoder.model(params)
    dec = decoder.model(params, enc)

    local decMethods = {'forwardConnect', 'backwardConnect', 'decoderConnect'}
    for key, value in pairs(decMethods) do self[value] = decoder[value]; end

    -- criterion
    if params.decoder == 'gen' then
        self.criterion = nn.ClassNLLCriterion();
        self.criterion.sizeAverage = false;
        self.criterion = nn.SequencerCriterion(
                                    nn.MaskZeroCriterion(self.criterion, 1));
    elseif params.decoder == 'disc' then
        self.criterion = nn.CrossEntropyCriterion()
    end

    -- wrap the models
    self.wrapper = nn.Sequential():add(enc):add(dec);

    -- initialize weights
    self.wrapper = require('model_utils/weight-init')(self.wrapper, params.weightInit);

    -- ship to gpu if necessary
    if params.gpuid >= 0 then
        self.wrapper = self.wrapper:cuda();
        self.criterion = self.criterion:cuda();
    end

    self.encoder = self.wrapper:get(1);
    self.decoder = self.wrapper:get(2);
    self.wrapperW, self.wrapperdW = self.wrapper:getParameters();

    self.wrapper:training();

    -- setup the optimizer
    self.optims = {};
    self.optims.learningRate = params.learningRate;
end

-------------------------------------------------------------------------------
-- One iteration of training -- forward and backward pass
function Model:trainIteration(dataloader)
    -- clear the gradients
    self.wrapper:zeroGradParameters();

    -- grab a training batch
    local batch = dataloader:getTrainBatch(self.params);

    -- call the internal function for model specific actions
    local curLoss = self:forwardBackward(batch);

    if self.params.decoder == 'gen' then
        -- count the number of tokens
        local numTokens = torch.sum(batch['answer_out']:gt(0));

        -- update the running average of loss
        if runningLoss > 0 then
            runningLoss = 0.95 * runningLoss + 0.05 * curLoss/numTokens;
        else
            runningLoss = curLoss/numTokens;
        end
    elseif self.params.decoder == 'disc' then
        -- update the running average of loss
        if runningLoss > 0 then
            runningLoss = 0.95 * runningLoss + 0.05 * curLoss
        else
            runningLoss = curLoss
        end
    end

    -- clamp gradients
    self.wrapperdW:clamp(-5.0, 5.0);

    -- update parameters
    adam(self.wrapperW, self.wrapperdW, self.optims);

    -- decay learning rate, if needed
    if self.optims.learningRate > self.params.minLRate then
        self.optims.learningRate = self.optims.learningRate *
                                        self.params.lrDecayRate;
    end
end
---------------------------------------------------------------------
-- validation performance on test/val
function Model:evaluate(dataloader, dtype)
    -- change to evaluate mode
    self.wrapper:evaluate();

    local curLoss = 0;
    local startId = 1;
    local numThreads = dataloader.numThreads[dtype];

    local numTokens = 0;
    while startId <= numThreads do
        -- print progress
        xlua.progress(startId, numThreads);

        -- grab a validation batch
        local batch, nextStartId
                        = dataloader:getTestBatch(startId, self.params, dtype);
        -- count the number of tokens
        numTokens = numTokens + torch.sum(batch['answer_out']:gt(0));
        -- forward pass to compute loss
        curLoss = curLoss + self:forwardBackward(batch, true);
        startId = nextStartId;
    end

    -- print the results
    curLoss = curLoss / numTokens;
    print(string.format('\n%s\tLoss: %f\t Perplexity: %f\n', dtype,
                        curLoss, math.exp(curLoss)));

    -- change back to training
    self.wrapper:training();
end

-- retrieval performance on val
function Model:retrieve(dataloader, dtype)
    -- change to evaluate mode
    self.wrapper:evaluate();

    local curLoss = 0;
    local startId = 1;
    local numThreads = dataloader.numThreads[dtype];
    print('numThreads', numThreads)

    local ranks = torch.Tensor(numThreads, self.params.maxQuesCount);
    ranks:fill(self.params.numOptions + 1);

    while startId <= numThreads do
        -- print progress
        xlua.progress(startId, numThreads);

        -- grab a batch
        local batch, nextStartId =
                        dataloader:getTestBatch(startId, self.params, dtype);

        -- Call retrieve function for specific model, and store ranks
        ranks[{{startId, nextStartId - 1}, {}}] = self:retrieveBatch(batch);
        startId = nextStartId;
    end

    print(string.format('\n%s - Retrieval:', dtype))
    utils.processRanks(ranks);

    -- change back to training
    self.wrapper:training();

    local retrieval = {};
    local ranks = torch.totable(ranks:double());
    for i = 1, #dataloader['unique_img_'..dtype] do
        for j = 1, 10 do
            table.insert(retrieval, {
                image_id = dataloader['unique_img_'..dtype][i];
                round_id = j;
                ranks = ranks[i][j]
            })
        end
    end
    -- collect garbage
    collectgarbage();

    return retrieval;
end

-- prediction on val/test
function Model:predict(dataloader, dtype)
    -- change to evaluate mode
    self.wrapper:evaluate();

    local curLoss = 0;
    local startId = 1;
    local numThreads = dataloader.numThreads[dtype];
    self.params.numOptions = 100;
    print('numThreads', numThreads)

    local ranks = torch.Tensor(numThreads, 10, self.params.numOptions);
    ranks:fill(self.params.numOptions + 1);

    while startId <= numThreads do
        -- print progress
        xlua.progress(startId, numThreads);

        -- grab a batch
        local batch, nextStartId =
                        dataloader:getTestBatch(startId, self.params, dtype);

        -- Call retrieve function for specific model, and store ranks
        ranks[{{startId, nextStartId - 1}, {}}] = self:retrieveBatch(batch)
                        :view(nextStartId - startId, -1, self.params.numOptions);
        startId = nextStartId;
    end

    -- change back to training
    self.wrapper:training();

    local prediction = {};
    local ranks = torch.totable(ranks:double());
    for i = 1, #dataloader['unique_img_'..dtype] do
        -- rank list for all rounds in val split and last round in test split
        if dtype == 'test' then
            table.insert(prediction, {
                image_id = dataloader['unique_img_'..dtype][i];
                round_id = dataloader[dtype..'_num_rounds'][i];
                ranks = ranks[i][dataloader[dtype..'_num_rounds'][i]]
            })
        else
            for j = 1, 10 do
                table.insert(prediction, {
                    image_id = dataloader['unique_img_'..dtype][i];
                    round_id = j;
                    ranks = ranks[i][j]
                })
            end
        end
    end
    -- collect garbage
    collectgarbage();

    return prediction;
end

-- forward + backward pass
function Model:forwardBackward(batch, onlyForward, encOutOnly)
    local onlyForward = onlyForward or false;
    local encOutOnly = encOutOnly or false
    local inputs = {}

    -- transpose for timestep first
    local batchQues = batch['ques_fwd']
    batchQues = batchQues:view(-1, batchQues:size(3)):t()
    table.insert(inputs, batchQues)

    if self.params.useIm == true then
        local imgFeats = batch['img_feat']
        -- if attention, then conv layer features
        if string.match(self.params.encoder, 'att') then
            imgFeats = imgFeats:view(-1, 1, params.imgSpatialSize, params.imgSpatialSize, params.imgFeatureSize)
            imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1, 1, 1)
            imgFeats = imgFeats:view(-1, params.imgSpatialSize, params.imgSpatialSize, params.imgFeatureSize)
        else
            imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize)
            imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1)
            imgFeats = imgFeats:view(-1, self.params.imgFeatureSize)
        end
        table.insert(inputs, imgFeats)
    end

    if self.params.useHistory == true then
        local batchHist = batch['hist']
        batchHist = batchHist:view(-1, batchHist:size(3)):t()
        table.insert(inputs, batchHist)
    end

    if string.match(self.params.encoder, 'mn') then
        local mask = torch.ones(10, 10):byte()
        for i = 1, 10 do
            for j = 1, 10 do
                if j <= i then
                    mask[i][j] = 0
                end
            end
        end
        if self.params.gpuid >= 0 then
            mask = mask:cuda()
        end
        local maskRepeat = torch.repeatTensor(mask, batch['hist']:size(1), 1)
        table.insert(inputs, maskRepeat)
    end

    -- encoder forward pass
    local encOut = self.encoder:forward(inputs)

    -- coupled enc-dec (only for gen)
    self.forwardConnect(self.encoder, self.decoder, encOut, batchQues:size(1));

    if encOutOnly == true then return encOut end

    -- decoder forward pass
    local curLoss = 0
    if self.params.decoder == 'gen' then
        local answerIn = batch['answer_in'];
        answerIn = answerIn:view(-1, answerIn:size(3)):t();

        local answerOut = batch['answer_out'];
        answerOut = answerOut:view(-1, answerOut:size(3)):t();

        local decOut = self.decoder:forward(answerIn);
        curLoss = self.criterion:forward(decOut, answerOut);

        -- backward pass
        if onlyForward == false then
            local gradCriterionOut = self.criterion:backward(decOut, answerOut);
            self.decoder:backward(answerIn, gradCriterionOut);

            --backward connect decoder and encoder (only for gen)
            local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
            self.encoder:backward(inputs, gradDecOut)
        end
    elseif self.params.decoder == 'disc' then
        local options = batch['options']
        local answerInd = batch['answer_ind']

        local decOut = self.decoder:forward({options, encOut})
        curLoss = self.criterion:forward(decOut, answerInd)

        -- backward pass
        if onlyForward == false then
            local gradCriterionOut = self.criterion:backward(decOut, answerInd)
            local t = self.decoder:backward({options, encOut}, gradCriterionOut)

            self.encoder:backward(inputs, t[2])
        end
    end

    return curLoss;
end

function Model:retrieveBatch(batch)
    local inputs = {}

    local batchQues = batch['ques_fwd'];
    batchQues = batchQues:view(-1, batchQues:size(3)):t();
    table.insert(inputs, batchQues)

    if self.params.useIm == true then
        local imgFeats = batch['img_feat']
        -- if attention, then conv layer features
        if string.match(self.params.encoder, 'att') then
            imgFeats = imgFeats:view(-1, 1, params.imgSpatialSize, params.imgSpatialSize, params.imgFeatureSize)
            imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1, 1, 1)
            imgFeats = imgFeats:view(-1, params.imgSpatialSize, params.imgSpatialSize, params.imgFeatureSize)
        else
            imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize)
            imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1)
            imgFeats = imgFeats:view(-1, self.params.imgFeatureSize)
        end
        table.insert(inputs, imgFeats)
    end

    if self.params.useHistory == true then
        local batchHist = batch['hist']
        batchHist = batchHist:view(-1, batchHist:size(3)):t()
        table.insert(inputs, batchHist)
    end

    if string.match(self.params.encoder, 'mn') then
        local mask = torch.ones(10, 10):byte()
        for i = 1, 10 do
            for j = 1, 10 do
                if j <= i then
                    mask[i][j] = 0
                end
            end
        end
        if self.params.gpuid >= 0 then
            mask = mask:cuda()
        end
        local maskRepeat = torch.repeatTensor(mask, batch['hist']:size(1), 1)
        table.insert(inputs, maskRepeat)
    end

    -- forward pass
    local encOut = self.encoder:forward(inputs)
    local batchSize = batchQues:size(2);

    if self.params.decoder == 'gen' then
        local optionIn = batch['option_in'];
        optionIn = optionIn:view(-1, optionIn:size(3), optionIn:size(4));

        local optionOut = batch['option_out'];
        optionOut = optionOut:view(-1, optionOut:size(3), optionOut:size(4));
        optionIn = optionIn:transpose(1, 2):transpose(2, 3);
        optionOut = optionOut:transpose(1, 2):transpose(2, 3);

        -- tensor holds the likelihood for all the options
        local optionLhood = torch.Tensor(self.params.numOptions, batchSize);

        -- repeat for each option and get gt rank
        for opId = 1, self.params.numOptions do
            -- forward connect encoder and decoder
            self.forwardConnect(self.encoder, self.decoder, encOut, batchQues:size(1));

            local curOptIn = optionIn[opId];
            local curOptOut = optionOut[opId];
            local decOut = self.decoder:forward(curOptIn);

            -- compute the probabilities for each answer, based on its tokens
            optionLhood[opId] = utils.computeLhood(curOptOut, decOut);
        end
        -- gtPosition can be nil if ground truth does not exist
        local gtPosition = batch['answer_ind'];

        -- return the ranks for this batch
        return utils.computeRanks(optionLhood:t(), gtPosition);
    elseif self.params.decoder == 'disc' then
        local options = batch['options']
        local decOut = self.decoder:forward({options, encOut})
        local gtPosition = batch['answer_ind'];

        -- return the ranks for this batch
        return utils.computeRanks(decOut, gtPosition)
    end

end

function Model:generateAnswers(dataloader, dtype, params)
    -- check decoder
    if self.params.decoder == 'disc' then
        print('Sampling/beam search only for generative model')
        os.exit()
    end

    -- setting the options for beam search / sampling
    params = params or {};

    -- sample or take max
    local sampleWords = params.sampleWords and params.sampleWords == 1 or false;
    local temperature = params.temperature or 1.0;
    local beamSize = params.beamSize or 5;
    local beamLen = params.beamLen or 20;

    print('Beam size', beamSize)
    print('Beam length', beamLen)

    -- endToken index
    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['<END>'];
    local numThreads = params.maxThreads or dataloader.numThreads[dtype];
    print('No. of threads', numThreads)

    local answerTable = {}
    for convId = 1, numThreads do
        xlua.progress(convId, numThreads);
        self.wrapper:evaluate()

        local inds = torch.LongTensor(1):fill(convId);
        local batch = dataloader:getIndexData(inds, self.params, dtype);
        local numQues = batch['ques_fwd']:size(1) * batch['ques_fwd']:size(2);

        local encOut = self:forwardBackward(batch, true, true)
        local threadAnswers = {}

        if sampleWords == false then
            -- do beam search for each example now
            for iter = 1, 10 do
                local encInSeq = batch['ques_fwd']:view(-1, batch['ques_fwd']:size(3)):t();
                encInSeq = encInSeq[{{},{iter}}]:squeeze():float()

                -- beams
                local beams = torch.LongTensor(beamLen, beamSize):zero();

                -- initial hidden states for the beam at current round of dialog
                local hiddenBeams = {};
                if self.encoder.rnnLayers ~=nil then
                    for level = 1, #self.encoder.rnnLayers do
                        if hiddenBeams[level] == nil then hiddenBeams[level] = {} end
                        hiddenBeams[level]['output'] = self.encoder.rnnLayers[level].output[batch['ques_fwd']:size(3)][iter];
                        hiddenBeams[level]['cell'] = self.encoder.rnnLayers[level].cell[batch['ques_fwd']:size(3)][iter];
                        if level == #self.encoder.rnnLayers then
                            hiddenBeams[#self.encoder.rnnLayers]['output'] = encOut[iter]
                        end
                        hiddenBeams[level]['output'] = torch.repeatTensor(hiddenBeams[level]['output'], beamSize, 1);
                        hiddenBeams[level]['cell'] = torch.repeatTensor(hiddenBeams[level]['cell'], beamSize, 1);
                    end
                    -- hiddenBeams[]['cell'] is beam_nums x 512
                    -- hiddenBeams[]['output'] is beam_nums x 512
                else
                    for level = 1, #self.decoder.rnnLayers do
                        if hiddenBeams[level] == nil then hiddenBeams[level] = {} end
                        if level == #self.decoder.rnnLayers then
                            hiddenBeams[level]['output'] = torch.repeatTensor(encOut[iter], beamSize, 1)
                        else
                            hiddenBeams[level]['output'] = torch.Tensor(beamSize, encOut:size(2)):zero()
                        end
                        hiddenBeams[level]['cell'] = hiddenBeams[level]['output']:clone():zero()
                    end
                end

                -- for first step, initialize with start symbols
                beams[1] = dataloader.word2ind['<START>'];
                scores = torch.DoubleTensor(beamSize):zero();
                finishBeams = {}; -- accumulate beams that are done

                for step = 2, beamLen do

                    -- candidates for the current iteration
                    cands = {};

                    -- if step == 2, explore only one beam (all are <START>)
                    local exploreSize = (step == 2) and 1 or beamSize;

                    -- first copy the hidden states to the decoder
                    for level = 1, #self.decoder.rnnLayers do
                        self.decoder.rnnLayers[level].userPrevOutput = hiddenBeams[level]['output']
                        self.decoder.rnnLayers[level].userPrevCell = hiddenBeams[level]['cell']
                    end

                    -- decoder forward pass
                    decOut = self.decoder:forward(beams[{{step-1}}]);
                    decOut = decOut:squeeze(); -- decOut is beam_nums x vocab_size

                    -- iterate separately for each possible word of beam
                    for wordId = 1, exploreSize do
                        local curHidden = {};
                        for level = 1, #self.decoder.rnnLayers do
                            if curHidden[level] == nil then curHidden[level] = {} end
                            curHidden[level]['output'] = self.decoder.rnnLayers[level].output[{{1},{wordId}}]:clone():squeeze(); -- rnnLayers[].output is 1 x beam_nums x 512
                            curHidden[level]['cell'] = self.decoder.rnnLayers[level].cell[{{1},{wordId}}]:clone():squeeze();
                        end

                        -- sort and get the top probabilities
                        if beamSize == 1 then
                            topProb, topInd = torch.topk(decOut, beamSize, true);
                        else
                            topProb, topInd = torch.topk(decOut[wordId], beamSize, true);
                        end

                        for candId = 1, beamSize do
                            local candBeam = beams[{{}, {wordId}}]:clone();
                            -- get the updated cost for each explored candidate, pool
                            candBeam[step] = topInd[candId];
                            if topInd[candId] == endToken then
                                table.insert(finishBeams, {beam = candBeam:double():squeeze(), length = step, score = scores[wordId] + topProb[candId]});
                            else
                                table.insert(cands, {score = scores[wordId] + topProb[candId],
                                                        beam = candBeam,
                                                        hidden = curHidden});
                            end
                        end
                    end

                    -- sort the candidates and stick to beam size
                    table.sort(cands, function (a, b) return a.score > b.score; end);

                    for candId = 1, math.min(#cands, beamSize) do
                        beams[{{}, {candId}}] = cands[candId].beam;

                        --recursive copy
                        for level = 1, #self.decoder.rnnLayers do
                            hiddenBeams[level]['output'][candId] = cands[candId].hidden[level]['output']:clone();
                            hiddenBeams[level]['cell'][candId] = cands[candId].hidden[level]['cell']:clone();
                        end

                        scores[candId] = cands[candId].score;
                    end
                end

                table.sort(finishBeams, function (a, b) return a.score > b.score; end);

                local quesWords = encInSeq:double():squeeze()
                local ansWords = finishBeams[1].beam:squeeze();

                local quesText = utils.idToWords(quesWords, dataloader.ind2word);
                local ansText = utils.idToWords(ansWords, dataloader.ind2word);

                table.insert(threadAnswers, {question = quesText, answer = ansText})
            end
        else
            local answerIn = torch.Tensor(1, numQues):fill(startToken)
            local answer = {answerIn:t():double()}
            for timeStep = 1, beamLen do
                -- one pass through decoder
                local decOut = self.decoder:forward(answerIn):squeeze()
                -- connect decoder to itself
                self.decoderConnect(self.decoder)

                local nextToken = torch.multinomial(torch.exp(decOut / temperature), 1)
                table.insert(answer, nextToken:double())
                answerIn:copy(nextToken)
            end
            answer = nn.JoinTable(-1):forward(answer)

            for iter = 1, 10 do
                local quesWords = batch['ques_fwd'][{{1}, {iter}, {}}]:squeeze():double()
                local ansWords = answer[{{iter}, {}}]:squeeze()

                local quesText = utils.idToWords(quesWords, dataloader.ind2word)
                local ansText = utils.idToWords(ansWords, dataloader.ind2word)

                table.insert(threadAnswers, {question = quesText, answer = ansText})
            end
        end
        self.wrapper:training()
        table.insert(answerTable, {image_id = dataloader['unique_img_'..dtype][convId], dialog = threadAnswers})
    end
    return answerTable
end

return Model;
