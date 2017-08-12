require 'hdf5'
require 'xlua'
local utils = require 'utils'

local dataloader = {};

-- read the data
-- params: object itself, command line options,
--          subset of data to load (train, val, test)
function dataloader:initialize(opt, subsets)
    -- read additional info like dictionary, etc
    print('DataLoader loading json file: ', opt.inputJson)
    info = utils.readJSON(opt.inputJson);
    for key, value in pairs(info) do dataloader[key] = value; end

    -- add <START> and <END> to vocabulary
    count = 0;
    for _ in pairs(dataloader['word2ind']) do count = count + 1; end
    dataloader['word2ind']['<START>'] = count + 1;
    dataloader['word2ind']['<END>'] = count + 2;
    count = count + 2;
    dataloader.vocabSize = count;
    print(string.format('Vocabulary size (with <START>,<END>): %d\n', count));

    -- construct ind2word
    local ind2word = {};
    for word, ind in pairs(dataloader['word2ind']) do
        ind2word[ind] = word;
    end
    dataloader['ind2word'] = ind2word;

    -- read questions, answers and options
    print('DataLoader loading h5 file: ', opt.inputQues)
    local quesFile = hdf5.open(opt.inputQues, 'r');

    print('DataLoader loading h5 file: ', opt.inputImg)
    local imgFile = hdf5.open(opt.inputImg, 'r');
    -- number of threads
    self.numThreads = {};

    for _, dtype in pairs(subsets) do
        -- read question related information
        self[dtype..'_ques'] = quesFile:read('ques_'..dtype):all();
        self[dtype..'_ques_len'] = quesFile:read('ques_length_'..dtype):all();

        -- read answer related information
        self[dtype..'_ans'] = quesFile:read('ans_'..dtype):all();
        self[dtype..'_ans_len'] = quesFile:read('ans_length_'..dtype):all();
        self[dtype..'_ans_ind'] = quesFile:read('ans_index_'..dtype):all():long();

        -- read image list, if image features are needed
        if opt.useIm then
            print('Reading image features..')
            local imgFeats = imgFile:read('/images_'..dtype):all();

            -- Normalize the image features (if needed)
            if opt.imgNorm == 1 then
                print('Normalizing image features..')
                local nm = torch.sqrt(torch.sum(torch.cmul(imgFeats, imgFeats), 2));
                imgFeats = torch.cdiv(imgFeats, nm:expandAs(imgFeats)):float();
            end
            self[dtype..'_img_fv'] = imgFeats;
            -- TODO: make it 1 indexed in processing code
            -- currently zero indexed, adjust manually
            self[dtype..'_img_pos'] = quesFile:read('img_pos_'..dtype):all():long();
            self[dtype..'_img_pos'] = self[dtype..'_img_pos'] + 1;
        end

        -- print information for data type
        print(string.format('%s:\n\tNo. of threads: %d\n\tNo. of rounds: %d'..
                            '\n\tMax ques len: %d'..'\n\tMax ans len: %d\n',
                                dtype, self[dtype..'_ques']:size(1),
                                        self[dtype..'_ques']:size(2),
                                        self[dtype..'_ques']:size(3),
                                        self[dtype..'_ans']:size(3)));

        -- record some stats
        if dtype == 'train' then
            self.numTrainThreads = self['train_ques']:size(1);
            self.numThreads['train'] = self.numTrainThreads;
        end
        if dtype == 'test' then
            self.numTestThreads = self['test_ques']:size(1);
            self.numThreads['test'] = self.numTestThreads;
        end
        if dtype == 'val' then
            self.numValThreads = self['val_ques']:size(1);
            self.numThreads['val'] = self.numValThreads;
        end

        -- record the options
        if dtype == 'train' or dtype == 'val' or dtype == 'test' then
            self[dtype..'_opt'] = quesFile:read('opt_'..dtype):all():long();
            self[dtype..'_opt_len'] = quesFile:read('opt_length_'..dtype):all();
            self[dtype..'_opt_list'] = quesFile:read('opt_list_'..dtype):all();
        end

        -- assume similar stats across multiple data subsets
        -- maximum number of questions per image, ideally 10
        self.maxQuesCount = self[dtype..'_ques']:size(2);
        -- maximum length of question
        self.maxQuesLen = self[dtype..'_ques']:size(3);
        -- maximum length of answer
        self.maxAnsLen = self[dtype..'_ans']:size(3);
        -- number of options, if read
        if self[dtype..'_opt'] then
            self.numOptions = self[dtype..'_opt']:size(3);
        end

        -- if history is needed
        if opt.useHistory then
            self[dtype..'_cap'] = quesFile:read('cap_'..dtype):all():long();
            self[dtype..'_cap_len'] = quesFile:read('cap_length_'..dtype):all();
        end
    end
    -- done reading, close files
    quesFile:close();
    imgFile:close();

    -- take desired flags/values from opt
    self.useHistory = opt.useHistory;
    self.concatHistory = opt.concatHistory;
    self.useIm = opt.useIm;
    self.maxHistoryLen = opt.maxHistoryLen or 60;

    -- prepareDataset for training
    for _, dtype in pairs(subsets) do self:prepareDataset(dtype); end
end

-- method to prepare questions and answers for retrieval
-- questions : right align
-- answers : prefix with <START> and <END>
function dataloader:prepareDataset(dtype)
    -- right align the questions
    print('Right aligning questions: '..dtype);
    self[dtype..'_ques_fwd'] = utils.rightAlign(self[dtype..'_ques'],
                                            self[dtype..'_ques_len']);

    -- if separate captions are needed
    if self.useHistory then self:processHistory(dtype); end
    -- prefix options with <START> and <END>, if not train
    -- if dtype ~= 'train' then self:processOptions(dtype); end
    self:processOptions(dtype)
    -- process answers
    self:processAnswers(dtype);
end

-- process answers
function dataloader:processAnswers(dtype)
    --prefix answers with <START>, <END>; adjust answer lengths
    local answers = self[dtype..'_ans'];
    local ansLen = self[dtype..'_ans_len'];

    local numConvs = answers:size(1);
    local numRounds = answers:size(2);
    local maxAnsLen = answers:size(3);

    local decodeIn = torch.LongTensor(numConvs, numRounds, maxAnsLen+1):zero();
    local decodeOut = torch.LongTensor(numConvs, numRounds, maxAnsLen+1):zero();

    -- decodeIn begins with <START>
    decodeIn[{{}, {}, 1}] = self.word2ind['<START>'];

    -- go over each answer and modify
    local endTokenId = self.word2ind['<END>'];
    for thId = 1, numConvs do
        for roundId = 1, numRounds do
            local length = ansLen[thId][roundId];

            -- only if nonzero
            if length > 0 then
                decodeIn[thId][roundId][{{2, length + 1}}]
                                = answers[thId][roundId][{{1, length}}];

                decodeOut[thId][roundId][{{1, length}}]
                                = answers[thId][roundId][{{1, length}}];
                decodeOut[thId][roundId][length+1] = endTokenId;
            else
                print(string.format('Warning: empty answer at (%d %d %d)',
                                                    thId, roundId, length))
            end
        end
    end

    self[dtype..'_ans_len'] = self[dtype..'_ans_len'] + 1;
    self[dtype..'_ans_in'] = decodeIn;
    self[dtype..'_ans_out'] = decodeOut;
end

-- process caption as history
function dataloader:processHistory(dtype)
    local captions = self[dtype..'_cap'];
    local questions = self[dtype..'_ques'];
    local quesLen = self[dtype..'_ques_len'];
    local capLen = self[dtype..'_cap_len'];
    local maxQuesLen = questions:size(3);

    local answers = self[dtype..'_ans'];
    local ansLen = self[dtype..'_ans_len'];
    local numConvs = answers:size(1);
    local numRounds = answers:size(2);
    local maxAnsLen = answers:size(3);

    local history, histLen;
    if self.concatHistory == true then
        self.maxHistoryLen = math.min(numRounds * (maxQuesLen + maxAnsLen), 300);

        history = torch.LongTensor(numConvs, numRounds,
                                        self.maxHistoryLen):zero();
        histLen = torch.LongTensor(numConvs, numRounds):zero();
    else
        history = torch.LongTensor(numConvs, numRounds,
                                        maxQuesLen+maxAnsLen):zero();
        histLen = torch.LongTensor(numConvs, numRounds):zero();
    end

    -- go over each question and append it with answer
    for thId = 1, numConvs do
        local lenC = capLen[thId];
        local lenH; -- length of history
        for roundId = 1, numRounds do
            if roundId == 1 then
                -- first round has caption as history
                history[thId][roundId][{{1, maxQuesLen + maxAnsLen}}]
                            = captions[thId][{{1, maxQuesLen + maxAnsLen}}];
                lenH = math.min(lenC, maxQuesLen + maxAnsLen);
            else
                local lenQ = quesLen[thId][roundId-1];
                local lenA = ansLen[thId][roundId-1];
                -- if concatHistory, string together all previous QAs
                if self.concatHistory == true then
                    history[thId][roundId][{{1, lenH}}]
                                = history[thId][roundId-1][{{1, lenH}}];
                    history[thId][roundId][{{lenH+1}}] = self.word2ind['<END>'];
                    if lenQ > 0 then
                        history[thId][roundId][{{lenH+2, lenH+1+lenQ}}]
                                    = questions[thId][roundId-1][{{1, lenQ}}];
                    end
                    if lenA > 0 then
                        history[thId][roundId][{{lenH+1+lenQ+1, lenH+1+lenQ+lenA}}]
                                    = answers[thId][roundId-1][{{1, lenA}}];
                    end
                    lenH = lenH + lenQ + lenA + 1
                -- else, history is just previous round QA
                else
                    if lenQ > 0 then
                        history[thId][roundId][{{1, lenQ}}]
                                = questions[thId][roundId-1][{{1, lenQ}}];
                    end
                    if lenA > 0 then
                        history[thId][roundId][{{lenQ + 1, lenQ + lenA}}]
                                    = answers[thId][roundId-1][{{1, lenA}}];
                    end
                    lenH = lenA + lenQ;
                end
            end
            -- save the history length
            histLen[thId][roundId] = lenH;
        end
    end

    -- right align history and then save
    print('Right aligning history: '..dtype);
    self[dtype..'_hist'] = utils.rightAlign(history, histLen);
    self[dtype..'_hist_len'] = histLen;
end

-- process options
function dataloader:processOptions(dtype)
    local lengths = self[dtype..'_opt_len'];
    local answers = self[dtype..'_ans'];
    local maxAnsLen = answers:size(3);
    local answers = self[dtype..'_opt_list'];
    local numConvs = answers:size(1);

    local ansListLen = answers:size(1);
    local decodeIn = torch.LongTensor(ansListLen, maxAnsLen + 1):zero();
    local decodeOut = torch.LongTensor(ansListLen, maxAnsLen + 1):zero();

    -- decodeIn begins with <START>
    decodeIn[{{}, 1}] = self.word2ind['<START>'];

    -- go over each answer and modify
    local endTokenId = self.word2ind['<END>'];
    for id = 1, ansListLen do
        -- print progress for number of images
        if id % 100 == 0 then
            xlua.progress(id, numConvs);
        end
        local length = lengths[id];

        -- only if nonzero
        if length > 0 then
            decodeIn[id][{{2, length + 1}}] = answers[id][{{1, length}}];

            decodeOut[id][{{1, length}}] = answers[id][{{1, length}}];
            decodeOut[id][length + 1] = endTokenId;
        else
            print(string.format('Warning: empty answer for %s at %d',
                                                            dtype, id))
        end
    end

    self[dtype..'_opt_len'] = self[dtype..'_opt_len'] + 1;
    self[dtype..'_opt_in'] = decodeIn;
    self[dtype..'_opt_out'] = decodeOut;

    collectgarbage();
end

-- method to grab the next training batch
function dataloader.getTrainBatch(self, params, batchSize)
    local size = batchSize or params.batchSize;
    local inds = torch.LongTensor(size):random(1, params.numTrainThreads);

    -- Index question, answers, image features for batch
    local batchOutput = self:getIndexData(inds, params, 'train')
    if params.decoder == 'disc' then
        local optionOutput = self:getIndexOption(inds, params, 'train')
        batchOutput['options'] = optionOutput:view(optionOutput:size(1)
                                    * optionOutput:size(2), optionOutput:size(3), -1)
        batchOutput['answer_ind'] = batchOutput['answer_ind']:view(batchOutput['answer_ind']
                                        :size(1) * batchOutput['answer_ind']:size(2))
    end

    return batchOutput
end

-- method to grab the next test/val batch, for evaluation of a given size
function dataloader.getTestBatch(self, startId, params, dtype)
    local batchSize = params.batchSize
    -- get the next start id and fill up current indices till then
    local nextStartId;
    if dtype == 'val' then
        nextStartId = math.min(self.numValThreads+1, startId + batchSize);
    end
    if dtype == 'test' then
        nextStartId = math.min(self.numTestThreads+1, startId + batchSize);
    end

    -- dumb way to get range (complains if cudatensor is default)
    local inds = torch.LongTensor(nextStartId - startId);
    for ii = startId, nextStartId - 1 do inds[ii - startId + 1] = ii; end

    -- Index question, answers, image features for batch
    local batchOutput = self:getIndexData(inds, params, dtype);
    local optionOutput = self:getIndexOption(inds, params, dtype);

    if params.decoder == 'disc' then
        batchOutput['options'] = optionOutput:view(optionOutput:size(1)
                                    * optionOutput:size(2), optionOutput:size(3), -1)
        batchOutput['answer_ind'] = batchOutput['answer_ind']:view(batchOutput['answer_ind']
                                        :size(1) * batchOutput['answer_ind']:size(2))
    elseif params.decoder == 'gen' then
        -- merge both the tables and return
        for key, value in pairs(optionOutput) do batchOutput[key] = value; end
    end

    return batchOutput, nextStartId;
end

-- get batch from data subset given the indices
function dataloader.getIndexData(self, inds, params, dtype)
    -- get the question lengths
    local batchQuesLen = self[dtype..'_ques_len']:index(1, inds);
    local maxQuesLen = torch.max(batchQuesLen);
    -- get questions
    local quesFwd = self[dtype..'_ques_fwd']:index(1, inds)
                                            [{{}, {}, {-maxQuesLen, -1}}];

    local history;
    if self.useHistory then
        local batchHistLen = self[dtype..'_hist_len']:index(1, inds);
        local maxHistLen = math.min(torch.max(batchHistLen), self.maxHistoryLen);
        history = self[dtype..'_hist']:index(1, inds)
                                    [{{}, {}, {-maxHistLen, -1}}];
    end

    local imgFeats;
    if self.useIm then
        local imgInds = self[dtype..'_img_pos']:index(1, inds);
        imgFeats = self[dtype..'_img_fv']:index(1, imgInds);
    end

    -- get the answer lengths
    local batchAnsLen = self[dtype..'_ans_len']:index(1, inds);
    local maxAnsLen = torch.max(batchAnsLen);
    -- answer labels (decode input and output)
    local answerIn = self[dtype..'_ans_in']
                                :index(1, inds)[{{}, {}, {1, maxAnsLen}}];
    local answerOut = self[dtype..'_ans_out']
                                :index(1, inds)[{{}, {}, {1, maxAnsLen}}];
    local answerInd = self[dtype..'_ans_ind']:index(1, inds);

    local output = {};
    if params.gpuid >= 0 then
        output['ques_fwd'] = quesFwd:cuda();
        output['answer_in'] = answerIn:cuda();
        output['answer_out'] = answerOut:cuda();
        output['answer_ind'] = answerInd:cuda();
        if history then output['hist'] = history:cuda(); end
        if caption then output['cap'] = caption:cuda(); end
        if imgFeats then output['img_feat'] = imgFeats:cuda(); end
    else
        output['ques_fwd'] = quesFwd:contiguous();
        output['answer_in'] = answerIn:contiguous();
        output['answer_out'] = answerOut:contiguous();
        output['answer_ind'] = answerInd:contiguous()
        if history then output['hist'] = history:contiguous(); end
        if caption then output['cap'] = caption:contiguous(); end
        if imgFeats then output['img_feat'] = imgFeats:contiguous(); end
    end

    return output;
end

-- get batch from options given the indices
function dataloader.getIndexOption(self, inds, params, dtype)
    local output = {};
    if params.decoder == 'gen' then
        local optionIn, optionOut

        local optInds = self[dtype..'_opt']:index(1, inds);
        local indVector = optInds:view(-1);

        local batchOptLen = self[dtype..'_opt_len']:index(1, indVector);
        local maxOptLen = torch.max(batchOptLen);

        optionIn = self[dtype..'_opt_in']:index(1, indVector);
        optionIn = optionIn:view(optInds:size(1), optInds:size(2),
                                                optInds:size(3), -1);
        optionIn = optionIn[{{}, {}, {}, {1, maxOptLen}}];

        optionOut = self[dtype..'_opt_out']:index(1, indVector);
        optionOut = optionOut:view(optInds:size(1), optInds:size(2),
                                                optInds:size(3), -1);
        optionOut = optionOut[{{}, {}, {}, {1, maxOptLen}}];

        if params.gpuid >= 0 then
            output['option_in'] = optionIn:cuda();
            output['option_out'] = optionOut:cuda();
        else
            output['option_in'] = optionIn:contiguous();
            output['option_out'] = optionOut:contiguous();
        end
    elseif params.decoder == 'disc' then
        local optInds = self[dtype .. '_opt']:index(1, inds)
        local indVector = optInds:view(-1)

        local optionIn = self[dtype .. '_opt_list']:index(1, indVector)

        optionIn = optionIn:view(optInds:size(1), optInds:size(2), optInds:size(3), -1)

        output = optionIn

        if params.gpuid >= 0 then
            output = output:cuda()
        end
    end

    return output;
end

return dataloader;
