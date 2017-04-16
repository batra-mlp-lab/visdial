-- new module to replace zero with a given value
local MaskTime, Parent = torch.class('nn.MaskTime', 'nn.Module')

function MaskTime:__init(featSize)
    Parent.__init(self);
    self.mask = torch.Tensor();
    self.seqLen = nil;
    self.featSize = featSize;
    self.gradInput = {torch.Tensor(), torch.Tensor()};
end

function MaskTime:updateOutput(input)
    local seqLen = input[1]:size(1);
    local batchSize = input[1]:size(2);

    -- expand the feature vector
    self.output:resizeAs(input[2]):copy(input[2]);
    self.output = self.output:view(1, batchSize, self.featSize);
    self.output = self.output:repeatTensor(seqLen, 1, 1);

    -- expand the word mask
    self.mask = input[1]:eq(0);
    self.mask = self.mask:view(seqLen, batchSize, 1)
                :expand(seqLen, batchSize, self.featSize);
    self.output[self.mask] = 0;

    return self.output;
end

function MaskTime:updateGradInput(input, gradOutput)
    -- the first component is zero gradients
    self.gradInput[1]:resizeAs(input[1]):zero();
    -- second component has zeroed out gradients
    -- sum along first dimension
    self.gradInput[2]:resizeAs(gradOutput):copy(gradOutput);
    self.gradInput[2][self.mask] = 0;
    self.gradInput[2] = self.gradInput[2]:sum(1):squeeze();

    return self.gradInput;
end
