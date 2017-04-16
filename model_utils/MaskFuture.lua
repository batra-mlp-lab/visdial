-- new module to replace zero with a given value
local MaskFuture, Parent = torch.class('nn.MaskFuture', 'nn.Module')

function MaskFuture:__init(numClasses)
    Parent.__init(self);
    self.mask = torch.Tensor(1, numClasses, numClasses):fill(1);
    -- extract the upper diagonal matrix
    self.mask[1] = torch.triu(self.mask[1], 1);
    self.gradInput = torch.Tensor();
    self.output = torch.Tensor();
end

function MaskFuture:updateOutput(input)
    local batchSize = input:size(1);

    self.output:resizeAs(input):copy(input);
    -- expand mask based on input
    self.output[self.mask:expandAs(input)] = 0;

    return self.output;
end

function MaskFuture:updateGradInput(input, gradOutput)
    -- the first component is zero gradients
    --self.gradInput:resizeAs(input):zero();
    -- zero out the gradients based on the mask
    self.gradInput:resizeAs(gradOutput):copy(gradOutput);
    self.gradInput[self.mask:expandAs(input)] = 0;

    return self.gradInput;
end
