-- new module to replace zero with a given value
local ReplaceZero, Parent = torch.class('nn.ReplaceZero', 'nn.Module')

function ReplaceZero:__init(constant)
    Parent.__init(self);
    if not constant then
        error('<ReplaceZero> constant must be specified')
    end
    self.constant = constant;
    self.mask = torch.Tensor();
end

function ReplaceZero:updateOutput(input)
    self.output:resizeAs(input):copy(input);
    self.mask = input:eq(0);
    self.output[self.mask] = self.constant;
    return self.output;
end

function ReplaceZero:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput):copy(gradOutput);
    -- remove the gradients at those points
    self.gradInput[self.mask] = 0;
    return self.gradInput;
end
