require 'torch';
require 'nn';

cmd = torch.CmdLine()
cmd:option('-loadPath', 'checkpoints/model.t7')
cmd:option('-savePath', 'checkpoints/model_cpu.t7')
cmd:option('-gpuid', 0)

opt = cmd:parse(arg)

-- check for new save path
if opt.savePath == 'checkpoints/model_cpu.t7' then
    opt.savePath = opt.loadPath .. '.cpu.t7'
end

print(opt)

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.backend == 'cudnn' then require 'cudnn' end
    cutorch.setDevice(opt.gpuid+1)
    torch.setdefaulttensortype('torch.CudaTensor');
else
    print('Gotta have a GPU to convert to CPU :(')
    os.exit()
end

print('Loading model')
model = torch.load(opt.loadPath)

-- convert modelW and optims to cpu
print('Shipping params to CPU')
if model.modelW:type() == 'torch.CudaTensor' then
    model.modelW = model.modelW:float()
end

for k,v in pairs(model.optims) do
    if torch.type(v) ~= 'number' and v:type() == 'torch.CudaTensor' then
        model.optims[k] = v:float()
    end
end

print('Saving to ' .. opt.savePath)
torch.save(opt.savePath, model)
