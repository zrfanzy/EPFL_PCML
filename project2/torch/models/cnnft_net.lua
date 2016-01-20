require 'nn'
--require 'cunn'

local backend_name = 'nn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end
   
local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(backend.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(backend.ReLU(true))
  return vgg
end

-- space as we can
local MaxPooling = nn.SpatialMaxPooling

vgg:add(nn.View(36864))
vgg:add(nn.Linear(36864, 512))
-- vgg:add(nn.ReLU(true))
 vgg:add(nn.Dropout(0.95))
vgg:add(nn.Linear(512,4))
-- vgg:add(nn.ReLU(true))

vgg:add(nn.View(4))





local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'cudnn.SpatialConvolution'
  init'nn.SpatialConvolution'
end

MSRinit(vgg)


return vgg
