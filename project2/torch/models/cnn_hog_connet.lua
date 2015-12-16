require 'nn'
model = nn.Sequential()
-- model:add(nn.JoinTable(1))
model:add(nn.View(42272,1,1))
model:add(nn.SpatialConvolution(42272, 3072, 1,1, 1,1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(3072, 4096, 1,1, 1,1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(4096,4, 1,1, 1,1))
model:add(nn.ReLU(true))
model:add(nn.View(4))

return model
