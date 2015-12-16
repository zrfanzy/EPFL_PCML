require 'nn'
net = nn.Sequential()
 net:add(nn.View(1024, 6, 6))
--   net:add(nn.ReLU(true))
--   net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   net:add(nn.SpatialConvolution(1024, 64, 6, 6, 1, 1))
   net:add(nn.ReLU(true))
   net:add(nn.Dropout(0.25))

   net:add(nn.SpatialConvolution(64, 4, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
--   net:add(nn.Dropout(0.5))


--   net:add(nn.SpatialConvolution(3072, 4096, 1, 1, 1, 1))
--   net:add(nn.ReLU(true))
--   net:add(nn.Dropout(0.5))
   

--   net:add(nn.SpatialConvolution(4096, 4, 1, 1, 1, 1))
   net:add(nn.View(4))

return net
