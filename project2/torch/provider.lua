--[[ useage: 
th -i provider.lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)

]]--


require 'debugger'
require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 4800
  local tesize = 1200
  local total_size = 6000
  local ft_dim =  36864
  local idx_ran = torch.randperm(6000)
  local dataset = 'hogft.bin' --'cnnft.bin'
  
  -- download dataset
  --if not paths.dirp('cifar-10-batches-t7') then
  --   local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
  --   local tar = paths.basename(www)
  --   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
  --end

  -- load dataset
  self.trainData = {
  --   data = torch.Tensor(trsize, 36865),
     data = torch.Tensor(trsize, ft_dim),
    
     labels = torch.Tensor(trsize):fill(0),
     size = function() return trsize end
  }
  self.testData = {
     data = torch.Tensor(tesize, ft_dim),
     labels = torch.Tensor(tesize):fill(0),
     size = function() return tesize end
  }


  local trainData = self.trainData
  print('dataset loading..')
  local X_train = torch.load('train/cnn_Xtrain.bin')
  local y_train = torch.load('train/cnn_ytrain.bin')
  local X_test = torch.load('train/cnn_Xtest.bin')
  local y_test = torch.load('train/cnn_ytest.bin')

print(y_test)
print(y_train)
--  local x = torch.load('train/Trainset_x_'..dataset)
--  local y = torch.load('train/Trainset_y_'..dataset)

  -- local x = torch.load('./train/Trainset_x.bin', 'ascii')
  -- local y = torch.load('./train/Trainset_y.bin','ascii')
  local testData = self.testData
  local x = X_train
   local y = y_train
  for i =1,trsize  do
    trainData.data[i] = x[i]
    trainData.labels[i] = y[i]
--[[
     -- print(trainData.labels[i]:size())
     -- trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
     -- trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
     trainData.data[i] = x[idx_ran[i]]


     -- print(idx_ran[i])
--     trainData.labels[i][y[idx_ran[i]]] = 1 -- convert 4 to 0001 for svm hinge loss   --y[idx_ran[i]]
     -- print(trainData.labels[i])
     -- trainData.labels[i] = y[idx_ran[i]]

  end
	--print(trainData.labels[i])
  -- trainData.labels = trainData.labels + 1
  local x = X_test
  local y = y_test

  for i = 1, tesize do
    testData.data[i] = x[i]
    testData.labels[i] = y[i]+1
--[[     testData.data[i] = x[idx_ran[i + trsize]]
--     testData.labels[i][y[idx_ran[i + trsize]]] = 1
     -- testData.labels[i] = y[idx_ran[i + trsize]]
  end
print(testData.data:size())
print(testData.labels:size())
print(trainData.data:size())
print(trainData.labels:size())
  -- testData.labels = testData.labels + 1

--[[
  local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
  self.testData = {
     data = subset.data:t():double(),
     labels = subset.labels[1]:double(),
     size = function() return tesize end
  }
  local testData = self.testData
  testData.labels = testData.labels + 1
]]--
  -- resize dataset (if using small version)
  trainData.data = trainData.data[{ {1,trsize} }]
  trainData.labels = trainData.labels[{ {1,trsize} }]

  testData.data = testData.data[{ {1,tesize} }]
  testData.labels = testData.labels[{ {1,tesize} }]

  -- reshape data
  trainData.data = trainData.data:reshape(trsize,ft_dim)
  testData.data = testData.data:reshape(tesize,ft_dim)
  print(torch.sum(trainData.labels))
  -- trainData.data = trainData.data:reshape(trsize,3,32,32)
  -- testData.data = testData.data:reshape(tesize,3,32,32)
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     -- local rgb = trainData.data[i]
     -- local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     -- yuv[1] = normalization(yuv[{{1}}])
     local feature = trainData.data[i]
     feature = normalization(feature)
     trainData.data[i] = feature
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- preprocess testSet
  for i = 1,testData:size() do
    xlua.progress(i, testData:size())
     -- rgb -> yuv
     -- local rgb = testData.data[i]
     -- local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     -- yuv[{1}] = normalization(yuv[{{1}}])
     local feature = testData.data[i]
     feature = normalization(feature)
     testData.data[i] = feature
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)
end
