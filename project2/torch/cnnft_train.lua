require 'xlua'
require 'optim'
require 'nn'
require 'save_model'

dofile './cnnprovider.lua'
dofile './hogprovider.lua'

local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 100)          batch size
   -r,--learningRate          (default 0.01)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 10)          epoch step
   --model                    (default overfeat_net)     model name
   --max_epoch                (default 30)           maximum number of iterations
   --resumeFlag		      (default 0)           if resume then load model from file
]]

print(opt)

last_loss = -10 -- init loss to a negetive value


print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()

-- IMPORTANT: CAST MODEL TO FLOAT!

-- load model from file

if resumeFlag == 1 then
  model:add(dofile('models/'..opt.model..'.lua'))
else
  print('load model, resume traning')
  model = torch.load('logs/cnn_0312/model.net')
end

print(model)
model:float()
print(c.blue '==>' ..' loading data')

database = '../train/cnnprovider.t7'
provider = torch.load(database)
print('load '..database)

-- load provider, class daving traning data and test data
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(4)


-- init log file to save the traing data in process
print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger_ave = optim.Logger(paths.concat(opt.save, 'epoch_test.log'))
trainLogger_ave = optim.Logger(paths.concat(opt.save, 'epoch_train.log'))
trainLogger_confusion = optim.Logger(paths.concat(opt.save, 'confusion_test.log'))
testLogger_confusion = optim.Logger(paths.concat(opt.save, 'confusion_test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion()
-- criterion = nn.CrossEntropyCriterion():cuda()
-- criterion = nn.ClassNLLCriterion() -- for multiclassification

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  -- local targets = torch.CudaTensor(opt.batchSize)
  local targets = torch.FloatTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()

  for t,v in ipairs(indices) do

    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))
    inputs = inputs:float()
    targets = targets:float()
--   print(inputs:size())
--   print(targets:size())


    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      -- print("debug: ")  
      model = model:float()
      local outputs = model:forward(torch.FloatTensor(inputs))
      targets = targets:float()
      outputs = outputs:float()
      criterion = criterion:float()
      
      -- compute error by criterion with output and targets:
      local f = criterion:forward(outputs, targets)

      -- compute gradient by backward
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)
      confusion:updateValids() 
      local current_acc = confusion.totalValid * 100
            
      print(confusion.totalValid * 100)

      if t%30 == 0
         then
            confusion:updateValids() 
            if last_loss > 0 then
                local cur_loss = confusion.totalValid * 100
                if cur_loss - last_loss > 5 then
                   local filename = paths.concat(opt.save, 'model.net')
                   print('increase by 5!') 
                   saveModel(model,filename) -- save model call function in 'save_model.lua'
                end
            end
            last_loss = confusion.totalValid
            print(confusion.totalValid * 100)
            print(confusion)     
--      testLogger:add{tostring(confusion)}
--      testLogger:add{current_acc, confusion.totalValid * 100}
         end
      
      return f,gradParameters
    
    end

    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))
  trainLogger_confusion:add{tostring(confusion)}
  train_acc = confusion.totalValid * 100
  trainLogger_ave:add{train_acc}
  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = opt.batchSize
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
    testLogger_ave:add{confusion.totalValid*100}
    testLogger_confusion:add{tostring(confusion)}

   local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()


    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))

    confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end
