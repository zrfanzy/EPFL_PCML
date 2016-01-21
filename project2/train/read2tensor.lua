--[[read csv file. 
if mat file, use matlab 'csvwrite' convert to csv file first

if python:
np.savetxt('X_train.csv',X_train,delimiter=',')
>>> np.savetxt('y_test.csv',y_test, delimiter=',')
>>> np.savetxt('y_train.csv',y_train, delimiter=',')


usega: dofile 'read2tensor.lua'
load x and y into memory, 
x: 6000x36865 tensor, y: 6000x1 tensor-- for cnnft.csv
x: 6000x5408 tensor, y: 6000x1 tensor -- for hogft.csv

]]--

info = lapp [[
  -c, --csv  (default "cnn_Xtest.csv") datafile in csv
  --ft (default "cnn")
  --ty (default "test")
]]
print(info)
require 'io'
require 'torch'

print(1)

if info.ft == "cnn" then
 ft_dim = 36864
else
 ft_dim = 5408
end

local function split(str, sep)
    sep = sep or ','
    fields={}
    local matchfunc = string.gmatch(str, "([^"..sep.."]+)")
    if not matchfunc then return {str} end
    for str in matchfunc do
        table.insert(fields, str)
    end
    return fields
end

file = io.open('testXcnn.csv','r')
-- file = io.open(info.ft..'_X'..info.ty..'.csv','r')
-- file = io.open('cnnft.csv','r')

i = 1
-- x = torch.Tensor(6000,36865)
-- temp = torch.Tensor(36865)
label = io.open(info.ft..'_y'..info.ty..'.csv','r')
if info.ty == 'test' then
 num = 1200
else
 num = 4800
end
num = 11453 -- for testing data
x = torch.Tensor(num,ft_dim)
temp = torch.Tensor(x:size()[2])

for line in file:lines() do
    -- print(type(line))
    col = split(line, ',')
    -- print(type(col))
    -- print(table.getn(col))
    for k = 1,x:size()[2] do 
    --    print(k)
	x[i][k] = tonumber(col[k])
    end
    -- print(2)
    -- x[i] = k
    i = i + 1
end
    print(x:size())
    print(i)
file:close()

i = 1
y = torch.Tensor(x:size()[1])

for line in label:lines() do
    y[i] = tonumber(line)
    i = i + 1
end
    print(i)

label:close()
x = x:float()
y = y:float()
-- save file as 'bin'
filename = string.split(info.csv,'csv')
-- torch.save(info.ft..'_X'..info.ty..'.bin',x)
torch.save('testXcnn.bin',x)
-- torch.save(info.ft..'_y'..info.ty..'.bin',y)
-- torch.save('cnn_ytest.bin',y)

