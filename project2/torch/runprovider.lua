

dofile('hogprovider.lua')
hogprovider = hogProvider()

torch.save('../train/hogprovider.t7',hogprovider)

dofile('cnnprovider.lua')
cnnprovider = cnnProvider()
torch.save('../train/cnnprovider.t7',cnnprovider)

