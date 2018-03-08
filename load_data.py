from utoolbox.container import Volume

path = "data/RFiSHp2aLFCYC/raw/488/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs.tif"
data = Volume(path, resolution=(0.3, 0.102, 0.102))
print(data.shape)
