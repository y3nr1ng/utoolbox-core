from utoolbox.container import Volume

raw_data = Volume("data/RFiSHp2aLFCYC/raw/488/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs.tif")
print(raw_data.shape)
print(raw_data.dtype)

from utoolbox.transform.deskew import deskew
deskew_data = deskew(raw_data, 32.5)
