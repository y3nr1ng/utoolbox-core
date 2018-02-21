from utoolbox.container import Volume

mem_data = Volume("RFiSHp2aLFCYC/decon/488/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif")
print(mem_data.shape)
print(mem_data.dtype)

from utoolbox.segmentation.slic import slic
segments = slic(mem_data)
print(segments.dtype)
