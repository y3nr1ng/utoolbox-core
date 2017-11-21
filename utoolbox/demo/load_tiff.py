from utoolbox.io import imopen

file_path = 'data/RAWtan1_3_3DSIMb_ch1_stack0001_561nm_0019400msec_0000215229msecAbs.tif'
with imopen(file_path, 'r') as fd:
    pass
