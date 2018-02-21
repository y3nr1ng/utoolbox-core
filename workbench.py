from utoolbox.container import Volume

mem_data = Volume('data/membrane_000.tif')
print(mem_data.shape)
print(mem_data.dtype)

from utoolbox.container.registry import containers
print(containers)
