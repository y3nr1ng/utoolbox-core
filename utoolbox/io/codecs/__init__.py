import pkgutil
import importlib

# iterate through valid subpackages
for _, name, _ in pkgutil.walk_packages(__path__):
    full_name = __name__ + '.' + name
    # import the package through __import__ wrapper
    importlib.import_module(full_name)
