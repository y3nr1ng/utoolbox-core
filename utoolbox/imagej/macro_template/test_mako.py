import os

from mako.template import Template
from mako.lookup import TemplateLookup

cwd = os.path.dirname(os.path.abspath(__file__))
tpl_dir = os.path.join(cwd, 'macro_template')
templates = TemplateLookup(directories=[tpl_dir])

template = Template(filename='macro.ijm', lookup=templates)
print(template.render(
    loop_files=False, batch_mode=True,
    file_list="SOURCE",
    #_prologue="PROLOGUE",
    body="HELLO WORLD!!!",
    #_epilogue="EPILOGUE"
))
