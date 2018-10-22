
args = getArgument()
parsed_args = split(args, ',')

% for arg in arg_list:
${arg} = parsed_args[${loop.index}];
% endfor
