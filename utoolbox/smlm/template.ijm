args = getArgument();
parsed_args = split(args, ',');

raw_file_list = File.openAsString(parsed_args[0]);
file_list = split(raw_file_list, '\n');

setBatchMode(true);
${camera_setup}
for (i = 0; i < file_list.length; i++) {
    open(file_list[i]);
    ${run_analysis}
    ${export_results}
    run("Close All");
}
setBatchMode(false);
