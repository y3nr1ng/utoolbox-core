
_file_list = File.openAsString(${file_list});
file_list = split(_file_list, '\n');

${camera_setup}

setBatchMode(true);
for(i = 0; i < file_list.length; i++) {
    path = split(file_list[i]);

    open(path);
    ${run_analysis}

    root = split(path, ".tif");
    path = root[0] + ".csv";

    ${export_results}

    run("Close All");
}
setBatchMode(false);

run("Quit");
