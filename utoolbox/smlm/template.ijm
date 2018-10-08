
_file_list = File.openAsString("${file_list}");
file_list = split(_file_list, '\n');

${camera_setup}

setBatchMode(true);
for(i = 0; i < file_list.length; i++) {
    path = file_list[i];
    print(path);

    open(path);
    ${run_analysis}

    ext_pos = indexOf(path, ".tif");
    path = substring(path, 0, ext_pos) + ".csv";
    print(path);

    ${export_results}

    run("Close All");
}
setBatchMode(false);

run("Quit");
