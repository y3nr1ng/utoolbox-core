
_file_list = File.openAsString("${file_list}");
file_list = split(_file_list, '\n');

${camera_setup}

setBatchMode(true);
for(i = 0; i < file_list.length; i++) {
    path = file_list[i];
    print(path);

    open(path);
    ${run_analysis}

    name = File.getName(path);
    name = substring(name, 0, lastIndexOf(name, ".")) + ".csv";
    path = "${dst_dir}" + File.separator + name;

    ${export_results}

    run("Close All");
}
setBatchMode(false);

run("Quit");
