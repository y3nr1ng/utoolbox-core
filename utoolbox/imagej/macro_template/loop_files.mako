
file_list = getFileList(src_dir)
for (i = 0; i < file_list.length; i++) {
    filename = split(file_list[i]);
    ${body}
}
