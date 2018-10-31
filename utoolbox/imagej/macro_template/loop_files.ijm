<%!
    def inherit(context):
        if context['batch_mode']:
            return "batch_mode.ijm"
        else:
            return "simple.ijm"
%>
<%inherit file="${inherit(context)}"/>

<%block name="prologue">
_file_list = File.openAsString(${file_list})
file_list = split(_file_list, '\n')
</%block>

for (i = 0; i < file_list.length; i++) {
    path = split(file_list[i]);
    ${next.body()}
}
