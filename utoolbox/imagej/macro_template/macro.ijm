<%!
    def inherit(context):
        if context['loop_files']:
            return "loop_files.ijm"
        elif context['batch_mode']:
            return "batch_mode.ijm"
        else:
            return "simple.ijm"
%>
<%inherit file="${inherit(context)}"/>

<%block name="prologue">
${parent.prologue()}
% if _prologue is not UNDEFINED:
${_prologue}
% endif
</%block>
${body}
<%block name="epilogue">
% if _epilogue is not UNDEFINED:
${_epilogue}
% endif
</%block>
