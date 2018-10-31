<%inherit file="base.ijm"/>

setBatchMode(true);
${next.body()}
setBatchMode(false);
