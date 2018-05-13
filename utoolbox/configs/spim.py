import re

class SPIMsettings(object):
    def __init__(self, fpath):
        self.path = fpath
        with open(fpath) as f:
            self.sections = self.identify_sections(f.read())

        self.parse_general()
        self.parse_waveform()
        self.parse_timing()

    def identify_sections(self, content):
        section_pattern = re.compile(
            '^(?:\*{5}\s{1}){3}\s*(?P<section>[^\*]+)(?:\s{1}\*{5}){3}',
            re.MULTILINE
        )

        pos = [
            (match.group('section').rstrip(), match.start())
            for match in section_pattern.finditer(content)
        ]
        n_sections = len(pos)

        sections = {}
        for index in range(n_sections):
            start = pos[index][1]
            end = pos[index+1][1] if index < n_sections-1 else len(content)
            sections[pos[index][0]] = content[start:end]
        return sections

    def parse_general(self):
        if 'General' not in self.sections:
            #TODO raise error
            raise ValueError

        general_patterns = {
            'timestamp': 'Date :\t(\d+/\d+/\d+ \d+:\d+:\d+ [A|P]M)',
            'acq_mode': 'Acq Mode :\t(.*)'
        }

        content = self.sections['General']
        self.sections['General'] = {
            k: re.findall(v, content, re.MULTILINE)[0]
            for k, v in general_patterns.items()
        }

    def parse_waveform(self):
        if 'Waveform' not in self.sections:
            #TODO raise error
            raise ValueError

        waveform_patterns = {
            'scan_mode': 'Z motion :\t(.*)',
            'sample_piezo_step':
                '^S PZT .* Interval \(um\), .* :\t\d+\.*\d*\t(\d+\.*\d*)\t\d+$',
            'obj_piezo_step':
                '^Z PZT .* Interval \(um\), .* :\t\d+\.*\d*\t(\d+\.*\d*)\t\d+$'
        }

        content = self.sections['Waveform']
        self.sections['Waveform'] = {
            k: re.findall(v, content, re.MULTILINE)[0]
            for k, v in waveform_patterns.items()
        }

    def parse_camera(self):
        if 'Camera' not in self.sections:
            #TODO raise error
            raise ValueError

        camera_patterns = {
            'model': 'Model :\t(.*)',
            'exposure': 'Exp(s) :\t(\d+\.\d+)',
            'roi': (),
            'binning': (),
        }

    def parse_timing(self):
        if 'Advanced Timing' not in self.sections:
            #TODO raise error
            raise ValueError

        timing_patterns = {
            'mode': 'Trigger Mode :\t(.*)'
        }

        content = self.sections['Advanced Timing']
        self.sections['Advanced Timing'] = {
            k: re.findall(v, content, re.MULTILINE)[0]
            for k, v in timing_patterns.items()
        }

class SPIMini(object):
    pass
