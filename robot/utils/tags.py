import sys
import datetime
import subprocess
import __main__
from os import path


def get_tag():
    out = []
    out += ['Time: ' + str(datetime.datetime.now()).split('.')[0]]
    out += ['commit id ' +
            subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()]
    out += ['python ' + ' '.join(sys.argv)]
    #d = path.abspath(__main__.__file__).split('/')
    #d = '/'.join(d[d.index('mp'):])
    #out += ['filepath: ' + d]
    return '\n'.join(out)
