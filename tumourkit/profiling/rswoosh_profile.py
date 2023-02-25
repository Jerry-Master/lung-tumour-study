"""
Copyright (C) 2023  Jose Pérez Cano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact information: joseperez2000@hotmail.es
"""
import cProfile
import io
import pstats
import sys
import os
import json

PROF_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(PROF_DIR)
sys.path.append(PKG_DIR)

from postprocessing.rswoosh import rswoosh
from utils.preprocessing import get_names, parse_path
from utils.postprocessing import create_comparator, merge_cells

RSWOOSH_DIR = parse_path(PKG_DIR) + 'tests/rswoosh/'
THRESHOLD = 1
NUM_FRONTIER = 15

names = get_names(RSWOOSH_DIR, '.input.json')
def exec_conversion(name):
    with open(RSWOOSH_DIR + name + '.input.json', 'r') as f:
        inp = json.load(f)
    with open(RSWOOSH_DIR + name + '.output.json', 'r') as f:
        expected_out = json.load(f)
    predicted_out = rswoosh(inp, create_comparator(THRESHOLD, NUM_FRONTIER), merge_cells)
    if not len(predicted_out) == len(expected_out):
        print(name)
        return False
    return True

for name in names:
    pr = cProfile.Profile()
    pr.enable()

    my_result = exec_conversion(name)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('rswoosh/rswoosh_profile_' + name + '.txt', 'w+') as f:
        f.write(s.getvalue())
