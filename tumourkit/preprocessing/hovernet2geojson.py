"""
Converts from HoVernet json to QuPath geojson.

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
from typing import Dict, Any, Tuple, List
import argparse
import geojson
from tqdm import tqdm
from ..utils.preprocessing import parse_path, get_names, create_dir, read_json, create_geojson


Point = Tuple[float,float]
Contour = List[Point]
def parse_contours(nuc: Dict[str, Any]) -> List[Contour]:
    """
    Input: hovernet json dictionary with nuclei information.
    Output: list of contours of cells as list of points.
            Each contour has the same point at position 0 and -1.
    """
    contours_ = []
    for inst in nuc:
        inst_info = nuc[inst]
        inst_type = inst_info['type']
        if inst_type == 1 or inst_type == 2:
            inst_contour = inst_info['contour']
            inst_contour.append(inst_contour[0])
            contours_.append((inst_contour, inst_type)) 
        elif inst_type is None:
            inst_contour = inst_info['contour']
            inst_contour.append(inst_contour[0])
            contours_.append((inst_contour, 3)) 
    return contours_


def save_contours(out_dir: str, name: str, contours: List[Contour]) -> None:
    """
    Saves geojson in a file. It doesn't change the format.
    """
    create_dir(parse_path(out_dir))
    with open(out_dir + name + '.geojson', 'w') as f:
        geojson.dump(contours, f, sort_keys=True, indent=2)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', type=str, default='./',
                        help='Path to json files.')
    parser.add_argument('--gson-dir', type=str, default='./',
                        help='Path to save files.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    json_dir = parse_path(args.json_dir)
    names = get_names(json_dir, '.json')
    for name in tqdm(names):
        json_path = json_dir + name + '.json'
        nuc = read_json(json_path)
        contours = parse_contours(nuc)
        features = create_geojson(contours)
        save_contours(parse_path(args.gson_dir), name, features)
