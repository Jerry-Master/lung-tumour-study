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
from typing import Dict, Any, Tuple, List, Optional
import argparse
import geojson
from tqdm import tqdm
from ..utils.preprocessing import parse_path, get_names, create_dir, read_json, create_geojson


Point = Tuple[float,float]
Contour = List[Point]
def parse_contours(nuc: Dict[str, Any], num_classes: Optional[int] = 2) -> List[Contour]:
    """
    Parses contours of cells from the given HoverNet JSON dictionary.

    :param nuc: A dictionary containing HoverNet nuclei information.
    :type nuc: Dict[str, Any]
    :param num_classes: The number of classes (default 2).
    :type num_classes: Optional[int]
    :return: A list of contours of cells as list of points. Each contour has the same point at position 0 and -1.
    :rtype: List[Contour]

    Each contour has the same point at position 0 and -1.
    If a cell has no class information, it is assumed to belong to the "segmented" class (class 3).
    """
    contours_ = []
    for inst in nuc:
        inst_info = nuc[inst]
        inst_type = inst_info['type']
        if inst_type >= 1 and inst_type <= num_classes:
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
    Save geojson in a file with given name in out_dir.
    
    :param out_dir: The directory where the file will be saved.
    :type out_dir: str
    :param name: The name of the file.
    :type name: str
    :param contours: The list of contours in the GeoJSON format.
    :type contours: List[Contour]
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
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    return parser


def main_with_args(args):
    json_dir = parse_path(args.json_dir)
    names = get_names(json_dir, '.json')
    for name in tqdm(names):
        json_path = json_dir + name + '.json'
        nuc = read_json(json_path)
        contours = parse_contours(nuc, args.num_classes)
        features = create_geojson(contours, args.num_classes)
        save_contours(parse_path(args.gson_dir), name, features)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
