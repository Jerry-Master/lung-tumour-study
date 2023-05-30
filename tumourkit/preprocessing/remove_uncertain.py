"""
Removes cells with label uncertain from the annotations.

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
import os
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
from argparse import Namespace
from ..utils.preprocessing import get_names, read_gson, save_geojson


def process_gson(gson: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters out uncertain classifications from a list of GeoJSON dictionaries.

    Given a list of GeoJSON dictionaries, this function filters out the GeoJSON objects
    that have an uncertain classification. The uncertain classification is determined
    based on the 'name' property under the 'classification' key in the GeoJSON dictionary.

    :param gson: The list of GeoJSON dictionaries.
    :type gson: List[Dict[str, Any]]
    :return: The filtered list of GeoJSON dictionaries without uncertain classifications.
    :rtype: List[Dict[str, Any]]
    """
    return list(filter(lambda x: x['properties']['classification']['name'] != 'uncertain', gson))


def main_with_args(args: Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    names = get_names(args.input_dir, '.geojson')
    for name in tqdm(names):
        gson = read_gson(name, args.input_dir)
        out_gson = process_gson(gson)
        save_geojson(out_gson, name, args.output_dir)
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='Folder where geojsons are with label uncertain.')
    parser.add_argument('--output-dir', type=str, help='Folder where to save geojsons without label uncertain.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
