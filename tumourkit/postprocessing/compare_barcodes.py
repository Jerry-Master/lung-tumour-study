"""
Computes q-Wassertein distance and bottleneck distance of persistence diagrams.
Input format: CSV (births and deaths)
Output format: TXT

Copyright (C) 2025  Jose Pérez Cano

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
import argparse
from argparse import Namespace


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--barcode-dir', type=str, help='Path to the csv with barcode information.')
    parser.add_argument('--output-dir', type=str, help='Path to folder where to save results.')
    return parser


def main_with_args(args: Namespace) -> None:
    pass


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)