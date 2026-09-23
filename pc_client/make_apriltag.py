#!/usr/bin/env python3

"""Generate a printable AprilTag 36h11 marker as SVG.

SVG is used rather than PNG or PDF because it carries real physical units, so
printing at 100% scale gives a tag whose measured size matches --size-mm
exactly. That matters: solvePnP scales its whole translation estimate by the
tag size, so a tag printed 5% small puts every distance 5% off.

The page also carries a 100 mm reference line. Measure it with a ruler before
trusting any pose output.
"""

import argparse

import cv2
import numpy

FAMILY = cv2.aruco.DICT_APRILTAG_36h11
GRID = 8  # 6x6 data bits plus a one-cell black border.


def parse_args():
    parser = argparse.ArgumentParser(description='Write a printable AprilTag 36h11 SVG.')
    parser.add_argument('--id', type=int, default=0, help='Tag id (0-586).')
    parser.add_argument('--size-mm', type=float, default=100.0,
                        help='Side of the outer black square in mm. This is the '
                             'number to pass to visual_state.py --tag-mm.')
    parser.add_argument('--out', default='apriltag_36h11.svg')
    return parser.parse_args()


def build_svg(tag_id, size_mm):
    dictionary = cv2.aruco.getPredefinedDictionary(FAMILY)
    if not 0 <= tag_id < dictionary.bytesList.shape[0]:
        raise SystemExit('Tag id must be 0..{}'.format(dictionary.bytesList.shape[0] - 1))

    grid = cv2.aruco.generateImageMarker(dictionary, tag_id, GRID)
    cell = size_mm / GRID
    quiet = cell  # One cell of white all round; the detector needs it.
    label_mm = 22.0
    page_w = size_mm + 2 * quiet
    page_h = page_w + label_mm

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">'.format(
            w=round(page_w, 4), h=round(page_h, 4)),
        '<rect x="0" y="0" width="{}" height="{}" fill="#ffffff"/>'.format(
            round(page_w, 4), round(page_h, 4)),
    ]

    # One rect per black cell. Cells are emitted at exact mm coordinates.
    for row in range(GRID):
        for col in range(GRID):
            if grid[row, col] == 0:
                parts.append(
                    '<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="#000000"/>'.format(
                        x=round(quiet + col * cell, 4),
                        y=round(quiet + row * cell, 4),
                        s=round(cell, 4) + 0.002))  # Hairline overlap kills seams.

    text_y = page_w + 7.0
    parts.append(
        '<text x="{x}" y="{y}" font-family="sans-serif" font-size="4.2" fill="#000000">'
        'AprilTag 36h11  id={i}  outer black square = {s:.1f} mm  '
        '(print at 100%, do not scale to fit)</text>'.format(
            x=round(quiet, 4), y=round(text_y, 4), i=tag_id, s=size_mm))

    # Reference line so the print scale can be checked against a ruler.
    line_y = page_w + 15.0
    line_x2 = quiet + 100.0
    if line_x2 <= page_w - 2:
        parts.append(
            '<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#000000" stroke-width="0.4"/>'
            .format(x1=round(quiet, 4), x2=round(line_x2, 4), y=round(line_y, 4)))
        for x in (quiet, line_x2):
            parts.append(
                '<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#000000" '
                'stroke-width="0.4"/>'.format(
                    x=round(x, 4), y1=round(line_y - 2, 4), y2=round(line_y + 2, 4)))
        parts.append(
            '<text x="{x}" y="{y}" font-family="sans-serif" font-size="4.2" '
            'fill="#000000">|&#8212; this line must measure exactly 100 mm '
            '&#8212;|</text>'.format(x=round(quiet, 4), y=round(line_y + 7, 4)))

    parts.append('</svg>')
    return '\n'.join(parts), page_w, page_h


def main():
    args = parse_args()
    svg, page_w, page_h = build_svg(args.id, args.size_mm)
    with open(args.out, 'w', encoding='utf-8') as handle:
        handle.write(svg)
    print('Wrote {}'.format(args.out))
    print('Page {:.1f} x {:.1f} mm; tag id {} at {:.1f} mm.'.format(
        page_w, page_h, args.id, args.size_mm))
    print('Open it in a browser and print at 100% scale, then check the 100 mm line.')
    print('Pass --tag-mm {:.1f} to visual_state.py.'.format(args.size_mm))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
