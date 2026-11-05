"""Compare screenshots using per-channel tolerance; requires Pillow."""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def compare(reference, actual, threshold=0):
    if not 0 <= threshold <= 255:
        raise ValueError('threshold must be between 0 and 255')
    a = Image.open(reference).convert('RGB')
    b = Image.open(actual).convert('RGB')
    if a.size != b.size:
        raise ValueError(f'image dimensions differ: {a.size} vs {b.size}')
    diff = ImageChops.difference(a, b)
    histogram = diff.histogram()
    channels = sum(histogram[i] for i in range(256 * 3) if i % 256 > threshold)
    # A pixel is different when any of its RGB channels exceeds the threshold.
    mask = diff.point([0 if i <= threshold else 255 for i in range(256)] * 3)
    r, g, b = mask.split()
    pixels = ImageChops.lighter(ImageChops.lighter(r, g), b).histogram()[255]
    count = a.width * a.height
    return diff, {'width': a.width, 'height': a.height, 'channel_threshold': threshold,
                  'different_pixels': pixels, 'different_fraction': pixels / count,
                  'different_channels': channels, 'mean_absolute_error': sum(ImageStat.Stat(diff).mean) / 3,
                  'max_channel_error': max(high for _, high in diff.getextrema())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('actual', type=Path)
    parser.add_argument('--diff', type=Path, required=True)
    parser.add_argument('--threshold', type=int, default=0, help='per-channel difference ignored, 0..255')
    parser.add_argument('--max-different-fraction', type=float, default=0.0)
    parser.add_argument('--reference-state', type=Path, help='JSON scene, event and camera metadata for the reference')
    parser.add_argument('--actual-state', type=Path, help='JSON scene, event and camera metadata for the actual image')
    args = parser.parse_args()
    if not 0 <= args.max_different_fraction <= 1:
        parser.error('--max-different-fraction must be between 0 and 1')
    try:
        aligned = False
        if args.reference_state and args.actual_state:
            a = json.loads(args.reference_state.read_text(encoding='utf-8'))
            b = json.loads(args.actual_state.read_text(encoding='utf-8'))
            keys = ('scene', 'event', 'camera')
            aligned = all(k in a and k in b and a[k] is not None and a[k] == b[k] for k in keys)
        diff, metrics = compare(args.reference, args.actual, args.threshold)
        diff.save(args.diff)
    except (OSError, ValueError, TypeError) as e:
        parser.error(str(e))
    metrics['matching_state_metadata'] = aligned
    if not aligned:
        print('warning: scene, event and camera metadata are missing or differ; pixel comparison may not be meaningful', file=sys.stderr)
    print(json.dumps(metrics, sort_keys=True))
    return int(metrics['different_fraction'] > args.max_different_fraction)


if __name__ == '__main__':
    raise SystemExit(main())

