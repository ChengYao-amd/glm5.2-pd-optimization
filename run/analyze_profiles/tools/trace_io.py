"""Small streaming reader for Kineto/Chrome JSON and JSON.GZ traces.

Import iter_events(path) from an analysis script. The CLI summarizes event
categories, or prints a bounded raw sample with --category. No kernel taxonomy.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import ijson
from ijson.common import ObjectBuilder


def iter_events(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    found = False
    builder = None
    with opener(path, 'rb') as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == 'traceEvents' and event == 'start_array':
                found = True
            if prefix == 'traceEvents.item' and event == 'start_map':
                builder = ObjectBuilder()
            if builder is not None:
                builder.event(event, value)
                if prefix == 'traceEvents.item' and event == 'end_map':
                    yield builder.value
                    builder = None
    if not found:
        raise ValueError(f'{path}: missing traceEvents array')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--category', help='Print raw events of this category')
    parser.add_argument('--limit', type=int, default=5, help='Sample size; 0 prints all')
    args = parser.parse_args()
    counts = Counter()
    printed = 0
    for event in iter_events(args.trace):
        counts[event.get('cat', '<none>')] += 1
        if args.category is not None and args.category == event.get('cat'):
            print(json.dumps(event))
            printed += 1
            if args.limit and printed >= args.limit:
                break  # A bounded preview does not validate the remainder.
    if not args.category:
        print(json.dumps({'file': str(args.trace), 'events': sum(counts.values()),
                          'categories': dict(counts)}, indent=2))


if __name__ == '__main__':
    main()
