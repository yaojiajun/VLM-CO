#!/usr/bin/env python3
"""
Generate 200,000 training images using the best performing visualization style.
Uses the draw function from saved model 09131853_ID_18185_0.4420
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=20, help='Number of parallel workers')
    parser.add_argument('--count', type=int, default=200000, help='Number of images to generate')
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent
    code_path = base_dir / 'saved_images/09131853_ID_18185_0.4420/gpt.py'
    json_path = base_dir / 'train_mvc_under30_5000.json'
    out_dir = base_dir / 'data/sft/train'

    # Verify source files exist
    if not code_path.exists():
        print(f"Error: Code file not found: {code_path}")
        sys.exit(1)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import render function
    sys.path.insert(0, str(base_dir))
    from render_images import render

    print(f"Starting generation of {args.count:,} images...")
    print(f"Code: {code_path}")
    print(f"Data: {json_path}")
    print(f"Output: {out_dir}")
    print(f"Workers: {args.workers}")
    print()

    # Run rendering
    n_done, errors, total = render(
        code_path=str(code_path),
        json_path=str(json_path),
        out_dir=str(out_dir),
        count=args.count,
        workers=args.workers,
        prefix='train'
    )

    print()
    print(f"Generation complete!")
    print(f"Images created: {n_done:,} / {total:,}")
    print(f"Errors: {errors}")

    if errors > 0:
        print(f"Warning: {errors} images failed to render")
        sys.exit(1)

if __name__ == '__main__':
    main()
