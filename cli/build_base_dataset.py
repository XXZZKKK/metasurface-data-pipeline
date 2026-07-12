from __future__ import annotations

from ..observations.base_dataset import build_cube_dataset, parse_args


def main():
    build_cube_dataset(parse_args())


if __name__ == "__main__":
    main()
