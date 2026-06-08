import argparse

from data.downloaders.registry import available_datasets, get_downloader


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare project datasets.")
    parser.add_argument(
        "--dataset",
        default="kvasir",
        choices=available_datasets(),
        help="Dataset to download.",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Directory where prepared datasets are stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload/recreate the prepared dataset if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    downloader_cls = get_downloader(args.dataset)
    downloader = downloader_cls(
        raw_root=args.raw_root,
        force=args.force,
    )
    dataset_dir = downloader.run()
    print(f"Dataset ready at: {dataset_dir}")


if __name__ == "__main__":
    main()
