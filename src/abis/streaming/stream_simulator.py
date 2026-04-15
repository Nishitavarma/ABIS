import time
import pandas as pd
from typing import Iterator, Dict, Any, Optional

from abis.utils.config import SETTINGS


def stream_csv(
    path,
    delay_seconds: float,
    max_rows: Optional[int]
) -> Iterator[Dict[str, Any]]:
    """
    Reads a CSV and yields one row at a time (like live data).
    Each yielded item is a dictionary.
    """
    df = pd.read_csv(path)

    # Limit rows for demo
    if max_rows is not None:
        df = df.head(max_rows)

    for _, row in df.iterrows():
        event = row.to_dict()
        yield event
        time.sleep(delay_seconds)


def main():
    print("📡 ABIS Stream Simulator starting...")
    print(f"📄 Reading dataset from: {SETTINGS.data_path}")

    for i, event in enumerate(
        stream_csv(
            SETTINGS.data_path,
            SETTINGS.stream_delay_seconds,
            SETTINGS.max_rows
        ),
        start=1
    ):
        preview = list(event.items())[:5]
        print(f"Event {i}: {preview} ...")


if __name__ == "__main__":
    main()
