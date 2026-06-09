"""
export_1b.py — transverse density profile PDF(x) for Fig. 1B.

Static-figure companion to animate_density_filtered.py (which renders the same
quantity as a movie). For the configuration (TARGET_VELOCITY, TARGET_WIDTH) and
the chosen INT set, all videos in the group are pooled at the snapshot window
around T_SNAPSHOT_S (+/- HALF_WINDOW_FRAMES) and histogrammed in x.

The wave propagates along y (see README); x is therefore the transverse
coordinate. The histogram is normalised so that the integral of p(x) dx in cm
is 1 (probability density per cm), matching the animation.

Output (output.DIR -> export_1b.OUT_NAME):
    x_cm           bin-centre transverse position [cm]
    density        PDF value [1/cm]
    n_samples      total particle-frames pooled into the histogram

Augmented by Claude Opus 4.8 (new exporter, parameters.json-driven).
"""

import numpy as np
import pandas as pd
import sqlite3 as sql

from pipeline_common import load_params

P = load_params()

DB_PATH        = P["database"]["path"]
OUTPUT_DIR     = P["output"]["DIR"]
FPS            = P["FPS"]
PX_TO_CM       = P["PX_TO_CM"]

E              = P["export_1b"]
SOURCE_TABLE   = E["SOURCE_TABLE"]
TARGET_VEL     = E["TARGET_VELOCITY"]
TARGET_WIDTH   = E["TARGET_WIDTH"]
INT_VALUES     = E["INT_VALUES"]
T_SNAPSHOT_S   = E["T_SNAPSHOT_S"]
HALF_WINDOW    = E["HALF_WINDOW_FRAMES"]
FRAME_WIDTH_PX = E["FRAME_WIDTH_PX"]
N_BINS_X       = E["N_BINS_X"]
OUT_NAME       = E["OUT_NAME"]

FRAME_CENTER = int(round(T_SNAPSHOT_S * FPS))
FRAME_START  = FRAME_CENTER - HALF_WINDOW
FRAME_END    = FRAME_CENTER + HALF_WINDOW


def main():
    print(f"Connecting to {DB_PATH}")
    conn = sql.connect(DB_PATH)

    int_list = ",".join(str(int(i)) for i in INT_VALUES)
    config_df = pd.read_sql(
        f"SELECT unique_name, velocity, width, INT FROM CONFIG "
        f"WHERE velocity = ? AND width = ? AND INT IN ({int_list})",
        conn, params=(TARGET_VEL, TARGET_WIDTH),
    )
    if config_df.empty:
        print(f"No videos for velocity={TARGET_VEL}, width={TARGET_WIDTH}, "
              f"INT in {INT_VALUES}.")
        conn.close()
        return

    vidids = config_df["unique_name"].tolist()
    placeholders = ",".join(["?"] * len(vidids))
    print(f"Pooling {len(vidids)} videos, frames {FRAME_START}-{FRAME_END}...")

    df = pd.read_sql(
        f"SELECT video, fixed_id, processed_frame_index, bounding_box_mid_x "
        f"FROM {SOURCE_TABLE} "
        f"WHERE video IN ({placeholders}) "
        f"AND processed_frame_index BETWEEN {FRAME_START} AND {FRAME_END}",
        conn, params=vidids,
    )
    df = df.dropna(subset=["fixed_id", "bounding_box_mid_x"])
    if df.empty:
        print("No particles in the snapshot window.")
        conn.close()
        return

    # x in px on a FRAME_WIDTH_PX grid -> cm. Edges fixed to the full frame so
    # the normalisation (integral p dx_cm = 1) is config-independent.
    edges_px = np.linspace(0, FRAME_WIDTH_PX, N_BINS_X + 1)
    edges_cm = edges_px * PX_TO_CM
    centers_cm = 0.5 * (edges_cm[1:] + edges_cm[:-1])

    x_px = df["bounding_box_mid_x"].clip(0, FRAME_WIDTH_PX).to_numpy()
    density, _ = np.histogram(x_px * PX_TO_CM, bins=edges_cm, density=True)

    out = pd.DataFrame({
        "x_cm": centers_cm,
        "density": density,
        "n_samples": len(x_px),
    })
    out_path = OUTPUT_DIR + OUT_NAME
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} bins, {len(x_px)} particle-frames pooled)")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
