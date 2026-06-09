"""
eval_orientation_on_interface.py — orientation distribution of FRONT particles
at the wave interface.

Script port of eval_orientation_on_interface.ipynb, wired to parameters.json and
pipeline_common so it shares the video-scope selection used by the rest of the
pipeline. Produces, per (velocity, width) configuration, a density histogram of
the heading `fixed_arrow` (DEGREES, 0-360) of particles classified as FRONT, and
optionally the clean polar bar plots used as orientation insets.

Output (output.DIR -> eval_orient.OUT_NAME):
    bin_centers   bin-centre heading [deg]
    histogram     density per (velocity, width)
    velocity      CONFIG velocity code (ms per 2 cm)
    width         physical channel width [cm]  (= CONFIG.width * 4)

Augmented by Claude Opus 4.8 (notebook -> pipeline script, parameters.json-driven).
"""

import os
import sqlite3 as sql
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_common import load_params, select_videos, sql_in_list

P = load_params()

DB_PATH      = P["database"]["path"]
OUTPUT_DIR   = P["output"]["DIR"]
Y_MIN        = P["Y_MIN"]

EO           = P["eval_orient"]
SOURCE_TABLE = EO["SOURCE_TABLE"]
CLASS        = EO["CLASSIFICATION"]
ORIENT_COL   = EO["ORIENTATION_COLUMN"]
N_BINS       = EO["N_BINS"]
RANGE_MIN    = EO["RANGE_MIN"]
RANGE_MAX    = EO["RANGE_MAX"]
OUT_NAME     = EO["OUT_NAME"]
SAVE_PLOTS   = EO.get("SAVE_PLOTS", True)
REGION       = EO["region"]


def main():
    print(f"Connecting to {DB_PATH}")
    conn = sql.connect(DB_PATH)

    _, videos = select_videos(conn, REGION)
    print(f"Valid videos in scope: {len(videos)}")
    if not videos:
        conn.close()
        return
    vlist = sql_in_list(videos)

    # We select classification == FRONT and gate only on the upper limit Y_MIN.
    #
    # NOTE: FRONT does not extend in the accumulation zone. By construction a
    # FRONT particle sits on the leading (bright) interface of the wave, ahead of
    # the dark accumulation band, so it never reaches y >= Y_BULK_LIMIT. The
    # accumulation gate (`keep`) used elsewhere in the pipeline is therefore
    # redundant here and is intentionally NOT applied: the FRONT classification
    # already restricts the sample to the not-yet-accumulated interface.
    df_obj = pd.read_sql_query(
        f"SELECT video, processed_frame_index, {ORIENT_COL} "
        f"FROM {SOURCE_TABLE} "
        f"WHERE classification = '{CLASS}' "
        f"  AND video IN ({vlist}) "
        f"  AND bounding_box_mid_y > {Y_MIN}",
        conn,
    )
    df_config = pd.read_sql_query("SELECT * FROM CONFIG", conn)
    conn.close()

    df = df_obj.merge(
        df_config, left_on="video", right_on="unique_name", how="left"
    )
    df = df[df[ORIENT_COL].notna()]
    df = df[df["velocity"] != 0]  # drop fillers / static-only rows
    if df.empty:
        print("No FRONT orientation data after filtering.")
        return

    bins = np.linspace(RANGE_MIN, RANGE_MAX, N_BINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    out_rows = []
    for (width, velocity), grp in df.groupby(["width", "velocity"]):
        hist, _ = np.histogram(
            grp[ORIENT_COL].astype(float), bins=bins, density=True
        )
        out_rows.append(pd.DataFrame({
            "bin_centers": centers,
            "histogram": hist,
            "velocity": velocity,
            "width": int(width) * 4,   # CONFIG.width -> physical cm
        }))

    final = pd.concat(out_rows, ignore_index=True)
    out_path = OUTPUT_DIR + OUT_NAME
    final.to_csv(out_path, index=False)
    print(f"Saved {os.path.abspath(out_path)} "
          f"({final[['velocity', 'width']].drop_duplicates().shape[0]} configs)")

    if SAVE_PLOTS:
        plot_dir = os.path.join(OUTPUT_DIR, "orient")
        os.makedirs(plot_dir, exist_ok=True)
        for (velocity, width), grp in final.groupby(["velocity", "width"]):
            theta = np.deg2rad(grp["bin_centers"].to_numpy())
            radii = grp["histogram"].to_numpy()
            fig, ax = plt.subplots(figsize=(2, 2), subplot_kw={"projection": "polar"})
            bar_w = np.deg2rad(360 / len(theta))
            bars = ax.bar(theta, radii, width=bar_w, bottom=0.0)
            if radii.max() > radii.min():
                norm = plt.Normalize(vmin=radii.min(), vmax=radii.max())
                for bar, r in zip(bars, radii):
                    bar.set_facecolor(plt.cm.Blues(0.3 + 0.7 * norm(r)))
                    bar.set_edgecolor("none")
            ax.set_rticks([]); ax.set_thetagrids([]); ax.grid(False)
            ax.spines["polar"].set_visible(False)
            ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
            plt.tight_layout()
            fname = os.path.join(
                plot_dir, f"colored_orientation_v{int(velocity)}_w{int(width)}.png"
            )
            plt.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=150,
                        transparent=True)
            plt.close()
        print(f"Saved polar orientation plots to {os.path.abspath(plot_dir)}/")

    print("Done.")


if __name__ == "__main__":
    main()
