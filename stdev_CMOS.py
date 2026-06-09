"""
stdev_CMOS.py — mean cluster spatial spread (sigma_x, sigma_y) vs cluster size.

Unified rewrite. The clustered set is built from CLUSTER_PERSISTENCE.clustered_eff
plus the shared accumulation gate, then restricted to CLUSTERS_PROPAGATING. The
px->cm conversion uses the global PX_TO_CM (1 px = PX_TO_CM cm), replacing the
hard-coded 640 px / 80 cm. Output is a CSV in output.DIR; an optional plot is
saved alongside it.
"""

import os
import sqlite3 as sql
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_common import (
    load_params, select_videos, sql_in_list,
    attach_cluster_state, restrict_to_propagating,
)

P = load_params()

DB_PATH           = P["database"]["path"]
OUTPUT_DIR        = P["output"]["DIR"]
PX_TO_CM          = P["PX_TO_CM"]
Y_MIN             = P["Y_MIN"]
Y_BULK_LIMIT      = P["accumulation_gate"]["Y_BULK_LIMIT"]
Y_CLUSTER_CONTACT = P["accumulation_gate"]["Y_CLUSTER_CONTACT"]

S                 = P["stdev"]
SOURCE_TABLE      = S["SOURCE_TABLE"]
FRAME_MAX         = S["FRAME_MAX"]
RESTRICT_PROP     = S["RESTRICT_PROPAGATING"]
COUNT_MIN         = S["COUNT_MIN"]
COUNT_MAX         = S["COUNT_MAX"]
OUT_NAME          = S["OUT_NAME"]
SAVE_PLOT         = S.get("SAVE_PLOT", True)
REGION            = S["region"]


def main():
    print(f"Connecting to {DB_PATH}")
    conn = sql.connect(DB_PATH)

    _, videos = select_videos(conn, REGION)
    print(f"Valid videos in scope: {len(videos)}")
    if not videos:
        conn.close()
        return
    vlist = sql_in_list(videos)

    df_pos = pd.read_sql_query(
        f"""SELECT processed_frame_index, fixed_id, video,
                   bounding_box_mid_x, bounding_box_mid_y
            FROM {SOURCE_TABLE}
            WHERE video IN ({vlist})
              AND processed_frame_index < {FRAME_MAX}
              AND bounding_box_mid_y >= {Y_MIN}""",
        conn,
    )

    df_nodes = pd.read_sql_query(
        f"""SELECT processed_frame_index, fixed_id, video, cluster_id, cluster_size
            FROM data_graphs_nodes
            WHERE video IN ({vlist}) AND processed_frame_index < {FRAME_MAX}""",
        conn,
    )

    df_persist = pd.read_sql_query(
        f"""SELECT video, processed_frame_index, fixed_id, clustered_eff
            FROM CLUSTER_PERSISTENCE
            WHERE video IN ({vlist}) AND processed_frame_index < {FRAME_MAX}""",
        conn,
    )

    master = attach_cluster_state(
        df_pos, df_nodes, df_persist, Y_BULK_LIMIT, Y_CLUSTER_CONTACT
    )
    master = master[(master["clustered_eff"] == 1) & (master["keep"] == 1)].copy()
    print(f"  Persistent, not-accumulated clustered particle-frames: {len(master)}")

    if RESTRICT_PROP:
        df_prop = pd.read_sql_query(
            f"""SELECT video, processed_frame_index, cluster_id
                FROM CLUSTERS_PROPAGATING
                WHERE video IN ({vlist})""", conn)
        master = restrict_to_propagating(master, df_prop)
        print(f"  After propagation filter: {len(master)}")

    if master.empty:
        print("No data after filtering; nothing written.")
        conn.close()
        return

    print("Computing per-cluster spatial spread...")
    cluster_shapes = (master.groupby(
        ["video", "processed_frame_index", "cluster_id"]
    ).agg(
        std_x=("bounding_box_mid_x", "std"),
        std_y=("bounding_box_mid_y", "std"),
        count=("fixed_id", "count"),
    ).reset_index().dropna())

    # px -> cm
    cluster_shapes["std_x"] *= PX_TO_CM
    cluster_shapes["std_y"] *= PX_TO_CM

    size_stats = (cluster_shapes.groupby("count")[["std_y", "std_x"]]
                  .mean().reset_index())
    size_stats = size_stats[
        (size_stats["count"] >= COUNT_MIN) & (size_stats["count"] <= COUNT_MAX)
    ]

    out_path = OUTPUT_DIR + OUT_NAME
    size_stats.to_csv(out_path, index=False)
    print(f"   Saved {os.path.abspath(out_path)}")

    if SAVE_PLOT:
        plt.figure(figsize=(10, 6))
        plt.plot(size_stats["count"], size_stats["std_y"], marker="s",
                 linestyle="-", label=r"Mean $\sigma_y$ (vertical)",
                 color="#1b9e77", alpha=0.8)
        plt.plot(size_stats["count"], size_stats["std_x"], marker="o",
                 linestyle="-", label=r"Mean $\sigma_x$ (horizontal)",
                 color="#d95f02", alpha=0.8)
        plt.title("Mean cluster spatial spread vs cluster size (propagating)")
        plt.xlabel("Number of particles in cluster")
        plt.ylabel("Mean standard deviation [cm]")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        png_path = OUTPUT_DIR + os.path.splitext(OUT_NAME)[0] + ".png"
        plt.savefig(png_path, dpi=150)
        print(f"   Saved plot {os.path.abspath(png_path)}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
