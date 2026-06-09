"""
save_phase_v2.py — phase-diagram exporters (fig5a orientation densities,
fig5b pushing/braking composition).

Unified rewrite. The clustered set is built from CLUSTER_PERSISTENCE.clustered_eff
plus the shared accumulation gate (pipeline_common.attach_cluster_state, using
parameters.json Y limits), then restricted to CLUSTERS_PROPAGATING. The inline
"cluster_size >= 3 + back-wall contact" logic of the previous version is gone;
the cluster definition is now identical to the rest of the pipeline.

Outputs (output.DIR):
  fig5a  per-bin orientation densities for FRONT vs DARK members.
  fig5b  joint (PUSHING, BRAKING) composition counts per cluster instance.
"""

import sqlite3 as sql
import pandas as pd
import numpy as np

from pipeline_common import (
    load_params, select_videos, sql_in_list,
    attach_cluster_state, restrict_to_propagating,
)

P = load_params()

DB_PATH           = P["database"]["path"]
OUTPUT_DIR        = P["output"]["DIR"]
Y_MIN             = P["Y_MIN"]
Y_BULK_LIMIT      = P["accumulation_gate"]["Y_BULK_LIMIT"]
Y_CLUSTER_CONTACT = P["accumulation_gate"]["Y_CLUSTER_CONTACT"]

PH                = P["phase"]
SOURCE_TABLE      = PH["SOURCE_TABLE"]
FRAME_MAX         = PH["FRAME_MAX"]
RESTRICT_PROP     = PH["RESTRICT_PROPAGATING"]
N_BINS_ORIENT     = PH["N_BINS_ORIENT"]
RECLASS_ANGLE_DEG = PH["RECLASS_ANGLE_DEG"]
OUT_5A            = OUTPUT_DIR + PH["OUT_NAME_5A"]
OUT_5B            = OUTPUT_DIR + PH["OUT_NAME_5B"]
REGION            = PH["region"]


def main():
    print(f"Connecting to {DB_PATH}")
    conn = sql.connect(DB_PATH)

    _, videos = select_videos(conn, REGION)
    print(f"Valid videos in scope: {len(videos)}")
    if not videos:
        conn.close()
        return
    vlist = sql_in_list(videos)

    # Positions + classification (carries the columns fig5 needs)
    df_pos = pd.read_sql_query(
        f"""SELECT processed_frame_index, fixed_id, video, fixed_arrow,
                   classification, bounding_box_mid_x, bounding_box_mid_y
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

    # Unified clustered state + accumulation gate, then keep persistent &
    # not-yet-accumulated members.
    master = attach_cluster_state(
        df_pos, df_nodes, df_persist, Y_BULK_LIMIT, Y_CLUSTER_CONTACT
    )
    master = master[(master["clustered_eff"] == 1) & (master["keep"] == 1)].copy()
    print(f"  Persistent, not-accumulated clustered particle-frames: {len(master)}")

    if RESTRICT_PROP:
        df_prop = pd.read_sql_query(
            f"""SELECT video, processed_frame_index, cluster_id, v_ratio
                FROM CLUSTERS_PROPAGATING
                WHERE video IN ({vlist})""", conn)
        master = restrict_to_propagating(master, df_prop)
        ninst = master.groupby(
            ["video", "processed_frame_index", "cluster_id"]
        ).ngroups
        print(f"  After propagation filter: {len(master)} particle-frames "
              f"in {ninst} cluster instances")

    if master.empty:
        print("No data after filtering; nothing written.")
        conn.close()
        return

    # -- FIGURE 5A: orientation densities (FRONT vs DARK) --
    print("1. Figure 5a (orientation densities)...")
    bins = np.linspace(0, 360, N_BINS_ORIENT + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    dens_front, _ = np.histogram(
        master.loc[master["classification"] == "FRONT", "fixed_arrow"],
        bins=bins, density=True)
    dens_dark, _ = np.histogram(
        master.loc[master["classification"] == "DARK", "fixed_arrow"],
        bins=bins, density=True)
    pd.DataFrame({
        "angle_bin_center": centers,
        "density_FRONT": dens_front,
        "density_DARK": dens_dark,
    }).to_csv(OUT_5A, index=False)
    print(f"   Saved {OUT_5A}")

    # -- FIGURE 5B: pushing vs braking composition --
    print("2. Figure 5b (pushing vs braking)...")
    df5b = master.copy()
    raw = df5b["fixed_arrow"].fillna(0) % 180 # wrapping
    df5b["orientation"] = np.where(raw > 90, 180 - raw, raw) #remap to the coordinate system of the main script.
    df5b["final_label"] = df5b["classification"].map( 
        {"FRONT": "PUSHING", "DARK": "BRAKING"})
    reclass = (df5b["classification"] == "FRONT") & \
              (df5b["orientation"] < RECLASS_ANGLE_DEG)
    df5b.loc[reclass, "final_label"] = "BRAKING"

    counts = (df5b.groupby(
        ["video", "processed_frame_index", "cluster_id", "final_label"]
    )["fixed_id"].count().unstack(fill_value=0))
    for col in ["PUSHING", "BRAKING"]:
        if col not in counts.columns:
            counts[col] = 0

    comp = (counts.groupby(["PUSHING", "BRAKING"]).size()
                  .reset_index(name="frequency"))
    comp["particle_weight"] = comp["frequency"] * (comp["PUSHING"] + comp["BRAKING"])
    out5b = comp[comp["frequency"] > 10].copy()
    out5b.to_csv(OUT_5B, index=False)
    print(f"   Saved {OUT_5B} ({len(out5b)} bins)")

    if len(out5b) > 5:
        x = out5b["PUSHING"].values
        y = out5b["BRAKING"].values
        w = out5b["frequency"].values
        slope = np.sum(w * x * y) / np.sum(w * x * x)
        print(f"   Weighted slope BRAKING/PUSHING (through origin): {slope:.3f}") #0.46 
        print(f"   (manuscript reports ~1/2)")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
