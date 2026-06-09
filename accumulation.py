"""
accumulation.py — steady-state dark-side accumulation per configuration.

Data for Fig. 1F (net accumulation phase diagram) and supplementary Fig. SI1.

For each video, at a snapshot time T_SNAPSHOT_S (+/- HALF_WINDOW_FRAMES, averaged),
count the particles on the dark side of the standing wave. A particle counts as
"dark" if it is NOT truly bright, where:

  bright candidate   : own y < Y_BULK_LIMIT
  reclassified dark  : the candidate belongs to a cluster (data_graphs_nodes)
                       that has ANY member at y >= Y_CLUSTER_CONTACT
                       (cluster-mediated contact with the accumulation zone)

    dark_count = TOTAL_PARTICLES - (truly bright count)

Counts are averaged over the snapshot window, then over video repeats of each
(velocity, width) configuration. The filler-only baseline (velocity == 0,
INT == FILLER_INT) is subtracted to give the NET accumulation gain `delta`, with
uncertainty propagated in quadrature.

Unified rewrite: the two y-lines are the shared accumulation gate
(parameters.json accumulation_gate.Y_BULK_LIMIT / Y_CLUSTER_CONTACT), identical
to crosser.py / pipeline_common.attach_cluster_state. All tunables live in
parameters.json; nothing is hard-coded here.

Augmented by Claude Opus 4.8 (parameters.json integration, gate unification).
"""

import numpy as np
import pandas as pd
import sqlite3 as sql

from pipeline_common import load_params

P = load_params()

DB_PATH           = P["database"]["path"]
OUTPUT_DIR        = P["output"]["DIR"]
FPS               = P["FPS"]
TOTAL_PARTICLES   = P["TOTAL_PARTICLES"]
Y_BRIGHT_LIMIT    = P["accumulation_gate"]["Y_BULK_LIMIT"]      # 512
Y_DARK_CONNECTION = P["accumulation_gate"]["Y_CLUSTER_CONTACT"]  # 560

A             = P["accumulation"]
SOURCE_TABLE  = A["SOURCE_TABLE"]
T_SNAPSHOT_S  = A["T_SNAPSHOT_S"]
HALF_WINDOW   = A["HALF_WINDOW_FRAMES"]
FILLER_INT    = A["FILLER_INT"]
OUT_NAME      = A["OUT_NAME"]

FRAME_CENTER = int(round(T_SNAPSHOT_S * FPS))
FRAME_START  = FRAME_CENTER - HALF_WINDOW
FRAME_END    = FRAME_CENTER + HALF_WINDOW


# ---------------------------------------------------------------------------
# Core counting (single frame, single video)
# ---------------------------------------------------------------------------
def count_bright_particles(df_obj_frame, df_nodes_frame):
    """
    Count unique bright fixed_ids at one frame: own y < Y_BRIGHT_LIMIT, then
    drop any candidate that is cluster-connected to a member at
    y >= Y_DARK_CONNECTION (cluster-mediated accumulation).
    """
    if df_obj_frame.empty:
        return 0

    bright_candidates = df_obj_frame[
        df_obj_frame["bounding_box_mid_y"] < Y_BRIGHT_LIMIT
    ]["fixed_id"].unique()
    if len(bright_candidates) == 0:
        return 0

    if df_nodes_frame.empty:
        return len(bright_candidates)

    nodes_with_y = df_nodes_frame.merge(
        df_obj_frame[["fixed_id", "bounding_box_mid_y"]], on="fixed_id", how="inner"
    )
    dark_clusters = nodes_with_y.loc[
        nodes_with_y["bounding_box_mid_y"] >= Y_DARK_CONNECTION, "cluster_id"
    ].unique()
    if len(dark_clusters) == 0:
        return len(bright_candidates)

    dark_connected_ids = df_nodes_frame.loc[
        df_nodes_frame["cluster_id"].isin(dark_clusters), "fixed_id"
    ].unique()
    truly_bright = np.setdiff1d(bright_candidates, dark_connected_ids)
    return len(truly_bright)


def main():
    print(f"Connecting to {DB_PATH}")
    conn = sql.connect(DB_PATH)

    print("Loading CONFIG...")
    config_df = pd.read_sql("SELECT * FROM CONFIG", con=conn)

    print("Loading data_graphs_nodes...")
    df_nodes = pd.read_sql(
        "SELECT video, processed_frame_index, fixed_id, cluster_id, cluster_size "
        "FROM data_graphs_nodes",
        con=conn,
    )

    print(f"Loading {SOURCE_TABLE} in window around t={T_SNAPSHOT_S}s "
          f"(frames {FRAME_START}-{FRAME_END})...")
    df_object = pd.read_sql(
        f"SELECT fixed_id, video, bounding_box_mid_y, processed_frame_index "
        f"FROM {SOURCE_TABLE} "
        f"WHERE processed_frame_index BETWEEN {FRAME_START} AND {FRAME_END}",
        con=conn,
    )
    df_object.dropna(subset=["fixed_id", "bounding_box_mid_y"], inplace=True)

    df_nodes_snap = df_nodes[
        (df_nodes["processed_frame_index"] >= FRAME_START)
        & (df_nodes["processed_frame_index"] <= FRAME_END)
    ].copy()

    frames_in_window = sorted(df_object["processed_frame_index"].unique())
    print(f"Window: {len(frames_in_window)} frames found")

    # --- per-video dark counts (snapshot-window average) ---
    print("Computing dark-side counts per video...")
    results = []
    for _, row in config_df.iterrows():
        vid       = row["unique_name"]
        velocity  = row["velocity"]
        width     = row["width"]
        intensity = row.get("INT", None)

        df_v = df_object[df_object["video"] == vid]
        df_n = df_nodes_snap[df_nodes_snap["video"] == vid]

        dark_counts = []
        for frame in frames_in_window:
            df_v_frame = df_v[df_v["processed_frame_index"] == frame]
            df_n_frame = df_n[df_n["processed_frame_index"] == frame]
            n_bright = count_bright_particles(df_v_frame, df_n_frame)
            dark_counts.append(TOTAL_PARTICLES - n_bright)

        results.append({
            "video": vid,
            "velocity": velocity,
            "width": width,
            "INT": intensity,
            "dark_count_mean": np.mean(dark_counts) if dark_counts else 0.0,
            "dark_count_std": np.std(dark_counts, ddof=1) if len(dark_counts) > 1 else 0.0,
        })

    results_df = pd.DataFrame(results)

    # --- average per (velocity, width) configuration (active runs only) ---
    print("Averaging per (velocity, width) configuration...")
    active_df = results_df[results_df["velocity"] != 0].copy()
    active_avg = active_df.groupby(["velocity", "width"]).agg(
        avg_dark=("dark_count_mean", "mean"),
        std_dark=("dark_count_mean", "std"),   # spread across video repeats
        n_videos=("dark_count_mean", "count"),
    ).reset_index()
    active_avg["std_dark"] = active_avg["std_dark"].fillna(0.0)

    # --- filler-only baseline: velocity == 0 AND INT == FILLER_INT ---
    filler_df = results_df[
        (results_df["velocity"] == 0) & (results_df["INT"] == FILLER_INT)
    ].copy()
    filler_avg = filler_df["dark_count_mean"].mean()
    filler_std = filler_df["dark_count_mean"].std(ddof=1) if len(filler_df) > 1 else 0.0
    print(f"\nFiller baseline (velocity=0, INT={FILLER_INT}): "
          f"avg = {filler_avg:.2f} +/- {filler_std:.2f}  (n={len(filler_df)})")

    # --- net gain over filler, with propagated uncertainty ---
    active_avg["filler_baseline"] = filler_avg
    active_avg["filler_std"]      = filler_std
    active_avg["delta"]           = active_avg["avg_dark"] - filler_avg
    active_avg["delta_std"]       = np.sqrt(active_avg["std_dark"] ** 2 + filler_std ** 2)

    print("\n--- Net dark-side accumulation gain over filler baseline ---")
    print(active_avg.to_string(index=False))

    out_path = OUTPUT_DIR + OUT_NAME
    active_avg.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(active_avg)} rows)")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
