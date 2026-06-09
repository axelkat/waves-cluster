"""
time_series.py — dark-side count vs time for the fixed-velocity / fixed-width
sweeps.

Data for Fig. 1D and Fig. 1E (accumulation transients).

Same dark-side counting rule as accumulation.py (own y < Y_BULK_LIMIT, minus
cluster-mediated contact with y >= Y_CLUSTER_CONTACT), but evaluated at EVERY
frame of the selected videos rather than at a single snapshot. Curves are then
averaged across the video repeats of each configuration.

Configurations are selected by the rules in parameters.json -> time_series.config_rules.
Each rule tags its matched videos with a `fixed` axis ('vel' = fixed velocity,
width varied; 'width' = fixed width, velocity varied), which is the panel split
for Fig. 1D vs 1E. A video may satisfy more than one rule and then appears under
both `fixed` groups (intended).

Unified rewrite: y-lines come from the shared accumulation gate; everything is
parameters.json-driven (no hard-coded DB path, particle count, or config lists).

Augmented by Claude Opus 4.8 (parameters.json integration, rule-driven selection).
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
Y_BRIGHT_LIMIT    = P["accumulation_gate"]["Y_BULK_LIMIT"]
Y_DARK_CONNECTION = P["accumulation_gate"]["Y_CLUSTER_CONTACT"]

T            = P["time_series"]
SOURCE_TABLE = T["SOURCE_TABLE"]
CONFIG_RULES = T["config_rules"]
OUT_NAME     = T["OUT_NAME"]


def count_bright_particles(df_obj_frame, df_nodes_frame):
    """Bright = own y < Y_BRIGHT_LIMIT, minus particles cluster-connected to a
    member at y >= Y_DARK_CONNECTION. (Identical rule to accumulation.py.)"""
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
    return len(np.setdiff1d(bright_candidates, dark_connected_ids))


def matches_rule(row, rule):
    """Test one CONFIG row against one config_rule (parameters.json)."""
    v = row.get("velocity", 0)
    w = row.get("width", 0)
    t = row.get("type", "")
    intensity = row.get("INT", None)

    if "type" in rule and t != rule["type"]:
        return False
    if "velocity" in rule and v != rule["velocity"]:
        return False
    if "width" in rule and w != rule["width"]:
        return False
    if "VEL_VALUES" in rule and v not in rule["VEL_VALUES"]:
        return False
    if "WIDTH_VALUES" in rule and w not in rule["WIDTH_VALUES"]:
        return False
    if "INT_VALUES" in rule and intensity not in rule["INT_VALUES"]:
        return False
    return True


def build_target_configs(config_df):
    """Apply every rule to every CONFIG row; one tagged record per (row, rule)."""
    targets = []
    for _, row in config_df.iterrows():
        for rule in CONFIG_RULES:
            if matches_rule(row, rule):
                targets.append({
                    "video": row["unique_name"],
                    "velocity": row.get("velocity", 0),
                    "width": row.get("width", 0),
                    "type": row.get("type", ""),
                    "INT": row.get("INT", None),
                    "fixed": rule["fixed"],
                })
    return pd.DataFrame(targets)


def main():
    print(f"Connecting to {DB_PATH}")
    conn = sql.connect(DB_PATH)

    print("Loading CONFIG and selecting target configurations...")
    config_df = pd.read_sql("SELECT * FROM CONFIG", con=conn)
    target_df = build_target_configs(config_df)
    if target_df.empty:
        print("No configurations matched the rules in parameters.json.")
        conn.close()
        return

    target_videos = target_df["video"].unique()
    vid_list = "','".join(target_videos)
    print(f"Found {len(target_videos)} matching videos. Loading timelines...")

    df_object = pd.read_sql(
        f"SELECT fixed_id, video, bounding_box_mid_y, processed_frame_index "
        f"FROM {SOURCE_TABLE} WHERE video IN ('{vid_list}')",
        con=conn,
    )
    df_object.dropna(subset=["fixed_id", "bounding_box_mid_y"], inplace=True)

    df_nodes = pd.read_sql(
        f"SELECT video, processed_frame_index, fixed_id, cluster_id "
        f"FROM data_graphs_nodes WHERE video IN ('{vid_list}')",
        con=conn,
    )

    print("Computing per-frame dark-side counts per video...")
    rows = []
    for vid in target_videos:
        df_v = df_object[df_object["video"] == vid]
        df_n = df_nodes[df_nodes["video"] == vid]
        for frame in sorted(df_v["processed_frame_index"].unique()):
            n_bright = count_bright_particles(
                df_v[df_v["processed_frame_index"] == frame],
                df_n[df_n["processed_frame_index"] == frame],
            )
            rows.append({
                "video": vid,
                "processed_frame_index": frame,
                "dark_count": TOTAL_PARTICLES - n_bright,
            })

    res_video_df = pd.DataFrame(rows)

    print("Averaging across configuration repeats...")
    merged = pd.merge(target_df, res_video_df, on="video")
    grouped = merged.groupby(
        ["velocity", "width", "type", "INT", "fixed", "processed_frame_index"]
    )
    final_df = grouped.agg(
        count=("dark_count", "mean"),
        std=("dark_count", "std"),
    ).reset_index()
    final_df["std"] = final_df["std"].fillna(0.0)
    final_df["time"] = final_df["processed_frame_index"] / FPS

    final_df = final_df[
        ["velocity", "width", "type", "processed_frame_index",
         "INT", "count", "std", "time", "fixed"]
    ]

    out_path = OUTPUT_DIR + OUT_NAME
    final_df.to_csv(out_path, index=False)
    print(f"Done. Saved {out_path} ({len(final_df)} rows).")

    conn.close()


if __name__ == "__main__":
    main()
