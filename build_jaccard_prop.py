"""
build_jaccard_prop.py — build the CLUSTERS_PROPAGATING table.

Unified rewrite. The "what is clustered" question is NOT recomputed here: the
inline dwell filter + 1 s purity post-filter of the previous version are
replaced by the persistent per-frame flag CLUSTER_PERSISTENCE.clustered_eff,
which already encodes MIN_CLUSTER_RUN_FRAMES (dwell) and FILL_RATIO_MIN
(temporal coherence). The accumulation gate uses the parameters.json Y limits
(accumulation_gate.Y_BULK_LIMIT / Y_CLUSTER_CONTACT), identical to crosser.py.

Per video:
  (1) Read clustered state + accumulation gate (pipeline_common.attach_cluster_state).
  (2) Reconstitute clusters from surviving (clustered_eff==1 & keep==1)
      particles, re-checking cluster_size >= MIN_CLUSTER_SIZE.
  (3) Track each cluster forward by Jaccard membership overlap across short
      lookahead windows.
  (4) Keep clusters whose centroid advances in y at ~the wave speed
      (within VELOCITY_TOLERANCE).

Output: one row per surviving (video, frame, cluster_id), with measured cluster
velocity (px/frame) and v_ratio = v_cluster / v_wave.

CONFIG.velocity is the time in MILLISECONDS to advance 2 cm, so
    v_w [px/frame] = 2000 / (velocity * PX_TO_CM * FPS).
"""

import sqlite3 as sql
import pandas as pd
import numpy as np
from collections import defaultdict

from pipeline_common import (
    load_params, select_videos, sql_in_list, attach_cluster_state,
)

P = load_params()

DB_PATH           = P["database"]["path"]
FPS               = P["FPS"]
PX_TO_CM          = P["PX_TO_CM"]
MIN_CLUSTER_SIZE  = P["clustering"]["MIN_CLUSTER_SIZE"]
Y_BULK_LIMIT      = P["accumulation_gate"]["Y_BULK_LIMIT"]
Y_CLUSTER_CONTACT = P["accumulation_gate"]["Y_CLUSTER_CONTACT"]

J                 = P["jaccard"]
SOURCE_TABLE      = J["SOURCE_TABLE"]
FRAME_MAX         = J["FRAME_MAX"]
DT_SHORT          = J["DT_SHORT"]
N_LOOKAHEADS      = J["N_LOOKAHEADS"]
MIN_VALID_POINTS  = J["MIN_VALID_POINTS"]
JACCARD_MIN       = J["JACCARD_MIN"]
VELOCITY_TOLERANCE = J["VELOCITY_TOLERANCE"]
OUT_TABLE         = J["OUT_TABLE"]
REGION            = J["region"]


def velocity_ms_to_pxframe(velocity_ms):
    """CONFIG.velocity (ms per 2 cm step) -> wave speed in px/frame."""
    return 2000.0 / (velocity_ms * PX_TO_CM * FPS)


# ---------------------------------------------------------------------------
# Jaccard helpers
# ---------------------------------------------------------------------------
def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def best_match(particles_t, clusters_at_target):
    best_id, best_j = None, 0.0
    for cid, members in clusters_at_target.items():
        j = jaccard(particles_t, members)
        if j > best_j:
            best_id, best_j = cid, j
    return best_id, best_j


# ---------------------------------------------------------------------------
# Tracking, per video
# ---------------------------------------------------------------------------
def track_clusters_for_video(df_unified, v_w_pxframe):
    """
    df_unified: one video, output of attach_cluster_state (has clustered_eff,
    keep, cluster_id, cluster_size, positions). Reconstitutes persistent,
    not-yet-accumulated clusters, Jaccard-tracks them, applies the wave-speed
    velocity filter.
    """
    gated = df_unified[
        (df_unified["clustered_eff"] == 1) & (df_unified["keep"] == 1)
    ].copy()
    if gated.empty:
        return []

    cluster_members  = defaultdict(dict)   # frame -> cid -> set(fixed_id)
    cluster_centroid = defaultdict(dict)   # frame -> cid -> (x, y)

    for (frame, cid), grp in gated.groupby(
        ["processed_frame_index", "cluster_id"]
    ):
        members = set(grp["fixed_id"].tolist())
        if len(members) < MIN_CLUSTER_SIZE:   # re-check after gating
            continue
        cluster_members[frame][cid] = members
        cluster_centroid[frame][cid] = (
            grp["bounding_box_mid_x"].mean(),
            grp["bounding_box_mid_y"].mean(),
        )

    if not cluster_members:
        return []

    results = []
    frames_sorted = sorted(cluster_members.keys())
    frames_set = set(frames_sorted)

    for t in frames_sorted:
        for cid, members_t in cluster_members[t].items():
            x0, y0 = cluster_centroid[t][cid]

            ts = [t]; ys = [y0]; xs = [x0]
            current_members = members_t

            for k in range(1, N_LOOKAHEADS + 1):
                t_next = t + k * DT_SHORT
                if t_next not in frames_set:
                    break
                target_clusters = cluster_members[t_next]
                if not target_clusters:
                    break
                best_cid, best_j = best_match(current_members, target_clusters)
                if best_cid is None or best_j < JACCARD_MIN:
                    break
                x_next, y_next = cluster_centroid[t_next][best_cid]
                ts.append(t_next); xs.append(x_next); ys.append(y_next)
                current_members = target_clusters[best_cid]

            if len(ts) < MIN_VALID_POINTS:
                continue

            slope, _ = np.polyfit(np.array(ts, float), np.array(ys, float), 1)

            v_low  = v_w_pxframe * (1 - VELOCITY_TOLERANCE)
            v_high = v_w_pxframe * (1 + VELOCITY_TOLERANCE)
            if not (v_low <= slope <= v_high):
                continue

            results.append({
                "processed_frame_index": t,
                "cluster_id": cid,
                "cluster_size": len(members_t),
                "centroid_x": x0,
                "centroid_y": y0,
                "v_cluster_pxframe": slope,
                "v_wave_pxframe": v_w_pxframe,
                "v_ratio": slope / v_w_pxframe if v_w_pxframe else np.nan,
                "n_track_points": len(ts),
                "track_duration_frames": ts[-1] - ts[0],
            })

    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print(f"Connecting to {DB_PATH}")
    conn = sql.connect(DB_PATH)

    df_config, videos = select_videos(conn, REGION)
    print(f"Valid videos in scope: {len(videos)}")
    if not videos:
        conn.close()
        return
    vlist = sql_in_list(videos)

    print("Fetching graph nodes...")
    df_nodes = pd.read_sql_query(
        f"""SELECT processed_frame_index, fixed_id, video, cluster_id, cluster_size
            FROM data_graphs_nodes
            WHERE video IN ({vlist}) AND processed_frame_index < {FRAME_MAX}""",
        conn,
    )

    print("Fetching positions...")
    df_pos = pd.read_sql_query(
        f"""SELECT processed_frame_index, fixed_id, video,
                   bounding_box_mid_x, bounding_box_mid_y
            FROM {SOURCE_TABLE}
            WHERE video IN ({vlist}) AND processed_frame_index < {FRAME_MAX}""",
        conn,
    ).dropna(subset=["fixed_id", "bounding_box_mid_x", "bounding_box_mid_y"])

    print("Fetching persistent cluster flags (CLUSTER_PERSISTENCE)...")
    df_persist = pd.read_sql_query(
        f"""SELECT video, processed_frame_index, fixed_id, clustered_eff
            FROM CLUSTER_PERSISTENCE
            WHERE video IN ({vlist}) AND processed_frame_index < {FRAME_MAX}""",
        conn,
    )

    # Unified clustered state + accumulation gate, computed once (vectorized).
    df_unified_all = attach_cluster_state(
        df_pos, df_nodes, df_persist, Y_BULK_LIMIT, Y_CLUSTER_CONTACT
    )

    video_to_velocity = dict(zip(df_config["unique_name"], df_config["velocity"]))

    all_results = []
    for video in videos:
        v_ms = video_to_velocity.get(video)
        if v_ms is None:
            continue
        v_w_pxframe = velocity_ms_to_pxframe(v_ms)
        v_w_cms = 2000.0 / v_ms

        df_v = df_unified_all[df_unified_all["video"] == video]
        n_gated = int(((df_v["clustered_eff"] == 1) & (df_v["keep"] == 1)).sum())
        print(f"  {video}: v_w = {v_w_cms:.2f} cm/s = {v_w_pxframe:.3f} px/frame, "
              f"gated clustered particle-frames = {n_gated}")
        if n_gated == 0:
            continue

        res = track_clusters_for_video(df_v, v_w_pxframe)
        for r in res:
            r["video"] = video
        all_results.extend(res)

    df_out = pd.DataFrame(all_results)
    print(f"\nTotal propagating cluster instances: {len(df_out)}")

    if len(df_out):
        cols = ["video", "processed_frame_index", "cluster_id", "cluster_size",
                "centroid_x", "centroid_y",
                "v_cluster_pxframe", "v_wave_pxframe", "v_ratio",
                "n_track_points", "track_duration_frames"]
        df_out = df_out[cols]

        print("\n--- Diagnostics ---")
        print(f"v_ratio mean: {df_out['v_ratio'].mean():.3f}")
        print(f"v_ratio std:  {df_out['v_ratio'].std():.3f}")
        print("cluster_size distribution:")
        print(df_out["cluster_size"].describe())

        df_out.to_sql(OUT_TABLE, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_cp_keys "
            f"ON {OUT_TABLE}(video, processed_frame_index, cluster_id)"
        )
        conn.commit()
        print(f"Saved {OUT_TABLE} with index.")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
