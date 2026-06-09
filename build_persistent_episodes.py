"""
Persistent cluster episode detection.

Per (video, fixed_id) trajectory, segments the clustered-frame time series into
episodes (gap-tolerant runs), judges each by span and fill ratio, and writes a
per-frame persistence flag back to the database.

Output table CLUSTER_PERSISTENCE, one row per observed (video, frame, fixed_id):
    is_clustered_frame  cluster_size >= MIN_CLUSTER_SIZE at this frame
    episode_id          per-particle episode index, -1 if in none
    episode_len_frames  span of this frame's episode (0 if none)
    fill_ratio          clustered fraction of that episode's span (NaN if none)
    is_persistent       1 if the frame's episode qualifies (span & fill), else 0
    clustered_eff       is_clustered_frame AND is_persistent

A per-episode diagnostic table CLUSTER_EPISODES is also written.
"""

import json
import sqlite3
import numpy as np
import pandas as pd


def load_params(path="parameters.json"):
    with open(path) as f:
        p = json.load(f)
    c = p["clustering"]
    return (
        p["database"]["path"],
        c["MIN_CLUSTER_SIZE"],
        c["MAX_GAP_FRAMES"],
        c["MIN_CLUSTER_RUN_FRAMES"],
        c["FILL_RATIO_MIN"],
        p["FPS"],
    )


def segment_trajectory(frames, clustered, max_gap):
    """
    frames:    int array of observed frame indices, ascending, unique.
    clustered: 0/1 array aligned to frames (cluster_size >= MIN at that frame).
    """
    n = frames.size
    ep_id = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return ep_id

    clustered_idx = np.flatnonzero(clustered == 1)
    if clustered_idx.size == 0:
        return ep_id

    current_ep = 0
    last_clustered_pos = clustered_idx[0]
    ep_first_pos = clustered_idx[0]
    ep_id[clustered_idx[0]] = current_ep

    for pos in clustered_idx[1:]:
        gap = frames[pos] - frames[last_clustered_pos] - 1 
        if gap <= max_gap:
            ep_id[ep_first_pos:pos + 1] = current_ep
            last_clustered_pos = pos
        else:
            current_ep += 1
            ep_first_pos = pos
            last_clustered_pos = pos
            ep_id[pos] = current_ep

    return ep_id


def detect_for_video(df_v, min_size, max_gap, min_run, fill_min):
    """
    df_v: rows for one video, columns processed_frame_index, fixed_id,
          cluster_size (NaN where unclustered/missing-after-reindex).
    Returns (frame_df, episode_df) for this video.
    """
    frame_parts = []
    episode_rows = []

    for fid, g in df_v.groupby("fixed_id", sort=False):
        g = g.sort_values("processed_frame_index")
        f_obs = g["processed_frame_index"].to_numpy()
        size_obs = g["cluster_size"].to_numpy()

        f0, f1 = int(f_obs.min()), int(f_obs.max()) #span the episode
        grid = np.arange(f0, f1 + 1)
        size_grid = np.full(grid.size, np.nan)
        size_grid[f_obs - f0] = size_obs

        clustered = (np.nan_to_num(size_grid, nan=0.0) >= min_size).astype(np.int8)
        ep_id = segment_trajectory(grid, clustered, max_gap)

        # Per-episode span / fill / verdict
        is_persistent = np.zeros(grid.size, dtype=np.int8) # we want to filter out 1 0 0 1 0 1 0 0 1 etc... candidates.
        ep_len = np.zeros(grid.size, dtype=np.int64)
        fill = np.full(grid.size, np.nan)

        for e in np.unique(ep_id[ep_id >= 0]):
            mask = ep_id == e
            ef = grid[mask]
            span = int(ef.max() - ef.min() + 1)
            n_clu = int(clustered[mask].sum())
            fr = n_clu / span if span > 0 else 0.0
            ok = (span >= min_run) and (fr >= fill_min)

            ep_len[mask] = span
            fill[mask] = fr
            if ok:
                is_persistent[mask] = 1

            sizes_here = size_grid[mask]
            sizes_here = sizes_here[np.isfinite(sizes_here)]
            episode_rows.append({
                "fixed_id": fid,
                "episode_id": int(e),
                "frame_first": int(ef.min()),
                "frame_last": int(ef.max()),
                "len_frames": span,
                "n_clustered": n_clu,
                "fill_ratio": fr,
                "max_size": float(sizes_here.max()) if sizes_here.size else np.nan,
                "mean_size": float(sizes_here.mean()) if sizes_here.size else np.nan,
                "persistent": int(ok),
            })

        # Keep only frames that were actually observed (drop synthetic grid rows)
        keep_obs = np.isin(grid, f_obs)
        frame_parts.append(pd.DataFrame({
            "fixed_id": fid,
            "processed_frame_index": grid[keep_obs],
            "is_clustered_frame": clustered[keep_obs],
            "episode_id": ep_id[keep_obs],
            "episode_len_frames": ep_len[keep_obs],
            "fill_ratio": fill[keep_obs],
            "is_persistent": is_persistent[keep_obs],
        }))

    frame_df = (pd.concat(frame_parts, ignore_index=True)
                if frame_parts else pd.DataFrame())
    episode_df = (pd.DataFrame(episode_rows)
                  if episode_rows else pd.DataFrame())
    return frame_df, episode_df


def main():
    (db_path, min_size, max_gap, min_run,
     fill_min, fps) = load_params()

    print(f"Connecting to {db_path}")
    conn = sqlite3.connect(db_path)

    print("Loading observed (video, frame, fixed_id) skeleton from data_object...")
    df_obj = pd.read_sql(
        "SELECT video, processed_frame_index, fixed_id FROM data_object",
        conn,
    )
    df_obj = df_obj.dropna(subset=["fixed_id"])

    print("Loading cluster sizes from data_graphs_nodes...")
    df_nodes = pd.read_sql(
        "SELECT video, processed_frame_index, fixed_id, cluster_size "
        "FROM data_graphs_nodes",
        conn,
    )

    df = df_obj.merge(
        df_nodes,
        on=["video", "processed_frame_index", "fixed_id"],
        how="left",
    )

    all_frames, all_eps = [], []
    videos = df["video"].unique()
    for i, vid in enumerate(videos, 1):
        df_v = df[df["video"] == vid]
        frame_df, episode_df = detect_for_video(
            df_v, min_size, max_gap, min_run, fill_min
        )
        if not frame_df.empty:
            frame_df.insert(0, "video", vid)
            all_frames.append(frame_df)
        if not episode_df.empty:
            episode_df.insert(0, "video", vid)
            all_eps.append(episode_df)
        print(f"  ({i}/{len(videos)}) {vid}: "
              f"{0 if frame_df.empty else int(frame_df['is_persistent'].sum())} "
              f"persistent frames")

    frames_out = pd.concat(all_frames, ignore_index=True)
    frames_out["clustered_eff"] = (
        frames_out["is_clustered_frame"] & frames_out["is_persistent"]
    ).astype(np.int8)

    eps_out = pd.concat(all_eps, ignore_index=True) if all_eps else pd.DataFrame()
    eps_out["len_sec"] = eps_out["len_frames"] / fps if not eps_out.empty else None

    print(f"\nWriting CLUSTER_PERSISTENCE ({len(frames_out)} rows)...")
    frames_out.to_sql("CLUSTER_PERSISTENCE", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cp "
        "ON CLUSTER_PERSISTENCE(video, processed_frame_index, fixed_id)"
    )

    if not eps_out.empty:
        print(f"Writing CLUSTER_EPISODES ({len(eps_out)} rows)...")
        eps_out.to_sql("CLUSTER_EPISODES", conn, if_exists="replace", index=False)

    conn.commit()

    n = len(frames_out)
    print("\n--- Summary ---")
    print(f"observed frames:        {n}")
    print(f"is_clustered_frame:     {int(frames_out['is_clustered_frame'].sum())} "
          f"({100*frames_out['is_clustered_frame'].mean():.1f}%)")
    print(f"clustered_eff:          {int(frames_out['clustered_eff'].sum())} "
          f"({100*frames_out['clustered_eff'].mean():.1f}%)")
    if not eps_out.empty:
        print(f"episodes:               {len(eps_out)}")
        print(f"persistent episodes:    {int(eps_out['persistent'].sum())}")
        print(f"fill_ratio median:      {eps_out['fill_ratio'].median():.3f}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
