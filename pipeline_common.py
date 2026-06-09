"""
pipeline_common.py — shared utilities for the wave-driven transport pipeline.

Single source of truth for everything that more than one script needs, so the
cluster definition and the accumulation gate are written down exactly once.

The "is this particle clustered?" question is answered upstream, once, by
build_persistent_episodes.py (column CLUSTER_PERSISTENCE.clustered_eff). NOTHING
downstream re-derives it: attach_cluster_state() only *reads* that flag and adds
the accumulation gate (keep) on top of it, using the Y limits in parameters.json.

Conventions (mirroring crosser.py / build_persistent_episodes.py):
    * parameters.json is the only place tunables live.
    * video scope is a {VEL_MIN, VEL_MAX, WIDTH_MIN, WIDTH_MAX} block, with
      both bounds inclusive (velocity in [VEL_MIN, VEL_MAX]).
    * keys are (video, processed_frame_index, fixed_id) for particle-frames and
      (video, processed_frame_index, cluster_id) for cluster-frames.
"""

import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def load_params(path="parameters.json"):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Video scope selection (CONFIG.velocity / CONFIG.width window)
# ---------------------------------------------------------------------------
def select_videos(conn, region):
    """
    region: dict with VEL_MIN, VEL_MAX, WIDTH_MIN, WIDTH_MAX (all inclusive).
    Returns (df_config, list_of_unique_names) for videos inside the window.
    """
    df_config = pd.read_sql_query(
        "SELECT unique_name, width, velocity FROM CONFIG", conn
    )
    sel = df_config[
        (df_config["width"]    >= region["WIDTH_MIN"]) &
        (df_config["width"]    <= region["WIDTH_MAX"]) &
        (df_config["velocity"] >= region["VEL_MIN"])   &
        (df_config["velocity"] <= region["VEL_MAX"])
    ]
    return df_config, sel["unique_name"].unique().tolist()


def sql_in_list(names):
    """Render a python list of strings as a SQL IN (...) body. Empty -> NULL."""
    if not names:
        return "NULL"
    return "'" + "', '".join(names) + "'"


# ---------------------------------------------------------------------------
# Canonical cluster state + accumulation gate (read, never recompute)
# ---------------------------------------------------------------------------
def attach_cluster_state(df_pos, df_nodes, df_persist,
                         y_bulk_limit, y_cluster_contact):
    """
    Augment a per-particle-frame position table with the unified cluster state.

    df_pos      must have: video, processed_frame_index, fixed_id,
                           bounding_box_mid_y  (bounding_box_mid_x optional).
    df_nodes     data_graphs_nodes subset: + cluster_id, cluster_size.
    df_persist   CLUSTER_PERSISTENCE subset: + clustered_eff (the persistent
                 per-frame cluster flag; the ONLY source of "is clustered").

    Adds:
        cluster_id, cluster_size   (NaN where the particle is not a graph node)
        clustered_eff              (0/1, missing -> 0)
        keep                       (1 == not yet accumulated, 0 == accumulated)

    keep == 0 if the particle's own y >= y_bulk_limit, OR it belongs to a
    persistent cluster whose top member (max y) is >= y_cluster_contact at that
    frame (cluster-mediated accumulation). This is the identical rule used in
    crosser.build_frames; it lives here so every consumer shares it verbatim.
    """
    df = df_pos.merge(
        df_nodes[["video", "processed_frame_index", "fixed_id",
                  "cluster_id", "cluster_size"]],
        on=["video", "processed_frame_index", "fixed_id"], how="left",
    )
    df = df.merge(
        df_persist[["video", "processed_frame_index", "fixed_id",
                    "clustered_eff"]],
        on=["video", "processed_frame_index", "fixed_id"], how="left",
    )
    df["clustered_eff"] = df["clustered_eff"].fillna(0).astype(np.int8)

    cond_self = df["bounding_box_mid_y"] >= y_bulk_limit

    clustered = df[df["clustered_eff"] == 1]
    if not clustered.empty:
        cluster_max_y = (
            clustered.groupby(
                ["video", "processed_frame_index", "cluster_id"]
            )["bounding_box_mid_y"]
            .max().rename("cluster_max_y").reset_index()
        )
        df = df.merge(
            cluster_max_y,
            on=["video", "processed_frame_index", "cluster_id"], how="left",
        )
    else:
        df["cluster_max_y"] = np.nan

    cond_contact = (df["clustered_eff"] == 1) & \
                   (df["cluster_max_y"] >= y_cluster_contact)
    df["keep"] = (~(cond_self | cond_contact)).astype(int)
    return df.drop(columns=["cluster_max_y"])


def restrict_to_propagating(df, df_propagating):
    """
    Inner-join to the cluster-level CLUSTERS_PROPAGATING table on
    (video, processed_frame_index, cluster_id). Keeps only particle-frames whose
    cluster was validated as wave-propagating by build_jaccard_prop.py.
    """
    keys = df_propagating[
        ["video", "processed_frame_index", "cluster_id"]
    ].drop_duplicates()
    return df.merge(
        keys, on=["video", "processed_frame_index", "cluster_id"], how="inner"
    )


# ---------------------------------------------------------------------------
# Smoothed vertical velocity (cm/s) — identical to crosser.calc_velocity
# ---------------------------------------------------------------------------
def calc_velocity(df, fps, px_to_cm, smooth_window):
    df = df.sort_values(["video", "fixed_id", "processed_frame_index"]).copy()
    grp = df.groupby(["video", "fixed_id"])
    df["diffy"] = grp["bounding_box_mid_y"].diff().fillna(0) * px_to_cm
    df["difft"] = grp["processed_frame_index"].diff().fillna(1) / fps
    df["vely_raw"] = df["diffy"] / df["difft"]
    df["vely"] = grp["vely_raw"].transform(
        lambda s: s.rolling(smooth_window, min_periods=1, center=True).mean()
    )
    return df


# ---------------------------------------------------------------------------
# Histogram / label helpers
# ---------------------------------------------------------------------------
def density_hist(series, edges, clip_range=None):
    """Density histogram of a series over fixed edges. Returns (hist, n) or None."""
    data = series.dropna()
    if clip_range is not None:
        data = data.clip(lower=clip_range[0], upper=clip_range[1])
    if data.empty:
        return None
    h, _ = np.histogram(data, bins=edges, density=True)
    return h, len(data)


def vel_label(velocity):
    """CONFIG.velocity (ms per 2 cm) -> wave speed label (cm/s, rounded)."""
    return np.round(1000.0 / velocity * 2, 2) if velocity != 0 else 0


def width_label(width):
    """CONFIG.width -> physical channel width label."""
    return width * 2 * 2
