"""
chirality.py - per-particle chirality (omega) in CLUSTERED vs UNCLUSTERED segments.

Hypothesis
----------
A free particle has a persistent intrinsic angular velocity omega_0 (chirality).
If clustering straightens trajectories, the effective angular velocity inside a
cluster (omega_clu) should be suppressed relative to the same particle's
free-state omega_unc.

Pipeline integration
--------------------
This is the chirality leg of the shared pipeline. Like crosser.py it sources the
cluster state from the precomputed CLUSTER_PERSISTENCE table
(build_persistent_episodes.py) instead of recomputing dwell logic inline, and it
uses the same accumulation-zone gating ('keep'). All tunables live in
parameters.json.

Method
------
For each (video, fixed_id), within contiguous keep==1 runs, cut non-overlapping
SEGMENT_FRAMES windows. Label each window by its mean persistent-cluster flag:
    mean(clustered_eff) > CLUSTERED_PURITY    -> CLUSTERED
    mean(clustered_eff) < UNCLUSTERED_PURITY  -> UNCLUSTERED
    otherwise                                 -> MIXED (dropped)
Drop immobile (path < IMMOBILE_PATH_CM) and NaN-heading segments. Fit
    theta(t) = omega * t + theta_0
from the unwrapped heading 'fixed_arrow' (stored in DEGREES, 0-360). Aggregate to
one omega per (particle, label), then pair particles having >= MIN_SEGS_PER_CLASS
segments in BOTH classes.

NB. calc_velocity / build_frames are byte-identical to crosser.py. If you want a
single source of truth, lift them into a shared pipeline_common.py and import in
both; kept inline here so each script runs standalone.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

pd.set_option('future.no_silent_downcasting', True)


# ---------------------------------------------------------------------------
# Parameters (shared with build_persistent_episodes.py and crosser.py)
# ---------------------------------------------------------------------------
def load_params(path="parameters.json"):
    with open(path) as f:
        return json.load(f)


P = load_params()

DB_PATH                = P["database"]["path"]
FPS                    = P["FPS"]
PX_TO_CM               = P["PX_TO_CM"]
VELOCITY_SMOOTH_WINDOW = P["VELOCITY_SMOOTH_WINDOW"]

Y_BULK_LIMIT           = P["accumulation_gate"]["Y_BULK_LIMIT"]
Y_CLUSTER_CONTACT      = P["accumulation_gate"]["Y_CLUSTER_CONTACT"]

_C = P["chirality"]
SEGMENT_SECONDS           = _C["SEGMENT_SECONDS"]
SEGMENT_FRAMES            = int(round(SEGMENT_SECONDS * FPS))
CLUSTERED_PURITY          = _C["CLUSTERED_PURITY"]
UNCLUSTERED_PURITY        = _C["UNCLUSTERED_PURITY"]
IMMOBILE_PATH_CM          = _C["IMMOBILE_PATH_CM"]
SUBTRACT_CLUSTER_ROTATION = _C["SUBTRACT_CLUSTER_ROTATION"]
MIN_SEGS_PER_CLASS        = _C["MIN_SEGS_PER_CLASS"]
OUT_PREFIX                = _C["OUT_PREFIX"]

_R = _C["region"]
VEL_MIN, VEL_MAX     = _R["VEL_MIN"], _R["VEL_MAX"]
WIDTH_MIN, WIDTH_MAX = _R["WIDTH_MIN"], _R["WIDTH_MAX"]
INT_VALUES           = _R["INT_VALUES"]

# MIN_CLUSTER_SIZE only needed for the optional rigid-rotation field.
MIN_CLUSTER_SIZE = P["clustering"]["MIN_CLUSTER_SIZE"]

OUTPUT_DIR = P["output"]["DIR"]


# ---------------------------------------------------------------------------
# Velocity  (identical to crosser.py)
# ---------------------------------------------------------------------------
def calc_velocity(df):
    df = df.sort_values(['video', 'fixed_id', 'processed_frame_index']).copy()
    grp = df.groupby(['video', 'fixed_id'])
    df['diffy'] = grp['bounding_box_mid_y'].diff().fillna(0) * PX_TO_CM
    df['difft'] = grp['processed_frame_index'].diff().fillna(1) / FPS
    df['vely_raw'] = df['diffy'] / df['difft']
    df['vely'] = grp['vely_raw'].transform(
        lambda s: s.rolling(VELOCITY_SMOOTH_WINDOW, min_periods=1, center=True).mean()
    )
    return df


# ---------------------------------------------------------------------------
# Per-frame table: velocity + persistent cluster flag + accumulation gating
# (identical to crosser.py; clustered_eff read from CLUSTER_PERSISTENCE)
# ---------------------------------------------------------------------------
def build_frames(df_v, df_nodes_v, df_persist_v):
    df = calc_velocity(df_v)

    # Particle identity is (video, fixed_id); cluster_id is likewise video-scoped.
    # Joins key on video so correctness does not depend on single-video slicing.
    df = df.merge(
        df_nodes_v[['video', 'processed_frame_index', 'fixed_id', 'cluster_id']],
        on=['video', 'processed_frame_index', 'fixed_id'], how='left'
    )
    df = df.merge(
        df_persist_v[['video', 'processed_frame_index', 'fixed_id', 'clustered_eff']],
        on=['video', 'processed_frame_index', 'fixed_id'], how='left'
    )
    df['clustered_eff'] = df['clustered_eff'].fillna(0).astype(np.int8)

    cond_self = df['bounding_box_mid_y'] >= Y_BULK_LIMIT

    clustered = df[df['clustered_eff'] == 1]
    if not clustered.empty:
        cluster_max_y = (
            clustered.groupby(['video', 'processed_frame_index', 'cluster_id'])['bounding_box_mid_y']
            .max().rename('cluster_max_y').reset_index()
        )
        df = df.merge(cluster_max_y,
                      on=['video', 'processed_frame_index', 'cluster_id'], how='left')
    else:
        df['cluster_max_y'] = np.nan

    cond_contact = (df['clustered_eff'] == 1) & (df['cluster_max_y'] >= Y_CLUSTER_CONTACT)
    df['keep'] = (~(cond_self | cond_contact)).astype(int)
    df = df.drop(columns=['cluster_max_y'])
    return df


# ---------------------------------------------------------------------------
# Chirality fit
# ---------------------------------------------------------------------------
def fit_omega(theta_unwrapped_rad, t_seconds):
    """Linear fit theta(t) = omega*t + theta_0, omega in rad/s.
    Returns (omega, residual_std)."""
    if theta_unwrapped_rad.size < 3:
        return np.nan, np.nan
    coeffs = np.polyfit(t_seconds, theta_unwrapped_rad, 1)
    omega = coeffs[0]
    fit = np.polyval(coeffs, t_seconds)
    resid = theta_unwrapped_rad - fit
    return float(omega), float(resid.std(ddof=1)) if resid.size > 1 else np.nan


# ---------------------------------------------------------------------------
# Segment extraction (per-segment omega from fixed_arrow IN DEGREES)
# ---------------------------------------------------------------------------
def extract_segments(df_frames, video_id, df_cluster_omega=None):
    """
    Non-overlapping SEGMENT_FRAMES windows per fixed_id, within contiguous
    keep==1 runs. Cluster purity is judged on the persistent flag clustered_eff.
    """
    if df_frames.empty:
        return pd.DataFrame()

    # Single-video contract: segmentation walks each track's frame sequence, and
    # (video, fixed_id) is the particle identity. Called once per video.
    assert df_frames['video'].nunique() <= 1, "extract_segments expects a single video"

    rows = []
    df_frames = df_frames.sort_values(['fixed_id', 'processed_frame_index'])

    for fid, g in df_frames.groupby('fixed_id', sort=False):
        g = g.reset_index(drop=True)
        if len(g) < SEGMENT_FRAMES:
            continue

        frames    = g['processed_frame_index'].to_numpy()
        keep      = g['keep'].to_numpy()
        clu       = g['clustered_eff'].to_numpy()          # persistent flag
        cl_id     = g['cluster_id'].to_numpy()
        x         = g['bounding_box_mid_x'].to_numpy() * PX_TO_CM
        y         = g['bounding_box_mid_y'].to_numpy() * PX_TO_CM
        theta_deg = g['fixed_arrow'].to_numpy(dtype=float)

        # sanity: fixed_arrow must be in degrees, not radians
        finite = theta_deg[np.isfinite(theta_deg)]
        if finite.size and np.nanmax(np.abs(finite)) <= 6.5:
            raise ValueError(
                f"fixed_arrow for video={video_id} fid={fid} appears to be in "
                f"radians (max |val| = {np.nanmax(np.abs(finite)):.3f}). "
                "This script expects degrees in [0, 360]."
            )

        # contiguous-frame runs with keep==1 on both endpoints
        contiguous_kept = (np.diff(frames) == 1) & (keep[:-1] == 1) & (keep[1:] == 1)
        run_starts = [0]
        for i, ok in enumerate(contiguous_kept):
            if not ok:
                run_starts.append(i + 1)
        run_starts.append(len(g))
        runs = [(run_starts[k], run_starts[k + 1]) for k in range(len(run_starts) - 1)]

        for r0, r1 in runs:
            run_len = r1 - r0
            if run_len < SEGMENT_FRAMES:
                continue
            if not (keep[r0:r1] == 1).all():
                continue

            n_segs = run_len // SEGMENT_FRAMES
            for s in range(n_segs):
                a = r0 + s * SEGMENT_FRAMES
                b = a + SEGMENT_FRAMES                       # exclusive
                cf = clu[a:b].mean()
                if cf > CLUSTERED_PURITY:
                    label = 'CLUSTERED'
                elif cf < UNCLUSTERED_PURITY:
                    label = 'UNCLUSTERED'
                else:
                    continue                                 # MIXED -> drop

                # immobility filter
                xs = x[a:b]; ys = y[a:b]
                path_len = np.hypot(np.diff(xs), np.diff(ys)).sum()
                if path_len < IMMOBILE_PATH_CM:
                    continue

                # heading: drop segments with any NaN orientation
                th_deg_seg = theta_deg[a:b]
                if np.isnan(th_deg_seg).any():
                    continue
                th_rad = np.deg2rad(th_deg_seg)
                dtheta = np.diff(th_rad)
                dtheta = (dtheta + np.pi / 2) % np.pi - np.pi / 2   # nematic wrap
                th_unw = np.concatenate([[th_rad[0]], th_rad[0] + np.cumsum(dtheta)])
                t_s = (frames[a:b] - frames[a]) / FPS
                omega_raw, resid = fit_omega(th_unw, t_s)

                # optional rigid cluster rotation subtraction
                omega_corr = omega_raw
                if SUBTRACT_CLUSTER_ROTATION and label == 'CLUSTERED' \
                        and df_cluster_omega is not None:
                    seg_cl_ids = cl_id[a:b]
                    vals, cnts = np.unique(
                        seg_cl_ids[~pd.isna(seg_cl_ids)], return_counts=True
                    )
                    if vals.size:
                        dom_cid = vals[np.argmax(cnts)]
                        sub = df_cluster_omega[
                            (df_cluster_omega['video'] == video_id) &
                            (df_cluster_omega['cluster_id'] == dom_cid) &
                            (df_cluster_omega['processed_frame_index'] >= frames[a]) &
                            (df_cluster_omega['processed_frame_index'] <= frames[b - 1])
                        ]
                        if not sub.empty:
                            omega_corr = omega_raw - float(sub['cluster_omega'].mean())

                rows.append({
                    'video':            video_id,
                    'fixed_id':         fid,
                    'seg_start_frame':  int(frames[a]),
                    'seg_end_frame':    int(frames[b - 1]),
                    'cluster_fraction': float(cf),
                    'label':            label,
                    'path_length_cm':   float(path_len),
                    'omega_raw':        float(omega_raw),
                    'omega':            float(omega_corr),
                    'theta_resid_std':  float(resid) if np.isfinite(resid) else np.nan,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optional: per-frame rigid cluster rotation field
# ---------------------------------------------------------------------------
def build_cluster_omega(df_video, df_nodes_v):
    tmp = df_video.merge(
        df_nodes_v[['video', 'processed_frame_index', 'fixed_id',
                    'cluster_id', 'cluster_size']],
        on=['video', 'processed_frame_index', 'fixed_id'], how='inner'
    )
    tmp = tmp[tmp['cluster_size'] >= MIN_CLUSTER_SIZE].sort_values(
        ['video', 'fixed_id', 'processed_frame_index']
    )
    tmp['fixed_arrow_rad'] = np.deg2rad(tmp['fixed_arrow'])
    tmp['dtheta'] = tmp.groupby(['video', 'fixed_id'])['fixed_arrow_rad'].diff()
    tmp['dtheta'] = (tmp['dtheta'] + np.pi) % (2 * np.pi) - np.pi
    tmp['dt'] = tmp.groupby(['video', 'fixed_id'])['processed_frame_index'].diff() / FPS
    tmp['omega_inst'] = tmp['dtheta'] / tmp['dt']
    return (
        tmp.dropna(subset=['omega_inst'])
           .groupby(['video', 'processed_frame_index', 'cluster_id'])['omega_inst']
           .mean().rename('cluster_omega').reset_index()
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    conn = sqlite3.connect(DB_PATH)

    vid_id_df = pd.read_sql("SELECT * FROM CONFIG", con=conn)
    n_all = len(vid_id_df)
    vid_id_df = vid_id_df[
        (vid_id_df['velocity'] >= VEL_MIN)   & (vid_id_df['velocity'] <= VEL_MAX) &
        (vid_id_df['width']    >= WIDTH_MIN) & (vid_id_df['width']    <= WIDTH_MAX) &
        (vid_id_df['INT'].isin(INT_VALUES))
    ].copy()
    print(f"Region: velocity in [{VEL_MIN},{VEL_MAX}], width in [{WIDTH_MIN},{WIDTH_MAX}], "
          f"INT in {INT_VALUES}")
    print(f"  CONFIG rows kept: {len(vid_id_df)}/{n_all}")
    if vid_id_df.empty:
        conn.close()
        return

    region_videos = vid_id_df['unique_name'].tolist()
    ph_all = ','.join(['?'] * len(region_videos))

    print("Loading cluster node data...")
    df_nodes = pd.read_sql(
        f"SELECT video, processed_frame_index, fixed_id, cluster_id, cluster_size "
        f"FROM data_graphs_nodes WHERE video IN ({ph_all})",
        con=conn, params=region_videos
    )

    print("Loading persistent clustering...")
    df_persist = pd.read_sql(
        f"SELECT video, processed_frame_index, fixed_id, clustered_eff "
        f"FROM CLUSTER_PERSISTENCE WHERE video IN ({ph_all})",
        con=conn, params=region_videos
    )

    seg_parts = []
    groups = vid_id_df.groupby(['velocity', 'width'])
    progress = tqdm.tqdm(groups, desc="Segmenting (chirality)", total=len(groups))

    for conf, group in progress:
        velocity, width = conf
        vidids = group['unique_name'].tolist()
        if not vidids:
            continue
        ph = ','.join(['?'] * len(vidids))
        try:
            df_video = pd.read_sql(
                f"SELECT fixed_id, video, bounding_box_mid_x, bounding_box_mid_y, "
                f"fixed_arrow, processed_frame_index FROM data_object "
                f"WHERE video IN ({ph})",
                conn, params=vidids
            )
        except Exception as e:
            print(f"Error querying data_object for {conf}: {e}")
            continue
        if df_video.empty:
            continue

        df_video.dropna(
            subset=['fixed_id', 'bounding_box_mid_x', 'bounding_box_mid_y'], inplace=True
        )
        if df_video.empty:
            continue

        df_nodes_v   = df_nodes[df_nodes['video'].isin(vidids)].copy()
        df_persist_v = df_persist[df_persist['video'].isin(vidids)].copy()

        df_cluster_omega = None
        if SUBTRACT_CLUSTER_ROTATION:
            df_cluster_omega = build_cluster_omega(df_video, df_nodes_v)

        for vid in vidids:
            df_vv = df_video[df_video['video'] == vid].copy()
            df_nn = df_nodes_v[df_nodes_v['video'] == vid].copy()
            df_pp = df_persist_v[df_persist_v['video'] == vid].copy()
            if df_vv.empty:
                continue
            try:
                frames = build_frames(df_vv, df_nn, df_pp)
                seg_df = extract_segments(frames, vid, df_cluster_omega=df_cluster_omega)
                if seg_df.empty:
                    continue
                seg_df['velocity'] = velocity
                seg_df['width']    = width
                seg_parts.append(seg_df)
            except Exception as e:
                print(f"Error processing video {vid}: {e}")
                continue

    conn.close()

    if not seg_parts:
        print("No segments produced.")
        return

    df_segs = pd.concat(seg_parts, ignore_index=True)
    df_segs.to_csv(OUTPUT_DIR + OUT_PREFIX + 'segments.csv', index=False)
    print(f"\nTotal segments (mobile, pure): {len(df_segs)}")
    print(df_segs['label'].value_counts().to_string())
    print(f"omega range [rad/s]: min={df_segs['omega'].min():+.3f}, "
          f"max={df_segs['omega'].max():+.3f}, median={df_segs['omega'].median():+.3f}")

    # --- per-particle aggregation ---
    per_p = (
        df_segs.groupby(['video', 'fixed_id', 'label'])
        .agg(n_segs=('omega', 'size'),
             omega_mean=('omega', 'mean'),
             omega_median=('omega', 'median'),
             omega_std=('omega', 'std'))
        .reset_index()
    )

    pivot_n   = per_p.pivot_table(index=['video', 'fixed_id'], columns='label',
                                  values='n_segs', fill_value=0)
    pivot_med = per_p.pivot_table(index=['video', 'fixed_id'], columns='label',
                                  values='omega_median')
    pivot_mn  = per_p.pivot_table(index=['video', 'fixed_id'], columns='label',
                                  values='omega_mean')

    for col in ['CLUSTERED', 'UNCLUSTERED']:
        for piv in (pivot_n, pivot_med, pivot_mn):
            if col not in piv.columns:
                piv[col] = np.nan if piv is not pivot_n else 0

    eligible = (
        (pivot_n['CLUSTERED']   >= MIN_SEGS_PER_CLASS) &
        (pivot_n['UNCLUSTERED'] >= MIN_SEGS_PER_CLASS)
    )
    paired = pd.DataFrame({
        'omega_clu_median': pivot_med['CLUSTERED'][eligible],
        'omega_unc_median': pivot_med['UNCLUSTERED'][eligible],
        'omega_clu_mean':   pivot_mn ['CLUSTERED'][eligible],
        'omega_unc_mean':   pivot_mn ['UNCLUSTERED'][eligible],
        'n_clu':            pivot_n  ['CLUSTERED'][eligible],
        'n_unc':            pivot_n  ['UNCLUSTERED'][eligible],
    }).reset_index()

    paired.to_csv(OUTPUT_DIR + OUT_PREFIX + 'paired.csv', index=False)
    print(f"\nParticles with >= {MIN_SEGS_PER_CLASS} segments in BOTH classes: {len(paired)}")
    if paired.empty:
        print("No paired particles. Lower MIN_SEGS_PER_CLASS/SEGMENT_SECONDS or widen region.")
        return

    x = paired['omega_unc_median'].to_numpy()
    y = paired['omega_clu_median'].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 1:
        print("No finite paired points to plot.")
        return

    # --- joint distribution plot ---
    sns.set_theme(style='whitegrid')
    g = sns.JointGrid(x=x, y=y, height=7, ratio=4)
    g.plot_joint(sns.scatterplot, s=25, alpha=0.6, color='#2c3e50',
                 edgecolor='white', linewidth=0.5)
    g.plot_marginals(sns.histplot, bins=40, color='#7f8c8d', fill=True,
                     alpha=0.5, element="step")

    lim = float(np.nanmax(np.abs(np.concatenate([x, y])))) * 1.1 if x.size else 1.0
    g.ax_joint.plot([-lim, lim], [-lim, lim], 'k--', lw=1.0, alpha=0.6, label='y = x')
    g.ax_joint.axhline(0, color='k', lw=0.6, alpha=0.4)
    g.ax_joint.axvline(0, color='k', lw=0.6, alpha=0.4)
    g.ax_joint.set_xlim(-lim, lim)
    g.ax_joint.set_ylim(-lim, lim)
    g.ax_joint.set_aspect('equal', adjustable='box')
    g.ax_joint.set_xlabel(r'$\omega_{\mathrm{unc}}$ (per-particle median)  [rad/s]')
    g.ax_joint.set_ylabel(r'$\omega_{\mathrm{clu}}$ (per-particle median)  [rad/s]')
    g.ax_joint.legend(frameon=False, loc='upper left')
    g.figure.suptitle(
        f'Paired chirality | region v in [{VEL_MIN},{VEL_MAX}], w in [{WIDTH_MIN},{WIDTH_MAX}]\n'
        f'cluster rotation subtracted: {SUBTRACT_CLUSTER_ROTATION} (n = {x.size} particles)',
        fontsize=11, y=1.03
    )
    plt.savefig(OUTPUT_DIR + OUT_PREFIX + 'paired.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUT_PREFIX}paired.png")
    print("Done.")


if __name__ == '__main__':
    main()
