"""
crosser.py - arrival and bulk velocity distributions, persistent-cluster aware.

Reconstructed so that the cluster state is *read* from the precomputed
CLUSTER_PERSISTENCE table (see build_persistent_episodes.py) rather than being
recomputed here with inline run/gap logic. All tunables live in parameters.json
(shared with the persistence builder), so the cluster definition used for
arrival classification and for the velocity distributions is identical to the
one written to the database.

Outputs (written into output.DIR):

  fig3a.csv   Per-config arrival cluster fraction. Of the particles that reach
              the accumulation zone, the fraction that arrived as part of a
              persistent cluster (majority-of-displacement rule, see below).

  fig3bb.csv  ARRIVAL cluster velocity distribution. For each crosser, the
              arrival velocity is the mean smoothed v_y over the last
              ARRIVAL_WINDOW_FRAMES frames before crossing. A crosser is
              labelled "clustered" by the displacement-majority rule over the
              last SPATIAL_WINDOW_PX pixels before crossing (more signed |dy|
              accumulated while clustered_eff==1 than while ==0). The histogram
              is built from the clustered arrivals only.

  fig3b.csv   ALL-cluster (bulk) velocity distribution. Per-frame v_y for every
              NOT-YET-ACCUMULATED frame (keep==1), split into clustered vs
              unclustered by the per-frame persistent flag clustered_eff. This
              is the population velocity distribution of persistent clusters
              that have not yet reached the accumulation zone.

Accumulation is deliberately NOT a bare y >= Y_BULK_LIMIT cut. A persistent
cluster whose leading member reaches Y_CLUSTER_CONTACT drags its whole
membership into the accumulation zone: this is used both to PROPAGATE the
crossing event to all cluster members (fig3a/fig3bb) and to EXCLUDE those
frames from the not-yet-accumulated bulk (fig3b). This is how the link between
clustering and accumulation enters, beyond the single-particle 512 line.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
import tqdm

pd.set_option('future.no_silent_downcasting', True)


# ---------------------------------------------------------------------------
# Parameters (single source of truth, shared with build_persistent_episodes.py)
# ---------------------------------------------------------------------------
def load_params(path="parameters.json"):
    with open(path) as f:
        return json.load(f)


P = load_params()

DB_PATH                = P["database"]["path"]
FPS                    = P["FPS"]
PX_TO_CM               = P["PX_TO_CM"]
VELOCITY_SMOOTH_WINDOW = P["VELOCITY_SMOOTH_WINDOW"]
Y_MIN                  = P["Y_MIN"]

# Cluster definition is enforced upstream (CLUSTER_PERSISTENCE.clustered_eff);
# MIN_CLUSTER_SIZE is kept here only for provenance / sanity printing.
MIN_CLUSTER_SIZE       = P["clustering"]["MIN_CLUSTER_SIZE"]

Y_BULK_LIMIT           = P["accumulation_gate"]["Y_BULK_LIMIT"]
Y_CLUSTER_CONTACT      = P["accumulation_gate"]["Y_CLUSTER_CONTACT"]

SPATIAL_WINDOW_PX      = P["arrival"]["SPATIAL_WINDOW_PX"]
ARRIVAL_WINDOW_FRAMES  = P["arrival"]["ARRIVAL_WINDOW_FRAMES"]

_H         = P["histogram"]
HIST_RANGE = (_H["RANGE_MIN"], _H["RANGE_MAX"])
ARR_EDGES  = np.linspace(HIST_RANGE[0], HIST_RANGE[1], _H["N_BINS_ARRIVAL"] + 1)
ARR_CENT   = 0.5 * (ARR_EDGES[1:] + ARR_EDGES[:-1])
BULK_EDGES = np.linspace(HIST_RANGE[0], HIST_RANGE[1], _H["N_BINS_BULK"] + 1)
BULK_CENT  = 0.5 * (BULK_EDGES[1:] + BULK_EDGES[:-1])

OUTPUT_DIR = P["output"]["DIR"]


# ---------------------------------------------------------------------------
# Label helpers (config code -> physical units), unchanged from originals
# ---------------------------------------------------------------------------
def _vel_label(velocity):
    return np.round(1000 / velocity * 2, 2) if velocity != 0 else 0


def _width_label(width):
    return width * 2 * 2


def _hist(series, edges):
    data = series.dropna().clip(lower=HIST_RANGE[0], upper=HIST_RANGE[1])
    if data.empty:
        return None
    h, _ = np.histogram(data, bins=edges, density=True)
    return h, len(data)


# ---------------------------------------------------------------------------
# Smoothed vertical velocity (cm/s)
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
# ---------------------------------------------------------------------------
def build_frames(df_v, df_nodes_v, df_persist_v):
    """
    Returns a per-frame DataFrame with:
        fixed_id, processed_frame_index, bounding_box_mid_x/y, vely,
        cluster_id      (from data_graphs_nodes; needed for contact propagation)
        clustered_eff   (persistent cluster flag from CLUSTER_PERSISTENCE)
        keep            (1 == not yet accumulated, 0 == in accumulation zone)

    keep == 0 if:
        - the particle's own y >= Y_BULK_LIMIT, OR
        - it is in a persistent cluster whose max member y >= Y_CLUSTER_CONTACT
          at that frame (cluster-mediated accumulation).
    """
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
# Crossings -> arrival classification (fig3a, fig3bb)
# ---------------------------------------------------------------------------
def find_crossings(df):
    """
    Detect arrivals into the accumulation zone and classify each by the
    displacement-majority rule over the last SPATIAL_WINDOW_PX pixels.

    Returns (arrivals, crossed_df):
        arrivals   crossers labelled clustered (isclustered==1) with a valid
                   arrival velocity; carries 'vely_arrival'.
        crossed_df all processed crossers (denominator for the cluster fraction).
    """
    df = df.copy()
    # Single-video contract: frame_prev and the per-track diff below assume one
    # video. The pipeline calls this once per video; guard against misuse.
    assert df['video'].nunique() <= 1, "find_crossings expects a single video"
    all_frames = np.sort(df['processed_frame_index'].unique())
    if len(all_frames) < 2:
        return pd.DataFrame(), pd.DataFrame()
    frame_prev = pd.Series(all_frames[:-1], index=all_frames[1:])

    df = df.sort_values(['video', 'fixed_id', 'processed_frame_index'])

    # --- standard single-particle crossing of the bulk limit ---
    df['above_threshold'] = (df['bounding_box_mid_y'] > Y_BULK_LIMIT).astype(int)
    df['crossed_standard'] = df.groupby(['video', 'fixed_id'])['above_threshold'].diff().fillna(0) > 0

    # --- cluster-mediated propagation ---
    # a persistent cluster whose leading member reaches the contact line marks
    # all its members at the previous frame as having crossed.
    eff = df[df['clustered_eff'] == 1]
    triggered = eff.loc[
        eff['bounding_box_mid_y'] >= Y_CLUSTER_CONTACT,
        ['video', 'processed_frame_index', 'cluster_id']
    ].dropna().drop_duplicates()

    df['crossed_propagated'] = False
    if not triggered.empty:
        members = eff.merge(
            triggered, on=['video', 'processed_frame_index', 'cluster_id'], how='inner'
        )[['video', 'processed_frame_index', 'fixed_id']].drop_duplicates()
        members['prev_frame'] = members['processed_frame_index'].map(frame_prev)
        members = members.dropna(subset=['prev_frame'])
        members['prev_frame'] = members['prev_frame'].astype(int)
        key = members[['video', 'prev_frame', 'fixed_id']].rename(
            columns={'prev_frame': 'processed_frame_index'}
        )
        key['crossed_propagated'] = True
        key = key.drop_duplicates()
        df = df.drop(columns=['crossed_propagated']).merge(
            key, on=['video', 'processed_frame_index', 'fixed_id'], how='left'
        )
        df['crossed_propagated'] = df['crossed_propagated'].fillna(False)

    df['crossed'] = df['crossed_standard'] | df['crossed_propagated']
    crossed_df = df[df['crossed']].drop_duplicates(subset=['video', 'fixed_id'], keep='first').copy()
    if crossed_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # --- approach window: last SPATIAL_WINDOW_PX pixels before crossing ---
    cross_info = crossed_df[['fixed_id', 'processed_frame_index', 'bounding_box_mid_y']].rename(
        columns={'processed_frame_index': 'cross_frame', 'bounding_box_mid_y': 'cross_y'}
    )
    cross_info['y_entry'] = cross_info['cross_y'] - SPATIAL_WINDOW_PX
    dwc = df.merge(cross_info, on='fixed_id', how='inner')

    below_entry = dwc[
        (dwc['bounding_box_mid_y'] <= dwc['y_entry']) &
        (dwc['processed_frame_index'] <= dwc['cross_frame'])
    ]
    if below_entry.empty:
        return pd.DataFrame(), crossed_df

    entry_frames = below_entry.groupby('fixed_id')['processed_frame_index'].max().rename('entry_frame')
    crossed_df = crossed_df[crossed_df['fixed_id'].isin(entry_frames.index)].copy()
    if crossed_df.empty:
        return pd.DataFrame(), crossed_df

    dwc = dwc.merge(entry_frames, on='fixed_id', how='inner')
    in_window = dwc[
        (dwc['processed_frame_index'] >= dwc['entry_frame']) &
        (dwc['processed_frame_index'] <= dwc['cross_frame'])
    ].sort_values(['fixed_id', 'processed_frame_index']).copy()
    if in_window.empty:
        return pd.DataFrame(), crossed_df

    # --- displacement-majority label over the last 50 px ---
    in_window['dy_cm'] = in_window.groupby('fixed_id')['bounding_box_mid_y'].diff().abs().fillna(0) * PX_TO_CM
    in_window['dy_clustered']   = in_window['dy_cm'] * in_window['clustered_eff']
    in_window['dy_unclustered'] = in_window['dy_cm'] * (1 - in_window['clustered_eff'])

    agg = in_window.groupby('fixed_id').agg(
        dy_clustered_cm=('dy_clustered', 'sum'),
        dy_unclustered_cm=('dy_unclustered', 'sum'),
        n_clustered_frames=('clustered_eff', 'sum'),
        n_window_frames=('clustered_eff', 'size'),
    )

    # --- arrival velocity: mean v_y over last ARRIVAL_WINDOW_FRAMES ---
    in_window['frames_to_cross'] = in_window['cross_frame'] - in_window['processed_frame_index']
    arrival_slice = in_window[in_window['frames_to_cross'] < ARRIVAL_WINDOW_FRAMES]
    vely_arrival = arrival_slice.groupby('fixed_id')['vely'].mean().rename('vely_arrival')
    agg = agg.join(vely_arrival, how='left')

    agg['n_unclustered_frames'] = agg['n_window_frames'] - agg['n_clustered_frames']
    agg['isclustered'] = (agg['dy_clustered_cm'] > agg['dy_unclustered_cm']).astype(int)

    crossed_df = crossed_df.merge(agg, on='fixed_id', how='left')

    arrivals = crossed_df[
        (crossed_df['isclustered'] == 1) & crossed_df['vely_arrival'].notna()
    ].copy()
    return arrivals, crossed_df


# ---------------------------------------------------------------------------
# Not-yet-accumulated bulk per-frame velocity samples (fig3b)
# ---------------------------------------------------------------------------
def extract_bulk_frames(df):
    """Per-frame v_y for kept (not yet accumulated) frames, labelled by the
    persistent per-frame cluster flag. No segment averaging."""
    d = df[df['keep'] == 1]
    if d.empty:
        return pd.DataFrame()
    return d[['fixed_id', 'processed_frame_index', 'bounding_box_mid_y',
              'vely', 'clustered_eff']].copy()


# ---------------------------------------------------------------------------
# Load tables
# ---------------------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)

vid_id_df = pd.read_sql("SELECT * FROM CONFIG", con=conn)

df_nodes = pd.read_sql(
    "SELECT video, processed_frame_index, fixed_id, cluster_id, cluster_size "
    "FROM data_graphs_nodes", con=conn
)

# persistent clustering, precomputed by build_persistent_episodes.py
df_persist = pd.read_sql(
    "SELECT video, processed_frame_index, fixed_id, clustered_eff "
    "FROM CLUSTER_PERSISTENCE", con=conn
)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
arrival_parts  = []
bulk_parts     = []
fig3a_rows     = []

groups = vid_id_df.groupby(['velocity', 'width'])
progress = tqdm.tqdm(groups, desc="Processing groups", total=len(groups))

for conf, group in progress:
    velocity, width = conf
    vidids = group['unique_name'].tolist()
    if not vidids:
        continue

    placeholders = ','.join(['?'] * len(vidids))
    try:
        df_video = pd.read_sql(
            f"SELECT fixed_id, video, bounding_box_mid_x, bounding_box_mid_y, "
            f"processed_frame_index FROM data_object WHERE video IN ({placeholders})",
            conn, params=vidids
        )
    except Exception:
        continue
    if df_video.empty:
        continue

    df_video.dropna(subset=['fixed_id', 'bounding_box_mid_x', 'bounding_box_mid_y'], inplace=True)
    df_video = df_video[df_video['bounding_box_mid_y'] >= Y_MIN].copy()
    if df_video.empty:
        continue

    df_nodes_g   = df_nodes[df_nodes['video'].isin(vidids)].copy()
    df_persist_g = df_persist[df_persist['video'].isin(vidids)].copy()

    total_arrivals = 0
    total_crossers = 0

    for vid in vidids:
        df_v  = df_video[df_video['video'] == vid].copy()
        df_n  = df_nodes_g[df_nodes_g['video'] == vid].copy()
        df_pe = df_persist_g[df_persist_g['video'] == vid].copy()
        if df_v.empty:
            continue
        try:
            frames = build_frames(df_v, df_n, df_pe)

            arrivals, crossed = find_crossings(frames)
            if not crossed.empty:
                total_crossers += len(crossed)
            if not arrivals.empty:
                arrivals = arrivals.assign(video=vid, velocity=velocity, wave=width)
                arrival_parts.append(arrivals)
                total_arrivals += len(arrivals)

            bulk = extract_bulk_frames(frames)
            if not bulk.empty:
                bulk = bulk.assign(video=vid, velocity=velocity, wave=width)
                bulk_parts.append(bulk)
        except Exception:
            continue

    if total_crossers > 0:
        fig3a_rows.append({
            'velocity': _vel_label(velocity),
            'width':    _width_label(width),
            'clustered_ratio': total_arrivals / total_crossers,
            'n_arrivals': total_arrivals,
            'n_crossers': total_crossers,
        })


print("\nProcessing done. Writing outputs...")

# --- fig3a: per-config arrival cluster fraction ---
if fig3a_rows:
    df_fig3a = pd.DataFrame(fig3a_rows)
    df_fig3a.to_csv(OUTPUT_DIR + 'fig3a.csv', index=False)
    print(f"Saved fig3a.csv ({len(df_fig3a)} configs)")

# --- fig3bb: arrival cluster velocity distribution (last 5 frames before cross) ---
if arrival_parts:
    arrivals_all = pd.concat(arrival_parts, ignore_index=True)
    rows = []
    for (velocity, width), g in arrivals_all.groupby(['velocity', 'wave']):
        res = _hist(g['vely_arrival'], ARR_EDGES)
        if res is None:
            continue
        hist, n = res
        rows.append(pd.DataFrame({
            'bin_centers': ARR_CENT,
            'hist': hist,
            'velocity': _vel_label(velocity),
            'width': _width_label(width),
            'n_samples': n,
        }))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(OUTPUT_DIR + 'fig3bb.csv', index=False)
        print("Saved fig3bb.csv (arrival cluster-velocity histograms)")

# --- fig3b: not-yet-accumulated bulk velocity distribution (clustered/unclustered) ---
if bulk_parts:
    bulk_all = pd.concat(bulk_parts, ignore_index=True)
    rows = []
    for (velocity, width), g in bulk_all.groupby(['velocity', 'wave']):
        for membership, label in [(1, 'clustered'), (0, 'unclustered')]:
            sub = g[g['clustered_eff'] == membership]
            if sub.empty:
                continue
            res = _hist(sub['vely'], BULK_EDGES)
            if res is None:
                continue
            hist, n = res
            rows.append(pd.DataFrame({
                'bin_centers': BULK_CENT,
                'hist': hist,
                'velocity': _vel_label(velocity),
                'width': _width_label(width),
                'type': label,
                'n_samples': n,
            }))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(OUTPUT_DIR + 'fig3b.csv', index=False)
        print("Saved fig3b.csv (bulk clustered/unclustered velocity histograms)")

conn.close()
print("Done.")
