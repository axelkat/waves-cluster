"""
Animation of the instantaneous PDF(x) for the configuration
(velocity = 750, width = 10), in the visual style of fig1b.

All videos in the group are pooled on a COMMON-TIME axis defined by
    t_common = processed_frame_index       (frame units)

For each common-time frame T, particles from every video where
processed_frame_index == T are pooled and histogrammed,
producing one PDF(x) per frame. Background gate is drawn from
DATA_BACKGROUND_CLASSIFIED at frame T.

Output features:
    - 3 s title card at the start (white background, summary text)
    - 2x playback speed (every second common frame, written at 30 fps)
    - logo overlay in the upper-right corner on every frame
    - white background everywhere
    - PDF strictly normalized in cm (integral p(x) dx_cm = 1)
    - Modified for fig1b style: y-axis capped at 0.11, gate height 0.08
    - 0.75x resolution scaling with tight layout to prevent label clipping
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.ticker import FuncFormatter

# =============================================================================
# User config
# =============================================================================
DB_PATH      = './database.db'
OUTPUT_FILE  = 'animations/v750_w10_density.mp4'
LOGO_PATH    = r'C:/Users/akatona/Pictures/logo.png'

TARGET_VELOCITY = 750
TARGET_WIDTH    = 10

# --- Geometry / time ---
FPS_LAB        = 30                # lab/data frame rate (common-time units)
FRAME_WIDTH_PX = 640
CHANNEL_LEN_CM = 80.0
PX_TO_CM       = CHANNEL_LEN_CM / FRAME_WIDTH_PX   # 0.125 cm/px

# --- Output / playback ---
OUT_FPS         = 30               # mp4 frame rate
PLAYBACK_SPEED  = 2                # 2x: take every 2nd common-time frame
TITLE_DURATION  = 3.0              # seconds

# --- Accumulation-zone exclusion ---
Y_BULK_LIMIT      = 512
Y_CLUSTER_CONTACT = 560

# --- Unified cluster definition ---
MIN_CLUSTER_SIZE         = 3
MIN_CLUSTER_DWELL_FRAMES = 30

# --- Histogram binning over y (plotted as "x [cm]") ---
PLOT_Y_MIN_CM = 7.0
PLOT_Y_MAX_CM = 65.0               # Match fig1b x-limit
N_BINS        = 30                 # Increased to match fig1b point density

# Setup bins explicitly in cm for correct normalization (1/cm)
BIN_EDGES_PX   = np.linspace(0, FRAME_WIDTH_PX, N_BINS + 1)
BIN_EDGES_CM   = BIN_EDGES_PX * PX_TO_CM
BIN_CENTERS_CM = 0.5 * (BIN_EDGES_CM[1:] + BIN_EDGES_CM[:-1])

# --- Plot style (from fig1b) ---
Y_MIN_PLOT      = -0.002
Y_MAX_PLOT      = 0.11             # Modified to requested 0.11 limit
GATE_HEIGHT     = 0.08             # Modified to requested 0.08 height
BASELINE_OFFSET = 0.0
PROFILE_COLOR   = '#990000'
GATE_REGION_TYPES = {'DARK', 'FRONT'}

FONT_AXIS_LABEL = 28
FONT_AXIS_TICK  = 25
AXIS_LINE_WIDTH = 2

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'font.size': FONT_AXIS_TICK,
    'axes.labelsize': FONT_AXIS_LABEL,
    'xtick.labelsize': FONT_AXIS_TICK,
    'ytick.labelsize': FONT_AXIS_TICK,
    'axes.linewidth': AXIS_LINE_WIDTH,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
})


# =============================================================================
# Custom formatter matching fig1b.py
# =============================================================================
def _fmt(value, _pos):
    if abs(value) < 1e-9:
        return "0"
    scaled = value * 1e2
    if abs(scaled - round(scaled)) < 1e-6:
        return f"{int(round(scaled))}"
    return f"{scaled:g}"


# =============================================================================
# Cluster / filtering logic
# =============================================================================
def add_unified_cluster_and_keep(df_obj, df_nodes_v):
    df = df_obj.merge(
        df_nodes_v[['processed_frame_index', 'fixed_id', 'cluster_id', 'cluster_size']],
        on=['processed_frame_index', 'fixed_id'], how='left',
    )

    df['cand_cluster'] = (
        df['cluster_id'].notna() & (df['cluster_size'] >= MIN_CLUSTER_SIZE)
    ).astype(int)

    df = df.sort_values(['fixed_id', 'processed_frame_index'])
    df['state_block_id'] = (
        df['cand_cluster'] != df.groupby('fixed_id')['cand_cluster'].shift()
    ).cumsum()

    block_span = df.groupby(['fixed_id', 'state_block_id'])['processed_frame_index'].agg(
        ['min', 'max']
    )
    block_span['dwell_frames'] = block_span['max'] - block_span['min'] + 1
    df = df.merge(
        block_span['dwell_frames'].reset_index(),
        on=['fixed_id', 'state_block_id'], how='left',
    )

    df['in_cluster'] = df['cand_cluster']
    df.loc[
        (df['in_cluster'] == 1) & (df['dwell_frames'] < MIN_CLUSTER_DWELL_FRAMES),
        'in_cluster',
    ] = 0
    df = df.drop(columns=['state_block_id', 'dwell_frames', 'cand_cluster'])

    cond_self = df['bounding_box_mid_y'] >= Y_BULK_LIMIT

    clustered = df[df['in_cluster'] == 1]
    if not clustered.empty:
        cluster_max_y = (
            clustered.groupby(['processed_frame_index', 'cluster_id'])['bounding_box_mid_y']
            .max().rename('cluster_max_y').reset_index()
        )
        df = df.merge(cluster_max_y, on=['processed_frame_index', 'cluster_id'], how='left')
    else:
        df['cluster_max_y'] = np.nan

    cond_cluster_contact = (
        (df['in_cluster'] == 1) & (df['cluster_max_y'] >= Y_CLUSTER_CONTACT)
    )

    df['keep'] = (~(cond_self | cond_cluster_contact)).astype(int)
    df = df.drop(columns=['cluster_max_y'])
    return df


# =============================================================================
# Background gate
# =============================================================================
def select_background_for_group(df_bg_all, target_velocity, target_width):
    bg_vw = df_bg_all[
        (df_bg_all['velocity'] == target_velocity) &
        (df_bg_all['width']    == target_width)
    ]
    if bg_vw.empty:
        raise RuntimeError(
            f"No DATA_BACKGROUND_CLASSIFIED rows for "
            f"velocity={target_velocity}, width={target_width}"
        )
    chosen = bg_vw['config_name'].iloc[0]
    print(f"Background: using config '{chosen}' for the gate")
    return bg_vw[bg_vw['config_name'] == chosen].copy()


def regions_at_common_frame(df_bg_cfg, t_common):
    mask = (df_bg_cfg['frame_start'] <= t_common) & (df_bg_cfg['frame_end'] > t_common)
    return df_bg_cfg.loc[mask].sort_values('y_start')


def plot_background_gate(ax, regions_df, offset, gate_height):
    if regions_df.empty:
        return
    gate_x, gate_y = [], []
    for _, r in regions_df.iterrows():
        x_start = r['y_start'] * PX_TO_CM
        x_end   = r['y_end']   * PX_TO_CM
        is_gate = r['region_type'] in GATE_REGION_TYPES
        if is_gate:
            ax.fill_between(
                [x_start, x_end], offset, offset + gate_height,
                color='#cccccc', alpha=0.8, zorder=1, edgecolor='none',
            )
        y_value = offset + gate_height if is_gate else offset
        gate_x.extend([x_start, x_end])
        gate_y.extend([y_value, y_value])
    ax.plot(gate_x, gate_y, color='#666666', linewidth=3.0, zorder=2)


# =============================================================================
# Logo helper
# =============================================================================
def load_logo(path):
    if not os.path.exists(path):
        print(f"[logo] file not found at '{path}' -- continuing without logo")
        return None
    try:
        img = mpimg.imread(path)
        return img
    except Exception as e:
        print(f"[logo] failed to load '{path}': {e}")
        return None


def add_logo(fig, logo_img, zoom=0.14, pad=0.015):
    if logo_img is None:
        return
    imagebox = OffsetImage(logo_img, zoom=zoom)
    ab = AnnotationBbox(
        imagebox, (1.0 - pad, 1.0 - pad),
        xycoords='figure fraction',
        box_alignment=(1.0, 1.0),
        frameon=False, zorder=100,
    )
    fig.add_artist(ab)


# =============================================================================
# Main
# =============================================================================
def main():
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Connecting to {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)

    # --- 1. CONFIG rows ---
    config_df = pd.read_sql(
        "SELECT unique_name, velocity, width, INT "
        "FROM CONFIG WHERE velocity = ? AND width = ? AND INT IN (255, 250)",
        conn, params=(TARGET_VELOCITY, TARGET_WIDTH),
    )
    if config_df.empty:
        raise RuntimeError(
            f"No videos in CONFIG for velocity={TARGET_VELOCITY}, width={TARGET_WIDTH}"
        )
    vidids = config_df['unique_name'].tolist()
    print(f"Found {len(vidids)} video(s):")
    for v in vidids:
        print(f"  {v}")

    n_experiments = len(vidids)
    placeholders = ','.join(['?'] * len(vidids))

    # --- 2. Particles ---
    print("Loading particle positions ...")
    df_obj = pd.read_sql(
        f"SELECT video, fixed_id, processed_frame_index, "
        f"bounding_box_mid_x, bounding_box_mid_y "
        f"FROM data_object WHERE video IN ({placeholders}) AND processed_frame_index < 1500 ",
        conn, params=vidids,
    )
    df_obj = df_obj.dropna(subset=['fixed_id', 'bounding_box_mid_y']).copy()

    # --- 3. Cluster nodes ---
    print("Loading cluster nodes ...")
    df_nodes = pd.read_sql(
        f"SELECT video, processed_frame_index, fixed_id, cluster_id, cluster_size "
        f"FROM data_graphs_nodes WHERE video IN ({placeholders})",
        conn, params=vidids,
    )

    # --- 4. Background ---
    print("Loading DATA_BACKGROUND_CLASSIFIED ...")
    df_bg_all = pd.read_sql(
        "SELECT config_name, width, velocity, frame_start, frame_end, "
        "y_start, y_end, region_type FROM DATA_BACKGROUND_CLASSIFIED "
        "WHERE velocity = ? AND width = ?",
        conn, params=(TARGET_VELOCITY, TARGET_WIDTH),
    )
    conn.close()

    # --- 5. Filtering ---
    print("Applying cluster definition + accumulation-zone exclusion ...")
    parts = []
    for vid in vidids:
        d_o = df_obj[df_obj['video'] == vid].copy()
        d_n = df_nodes[df_nodes['video'] == vid].copy()
        if d_o.empty:
            continue
        parts.append(add_unified_cluster_and_keep(d_o, d_n).assign(video=vid))
    if not parts:
        raise RuntimeError("No data after filtering.")
    df_all = pd.concat(parts, ignore_index=True)
    df_kept = df_all[df_all['keep'] == 1].copy()

    # --- 6. Common time mapping (No delta used) ---
    df_kept['t_common'] = df_kept['processed_frame_index'].astype(int)

    # --- 7. Common-time axis (every-other for 2x speed) ---
    t_max = int(df_kept['t_common'].max()) if not df_kept.empty else 0
    common_frames = list(range(0, t_max + 1, PLAYBACK_SPEED))
    print(f"Common-time axis: T = 0 .. {t_max}, "
          f"{len(common_frames)} sampled frames (stride={PLAYBACK_SPEED})")

    # --- 8. Per-frame pooled histograms ---
    print("Pre-computing pooled per-frame histograms ...")
    
    # Convert y-coordinates to cm prior to grouping to guarantee exact 1/cm normalization
    df_kept['y_cm'] = df_kept['bounding_box_mid_y'] * PX_TO_CM
    grp = df_kept.groupby('t_common')['y_cm']
    
    hist_per_T = {}
    for T in common_frames:
        try:
            ys_cm = grp.get_group(T).values
        except KeyError:
            ys_cm = np.array([])
            
        if len(ys_cm) == 0:
            hist_per_T[T] = np.zeros(N_BINS)
        else:
            # Using BIN_EDGES_CM ensures integration against dx (cm) equals 1
            h, _ = np.histogram(ys_cm, bins=BIN_EDGES_CM, density=True)
            hist_per_T[T] = h

    # --- 9. Background regions per frame ---
    df_bg_cfg = select_background_for_group(df_bg_all, TARGET_VELOCITY, TARGET_WIDTH)
    print("Pre-computing per-frame background regions ...")
    regions_per_T = {T: regions_at_common_frame(df_bg_cfg, T) for T in common_frames}

    # --- 10. Logo ---
    logo_img = load_logo(LOGO_PATH)

    # --- 11. Title-card frame count ---
    n_title_frames = int(round(TITLE_DURATION * OUT_FPS))
    n_data_frames  = len(common_frames)
    n_total_frames = n_title_frames + n_data_frames
    print(f"Title frames: {n_title_frames}  (at {OUT_FPS} fps for {TITLE_DURATION}s)")
    print(f"Data frames:  {n_data_frames}")

    # --- 12. Build figure ---
    # Downscaled by 0.75x: (10, 7) -> (7.5, 5.25)
    fig, ax = plt.subplots(figsize=(7.5, 5.25), dpi=200)
    fig.patch.set_facecolor('white')
    
    # Explicit subplots_adjust fixes the bug with large X-axis text getting clipped
    fig.subplots_adjust(bottom=0.22, left=0.20, right=0.95, top=0.90)

    # Persistent logo on the figure (drawn on every saved frame)
    add_logo(fig, logo_img, zoom=0.14, pad=0.015)

    # Pre-render text strings
    n_agents = 80 
    wavelength_cm = 40
    wave_velocity_cm_s = 2.7
    title_lines = [
        f"Average of {n_experiments} experiments with {n_agents} macroscopic",
        "photoactive agents with travelling wave.",
        "",
        "Probability density function of number of agents",
        "as a function of their x coordinate.",
        "",
        f"Wavelength: {wavelength_cm} cm",
        f"Wave velocity: {wave_velocity_cm_s} cm/s",
    ]

    def draw_title(idx):
        ax.clear()
        ax.set_facecolor('white')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        y_start = 0.78
        line_step = 0.085
        for i, line in enumerate(title_lines):
            y = y_start - i * line_step
            # Title text sized down slightly to fit the smaller layout seamlessly
            if i == 0 or i == 1:
                fs, weight = 18, 'bold'
            elif i in (3, 4):
                fs, weight = 14, 'normal'
            elif i in (6, 7):
                fs, weight = 16, 'normal'
            else:
                continue 
            ax.text(0.5, y, line,
                    fontsize=fs, fontweight=weight,
                    ha='center', va='center',
                    color='black', transform=ax.transAxes)

    def draw_data(data_idx):
        ax.clear()
        ax.axis('on')
        ax.set_facecolor('white')
        T = common_frames[data_idx]

        plot_background_gate(ax, regions_per_T[T],
                             offset=BASELINE_OFFSET, gate_height=GATE_HEIGHT)

        density = hist_per_T[T] + BASELINE_OFFSET
        
        density_clipped = np.minimum(density, Y_MAX_PLOT)
        ax.plot(BIN_CENTERS_CM, density_clipped,
                color=PROFILE_COLOR, linewidth=2.5,
                marker='.', markersize=10, zorder=10)

        ax.axhline(y=BASELINE_OFFSET, color=PROFILE_COLOR,
                   linestyle='--', linewidth=2.0, alpha=0.7, zorder=5)

        ax.axvline(x=Y_BULK_LIMIT * PX_TO_CM,
                   color='black', linestyle=':', linewidth=1.5,
                   alpha=0.5, zorder=4)

        # Inline time label styled like fig1b.py (raised by 0.005 above gate top)
        t_sec = T / FPS_LAB
        ax.text(PLOT_Y_MIN_CM, GATE_HEIGHT + 0.005,
                f"  t = {int(round(t_sec))} s", fontsize=18,
                fontweight='bold', color='black', verticalalignment='bottom', zorder=100)

        # Axis configurations
        ax.set_xlabel(r"$x$ [cm]", fontsize=FONT_AXIS_LABEL)
        ax.set_ylabel(r"$p(x)$", fontsize=FONT_AXIS_LABEL, labelpad=10)
        ax.set_xlim(PLOT_Y_MIN_CM, PLOT_Y_MAX_CM)
        ax.set_ylim(Y_MIN_PLOT, Y_MAX_PLOT)
        
        # X Ticks
        ax.set_xticks([10, 20, 30, 40, 50, 60])
        ax.tick_params(axis='x', labelsize=FONT_AXIS_TICK, pad=10)
        
        # Y Ticks (spaced for max limit = 0.11, auto formatting matching fig1b.py)
        ax.set_yticks([0.0, 0.05, 0.10])
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
        ax.tick_params(axis='y', labelsize=22)
        
        # Exponent for Y axis
        ax.annotate(
            r"$10^{-2}\times$",
            xy=(-0.05, 0.92), xycoords='axes fraction',
            xytext=(6, 8), textcoords='offset points',
            ha='right', va='bottom', fontsize=16,
        )

        # Grids matching fig1b
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.1)
        ax.yaxis.grid(True, which='major', linestyle=':', color='#888888', alpha=0.5, linewidth=1.0)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)

        if data_idx > 0 and data_idx % max(1, n_data_frames // 10) == 0:
            pct = 100 * data_idx // n_data_frames
            print(f"  Progress (data): {data_idx}/{n_data_frames} ({pct}%)")

    def draw_frame(global_idx):
        if global_idx < n_title_frames:
            draw_title(global_idx)
        else:
            draw_data(global_idx - n_title_frames)

    print(f"Rendering {n_total_frames} frames ...")
    anim = FuncAnimation(fig, draw_frame, frames=n_total_frames, repeat=False)

    writer = FFMpegWriter(fps=OUT_FPS, codec='libx264', bitrate=2400)
    try:
        anim.save(OUTPUT_FILE, writer=writer)
        print(f"Saved: {OUTPUT_FILE}")
    except Exception as e:
        print(f"ERROR saving animation: {e}")
        print("Make sure FFmpeg is installed and on PATH.")
    plt.close(fig)


if __name__ == "__main__":
    main()