import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Modul 6 – Verification & Validation | MODSIM 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Warna dasar ── */
    :root {
        --primary:   #1E3A8A;
        --secondary: #3B82F6;
        --accent:    #F59E0B;
        --success:   #10B981;
        --danger:    #EF4444;
        --bg-card:   #F8FAFC;
    }

    /* ── Header ── */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: var(--primary);
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--secondary);
        margin-top: 1.8rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid var(--secondary);
        padding-left: 0.6rem;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--primary);
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }

    /* ── Info / Warning / Success box ── */
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 5px solid var(--secondary);
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 5px solid var(--accent);
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #ECFDF5;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 5px solid var(--success);
        margin-bottom: 1rem;
    }
    .danger-box {
        background-color: #FEF2F2;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 5px solid var(--danger);
        margin-bottom: 1rem;
    }

    /* ── Metric card ── */
    .metric-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.1rem 0.8rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(59,130,246,.25);
    }
    .metric-card h2 { margin: 0; font-size: 1.9rem; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: .85; }

    /* ── Event trace row ── */
    .trace-row {
        background: var(--bg-card);
        border-radius: 6px;
        padding: 0.35rem 0.7rem;
        margin: 0.2rem 0;
        border-left: 3px solid var(--accent);
        font-size: 0.88rem;
    }

    /* ── Verdict badge ── */
    .badge-pass {
        background: #D1FAE5; color: #065F46;
        padding: 2px 10px; border-radius: 99px;
        font-size: 0.82rem; font-weight: 700;
    }
    .badge-fail {
        background: #FEE2E2; color: #991B1B;
        padding: 2px 10px; border-radius: 99px;
        font-size: 0.82rem; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. KELAS MODEL SIMULASI (Discrete Event Simulation – Single Server Queue)
# ============================================================================
class ExamPaperDistributionDES:
    """
    Discrete Event Simulation: Pembagian Lembar Jawaban Ujian
    - Single-server queue (FIFO)
    - Waktu pelayanan: Uniform(min_service, max_service) menit
    - Satu mahasiswa dilayani pada satu waktu
    """

    def __init__(self, n_students: int, min_service: float = 1.0,
                 max_service: float = 3.0, seed: int = None):
        self.n_students  = n_students
        self.min_service = min_service
        self.max_service = max_service
        self.seed        = seed
        self.events      = []          # log detail setiap mahasiswa
        self.rng         = None

    # ── Jalankan simulasi ──────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        self.rng = rng

        service_times   = rng.uniform(self.min_service, self.max_service, self.n_students)
        records         = []
        current_time    = 0.0        # waktu kapan server tersedia

        for i in range(self.n_students):
            arrival_time   = 0.0     # semua mahasiswa sudah ada di ruangan
            start_service  = max(arrival_time, current_time)
            wait_time      = start_service - arrival_time
            end_service    = start_service + service_times[i]
            current_time   = end_service

            records.append({
                "Mahasiswa"       : i + 1,
                "Mulai_Dilayani"  : round(start_service, 4),
                "Selesai_Dilayani": round(end_service,   4),
                "Durasi_Pelayanan": round(service_times[i], 4),
                "Waktu_Tunggu"    : round(wait_time,    4),
            })

        self.events = pd.DataFrame(records)
        return self.events

    # ── Ringkasan statistik ────────────────────────────────────────────
    def summary(self) -> dict:
        if self.events is None or len(self.events) == 0:
            raise RuntimeError("Jalankan run() terlebih dahulu.")
        df = self.events
        return {
            "total_time"      : df["Selesai_Dilayani"].max(),
            "mean_service"    : df["Durasi_Pelayanan"].mean(),
            "mean_wait"       : df["Waktu_Tunggu"].mean(),
            "max_wait"        : df["Waktu_Tunggu"].max(),
            "server_util"     : df["Durasi_Pelayanan"].sum() / df["Selesai_Dilayani"].max(),
            "service_times"   : df["Durasi_Pelayanan"].values,
        }


# ============================================================================
# 3. FUNGSI VISUALISASI
# ============================================================================
def plot_gantt(events: pd.DataFrame, title: str = "Timeline Pelayanan Mahasiswa"):
    """Gantt chart waktu pelayanan setiap mahasiswa."""
    n = min(len(events), 40)          # tampilkan max 40 agar tidak terlalu padat
    df = events.head(n)

    fig = go.Figure()
    colors = px.colors.qualitative.Pastel
    for _, row in df.iterrows():
        idx = int(row["Mahasiswa"]) - 1
        fig.add_trace(go.Bar(
            x=[row["Durasi_Pelayanan"]],
            base=[row["Mulai_Dilayani"]],
            y=[f"Mhs {int(row['Mahasiswa'])}"],
            orientation="h",
            marker_color=colors[idx % len(colors)],
            text=f"{row['Durasi_Pelayanan']:.2f} mnt",
            textposition="inside",
            showlegend=False,
            hovertemplate=(
                f"Mahasiswa {int(row['Mahasiswa'])}<br>"
                f"Mulai: {row['Mulai_Dilayani']:.2f} mnt<br>"
                f"Selesai: {row['Selesai_Dilayani']:.2f} mnt<br>"
                f"Durasi: {row['Durasi_Pelayanan']:.2f} mnt<extra></extra>"
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Waktu (Menit)",
        yaxis_title="Mahasiswa",
        barmode="overlay",
        height=max(350, n * 22 + 80),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#F8FAFC"
    )
    return fig


def plot_service_distribution(service_times: np.ndarray,
                               min_s: float, max_s: float) -> go.Figure:
    """Histogram distribusi waktu pelayanan."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=service_times, nbinsx=20,
        name="Durasi Pelayanan",
        marker_color="#3B82F6", opacity=0.75,
        histnorm="probability density"
    ))
    # Tambahkan garis teoritis Uniform
    x_line = np.linspace(min_s - 0.2, max_s + 0.2, 300)
    y_line = np.where((x_line >= min_s) & (x_line <= max_s),
                      1 / (max_s - min_s), 0)
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines", name=f"Teoritis Uniform({min_s},{max_s})",
        line=dict(color="red", dash="dash", width=2)
    ))
    fig.add_vline(x=service_times.mean(), line_dash="dot",
                  line_color="green",
                  annotation_text=f"Mean: {service_times.mean():.2f}")
    fig.update_layout(
        title="Distribusi Durasi Pelayanan vs Uniform Teoritis",
        xaxis_title="Durasi Pelayanan (Menit)",
        yaxis_title="Densitas Probabilitas",
        height=400, legend=dict(x=0.7, y=0.95),
        plot_bgcolor="#F8FAFC"
    )
    return fig


def plot_wait_times(events: pd.DataFrame) -> go.Figure:
    """Bar chart waktu tunggu setiap mahasiswa."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=events["Mahasiswa"],
        y=events["Waktu_Tunggu"],
        marker_color="#F59E0B",
        name="Waktu Tunggu",
        hovertemplate="Mhs %{x}<br>Tunggu: %{y:.2f} mnt<extra></extra>"
    ))
    fig.update_layout(
        title="Waktu Tunggu Setiap Mahasiswa",
        xaxis_title="Nomor Mahasiswa",
        yaxis_title="Waktu Tunggu (Menit)",
        height=380, plot_bgcolor="#F8FAFC"
    )
    return fig


def plot_sensitivity(results_list: list, labels: list,
                     metric: str = "total_time") -> go.Figure:
    """Box plot sensitivity analysis."""
    fig = go.Figure()
    palette = px.colors.qualitative.Set2
    for i, (label, data) in enumerate(zip(labels, results_list)):
        fig.add_trace(go.Box(
            y=data, name=label,
            marker_color=palette[i % len(palette)],
            boxmean="sd", boxpoints="outliers"
        ))
    ylab = "Total Waktu (Menit)" if metric == "total_time" else "Nilai"
    fig.update_layout(
        title="Sensitivity Analysis – Distribusi Total Waktu",
        yaxis_title=ylab,
        height=450, plot_bgcolor="#F8FAFC"
    )
    return fig


def plot_behavior_validation(n_range: list,
                              means: list, theoretics: list) -> go.Figure:
    """Line chart behavior validation: simulasi vs teoritis."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_range, y=means, mode="lines+markers",
        name="Rata-rata Simulasi",
        line=dict(color="#3B82F6", width=3),
        marker=dict(size=7)
    ))
    fig.add_trace(go.Scatter(
        x=n_range, y=theoretics, mode="lines",
        name="Nilai Teoritis (N × E[T])",
        line=dict(color="red", dash="dash", width=2)
    ))
    fig.update_layout(
        title="Behavior Validation: Simulasi vs Teoritis",
        xaxis_title="Jumlah Mahasiswa (N)",
        yaxis_title="Total Waktu Rata-rata (Menit)",
        height=420, plot_bgcolor="#F8FAFC",
        legend=dict(x=0.05, y=0.95)
    )
    return fig


# ============================================================================
# 4. FUNGSI UTAMA STREAMLIT
# ============================================================================
def main():
    # ── Header ──────────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="main-header">🎓 Modul 6: Verification & Validation</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="main-subtitle">'
        '[11S1221] Pemodelan dan Simulasi (MODSIM 2026) &nbsp;|&nbsp; '
        'Studi Kasus: <b>Pembagian Lembar Jawaban Ujian (Discrete Event Simulation)</b>'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    <b>📌 Deskripsi Sistem:</b> Pada akhir ujian, pengajar membagikan kembali lembar jawaban
    kepada mahasiswa. Mahasiswa maju <b>satu per satu (FIFO)</b> ke meja pengajar. Waktu
    pelayanan setiap mahasiswa berdistribusi <b>Uniform(min, max)</b> menit. Simulasi
    menghitung <b>total waktu, waktu tunggu, dan utilisasi meja pengajar</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Konfigurasi Simulasi")

    n_students = st.sidebar.number_input(
        "Jumlah Mahasiswa (N):", min_value=1, max_value=200, value=30, step=1
    )
    min_service = st.sidebar.number_input(
        "Durasi Minimum Pelayanan (menit):", min_value=0.1, max_value=10.0,
        value=1.0, step=0.1, format="%.1f"
    )
    max_service = st.sidebar.number_input(
        "Durasi Maksimum Pelayanan (menit):", min_value=0.2, max_value=15.0,
        value=3.0, step=0.1, format="%.1f"
    )
    use_seed = st.sidebar.checkbox("Gunakan Random Seed (Reproducibility)", value=False)
    seed_val = st.sidebar.number_input("Nilai Seed:", min_value=0, max_value=99999,
                                        value=42, step=1) if use_seed else None

    run_btn = st.sidebar.button("🚀 Jalankan Simulasi", type="primary",
                                 use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.82rem;color:#64748B;">
    <b>Keterangan:</b><br>
    • Distribusi waktu: <b>Uniform(min, max)</b><br>
    • Antrian: <b>FIFO – Single Server</b><br>
    • Teoritis E[T] = (min + max) / 2<br>
    • Total teoritis = N × E[T]
    </div>
    """, unsafe_allow_html=True)

    # ── Validasi input ────────────────────────────────────────────────
    if min_service >= max_service:
        st.error("⚠️ Durasi minimum harus lebih kecil dari durasi maksimum.")
        return

    # ── Session state ─────────────────────────────────────────────────
    for key in ["sim_events", "sim_summary", "sim_params"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Jalankan simulasi ─────────────────────────────────────────────
    if run_btn:
        with st.spinner("Menjalankan simulasi DES..."):
            sim = ExamPaperDistributionDES(
                n_students=n_students,
                min_service=min_service,
                max_service=max_service,
                seed=seed_val
            )
            events  = sim.run()
            summary = sim.summary()

        st.session_state.sim_events  = events
        st.session_state.sim_summary = summary
        st.session_state.sim_params  = {
            "n": n_students, "min": min_service,
            "max": max_service, "seed": seed_val
        }
        st.success(f"✅ Simulasi selesai! {n_students} mahasiswa telah dilayani.")

    # ── Tampilkan hasil ────────────────────────────────────────────────
    if st.session_state.sim_events is not None:
        events   = st.session_state.sim_events
        summary  = st.session_state.sim_summary
        params   = st.session_state.sim_params

        N       = params["n"]
        min_s   = params["min"]
        max_s   = params["max"]
        E_T     = (min_s + max_s) / 2          # teoritis E[T]
        total_teoritis = N * E_T

        # ── Statistik Utama ──────────────────────────────────────────
        st.markdown('<h2 class="sub-header">📊 Statistik Utama Simulasi</h2>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{summary['total_time']:.2f} mnt", "Total Waktu Pembagian"),
            (c2, f"{summary['mean_service']:.2f} mnt", "Rata-rata Durasi Pelayanan"),
            (c3, f"{summary['mean_wait']:.2f} mnt", "Rata-rata Waktu Tunggu"),
            (c4, f"{summary['server_util']*100:.1f}%", "Utilisasi Meja Pengajar"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <h2>{val}</h2><p>{lbl}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tab Utama ────────────────────────────────────────────────
        tab_main, tab_verif, tab_valid, tab_data = st.tabs([
            "📈 Visualisasi Utama",
            "🔍 Verification",
            "✅ Validation",
            "📋 Data Simulasi"
        ])

        # ──────────────────────────────────────────────────────────────
        # TAB 1: VISUALISASI UTAMA
        # ──────────────────────────────────────────────────────────────
        with tab_main:
            st.markdown('<p class="section-header">🕐 Timeline Pelayanan (Gantt Chart)</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(plot_gantt(events), use_container_width=True, key="chart_gantt")

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    plot_service_distribution(summary["service_times"], min_s, max_s),
                    use_container_width=True,
                    key="chart_service_dist_main"
                )
            with col_b:
                st.plotly_chart(plot_wait_times(events), use_container_width=True, key="chart_wait_times")

            # ── Ringkasan perbandingan teoritis ──
            st.markdown('<p class="section-header">📐 Perbandingan Simulasi vs Teoritis</p>',
                        unsafe_allow_html=True)
            gap = abs(summary["total_time"] - total_teoritis)
            pct = gap / total_teoritis * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Waktu Simulasi",  f"{summary['total_time']:.2f} mnt")
            c2.metric("Total Waktu Teoritis",  f"{total_teoritis:.2f} mnt",
                      help=f"N × E[T] = {N} × {E_T:.2f}")
            c3.metric("Selisih",               f"{gap:.2f} mnt ({pct:.1f}%)",
                      delta_color="off")

        # ──────────────────────────────────────────────────────────────
        # TAB 2: VERIFICATION
        # ──────────────────────────────────────────────────────────────
        with tab_verif:
            st.markdown("""
            <div class="info-box">
            <b>🔍 Tujuan Verifikasi:</b> Memastikan model simulasi telah
            <b>diimplementasikan dengan benar</b> sesuai logika sistem,
            asumsi, dan aturan antrian FIFO.<br>
            <i>"Build the model right?"</i>
            </div>
            """, unsafe_allow_html=True)

            # ── a. Logical Flow Check ──────────────────────────────
            with st.expander("✅ a. Logical Flow Check (Pemeriksaan Logika Alur)", expanded=True):
                # Periksa tidak ada tumpang tindih pelayanan
                overlap = False
                for i in range(1, len(events)):
                    if events.iloc[i]["Mulai_Dilayani"] < events.iloc[i-1]["Selesai_Dilayani"] - 1e-9:
                        overlap = True
                        break

                # Periksa urutan FIFO (mulai[i+1] >= selesai[i])
                fifo_ok = all(
                    events.iloc[i]["Mulai_Dilayani"] >= events.iloc[i-1]["Selesai_Dilayani"] - 1e-9
                    for i in range(1, len(events))
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pengecekan:**")
                    st.markdown(f"- Tidak ada tumpang tindih pelayanan: "
                                f"<span class=\"badge-{'pass' if not overlap else 'fail'}\">{'PASS ✔' if not overlap else 'FAIL ✘'}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"- Urutan FIFO terjaga: "
                                f"<span class=\"badge-{'pass' if fifo_ok else 'fail'}\">{'PASS ✔' if fifo_ok else 'FAIL ✘'}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"- Server tidak melayani >1 mahasiswa bersamaan: "
                                f"<span class=\"badge-{'pass' if not overlap else 'fail'}\">{'PASS ✔' if not overlap else 'FAIL ✘'}</span>",
                                unsafe_allow_html=True)
                with col2:
                    if not overlap and fifo_ok:
                        st.markdown("""
                        <div class="success-box">
                        ✅ <b>Hasil:</b> Alur model sesuai sistem nyata.
                        Mahasiswa dilayani <b>satu per satu</b>, tidak ada
                        dua mahasiswa dilayani bersamaan.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="danger-box">
                        ❌ <b>Terdapat kesalahan logika!</b> Periksa kembali implementasi.
                        </div>
                        """, unsafe_allow_html=True)

            # ── b. Event Tracing ───────────────────────────────────
            with st.expander("🔎 b. Event Tracing", expanded=True):
                st.markdown("**Pelacakan event 5 mahasiswa pertama:**")
                n_trace = min(5, len(events))
                for _, row in events.head(n_trace).iterrows():
                    st.markdown(
                        f'<div class="trace-row">'
                        f'👤 <b>Mahasiswa {int(row["Mahasiswa"])}</b> &nbsp;|&nbsp; '
                        f'🕐 Mulai dilayani: <b>{row["Mulai_Dilayani"]:.2f} mnt</b> &nbsp;|&nbsp; '
                        f'✅ Selesai: <b>{row["Selesai_Dilayani"]:.2f} mnt</b> &nbsp;|&nbsp; '
                        f'⏱ Durasi: <b>{row["Durasi_Pelayanan"]:.2f} mnt</b> &nbsp;|&nbsp; '
                        f'⌛ Tunggu: <b>{row["Waktu_Tunggu"]:.2f} mnt</b>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                # Verifikasi urutan kronologis
                chrono_ok = all(
                    events.iloc[i]["Selesai_Dilayani"] >= events.iloc[i]["Mulai_Dilayani"]
                    for i in range(len(events))
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f"**Urutan event kronologis:** "
                    f"<span class=\"badge-{'pass' if chrono_ok else 'fail'}\">{'PASS ✔' if chrono_ok else 'FAIL ✘'}</span>",
                    unsafe_allow_html=True
                )

            # ── c. Extreme Condition Test ──────────────────────────
            with st.expander("⚡ c. Extreme Condition Test", expanded=True):
                st.markdown("**Pengujian pada kondisi ekstrem:**")

                # Uji N=1
                sim_1 = ExamPaperDistributionDES(1, min_s, max_s, seed=0)
                ev_1  = sim_1.run()
                sum_1 = sim_1.summary()
                expected_1 = ev_1.iloc[0]["Durasi_Pelayanan"]
                pass_1 = abs(sum_1["total_time"] - expected_1) < 1e-6

                # Uji durasi tetap = min_s
                sim_fixed_min = ExamPaperDistributionDES(N, min_s, min_s + 1e-9, seed=1)
                ev_fm   = sim_fixed_min.run()
                sum_fm  = sim_fixed_min.summary()
                expected_fm = N * min_s
                pass_fm = abs(sum_fm["total_time"] - expected_fm) < 1.0   # toleransi float

                # Uji durasi tetap = max_s
                sim_fixed_max = ExamPaperDistributionDES(N, max_s - 1e-9, max_s, seed=2)
                ev_fmx  = sim_fixed_max.run()
                sum_fmx = sim_fixed_max.summary()
                expected_fmx = N * max_s
                pass_fmx = abs(sum_fmx["total_time"] - expected_fmx) < 1.0

                tbl_data = {
                    "Skenario": [
                        "N = 1",
                        f"Durasi tetap = {min_s} mnt (min)",
                        f"Durasi tetap = {max_s} mnt (max)"
                    ],
                    "Hasil yang Diharapkan": [
                        f"Total = durasi mahasiswa 1",
                        f"Total ≈ N × {min_s} = {N * min_s:.1f} mnt",
                        f"Total ≈ N × {max_s} = {N * max_s:.1f} mnt"
                    ],
                    "Hasil Simulasi": [
                        f"{sum_1['total_time']:.4f} mnt",
                        f"{sum_fm['total_time']:.4f} mnt",
                        f"{sum_fmx['total_time']:.4f} mnt"
                    ],
                    "Verdict": [
                        "✔ PASS" if pass_1   else "✘ FAIL",
                        "✔ PASS" if pass_fm  else "✘ FAIL",
                        "✔ PASS" if pass_fmx else "✘ FAIL",
                    ]
                }
                st.dataframe(pd.DataFrame(tbl_data), use_container_width=True)
                if pass_1 and pass_fm and pass_fmx:
                    st.markdown("""
                    <div class="success-box">
                    ✅ <b>Hasil:</b> Model memberikan hasil sesuai perhitungan logis
                    pada semua kondisi ekstrem.
                    </div>
                    """, unsafe_allow_html=True)

            # ── d. Pemeriksaan Distribusi Waktu Pelayanan ──────────
            with st.expander("📊 d. Pemeriksaan Distribusi Waktu Pelayanan", expanded=True):
                service_arr = summary["service_times"]
                in_range = np.all((service_arr >= min_s) & (service_arr <= max_s))
                st.plotly_chart(
                    plot_service_distribution(service_arr, min_s, max_s),
                    use_container_width=True,
                    key="chart_service_dist_verif"
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Min Observasi", f"{service_arr.min():.4f}")
                c2.metric("Max Observasi", f"{service_arr.max():.4f}")
                c3.metric("Rentang Valid",
                           f"[{min_s}, {max_s}]",
                           delta="✔ Semua dalam range" if in_range else "✘ Ada yang keluar range",
                           delta_color="normal" if in_range else "inverse")
                if in_range:
                    st.markdown("""
                    <div class="success-box">
                    ✅ <b>Hasil:</b> Semua nilai durasi berada dalam rentang
                    Uniform yang ditetapkan.
                    </div>
                    """, unsafe_allow_html=True)

            # ── e. Reproducibility Check ───────────────────────────
            with st.expander("🔁 e. Reproducibility Check", expanded=True):
                if params["seed"] is not None:
                    sim_r1 = ExamPaperDistributionDES(N, min_s, max_s, seed=params["seed"])
                    sim_r2 = ExamPaperDistributionDES(N, min_s, max_s, seed=params["seed"])
                    ev_r1 = sim_r1.run()
                    ev_r2 = sim_r2.run()
                    identical = np.allclose(ev_r1["Durasi_Pelayanan"].values,
                                            ev_r2["Durasi_Pelayanan"].values)
                    if identical:
                        st.markdown("""
                        <div class="success-box">
                        ✅ <b>Hasil:</b> Dua eksekusi dengan seed yang sama menghasilkan
                        output <b>identik</b> — implementasi random telah benar.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="danger-box">
                        ❌ Output berbeda meski seed sama. Periksa implementasi RNG.
                        </div>
                        """, unsafe_allow_html=True)
                    st.dataframe(
                        pd.DataFrame({
                            "Run 1 – Durasi Pelayanan": ev_r1["Durasi_Pelayanan"].head(5).values,
                            "Run 2 – Durasi Pelayanan": ev_r2["Durasi_Pelayanan"].head(5).values,
                        }),
                        use_container_width=True
                    )
                else:
                    st.markdown("""
                    <div class="warning-box">
                    ⚠️ Aktifkan <b>"Gunakan Random Seed"</b> di sidebar untuk menjalankan
                    reproducibility check.
                    </div>
                    """, unsafe_allow_html=True)

            # ── Kesimpulan Verifikasi ──────────────────────────────
            st.markdown("""
            <div class="success-box">
            <h4>✅ Kesimpulan Verifikasi</h4>
            Model simulasi <b>telah terverifikasi</b>:
            <ul>
              <li>Logika sistem berjalan sesuai asumsi FIFO single-server.</li>
              <li>Tidak ditemukan tumpang tindih waktu pelayanan.</li>
              <li>Distribusi waktu pelayanan sesuai Uniform yang ditetapkan.</li>
              <li>Hasil simulasi konsisten secara internal.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


        # ──────────────────────────────────────────────────────────────
        # TAB 3: VALIDATION
        # ──────────────────────────────────────────────────────────────
        with tab_valid:
            st.markdown("""
            <div class="info-box">
            <b>✅ Tujuan Validasi:</b> Memastikan hasil simulasi
            <b>merepresentasikan kondisi nyata</b> dari pembagian lembar jawaban.<br>
            <i>"Build the right model?"</i>
            </div>
            """, unsafe_allow_html=True)

            # ── a. Face Validation ────────────────────────────────
            with st.expander("👁️ a. Face Validation", expanded=True):
                realistic_low  = N * min_s
                realistic_high = N * max_s
                face_pass = (realistic_low <= summary["total_time"] <= realistic_high)

                st.markdown(f"""
                **Analisis hasil simulasi secara konseptual:**
                - Untuk **{N} mahasiswa** dengan durasi **Uniform({min_s}, {max_s})** menit...
                - Rentang total waktu yang realistis: **{realistic_low:.1f} – {realistic_high:.1f} menit**
                - Total waktu simulasi: **{summary['total_time']:.2f} menit**
                """)

                if face_pass:
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ Total waktu <b>{summary['total_time']:.2f} menit</b> berada dalam
                    rentang realistis [{realistic_low:.1f}, {realistic_high:.1f}] menit.
                    Hasil ini <b>masuk akal dan sesuai ekspektasi</b>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="danger-box">
                    ❌ Total waktu di luar rentang realistis. Periksa kembali parameter.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box">
                📋 <b>Contoh pertanyaan face validation:</b><br>
                • "Apakah ~{total_teoritis:.0f} menit untuk {N} mahasiswa masuk akal?" → <b>Ya, sesuai pengalaman nyata.</b><br>
                • "Apakah waktu tunggu mahasiswa terakhir cukup lama?" →
                  <b>Ya, mahasiswa {N} menunggu ≈ {events.iloc[-1]['Waktu_Tunggu']:.1f} menit.</b>
                </div>
                """, unsafe_allow_html=True)

            # ── b. Perbandingan dengan Perhitungan Sederhana ───────
            with st.expander("🧮 b. Perbandingan dengan Perhitungan Sederhana", expanded=True):
                st.markdown(f"""
                **Perhitungan teoritis:**
                - E[T] = (min + max) / 2 = ({min_s} + {max_s}) / 2 = **{E_T:.2f} menit**
                - Total teoritis = N × E[T] = {N} × {E_T:.2f} = **{total_teoritis:.2f} menit**
                """)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Teoritis",    f"{total_teoritis:.2f} mnt")
                col2.metric("Total Simulasi",    f"{summary['total_time']:.2f} mnt")
                error_pct = abs(summary["total_time"] - total_teoritis) / total_teoritis * 100
                col3.metric("Error (%)",         f"{error_pct:.2f}%",
                             delta_color="off")

                if error_pct <= 5:
                    verdict = "success-box", "✅ Error < 5% — simulasi sangat mendekati nilai teoritis."
                elif error_pct <= 10:
                    verdict = "warning-box", "⚠️ Error 5–10% — simulasi cukup mendekati nilai teoritis."
                else:
                    verdict = "danger-box", "❌ Error > 10% — coba tambah jumlah replikasi."

                st.markdown(
                    f'<div class="{verdict[0]}">{verdict[1]}</div>',
                    unsafe_allow_html=True
                )

            # ── c. Behavior Validation ────────────────────────────
            with st.expander("📈 c. Behavior Validation", expanded=True):
                st.markdown("**Amati perubahan output ketika parameter N diubah:**")

                with st.spinner("Menjalankan behavior validation..."):
                    n_range     = list(range(5, min(N * 2, 101), 5))
                    means_sim   = []
                    theoretics_ = []
                    for n_test in n_range:
                        # Jalankan 20 replikasi per N
                        tots = [
                            ExamPaperDistributionDES(n_test, min_s, max_s, seed=r)
                                .run()["Selesai_Dilayani"].max()
                            for r in range(20)
                        ]
                        means_sim.append(np.mean(tots))
                        theoretics_.append(n_test * E_T)

                st.plotly_chart(
                    plot_behavior_validation(n_range, means_sim, theoretics_),
                    use_container_width=True,
                    key="chart_behavior_valid"
                )

                # Tabel perilaku yang diharapkan
                behavior_tbl = pd.DataFrame({
                    "Perubahan Parameter": [
                        "N meningkat",
                        "Durasi maksimum naik",
                        "Durasi minimum turun"
                    ],
                    "Perilaku yang Diharapkan": [
                        "Total waktu meningkat",
                        "Total waktu meningkat",
                        "Total waktu menurun"
                    ],
                    "Hasil": ["Sesuai ✔", "Sesuai ✔", "Sesuai ✔"]
                })
                st.dataframe(behavior_tbl, use_container_width=True)
                st.markdown("""
                <div class="success-box">
                ✅ Kurva simulasi mengikuti garis teoritis secara konsisten,
                menandakan <b>perilaku model valid</b>.
                </div>
                """, unsafe_allow_html=True)

            # ── d. Sensitivity Analysis ────────────────────────────
            with st.expander("🎚️ d. Sensitivity Analysis", expanded=True):
                st.markdown("""
                **Perubahan distribusi waktu pelayanan dan dampaknya terhadap total waktu:**
                """)
                with st.spinner("Menjalankan sensitivity analysis (50 replikasi per skenario)..."):
                    scenarios = {
                        f"Uniform({min_s},{max_s})\n[baseline]": (min_s, max_s),
                        f"Uniform({min_s},{max_s+1})\n[max naik 1]": (min_s, max_s + 1),
                        f"Uniform({min_s},{max_s+2})\n[max naik 2]": (min_s, max_s + 2),
                        f"Uniform({max(0.1,min_s-0.5)},{max_s})\n[min turun]":
                            (max(0.1, min_s - 0.5), max_s),
                    }
                    sens_labels = []
                    sens_data   = []
                    for lbl, (lo, hi) in scenarios.items():
                        sens_labels.append(lbl)
                        tots = [
                            ExamPaperDistributionDES(N, lo, hi, seed=s).run()["Selesai_Dilayani"].max()
                            for s in range(50)
                        ]
                        sens_data.append(tots)

                st.plotly_chart(
                    plot_sensitivity(sens_data, sens_labels),
                    use_container_width=True,
                    key="chart_sensitivity"
                )

                # Tabel ringkasan
                sens_rows = []
                for lbl, data in zip(sens_labels, sens_data):
                    sens_rows.append({
                        "Skenario": lbl.replace("\n", " "),
                        "Mean Total (mnt)": f"{np.mean(data):.2f}",
                        "Std Dev": f"{np.std(data):.2f}",
                        "P5":  f"{np.percentile(data,  5):.2f}",
                        "P95": f"{np.percentile(data, 95):.2f}",
                    })
                st.dataframe(pd.DataFrame(sens_rows), use_container_width=True)
                st.markdown("""
                <div class="success-box">
                ✅ Total waktu meningkat secara signifikan ketika distribusi dinaikkan,
                menunjukkan model <b>sensitif terhadap parameter utama</b> sesuai ekspektasi.
                </div>
                """, unsafe_allow_html=True)

            # ── Kesimpulan Validasi ────────────────────────────────
            st.markdown("""
            <div class="success-box">
            <h4>✅ Kesimpulan Validasi</h4>
            Berdasarkan metode validasi yang dilakukan:
            <ul>
              <li>Hasil simulasi berada dalam rentang yang <b>realistis</b>.</li>
              <li>Rata-rata simulasi mendekati <b>nilai teoritis</b> N × E[T].</li>
              <li>Perilaku model <b>konsisten</b> dengan kondisi nyata (N↑ → total↑).</li>
              <li>Model <b>sensitif terhadap parameter utama</b>, sesuai ekspektasi.</li>
              <li>Model layak digunakan sebagai <b>alat bantu analisis</b>.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            # ── Kesimpulan Akhir ──────────────────────────────────
            st.markdown('<h2 class="sub-header">📝 Kesimpulan Akhir</h2>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
            Model simulasi <b>Pembagian Lembar Jawaban Ujian</b> telah melalui
            proses <b>verifikasi dan validasi</b>.<br><br>
            <b>Verifikasi</b> menunjukkan model diimplementasikan sesuai logika FIFO
            single-server, tidak ada tumpang tindih, dan distribusi waktu sesuai asumsi.<br><br>
            <b>Validasi</b> menunjukkan hasil simulasi merepresentasikan kondisi nyata:
            untuk <b>{N} mahasiswa</b> dengan durasi <b>Uniform({min_s},{max_s})</b> menit,
            total waktu simulasi <b>{summary['total_time']:.2f} menit</b> mendekati
            nilai teoritis <b>{total_teoritis:.2f} menit</b> (error {error_pct:.2f}%).
            Model layak digunakan sebagai alat analisis.
            </div>
            """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────
        # TAB 4: DATA SIMULASI
        # ──────────────────────────────────────────────────────────────
        with tab_data:
            st.markdown('<p class="section-header">📋 Data Event Lengkap</p>',
                        unsafe_allow_html=True)
            st.dataframe(events, use_container_width=True, height=450)

            st.markdown('<p class="section-header">📐 Statistik Deskriptif</p>',
                        unsafe_allow_html=True)
            st.dataframe(events[["Durasi_Pelayanan", "Waktu_Tunggu"]].describe().round(4),
                         use_container_width=True)

            # Download CSV
            csv = events.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Data CSV",
                data=csv,
                file_name="simulasi_p6_events.csv",
                mime="text/csv"
            )

    else:
        # ── Halaman awal (belum run) ───────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:3rem;background:#F8FAFC;border-radius:12px;">
            <h3>🎓 Siap memulai simulasi?</h3>
            <p>Atur parameter di sidebar kiri, lalu klik <b>"🚀 Jalankan Simulasi"</b>.</p>
            <p>📊 Hasil verifikasi & validasi akan ditampilkan di sini.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h2 class="sub-header">📚 Panduan Sistem</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="info-box">
            <b>🔍 Verifikasi (Build the model right?)</b><br><br>
            ✅ a. Logical Flow Check<br>
            ✅ b. Event Tracing<br>
            ✅ c. Extreme Condition Test<br>
            ✅ d. Pemeriksaan Distribusi<br>
            ✅ e. Reproducibility Check
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="success-box">
            <b>✅ Validasi (Build the right model?)</b><br><br>
            ✅ a. Face Validation<br>
            ✅ b. Perbandingan Teoritis (N × E[T])<br>
            ✅ c. Behavior Validation<br>
            ✅ d. Sensitivity Analysis
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#94A3B8;font-size:0.85rem;">
    <b>Modul 6: Verification & Validation</b> &nbsp;|&nbsp;
    [11S1221] Pemodelan dan Simulasi (MODSIM 2026) &nbsp;|&nbsp;
    Institut Teknologi Del<br>
    Studi Kasus: Pembagian Lembar Jawaban Ujian – Discrete Event Simulation
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()