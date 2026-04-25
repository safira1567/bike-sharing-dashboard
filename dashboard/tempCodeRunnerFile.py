import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load & preprocess data ───────────────────────────────────────────────────
@st.cache_data
def load_data():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Data harian
    day = pd.read_csv(os.path.join(base_dir, "main_data.csv"), parse_dates=["dteday"])

    season_map  = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    weather_map = {
        1: "Clear/Partly Cloudy",
        2: "Mist/Cloudy",
        3: "Light Rain/Snow",
        4: "Heavy Rain/Snow"
    }
    yr_map      = {0: "2011", 1: "2012"}
    workday_map = {0: "Non-Working Day", 1: "Working Day"}

    day["season_label"] = pd.Categorical(
        day["season"].map(season_map),
        categories=["Spring", "Summer", "Fall", "Winter"], ordered=True)
    day["weather_label"] = pd.Categorical(
        day["weathersit"].map(weather_map),
        categories=["Clear/Partly Cloudy", "Mist/Cloudy",
                    "Light Rain/Snow", "Heavy Rain/Snow"], ordered=True)
    day["yr_label"]         = day["yr"].map(yr_map)
    day["workingday_label"] = day["workingday"].map(workday_map)

    p33 = day["cnt"].quantile(0.33)
    p66 = day["cnt"].quantile(0.66)
    day["demand_cluster"] = pd.cut(
        day["cnt"], bins=[-np.inf, p33, p66, np.inf],
        labels=["Low Demand", "Moderate Demand", "High Demand"])

    # Data per jam
    hour = pd.read_csv(os.path.join(base_dir, "hour_data.csv"), parse_dates=["dteday"])
    hour["season_label"] = pd.Categorical(
        hour["season"].map(season_map),
        categories=["Spring", "Summer", "Fall", "Winter"], ordered=True)
    hour["weather_label"] = pd.Categorical(
        hour["weathersit"].map(weather_map),
        categories=["Clear/Partly Cloudy", "Mist/Cloudy",
                    "Light Rain/Snow", "Heavy Rain/Snow"], ordered=True)
    hour["yr_label"]         = hour["yr"].map(yr_map)
    hour["workingday_label"] = hour["workingday"].map(workday_map)

    return day, hour

day_df, hour_df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(" Filter Data")

    year_opts = ["Semua"] + sorted(day_df["yr_label"].unique().tolist())
    sel_year  = st.selectbox("Tahun", year_opts)

    season_opts = ["Semua"] + day_df["season_label"].cat.categories.tolist()
    sel_season  = st.multiselect("Musim", season_opts, default=["Semua"])

    weather_opts = ["Semua"] + day_df["weather_label"].cat.categories.tolist()
    sel_weather  = st.multiselect("Kondisi Cuaca", weather_opts, default=["Semua"])

    st.markdown("---")
    st.caption("Proyek Analisis Data — Bike Sharing\nSafira Salsabila · 2025")


def apply_filters(df):
    d = df.copy()
    if sel_year != "Semua":
        d = d[d["yr_label"] == sel_year]
    if "Semua" not in sel_season and sel_season:
        d = d[d["season_label"].isin(sel_season)]
    if "Semua" not in sel_weather and sel_weather:
        d = d[d["weather_label"].isin(sel_weather)]
    return d


dff    = apply_filters(day_df)
hour_f = apply_filters(hour_df)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Bike Sharing — Dashboard Analisis Data")
st.markdown(
    "Dashboard ini merangkum hasil analisis penyewaan sepeda harian & per jam "
    "dari **Capital Bikeshare, Washington D.C.** periode **2011–2012**."
)

# ── KPI Metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Hari Dianalisis", f"{len(dff):,} hari")
c2.metric("Total Penyewaan",       f"{dff['cnt'].sum():,}")
c3.metric("Rata-rata/Hari",        f"{dff['cnt'].mean():,.0f}")
yoy = (
    (day_df[day_df['yr'] == 1]['cnt'].mean() - day_df[day_df['yr'] == 0]['cnt'].mean())
    / day_df[day_df['yr'] == 0]['cnt'].mean() * 100
)
c4.metric("Pertumbuhan YoY", f"{yoy:.1f}%")

st.markdown("---")

# ── Tab Layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    " Musim & Cuaca",
    " Segmentasi Pengguna",
    " Pola Jam (Hour Data)",
    " Demand Clustering"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Musim & Cuaca
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Pertanyaan 1: Pola Penyewaan Berdasarkan Musim & Kondisi Cuaca")
    st.info("Musim dan kondisi cuaca mana yang menghasilkan volume penyewaan tertinggi dan terendah?")

    if dff.empty:
        st.warning("Tidak ada data untuk filter yang dipilih.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            season_agg = (
                dff.groupby("season_label", observed=True)["cnt"]
                .agg(["mean", "sem"]).reset_index()
            )
            palette = {"Spring": "#4CAF50", "Summer": "#FF9800",
                       "Fall": "#E53935", "Winter": "#1565C0"}
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(
                season_agg["season_label"].astype(str), season_agg["mean"],
                color=[palette.get(s, "#999") for s in season_agg["season_label"].astype(str)],
                width=0.55, edgecolor="white",
                yerr=season_agg["sem"] * 1.96, capsize=5,
                error_kw={"elinewidth": 1.4, "ecolor": "#555"}
            )
            for bar, val in zip(bars, season_agg["mean"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
                        f"{val:,.0f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
            ax.set_title("Rata-rata Penyewaan per Musim\n(error bar = 95% CI)",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Musim")
            ax.set_ylabel("Rata-rata cnt")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            weather_agg = (
                dff.groupby("weather_label", observed=True)["cnt"]
                .agg(["mean", "sem", "count"]).reset_index()
            )
            wpal = {
                "Clear/Partly Cloudy": "#FDD835",
                "Mist/Cloudy":         "#90A4AE",
                "Light Rain/Snow":     "#42A5F5",
                "Heavy Rain/Snow":     "#1565C0"
            }
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(
                weather_agg["weather_label"].astype(str), weather_agg["mean"],
                color=[wpal.get(w, "#BDBDBD") for w in weather_agg["weather_label"].astype(str)],
                height=0.5, edgecolor="white",
                xerr=weather_agg["sem"] * 1.96, capsize=4,
                error_kw={"elinewidth": 1.4, "ecolor": "#555"}
            )
            for i, row in weather_agg.iterrows():
                ax.text(row["mean"] + 80, i,
                        f"{row['mean']:,.0f} (n={row['count']})",
                        va="center", fontsize=8)
            ax.set_title("Rata-rata Penyewaan per Kondisi Cuaca\n(error bar = 95% CI)",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Rata-rata cnt")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Heatmap: Musim × Kondisi Cuaca")
        pivot_heat = dff.pivot_table(
            values="cnt", index="season_label",
            columns="weather_label", aggfunc="mean", observed=True)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        sns.heatmap(pivot_heat, annot=True, fmt=".0f", cmap="YlOrRd",
                    linewidths=0.5, linecolor="white", ax=ax,
                    cbar_kws={"label": "Rata-rata cnt", "shrink": 0.8})
        ax.set_title("Heatmap Rata-rata Penyewaan: Musim × Cuaca",
                     fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.success("**Kesimpulan:** Fall + Clear adalah kondisi optimal. "
                   "Spring + hujan menghasilkan permintaan terendah.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Segmentasi Pengguna
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Pertanyaan 2: Casual vs Registered — Hari Kerja vs Non-Kerja")
    st.info("Apakah pola pengguna kasual dan terdaftar berbeda secara signifikan "
            "antara hari kerja dan non-kerja?")

    if dff.empty:
        st.warning("Tidak ada data untuk filter yang dipilih.")
    else:
        user_agg = dff.groupby("workingday_label")[["casual", "registered", "cnt"]].mean()
        cats = user_agg.index.tolist()
        x = np.arange(len(cats))
        w = 0.35

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            b1 = ax.bar(x - w / 2, user_agg["casual"],     w,
                        label="Casual",     color="#FF7043", edgecolor="white", alpha=0.9)
            b2 = ax.bar(x + w / 2, user_agg["registered"], w,
                        label="Registered", color="#1976D2", edgecolor="white", alpha=0.9)
            for bar, v in list(zip(b1, user_agg["casual"])) + list(zip(b2, user_agg["registered"])):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                        f"{v:,.0f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(cats, fontsize=9)
            ax.set_ylabel("Rata-rata Pengguna/Hari")
            ax.legend(title="Tipe Pengguna", fontsize=8)
            ax.set_title("Rata-rata Casual vs Registered\nper Tipe Hari",
                         fontsize=10, fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            pct_c = (user_agg["casual"]     / user_agg["cnt"] * 100).values
            pct_r = (user_agg["registered"] / user_agg["cnt"] * 100).values
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(cats, pct_c, label="Casual",     color="#FF7043", edgecolor="white", alpha=0.9)
            ax.bar(cats, pct_r, label="Registered", color="#1976D2", edgecolor="white",
                   alpha=0.9, bottom=pct_c)
            for i, (vc, vr) in enumerate(zip(pct_c, pct_r)):
                ax.text(i, vc / 2,      f"{vc:.1f}%",
                        ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold")
                ax.text(i, vc + vr / 2, f"{vr:.1f}%",
                        ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold")
            ax.set_ylabel("Proporsi (%)")
            ax.legend(title="Tipe Pengguna", fontsize=8)
            ax.set_title("Proporsi Casual vs Registered\nper Tipe Hari",
                         fontsize=10, fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Mann-Whitney test
        casual_wd  = dff[dff["workingday"] == 1]["casual"]
        casual_nwd = dff[dff["workingday"] == 0]["casual"]
        reg_wd     = dff[dff["workingday"] == 1]["registered"]
        reg_nwd    = dff[dff["workingday"] == 0]["registered"]
        if len(casual_wd) > 0 and len(casual_nwd) > 0:
            _, p_c = stats.mannwhitneyu(casual_wd, casual_nwd, alternative="two-sided")
            _, p_r = stats.mannwhitneyu(reg_wd,    reg_nwd,    alternative="two-sided")
            col_a, col_b = st.columns(2)
            col_a.metric("Mann-Whitney p (Casual)",
                         f"{p_c:.4f}",
                         delta="Signifikan" if p_c < 0.05 else "Tidak Signifikan")
            col_b.metric("Mann-Whitney p (Registered)",
                         f"{p_r:.4f}",
                         delta="Signifikan" if p_r < 0.05 else "Tidak Signifikan")

        st.success("**Kesimpulan:** Registered ~80% di hari kerja (komuter); "
                   "Casual ~30% di non-kerja (rekreasi). "
                   "Perbedaan signifikan secara statistik.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Pola Jam (Hour Data)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Pertanyaan 3: Pola Jam Sibuk (Data Per Jam)")
    st.info("Jam berapa permintaan puncak terjadi? "
            "Apakah pola hari kerja berbeda dengan weekend?")

    hourly_stats = hour_f.groupby(["workingday", "hr"])["cnt"].mean().reset_index()
    hourly_stats["day_type"] = hourly_stats["workingday"].map(
        {1: "Hari Kerja", 0: "Non-Kerja/Weekend"})

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = {"Hari Kerja": "#1565C0", "Non-Kerja/Weekend": "#E53935"}
    markers = {"Hari Kerja": "o",       "Non-Kerja/Weekend": "s"}

    for day_type, grp in hourly_stats.groupby("day_type"):
        ax.plot(grp["hr"], grp["cnt"],
                marker=markers[day_type], linewidth=2.2,
                color=colors[day_type], label=day_type,
                markersize=5, markerfacecolor="white", markeredgewidth=2)

    # Anotasi puncak hari kerja
    wd_data = hourly_stats[hourly_stats["day_type"] == "Hari Kerja"]
    if not wd_data.empty:
        peak1 = wd_data.loc[wd_data["cnt"].idxmax()]
        ax.annotate(
            f"Puncak Komuter\n{int(peak1['hr']):02d}:00 ({peak1['cnt']:.0f})",
            xy=(peak1["hr"], peak1["cnt"]),
            xytext=(peak1["hr"] - 3.5, peak1["cnt"] + 25),
            arrowprops=dict(arrowstyle="->", color="#1565C0"),
            fontsize=8.5, color="#1565C0", fontweight="bold"
        )

    # Anotasi puncak non-kerja
    nwd_data = hourly_stats[hourly_stats["day_type"] == "Non-Kerja/Weekend"]
    if not nwd_data.empty:
        peak2 = nwd_data.loc[nwd_data["cnt"].idxmax()]
        ax.annotate(
            f"Puncak Rekreasi\n{int(peak2['hr']):02d}:00 ({peak2['cnt']:.0f})",
            xy=(peak2["hr"], peak2["cnt"]),
            xytext=(peak2["hr"] + 1, peak2["cnt"] + 25),
            arrowprops=dict(arrowstyle="->", color="#E53935"),
            fontsize=8.5, color="#E53935", fontweight="bold"
        )

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, fontsize=7.5)
    ax.set_title("Rata-rata Penyewaan per Jam: Hari Kerja vs Non-Kerja/Weekend",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Jam")
    ax.set_ylabel("Rata-rata cnt")
    ax.legend(title="Tipe Hari", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Heatmap jam × hari dalam seminggu
    st.subheader("Heatmap: Jam × Hari dalam Seminggu")
    hour_f2 = hour_f.copy()
    hour_f2["weekday_label"] = hour_f2["weekday"].map(
        {0: "Sen", 1: "Sel", 2: "Rab", 3: "Kam", 4: "Jum", 5: "Sab", 6: "Min"})
    pivot_hr = hour_f2.pivot_table(
        values="cnt", index="hr", columns="weekday_label", aggfunc="mean")
    day_order = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
    pivot_hr  = pivot_hr[[d for d in day_order if d in pivot_hr.columns]]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(pivot_hr, cmap="RdYlGn", linewidths=0.3, linecolor="white",
                ax=ax, cbar_kws={"label": "Rata-rata cnt", "shrink": 0.8}, annot=False)
    ax.set_title("Heatmap Rata-rata Penyewaan: Jam × Hari",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Hari")
    ax.set_ylabel("Jam")
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.success("**Kesimpulan:** Hari kerja: puncak bimodal 08:00 & 17:00 (komuter). "
               "Weekend: puncak unimodal 11:00–14:00 (rekreasi).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Demand Clustering
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Analisis Lanjutan: Demand Clustering Manual")
    st.markdown(
        "Hari diklasifikasikan ke dalam tiga klaster berdasarkan persentil ke-33 dan ke-66 "
        "dari distribusi total penyewaan (`cnt`), sebagai pendekatan segmentasi berbasis permintaan."
    )

    if dff.empty:
        st.warning("Tidak ada data untuk filter yang dipilih.")
    else:
        cluster_summary = dff.groupby("demand_cluster", observed=True).agg(
            N_Hari=("cnt", "count"),
            Rata_rata_cnt=("cnt", "mean"),
            Median_cnt=("cnt", "median"),
            Rata_rata_temp=("temp", "mean"),
            Pct_WorkingDay=("workingday", "mean")
        ).round(3)
        cluster_summary["Pct_WorkingDay (%)"] = (cluster_summary["Pct_WorkingDay"] * 100).round(1)
        cluster_summary.drop(columns="Pct_WorkingDay", inplace=True)
        st.dataframe(
            cluster_summary.style.background_gradient(cmap="RdYlGn"),
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            sc = (
                dff.groupby(["demand_cluster", "season_label"], observed=True)
                .size().unstack(fill_value=0)
            )
            sc_pct = sc.div(sc.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            sc_pct.plot(kind="bar", stacked=True, ax=ax,
                        color=["#4CAF50", "#FF9800", "#E53935", "#1565C0"],
                        edgecolor="white", linewidth=0.5)
            ax.set_title("Komposisi Musim per Klaster", fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Proporsi (%)")
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="Musim", bbox_to_anchor=(1.01, 1),
                      loc="upper left", fontsize=7)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            cluster_order = ["Low Demand", "Moderate Demand", "High Demand"]
            data_cls = [
                dff[dff["demand_cluster"] == c]["cnt"].dropna().values
                for c in cluster_order
            ]
            fig, ax = plt.subplots(figsize=(6, 4))
            bp = ax.boxplot(data_cls, labels=cluster_order, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 2})
            for patch, col in zip(bp["boxes"], ["#EF5350", "#FFA726", "#66BB6A"]):
                patch.set_facecolor(col)
                patch.set_alpha(0.8)
            ax.set_title("Distribusi cnt per Klaster", fontsize=10, fontweight="bold")
            ax.set_ylabel("Jumlah Penyewaan (cnt)")
            ax.tick_params(axis="x", rotation=15)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.success("**Kesimpulan:** High Demand dominan di Fall & Summer dengan % hari kerja tinggi. "
                   "Low Demand identik dengan Spring & Winter.")