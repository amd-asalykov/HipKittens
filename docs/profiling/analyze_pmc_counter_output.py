#!/usr/bin/env python3
# Analyze rocprofv3 split runs (profiles_1/2/3/4) and compute memory-bound signals.

import os, glob
import numpy as np
import pandas as pd

# ==== Inputs: expect four separate runs ======================================
INPUTS = [
    "out/profiles_1_counter_collection.csv",
    "out/profiles_2_counter_collection.csv",
    "out/profiles_3_counter_collection.csv",
    "out/profiles_4_counter_collection.csv",
    "profiles_1_counter_collection.csv",
    "profiles_2_counter_collection.csv",
    "profiles_3_counter_collection.csv",
    "profiles_4_counter_collection.csv",
    "out/profiles_1*counter*csv",
    "out/profiles_2*counter*csv",
    "out/profiles_3*counter*csv",
    "out/profiles_4*counter*csv",
]
OUTDIR = "analysis_out"

# Hard-coded kernel regex -> label
KERNEL_MAP = {
    r"tk_fused_rotary": "TK kernel",
    r"kn_entry_1c_sbhd_cached": "AITer kernel",
}

# Column name compatibility
COLUMN_ALIASES = {
    "Counter_Name": ["Counter_Name", "CounterName"],
    "Counter_Value": ["Counter_Value", "CounterValue", "Value"],
    "Kernel_Name": ["Kernel_Name", "KernelName", "Name"],
    "Dispatch_Id": ["Dispatch_Id", "DispatchID", "DispatchId", "Dispatch Id"],
}

# ---------- utils ----------
def pick_inputs():
    paths = []
    for pat in INPUTS:
        if os.path.exists(pat):
            paths.append(pat)
        else:
            paths.extend(glob.glob(pat))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("profiles_{1,2,3,4}_counter_collection.csv not found.")
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
    if not dfs:
        raise FileNotFoundError("No readable CSVs.")
    return pd.concat(dfs, ignore_index=True)

def coalesce_columns(df):
    ren = {}
    for canon, alts in COLUMN_ALIASES.items():
        if canon in df.columns:
            continue
        for a in alts:
            if a in df.columns:
                ren[a] = canon
                break
    if ren:
        df = df.rename(columns=ren)
    miss = [c for c in ["Counter_Name","Counter_Value","Kernel_Name","Dispatch_Id"] if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")
    return df

def safe_ratio(n, d):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = n.astype(float) / d.astype(float)
    return pd.Series(r).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def qstats(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if s.size == 0:
        return dict(mean=0.0, p50=0.0, p90=0.0)
    return dict(mean=float(s.mean()), p50=float(s.quantile(0.5)), p90=float(s.quantile(0.9)))

# ---------- main ----------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    df = pick_inputs()
    df = coalesce_columns(df)

    # Pivot: one row per dispatch/kernel, all counters as columns
    pivot = (df.pivot_table(index=["Dispatch_Id","Kernel_Name"],
                            columns="Counter_Name",
                            values="Counter_Value",
                            aggfunc="first")
               .reset_index()
               .rename_axis(None, axis=1))
    for c in pivot.columns:
        if c not in ("Dispatch_Id","Kernel_Name"):
            pivot[c] = pd.to_numeric(pivot[c], errors="coerce")

    # ===== Derived metrics (only uses counters from your final commands) =====
    # L1i / L1d quality
    pivot["Icache_Hit_Rate"]  = safe_ratio(pivot.get("SQC_ICACHE_HITS",0),  pivot.get("SQC_ICACHE_REQ",1))
    pivot["Icache_Miss_Rate"] = safe_ratio(pivot.get("SQC_ICACHE_MISSES",0),pivot.get("SQC_ICACHE_REQ",1))
    pivot["Icache_MPKI"]      = 1000.0 * safe_ratio(pivot.get("SQC_ICACHE_MISSES",0), pivot.get("SQ_INSTS",1))

    pivot["Scache_Hit_Rate"]  = safe_ratio(pivot.get("SQC_DCACHE_HITS",0),  pivot.get("SQC_DCACHE_REQ",1))
    pivot["Scache_Miss_Rate"] = safe_ratio(pivot.get("SQC_DCACHE_MISSES",0),pivot.get("SQC_DCACHE_REQ",1))

    # Instruction mix shares
    total_insts = pivot.get("SQ_INSTS",1)
    pivot["Frac_INSTS_VALU"] = safe_ratio(pivot.get("SQ_INSTS_VALU",0), total_insts)
    pivot["Frac_INSTS_SALU"] = safe_ratio(pivot.get("SQ_INSTS_SALU",0), total_insts)
    pivot["Frac_INSTS_SMEM"] = safe_ratio(pivot.get("SQ_INSTS_SMEM",0), total_insts)
    pivot["Frac_INSTS_VMEM"] = safe_ratio(pivot.get("SQ_INSTS_VMEM",0), total_insts)

    # Activity / waves
    pivot["SQ_Busy_Ratio"] = safe_ratio(pivot.get("SQ_BUSY_CYCLES",0), pivot.get("SQ_CYCLES",1))
    pivot["Waves"]         = pivot.get("SQ_WAVES",0).astype(float)
    pivot["SPI_Waves"]     = pivot.get("SPI_CSN_WAVE",0).astype(float)

    # Dispatch granularity
    pivot["Workgroups"]   = pivot.get("SPI_CSN_NUM_THREADGROUPS",0).astype(float)
    pivot["Waves_per_WG"] = safe_ratio(pivot.get("SPI_CSN_WAVE",0), pivot.get("SPI_CSN_NUM_THREADGROUPS",1))

    # LDS conflicts
    pivot["LDS_Conflict_Ratio"] = safe_ratio(pivot.get("SQ_LDS_BANK_CONFLICT",0), pivot.get("SQ_INSTS_LDS",1))

    # VMEM “bound” signals (active vs any; and wait-any)
    pivot["Frac_Active_VMEM"] = safe_ratio(pivot.get("SQ_ACTIVE_INST_VMEM",0), pivot.get("SQ_ACTIVE_INST_ANY",1))
    # If SQ_WAVE_CYCLES is absent, fall back to SQ_CYCLES for the wait denominator
    pivot["Frac_Wait_Any"]    = safe_ratio(pivot.get("SQ_WAIT_INST_ANY",0), pivot.get("SQ_WAVE_CYCLES", pivot.get("SQ_CYCLES",1)))

    # L2 (from *_sum counters)
    pivot["L2_Hit_Rate"]   = safe_ratio(pivot.get("TCC_HIT_sum",0),  pivot.get("TCC_REQ_sum",1))
    pivot["L2_Miss_Rate"]  = safe_ratio(pivot.get("TCC_MISS_sum",0), pivot.get("TCC_REQ_sum",1))
    pivot["L2_Atomic_Frac"] = safe_ratio(pivot.get("TCC_ATOMIC_sum",0), pivot.get("TCC_REQ_sum",1))

    # Vector L1 totals (mix, not hit rate)
    pivot["vL1_Read_Frac"]  = safe_ratio(pivot.get("TCP_TOTAL_READ_sum",0),  pivot.get("TCP_TOTAL_ACCESSES_sum",1))
    pivot["vL1_Write_Frac"] = safe_ratio(pivot.get("TCP_TOTAL_WRITE_sum",0), pivot.get("TCP_TOTAL_ACCESSES_sum",1))

    # ===== Print per-kernel headline stats & save CSVs ========================
    headline = [
        "Icache_Hit_Rate","Icache_Miss_Rate","Icache_MPKI",
        "Scache_Hit_Rate","Scache_Miss_Rate",
        "Frac_INSTS_VMEM","Frac_Active_VMEM","Frac_Wait_Any",
        "L2_Hit_Rate","L2_Miss_Rate","L2_Atomic_Frac",
        "vL1_Read_Frac","vL1_Write_Frac",
        "LDS_Conflict_Ratio","SQ_Busy_Ratio","Waves_per_WG"
    ]

    rows, per_rows = [], []
    for pat, label in KERNEL_MAP.items():
        sub = pivot[pivot["Kernel_Name"].str.contains(pat, regex=True, na=False)].copy()
        print(f"\n====== Analyzing {label} ======")
        if sub.empty:
            print("No data.")
            rows.append({"Kernel_Label": label, "Dispatches": 0})
            continue

        print(f"Dispatches: {len(sub)}")
        sub["Kernel_Label"] = label
        per_rows.append(sub)

        # Print concise stats
        for m in headline:
            if m in sub.columns and sub[m].notna().any():
                s = qstats(sub[m])
                print(f"{m}: mean {s['mean']:.3f} | p50 {s['p50']:.3f} | p90 {s['p90']:.3f}")

        # Build summary row
        row = {"Kernel_Label": label, "Dispatches": int(len(sub))}
        for m in headline:
            s = qstats(sub[m]) if m in sub.columns else dict(mean=0.0,p50=0.0,p90=0.0)
            row[f"{m}_mean"] = s["mean"]; row[f"{m}_p50"] = s["p50"]; row[f"{m}_p90"] = s["p90"]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("Kernel_Label").reset_index(drop=True)
    per = pd.concat(per_rows, ignore_index=True) if per_rows else pivot.assign(Kernel_Label="(unmatched)")

    # Columns to keep in per-dispatch CSV
    keep = ["Kernel_Label","Dispatch_Id","Kernel_Name"] + headline + [
        # raw counters (only those you collect in your final commands)
        "SQC_ICACHE_REQ","SQC_ICACHE_HITS","SQC_ICACHE_MISSES",
        "SQC_DCACHE_REQ","SQC_DCACHE_HITS","SQC_DCACHE_MISSES",
        "SQ_INSTS","SQ_INSTS_VALU","SQ_INSTS_SALU","SQ_INSTS_SMEM","SQ_INSTS_VMEM",
        "SQ_ACTIVE_INST_ANY","SQ_ACTIVE_INST_VMEM","SQ_WAIT_INST_ANY","SQ_WAVE_CYCLES","SQ_CYCLES",
        "SQ_WAVES","SPI_CSN_WAVE","SPI_CSN_NUM_THREADGROUPS",
        "SQ_LDS_BANK_CONFLICT","SQ_INSTS_LDS",
        "TCC_REQ_sum","TCC_HIT_sum","TCC_MISS_sum","TCC_ATOMIC_sum",
        "TCP_TOTAL_ACCESSES_sum","TCP_TOTAL_READ_sum","TCP_TOTAL_WRITE_sum",
    ]
    keep = [c for c in keep if c in per.columns]

    os.makedirs(OUTDIR, exist_ok=True)
    summary.to_csv(os.path.join(OUTDIR, "kernel_summary.csv"), index=False)
    per[keep].sort_values(["Kernel_Label","Dispatch_Id"]).to_csv(
        os.path.join(OUTDIR, "per_dispatch_metrics.csv"), index=False
    )

    print("\nWrote:", os.path.join(OUTDIR, "kernel_summary.csv"))
    print("Wrote:", os.path.join(OUTDIR, "per_dispatch_metrics.csv"))

if __name__ == "__main__":
    main()
