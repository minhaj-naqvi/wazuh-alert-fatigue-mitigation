#!/usr/bin/env python3
import os
import json
import re
import hashlib
import pandas as pd
import joblib

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "/var/ossec/ml/priv_esc_sysmon_model.joblib" 
STATE_PATH = "/var/ossec/ml/ml_triage_state.txt"           
ALERTS_PATH = "/var/ossec/logs/alerts/alerts.json"         
OUT_PATH = "/var/ossec/logs/alerts/alerts_ml.jsonl"        

# -----------------------------
# Masking regexes (same logic as training)
# -----------------------------
IPV4 = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
IPV6 = re.compile(r'\b(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}\b')
PATH_WIN = re.compile(r'[A-Za-z]:\\\\[^\s"]+')
HOST = re.compile(r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
GUID = re.compile(r'\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?')
SID = re.compile(r'S-1-\d+-\d+(?:-\d+)+')
USER_GENERAL = re.compile(r'\b[a-zA-Z][a-zA-Z0-9_\-]{2,20}\b') 

def mask_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = IPV4.sub("<IP4>", s)
    s = IPV6.sub("<IP6>", s)
    s = PATH_WIN.sub("<PATH>", s)
    s = HOST.sub("<HOST>", s)
    s = GUID.sub("<GUID>", s)
    s = SID.sub("<USER>", s)
    s = USER_GENERAL.sub("<USER>", s)
    return s.lower().strip()

# -----------------------------
# Helpers for Wazuh Sysmon
# -----------------------------
def is_sysmon(row) -> bool:
    """
    Detect Wazuh-parsed Sysmon events: data.win.system + data.win.eventdata exist.
    """
    data = row.get("data", {})
    if not isinstance(data, dict):
        return False
    win = data.get("win", {})
    return isinstance(win, dict) and "system" in win and "eventdata" in win

def map_wazuh_sysmon(row):
    """
    Map a raw Wazuh Sysmon alert row into unified schema for inference.
    """
    data = row.get("data", {})
    win = data.get("win", {})
    system = win.get("system", {})
    ev = win.get("eventdata", {})

    agent = row.get("agent", {}) or {}
    rule = row.get("rule", {}) or {}

    ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
    computer = agent.get("name")

    event_id = system.get("EventID")
    image = ev.get("Image")
    command_line = ev.get("CommandLine")
    target_object = ev.get("TargetObject")
    message = data.get("message", "")  #human-readable message
    level = rule.get("level")

    return {
        "ts": ts,
        "computer": computer,
        "event_id": event_id,
        "image": image,
        "command_line": command_line,
        "target_object": target_object,
        "message": message,
        "level": level,
    }

def make_alert_uid(raw_alert: dict) -> str:
    """
    Build a stable hash ID from the original alert content.
    This lets you pivot/search in wazuh-alerts-* by a unique value.
    """
    try:
        s = json.dumps(raw_alert, sort_keys=True)
    except TypeError:
        s = str(raw_alert)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -----------------------------
# Main scoring logic
# -----------------------------
def main():
    clf = joblib.load(MODEL_PATH)

    # Read last file position
    last_pos = 0
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                last_pos = int(f.read().strip() or 0)
        except Exception:
            last_pos = 0

    # Read new lines from alerts.json
    new_rows = []
    new_pos = last_pos
    if not os.path.exists(ALERTS_PATH):
        print(f"[ml_triage] alerts.json not found at {ALERTS_PATH}")
        return

    with open(ALERTS_PATH, "r") as f:
        f.seek(last_pos)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                new_rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed line
                continue
        new_pos = f.tell()

    # Update state even if nothing to process
    with open(STATE_PATH, "w") as f:
        f.write(str(new_pos))

    if not new_rows:
        print("[ml_triage] No new alerts to process.")
        return

    df_raw = pd.DataFrame(new_rows)

    # Keep only Sysmon alerts
    df_sys = df_raw[df_raw.apply(is_sysmon, axis=1)].copy()
    if df_sys.empty:
        print("[ml_triage] No new Sysmon alerts among new rows.")
        return

    # Map to unified schema (for model features)
    df_feat = df_sys.apply(map_wazuh_sysmon, axis=1, result_type="expand")

    # Drop rows with invalid timestamps
    df_feat = df_feat[~df_feat["ts"].isna()]
    if df_feat.empty:
        print("[ml_triage] No valid timestamps after mapping.")
        return

    # Convert types
    df_feat["event_id"] = pd.to_numeric(df_feat["event_id"], errors="coerce").fillna(-1).astype(int)
    df_feat["level"] = pd.to_numeric(df_feat["level"], errors="coerce").fillna(0).astype(int)

    # Time features
    df_feat["hour"] = df_feat["ts"].dt.hour
    df_feat["weekday"] = df_feat["ts"].dt.dayofweek

    # Mask text fields for ML
    for col in ["image", "command_line", "target_object", "message"]:
        df_feat[col] = df_feat[col].fillna("").apply(mask_text)

    # Build combined text field (same as training)
    df_feat["text_all"] = (
        df_feat["image"].fillna("") + " " +
        df_feat["command_line"].fillna("") + " " +
        df_feat["target_object"].fillna("") + " " +
        df_feat["message"].fillna("")
    ).str.strip()

    # Prepare features for the model (same as training)
    feat_cols = ["text_all", "event_id", "level", "hour", "weekday"]
    X_new = df_feat[feat_cols]

    # Inference
    proba = clf.predict_proba(X_new)[:, 1]
    pred = clf.predict(X_new)

    df_feat["ml_score"] = proba
    df_feat["ml_label"] = pred

    # Align df_sys (raw) and df_feat (features) by index
    # so we can attach the original alert & UID
    df_sys = df_sys.loc[df_feat.index]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "a") as f:
        for idx in df_feat.index:
            rec_feat = df_feat.loc[idx]
            raw_alert = df_sys.loc[idx].to_dict()

            # Build unique UID from the original alert
            uid = make_alert_uid(raw_alert)

            out = {
                "ts": rec_feat["ts"].isoformat(),
                "computer": rec_feat["computer"],
                "event_id": int(rec_feat["event_id"]),
                "level": int(rec_feat["level"]),
                "hour": int(rec_feat["hour"]),
                "weekday": int(rec_feat["weekday"]),
                "ml_score": float(rec_feat["ml_score"]),
                "ml_label": int(rec_feat["ml_label"]),
                "alert_uid": uid,
                "original_alert": raw_alert,  # full unmasked Wazuh alert for analyst consumption
            }
            f.write(json.dumps(out) + "\n")

    print(f"[ml_triage] Processed {len(df_feat)} new Sysmon alerts. Saved to {OUT_PATH}.")


if __name__ == "__main__":
    main()