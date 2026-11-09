import os
import re
import json
import math
import time
import hashlib
import hmac
from datetime import datetime, timezone, timedelta

import numpy as np
import requests
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, no_update
from dash.exceptions import PreventUpdate

try:
    from sgp4.api import Satrec, jday
    SGP4_OK = True
except Exception:
    SGP4_OK = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from skyfield.api import Loader
    SKYFIELD_OK = True
except Exception:
    SKYFIELD_OK = False


# ============================================================
# DEEP SPACE COMMUNICATION SIMULATOR
# Dynamic CelesTrak satellite catalog + SGP4 propagation
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")
SAT_CACHE_FILE = os.path.join(DATA_DIR, "satellite_catalog.json")
HISTORY_FILE = os.path.join(DATA_DIR, "mission_history.json")

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php"
HEADERS = {
    "User-Agent": "DeepSpaceCommunicationSimulator/2.0 (academic research dashboard)"
}

C = 299792.458  # km/s
EARTH_RADIUS = 6378.137

# Public GP groups. The application combines groups and removes duplicates.
SATELLITE_GROUPS = [
    ("Active Satellites", "active"),
    ("Starlink", "starlink"),
    ("OneWeb", "oneweb"),
    ("GPS", "gps-ops"),
    ("GLONASS", "glo-ops"),
    ("Galileo", "galileo"),
    ("Beidou", "beidou"),
    ("Iridium", "iridium"),
    ("Globalstar", "globalstar"),
    ("Weather", "weather"),
    ("NOAA", "noaa"),
    ("GOES", "goes"),
    ("Earth Resources", "resource"),
    ("Space Stations", "stations"),
]

# Ground stations used by the simulator.
GROUND_STATIONS = {
    "Byalalu, India": (13.4020, 77.4880),
    "Goldstone, USA": (35.4266, -116.8900),
    "Madrid, Spain": (40.4314, -4.2486),
    "Canberra, Australia": (-35.4014, 148.9817),
}

# A small set of known public satellites is used only as a first
# selection. Their positions are NEVER hard-coded; positions come
# from their fetched GP/TLE data.
DEFAULT_SATELLITES = [
    "ISS (ZARYA)",
    "HST",
    "CSS (TIANHE)",
    "STARLINK-1000",
]


# ============================================================
# FILE / AUTH HELPERS
# ============================================================

def hash_password(password):
    """PBKDF2 password hash; stores salt and digest in one string."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 240_000)
    return "pbkdf2_sha256$" + salt.hex() + "$" + digest.hex()


def verify_password(password, stored):
    """Verify current PBKDF2 hashes and upgrade-compatible legacy SHA-256 hashes."""
    if not stored:
        return False
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, salt_hex, digest_hex = stored.split("$", 2)
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 240_000)
            return hmac.compare_digest(actual.hex(), digest_hex)
        except (ValueError, TypeError):
            return False
    # Compatibility with accounts created by the earlier project version.
    return hmac.compare_digest(hashlib.sha256(password.encode("utf-8")).hexdigest(), stored)


def get_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Repair the old list format automatically.
        if isinstance(data, list):
            converted = {}
            for item in data:
                if isinstance(item, dict):
                    username = str(item.get("username", "")).strip().lower()
                    if username:
                        converted[username] = item
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(converted, f, indent=2)
            return converted
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_user(username, password):
    users = get_users()
    username = username.strip().lower()
    users[username] = {
        "username": username,
        "password": hash_password(password),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def authenticate(username, password):
    users = get_users()
    username = username.strip().lower()
    user = users.get(username)
    return bool(user and verify_password(password, user.get("password", "")))


def reset_password(username, new_password):
    users = get_users()
    username = username.strip().lower()
    if username not in users:
        return False
    users[username]["password"] = hash_password(new_password)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    return True


# ============================================================
# MISSION HISTORY
# ============================================================

def read_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def add_history(event, details):
    rows = read_history()
    rows.insert(0, {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "event": event,
        "details": details,
    })
    rows = rows[:100]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


# ============================================================
# CelesTrak DYNAMIC CATALOG
# ============================================================

def parse_tle_text(text):
    """Parse 3-line legacy TLE records."""
    lines = [x.rstrip() for x in text.splitlines() if x.strip()]
    result = []
    i = 0
    while i < len(lines):
        if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name = lines[i].strip()
            l1 = lines[i + 1].strip()
            l2 = lines[i + 2].strip()
            result.append({"name": name, "line1": l1, "line2": l2})
            i += 3
        else:
            i += 1
    return result


def fetch_group(group):
    params = {"GROUP": group, "FORMAT": "TLE"}
    try:
        r = requests.get(CELESTRAK_GP, params=params, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            return parse_tle_text(r.text)
    except Exception:
        pass
    return []


def load_satellite_catalog(force=False):
    """
    Downloads current public GP/TLE groups from CelesTrak.
    Failed groups are skipped. Existing cache remains usable.
    No satellite position is fabricated.
    """
    cache = {}
    if os.path.exists(SAT_CACHE_FILE) and not force:
        try:
            with open(SAT_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if isinstance(cache, dict) and cache:
                # Refresh periodically.
                stamp = cache.get("_meta", {}).get("timestamp", 0)
                if time.time() - stamp < 1800:
                    return cache.get("satellites", [])
        except Exception:
            pass

    combined = {}
    for label, group in SATELLITE_GROUPS:
        rows = fetch_group(group)
        for row in rows:
            name = row["name"].strip()
            key = (name, row["line1"], row["line2"])
            combined[key] = {
                **row,
                "group": label,
                "source": "CelesTrak GP/TLE",
                "fetched_utc": datetime.now(timezone.utc).isoformat(),
            }

    satellites = list(combined.values())

    # Preserve an existing successful cache if the network is temporarily unavailable.
    if not satellites and os.path.exists(SAT_CACHE_FILE):
        try:
            with open(SAT_CACHE_FILE, "r", encoding="utf-8") as f:
                old = json.load(f)
            return old.get("satellites", [])
        except Exception:
            return []

    with open(SAT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "timestamp": time.time(),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "source": "CelesTrak",
            },
            "satellites": satellites,
        }, f)

    return satellites


SATELLITES = load_satellite_catalog()
SATELLITE_MAP = {s["name"]: s for s in SATELLITES}

# If CelesTrak was unavailable at startup, the UI still starts and can refresh.
if not SATELLITES:
    SATELLITE_MAP = {}


def extract_norad(line1):
    try:
        return int(line1[2:7])
    except Exception:
        return None


def satellite_options():
    names = sorted(SATELLITE_MAP.keys(), key=str.upper)
    return [{"label": f"{n}  •  NORAD {extract_norad(SATELLITE_MAP[n]['line1']) or '—'}",
             "value": n} for n in names]


# ============================================================
# SGP4 PROPAGATION
# ============================================================

def satellite_object(record):
    if not SGP4_OK:
        return None
    try:
        return Satrec.twoline2rv(record["line1"], record["line2"])
    except Exception:
        return None


def propagate_record(record, dt=None):
    """Return TEME-like position/velocity from the actual SGP4 record."""
    if not SGP4_OK:
        return None

    sat = satellite_object(record)
    if sat is None:
        return None

    dt = dt or datetime.now(timezone.utc)
    jd, fr = jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
    )
    error, position, velocity = sat.sgp4(jd, fr)
    if error != 0:
        return None

    return np.array(position, dtype=float), np.array(velocity, dtype=float)


def gmst_from_datetime(dt):
    # Standard approximate GMST calculation for converting propagated
    # Earth-fixed longitude for visualization.
    unix = dt.timestamp()
    jd = unix / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T*T
        - T*T*T / 38710000.0
    )
    return math.radians(gmst % 360)


def eci_to_geodetic(position_km, dt):
    x, y, z = position_km
    r = math.sqrt(x*x + y*y + z*z)
    lat = math.degrees(math.asin(z / r))
    lon_eci = math.atan2(y, x)
    lon = math.degrees(lon_eci - gmst_from_datetime(dt))
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    alt = r - EARTH_RADIUS
    return lat, lon, alt


def orbit_trail(record, minutes=90, points=180):
    times = [
        datetime.now(timezone.utc) + timedelta(
            minutes=(-minutes / 2) + (minutes * i / max(points - 1, 1))
        )
        for i in range(points)
    ]
    lats, lons, alts = [], [], []
    valid = 0
    for dt in times:
        out = propagate_record(record, dt)
        if out is None:
            lats.append(None)
            lons.append(None)
            alts.append(None)
        else:
            p, _ = out
            lat, lon, alt = eci_to_geodetic(p, dt)
            lats.append(lat)
            lons.append(lon)
            alts.append(alt)
            valid += 1
    return times, lats, lons, alts, valid


# ============================================================
# EARTH 3D
# ============================================================

def sphere_xyz(radius=EARTH_RADIUS, n=50):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def geo_xyz(lat, lon, radius=EARTH_RADIUS):
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return (
        radius * np.cos(lat_r) * np.cos(lon_r),
        radius * np.cos(lat_r) * np.sin(lon_r),
        radius * np.sin(lat_r),
    )


def make_earth_figure(record=None, station_name="Byalalu, India"):
    fig = go.Figure()
    x, y, z = sphere_xyz()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#061b2e"], [0.45, "#0b4f71"], [1, "#1e9ad6"]],
        showscale=False,
        opacity=0.98,
        hoverinfo="skip",
        name="Earth",
    ))

    # Ground stations.
    gx, gy, gz = [], [], []
    for name, (lat, lon) in GROUND_STATIONS.items():
        xx, yy, zz = geo_xyz(lat, lon, EARTH_RADIUS + 35)
        gx.append(xx); gy.append(yy); gz.append(zz)
    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz,
        mode="markers+text",
        text=list(GROUND_STATIONS.keys()),
        textposition="top center",
        marker=dict(size=6, color="#00e5ff"),
        name="Ground Stations",
    ))

    if record and SGP4_OK:
        now = datetime.now(timezone.utc)
        current = propagate_record(record, now)
        if current:
            pos, vel = current
            altitude = np.linalg.norm(pos) - EARTH_RADIUS
            scale = (EARTH_RADIUS + max(altitude, 100)) / np.linalg.norm(pos)
            sx, sy, sz = pos * scale

            fig.add_trace(go.Scatter3d(
                x=[sx], y=[sy], z=[sz],
                mode="markers+text",
                text=[record["name"]],
                textposition="top center",
                marker=dict(size=9, color="#ffffff"),
                name="Live Satellite",
                customdata=[[extract_norad(record["line1"])]],
                hovertemplate="NORAD %{customdata[0]}<extra></extra>",
            ))

            # SGP4 orbit trail.
            _, lats, lons, alts, valid = orbit_trail(record)
            ox, oy, oz = [], [], []
            for lat, lon, alt in zip(lats, lons, alts):
                if lat is None:
                    continue
                xx, yy, zz = geo_xyz(lat, lon, EARTH_RADIUS + max(alt, 100))
                ox.append(xx); oy.append(yy); oz.append(zz)
            if ox:
                fig.add_trace(go.Scatter3d(
                    x=ox, y=oy, z=oz,
                    mode="lines",
                    line=dict(color="#00ff99", width=3),
                    name=f"SGP4 Orbit Trail ({valid} points)",
                ))

            # Communication line and wave markers.
            stlat, stlon = GROUND_STATIONS.get(station_name, GROUND_STATIONS["Byalalu, India"])
            a = np.array(geo_xyz(stlat, stlon, EARTH_RADIUS + 35))
            b = np.array([sx, sy, sz])
            fig.add_trace(go.Scatter3d(
                x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                mode="lines",
                line=dict(color="#ffcc00", width=5),
                name="Communication Link",
            ))
            for t in np.linspace(0.15, 0.85, 5):
                q = a * (1-t) + b * t
                fig.add_trace(go.Scatter3d(
                    x=[q[0]], y=[q[1]], z=[q[2]],
                    mode="markers",
                    marker=dict(size=5, color="#ffcc00"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    fig.update_layout(
        paper_bgcolor="#030914",
        plot_bgcolor="#030914",
        font=dict(color="#dff7ff"),
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#030914",
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.55, z=0.95)),
        ),
        legend=dict(bgcolor="rgba(0,0,0,.45)", orientation="h"),
    )
    return fig


# ============================================================
# JPL EPHEMERIS / SOLAR SYSTEM
# ============================================================

EPHEMERIS = None
EPHEMERIS_STATUS = "JPL ephemeris not loaded"

def load_jpl_ephemeris():
    global EPHEMERIS, EPHEMERIS_STATUS
    if not SKYFIELD_OK:
        EPHEMERIS_STATUS = "Skyfield not installed"
        return
    try:
        loader = Loader(os.path.join(DATA_DIR, "skyfield"))
        EPHEMERIS = loader("de440s.bsp")
        EPHEMERIS_STATUS = "JPL DE440s loaded"
    except Exception as e:
        EPHEMERIS_STATUS = f"JPL ephemeris unavailable: {type(e).__name__}"

load_jpl_ephemeris()


def earth_mars_distance_km(when=None):
    """Return Earth-centre to Mars-barycentre distance from the loaded JPL kernel."""
    if EPHEMERIS is None:
        return None
    try:
        when = when or datetime.now(timezone.utc)
        t = EPHEMERIS.ts.from_datetime(when)
        earth = EPHEMERIS["earth"].at(t).position.km
        mars = EPHEMERIS["mars barycenter"].at(t).position.km
        return float(np.linalg.norm(np.asarray(mars) - np.asarray(earth)))
    except Exception:
        return None


def make_solar_figure():
    fig = go.Figure()
    if EPHEMERIS is None:
        fig.add_annotation(
            text="JPL DE440s ephemeris unavailable. Install Skyfield and allow the kernel to download.",
            x=.5, y=.5, showarrow=False, font=dict(size=15, color="#ffcc00")
        )
        fig.update_layout(paper_bgcolor="#030914", plot_bgcolor="#030914")
        return fig

    try:
        ts = EPHEMERIS.ts
        t = ts.from_datetime(datetime.now(timezone.utc))
        sun = EPHEMERIS["sun"]
        earth = EPHEMERIS["earth"]
        mars = EPHEMERIS["mars barycenter"]
        planets = [
            ("Mercury", EPHEMERIS["mercury"]),
            ("Venus", EPHEMERIS["venus"]),
            ("Earth", earth),
            ("Mars", mars),
            ("Jupiter", EPHEMERIS["jupiter barycenter"]),
            ("Saturn", EPHEMERIS["saturn barycenter"]),
        ]

        # Plot in AU using heliocentric positions at current UTC.
        au_km = 149597870.7
        for name, body in planets:
            p = body.at(t).observe(sun).apparent().position.km
            # observe() here is body relative to observer; use difference via vectors.
            # Recalculate robustly as body barycentric - sun barycentric.
            pb = body.at(t).position.km
            ps = sun.at(t).position.km
            rel = (pb - ps) / au_km
            fig.add_trace(go.Scatter3d(
                x=[rel[0]], y=[rel[1]], z=[rel[2]],
                mode="markers+text",
                text=[name],
                textposition="top center",
                marker=dict(size=9 if name == "Earth" else 7),
                name=name,
            ))

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            text=["Sun"],
            textposition="top center",
            marker=dict(size=15),
            name="Sun",
        ))
    except Exception as e:
        fig.add_annotation(text=f"Ephemeris rendering error: {e}", x=.5, y=.5, showarrow=False)

    fig.update_layout(
        paper_bgcolor="#030914",
        plot_bgcolor="#030914",
        font=dict(color="#dff7ff"),
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            bgcolor="#030914",
            xaxis=dict(title="AU", gridcolor="#183148"),
            yaxis=dict(title="AU", gridcolor="#183148"),
            zaxis=dict(title="AU", gridcolor="#183148"),
            aspectmode="data",
        ),
        legend=dict(bgcolor="rgba(0,0,0,.4)"),
    )
    return fig


# ============================================================
# COMMUNICATION MODEL
# ============================================================

def link_metrics(distance_km):
    frequency_hz = 2.2e9
    tx_dbm = 20.0
    antenna_gain_db = 24.0
    system_loss_db = 3.0
    attenuation_db = 2.0
    noise_dbm = -100.0
    bandwidth_hz = 1e6

    fspl = 20 * math.log10(max(distance_km, 1)) + 20 * math.log10(frequency_hz) - 147.55
    rx_dbm = tx_dbm + antenna_gain_db - fspl - system_loss_db - attenuation_db
    snr_db = rx_dbm - noise_dbm
    snr_linear = 10 ** (snr_db / 10)
    ber = 0.5 * math.erfc(math.sqrt(max(snr_linear, 0)))
    packet_loss = 1 - (1 - min(max(ber, 0), 0.5)) ** 8000
    packet_loss = min(max(packet_loss, 0), 1)
    throughput = bandwidth_hz * max(0, 1 - packet_loss)

    return {
        "distance": distance_km,
        "delay": distance_km / C,
        "fspl": fspl,
        "rx": rx_dbm,
        "snr": snr_db,
        "ber": ber,
        "packet_loss": packet_loss,
        "throughput": throughput,
        "attenuation": attenuation_db,
        "noise": noise_dbm,
    }


def get_satellite_metrics(record, station_name):
    if not record or not SGP4_OK:
        return None
    now = datetime.now(timezone.utc)
    out = propagate_record(record, now)
    if not out:
        return None
    pos, vel = out
    lat, lon, alt = eci_to_geodetic(pos, now)
    slat, slon = GROUND_STATIONS.get(station_name, GROUND_STATIONS["Byalalu, India"])
    gs = np.array(geo_xyz(slat, slon, EARTH_RADIUS))
    # Visualization ECI position is approximate Earth-fixed conversion for the link geometry.
    r = np.linalg.norm(pos)
    sat_scaled = pos / r * (EARTH_RADIUS + alt)
    distance = float(np.linalg.norm(sat_scaled - gs))
    m = link_metrics(distance)
    speed = float(np.linalg.norm(vel))
    return {
        **m,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "velocity": speed,
        "norad": extract_norad(record["line1"]),
        "epoch": record["line1"][18:32] if len(record["line1"]) >= 32 else "—",
    }


def make_metrics_figure(distance_km):
    distances = np.linspace(max(100, distance_km * 0.5), distance_km * 1.5, 60)
    losses = [link_metrics(float(d))["packet_loss"] * 100 for d in distances]
    delays = [link_metrics(float(d))["delay"] for d in distances]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=distances, y=losses,
        mode="lines",
        name="Packet Loss %",
        line=dict(color="#ff4d6d", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=distances, y=[d * 1000 for d in delays],
        mode="lines",
        name="One-way Delay ms",
        line=dict(color="#00e5ff", width=3),
        yaxis="y2",
    ))
    fig.update_layout(
        paper_bgcolor="#030914",
        plot_bgcolor="#030914",
        font=dict(color="#dff7ff"),
        margin=dict(l=55, r=55, t=20, b=45),
        xaxis=dict(title="Range (km)", gridcolor="#163047"),
        yaxis=dict(title="Packet Loss (%)", gridcolor="#163047"),
        yaxis2=dict(title="Delay (ms)", overlaying="y", side="right"),
        legend=dict(bgcolor="rgba(0,0,0,.3)"),
    )
    return fig


ML_EVALUATION = {"split": "80% train / 20% test", "mae": None, "r2": None, "samples": 0}
_ML_MODEL = None
_ML_MODEL_READY = False


def train_ml_model():
    """Train an educational RandomForest regressor on analytical link-model samples.

    The dataset is synthetic/model-derived, not historical spacecraft telemetry.
    The 80/20 split is reported transparently in the UI.
    """
    global _ML_MODEL, _ML_MODEL_READY, ML_EVALUATION
    xs = np.linspace(500.0, 800000.0, 400).reshape(-1, 1)
    ys = np.array([link_metrics(float(x))["packet_loss"] * 100.0 for x in xs[:, 0]])
    ML_EVALUATION = {"split": "80% train / 20% test", "mae": None, "r2": None, "samples": int(len(xs))}
    if SKLEARN_OK:
        x_train, x_test, y_train, y_test = train_test_split(
            xs, ys, test_size=0.20, random_state=42
        )
        model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=1)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        ML_EVALUATION["mae"] = float(mean_absolute_error(y_test, predictions))
        ML_EVALUATION["r2"] = float(r2_score(y_test, predictions))
        _ML_MODEL = model
        _ML_MODEL_READY = True
    return ML_EVALUATION


def ml_prediction(distance_km):
    global _ML_MODEL, _ML_MODEL_READY
    if not _ML_MODEL_READY:
        train_ml_model()
    if SKLEARN_OK and _ML_MODEL is not None:
        pred = float(_ML_MODEL.predict(np.array([[float(distance_km)]]))[0])
    else:
        pred = float(link_metrics(distance_km)["packet_loss"] * 100)
    return max(0.0, min(100.0, pred))


train_ml_model()


# ============================================================
# COMMUNICATION WINDOW PREDICTION + PER-SATELLITE ANALYSIS
# ============================================================

_WINDOW_CACHE = {}
_WINDOW_CACHE_TTL = 60


def satellite_elevation_deg(record, station_name, when):
    """Approximate topocentric elevation from SGP4 TEME state and Earth rotation."""
    propagated = propagate_record(record, when)
    if propagated is None:
        return None
    pos_teme, _ = propagated
    theta = gmst_from_datetime(when)
    cth, sth = math.cos(theta), math.sin(theta)
    # TEME to Earth-fixed rotation approximation for educational pass prediction.
    sat_ecef = np.array([cth * pos_teme[0] + sth * pos_teme[1],
                         -sth * pos_teme[0] + cth * pos_teme[1], pos_teme[2]], dtype=float)
    lat_deg, lon_deg = GROUND_STATIONS.get(station_name, GROUND_STATIONS["Byalalu, India"])
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    r = EARTH_RADIUS
    gs = np.array([r * math.cos(lat) * math.cos(lon),
                   r * math.cos(lat) * math.sin(lon), r * math.sin(lat)], dtype=float)
    los = sat_ecef - gs
    norm = float(np.linalg.norm(los))
    if norm <= 0:
        return None
    up = gs / float(np.linalg.norm(gs))
    return math.degrees(math.asin(float(np.clip(np.dot(los, up) / norm, -1.0, 1.0))))


def predict_communication_windows(record, station_name, hours=6, step_seconds=120, min_elevation_deg=10):
    """Predict approximate upcoming visibility windows from the selected TLE/SGP4 record."""
    if not record or not SGP4_OK:
        return []
    cache_key = (record.get("name"), station_name, int(hours), int(step_seconds))
    now_ts = time.time()
    cached = _WINDOW_CACHE.get(cache_key)
    if cached and now_ts - cached[0] < _WINDOW_CACHE_TTL:
        return cached[1]
    now = datetime.now(timezone.utc)
    samples = []
    count = int(hours * 3600 / step_seconds)
    for i in range(count + 1):
        when = now + timedelta(seconds=i * step_seconds)
        elev = satellite_elevation_deg(record, station_name, when)
        samples.append((when, elev if elev is not None else -90.0))
    windows, open_start = [], None
    previous = None
    for when, elev in samples:
        visible = elev >= min_elevation_deg
        if visible and open_start is None:
            open_start = when
        elif not visible and open_start is not None:
            windows.append({"start": open_start, "end": when, "duration_min": (when-open_start).total_seconds()/60})
            open_start = None
        previous = visible
    if open_start is not None:
        windows.append({"start": open_start, "end": samples[-1][0], "duration_min": (samples[-1][0]-open_start).total_seconds()/60})
    _WINDOW_CACHE[cache_key] = (now_ts, windows)
    return windows


def satellite_analysis_text(record, station_name, metrics):
    """Three transparent steps for the selected satellite."""
    if not record or not metrics:
        return "Select a satellite with a valid live CelesTrak TLE to run analysis."
    quality = "GOOD" if metrics["snr"] >= 12 and metrics["packet_loss"] < 0.05 else ("MODERATE" if metrics["snr"] >= 3 and metrics["packet_loss"] < 0.25 else "POOR")
    windows = predict_communication_windows(record, station_name, hours=6)
    next_window = "No pass above 10° elevation found in the next 6 hours at the current sampling interval."
    if windows:
        w = windows[0]
        next_window = f"{w['start'].strftime('%H:%M:%S UTC')}–{w['end'].strftime('%H:%M:%S UTC')} (about {w['duration_min']:.0f} min)"
    return html.Div([
        html.Div([html.B("STEP 1 — ORBIT STATE"), html.Div(f"Live TLE + SGP4 propagation; altitude {metrics['alt']:.1f} km, speed {metrics['velocity']:.3f} km/s.")], className="analysis-step"),
        html.Div([html.B("STEP 2 — LINK BUDGET"), html.Div(f"Range {metrics['distance']:,.1f} km; one-way light time {metrics['delay']:.3f} s; SNR {metrics['snr']:.2f} dB; BER {metrics['ber']:.3e}; estimated packet loss {metrics['packet_loss']*100:.4f}%.")], className="analysis-step"),
        html.Div([html.B("STEP 3 — LINK DECISION"), html.Div(f"{quality} link under the configured educational thresholds. Ground station: {station_name}. Next approximate visibility window: {next_window}")], className="analysis-step"),
        html.Div("Window estimates use a 10° elevation mask, current TLE and approximate TEME-to-Earth-fixed rotation; validate against a mission-grade orbit tool before operational use.", className="analysis-note")
    ])


# ============================================================
# DTN
# ============================================================

DTN_STORE = []


def dtn_store(packet, node):
    DTN_STORE.append({
        "packet": packet,
        "node": node,
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "status": "STORED",
    })


# ============================================================
# CHATBOT
# ============================================================

def chatbot_answer(text):
    q = text.lower().strip()
    if not q:
        return "Mission AI online. Ask about satellites, SGP4, CelesTrak, delay, BER, DTN, or Earth–Mars communication."
    if "sgp4" in q or "propagat" in q:
        return "SGP4 propagates a satellite's orbit from its current two-line orbital elements. This dashboard obtains GP/TLE data from CelesTrak and calculates the propagated state instead of using hard-coded satellite positions."
    if "celestrak" in q:
        return "CelesTrak is the public orbital-data source used by this simulator. The catalog is refreshed and cached locally so the dashboard can continue operating when a temporary request fails."
    if "mars" in q or "earth" in q:
        return "The planetary view uses JPL DE440s through Skyfield when available. Earth–Mars communication metrics are calculated from the current ephemeris distance and the link model."
    if "dtn" in q:
        return "DTN uses store-and-forward behavior: a packet can be stored at an intermediate node and forwarded when the next communication opportunity is available."
    if "ber" in q or "packet" in q:
        return "BER is derived from the modeled SNR, and packet loss is estimated from BER across an 8,000-bit packet in the current educational link model."
    if "satellite" in q:
        return f"The catalog currently contains {len(SATELLITE_MAP):,} loaded public GP records. Use the satellite search box to select a satellite and propagate it."
    return "Mission AI: I can explain the satellite catalog, SGP4 propagation, ground-station links, JPL ephemeris, communication metrics, packet flow, and DTN."


# ============================================================
# UI HELPERS
# ============================================================

def metric_card(label, value, unit=""):
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div([
            html.Span(value, className="metric-value"),
            html.Span(unit, className="metric-unit"),
        ])
    ], className="metric-card")


def empty_figure(message="Waiting for mission data..."):
    fig = go.Figure()
    fig.add_annotation(text=message, x=.5, y=.5, showarrow=False,
                       font=dict(size=16, color="#8ba9bb"))
    fig.update_layout(paper_bgcolor="#030914", plot_bgcolor="#030914",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ============================================================
# DASH APP
# ============================================================

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Deep Space Communication Simulator",
    update_title="Mission Control updating..."
)

app.layout = html.Div([
    dcc.Store(id="session", data={"logged_in": False, "username": ""}),
    dcc.Store(id="auth-mode", data="login"),
    dcc.Interval(id="live-interval", interval=5000, n_intervals=0),

    # ---------------- AUTH ----------------
    html.Div(id="auth-container", className="auth-shell", children=[
        html.Div(className="auth-stars"),
        html.Div(className="auth-card", children=[
            html.Div("DEEP SPACE", className="auth-kicker"),
            html.H1("COMMUNICATION SIMULATOR", className="auth-title"),
            html.Div("MISSION CONTROL ACCESS", className="auth-subtitle"),

            html.Div(id="auth-message", className="auth-message"),

            html.Div(id="login-form", children=[
                html.Label("USERNAME"),
                dcc.Input(id="login-username", type="text", placeholder="Enter username", className="auth-input"),
                html.Label("PASSWORD"),
                dcc.Input(id="login-password", type="password", placeholder="Enter password", className="auth-input"),
                html.Button("LOGIN", id="login-button", className="auth-button"),
                html.Button("CREATE ACCOUNT", id="show-signup", className="auth-link-button"),
                html.Button("FORGOT PASSWORD", id="show-reset", className="auth-link-button"),
            ]),

            html.Div(id="signup-form", style={"display": "none"}, children=[
                html.Label("USERNAME"),
                dcc.Input(id="signup-username", type="text", placeholder="Choose username", className="auth-input"),
                html.Label("PASSWORD"),
                dcc.Input(id="signup-password", type="password", placeholder="Create password", className="auth-input"),
                html.Label("CONFIRM PASSWORD"),
                dcc.Input(id="signup-confirm", type="password", placeholder="Confirm password", className="auth-input"),
                html.Button("SIGN UP", id="signup-button", className="auth-button"),
                html.Button("BACK TO LOGIN", id="back-login", className="auth-link-button"),
            ]),

            html.Div(id="reset-form", style={"display": "none"}, children=[
                html.Label("USERNAME"),
                dcc.Input(id="reset-username", type="text", placeholder="Username", className="auth-input"),
                html.Label("NEW PASSWORD"),
                dcc.Input(id="reset-password", type="password", placeholder="New password", className="auth-input"),
                html.Label("CONFIRM PASSWORD"),
                dcc.Input(id="reset-confirm", type="password", placeholder="Confirm password", className="auth-input"),
                html.Button("CHANGE PASSWORD", id="reset-button", className="auth-button"),
                html.Button("BACK TO LOGIN", id="reset-back-login", className="auth-link-button"),
            ]),

            html.Div("SGP4 • CelesTrak • JPL EPHEMERIS • DTN", className="auth-footer")
        ])
    ]),

    # ---------------- DASHBOARD ----------------
    html.Div(id="dashboard-container", className="dashboard-shell", style={"display": "none"}, children=[
        html.Div(className="topbar", children=[
            html.Div(className="brand", children=[
                html.Div("◉", className="brand-mark"),
                html.Div([
                    html.Div("DEEP SPACE", className="brand-small"),
                    html.Div("COMMUNICATION SIMULATOR", className="brand-main"),
                ])
            ]),
            html.Div(className="mission-clock", children=[
                html.Div(id="utc-clock", children="UTC --:--:--"),
                html.Div("MISSION CONTROL ONLINE", className="online")
            ]),
            html.Button("LOGOUT", id="logout-button", className="logout-button")
        ]),

        html.Div(className="status-strip", children=[
            html.Div(["●  CelesTrak: ", html.Span(id="tle-status")]),
            html.Div(["●  Propagation: ", html.Span(id="prop-status")]),
            html.Div(["●  Planetary: ", html.Span(id="planet-status")]),
            html.Div(["●  Catalog: ", html.Span(id="catalog-count")]),
        ]),

        html.Div(className="control-bar", children=[
            html.Div(className="control-item search-control", children=[
                html.Label("SATELLITE CATALOG"),
                dcc.Dropdown(
                    id="satellite-select",
                    options=satellite_options(),
                    value=(DEFAULT_SATELLITES[0] if DEFAULT_SATELLITES[0] in SATELLITE_MAP else
                           (next(iter(SATELLITE_MAP)) if SATELLITE_MAP else None)),
                    searchable=True,
                    clearable=False,
                    placeholder="Search any satellite / NORAD object...",
                    className="satellite-dropdown",
                )
            ]),
            html.Div(className="control-item", children=[
                html.Label("GROUND STATION"),
                dcc.Dropdown(
                    id="station-select",
                    options=[{"label": x, "value": x} for x in GROUND_STATIONS],
                    value="Byalalu, India",
                    clearable=False,
                    className="satellite-dropdown",
                )
            ]),
            html.Div(className="control-item mini-control", children=[
                html.Label("REFRESH"),
                html.Button("REFRESH CATALOG", id="refresh-catalog", className="refresh-button")
            ])
        ]),

        html.Div(className="hero-grid", children=[
            html.Div(className="panel earth-panel", children=[
                html.Div(className="panel-head", children=[
                    html.Div([
                        html.Div("ORBITAL COMMAND", className="panel-kicker"),
                        html.H2("LIVE SATELLITE TRACKING"),
                    ]),
                    html.Div(id="satellite-live-badge", className="live-badge")
                ]),
                dcc.Graph(id="earth-graph", config={"displayModeBar": False}, className="big-graph"),
                html.Div(id="satellite-details", className="sat-details")
            ]),
            html.Div(className="panel solar-panel", children=[
                html.Div(className="panel-head", children=[
                    html.Div([
                        html.Div("PLANETARY NAVIGATION", className="panel-kicker"),
                        html.H2("JPL EPHEMERIS"),
                    ]),
                    html.Div("DE440s", className="kernel-badge")
                ]),
                dcc.Graph(id="solar-graph", config={"displayModeBar": False}, className="solar-graph"),
                html.Div(id="planet-detail", className="planet-detail")
            ])
        ]),

        html.Div(id="metric-row", className="metric-row", children=[
            metric_card("ALTITUDE", "—", "km"),
            metric_card("VELOCITY", "—", "km/s"),
            metric_card("RANGE", "—", "km"),
            metric_card("ONE-WAY DELAY", "—", "ms"),
            metric_card("SNR", "—", "dB"),
            metric_card("PACKET LOSS", "—", "%"),
            metric_card("BER", "—", ""),
            metric_card("THROUGHPUT", "—", "Mbps"),
        ]),

        html.Div(className="panel analysis-panel", children=[
            html.Div(className="panel-head", children=[
                html.Div([html.Div("PER-SATELLITE DECISION SUPPORT", className="panel-kicker"),
                          html.H2("THREE-STEP LINK ANALYSIS & COMMUNICATION WINDOWS")])
            ]),
            html.Div(id="communication-analysis", className="communication-analysis"),
            html.Div(id="communication-windows", className="communication-windows")
        ]),

        html.Div(className="analytics-grid", children=[
            html.Div(className="panel analytics-panel", children=[
                html.Div(className="panel-head", children=[
                    html.Div([
                        html.Div("LINK ANALYTICS", className="panel-kicker"),
                        html.H2("RANGE / DELAY / PACKET LOSS")
                    ])
                ]),
                dcc.Graph(id="metrics-graph", config={"displayModeBar": False}, className="analytics-graph")
            ]),
            html.Div(className="panel telemetry-panel", children=[
                html.Div(className="panel-head", children=[
                    html.Div([
                        html.Div("TELEMETRY", className="panel-kicker"),
                        html.H2("PROPAGATED STATE")
                    ])
                ]),
                html.Pre(id="telemetry-output", className="telemetry-box")
            ])
        ]),

        html.Div(className="modules-grid", children=[
            html.Div(className="panel module-panel", children=[
                html.Div("PACKET COMMUNICATION", className="panel-kicker"),
                html.H2("MISSION PACKET"),
                dcc.Input(id="packet-input", type="text", value="DEEP SPACE TEST PACKET",
                          className="module-input"),
                html.Button("TRANSMIT PACKET", id="send-packet", className="module-button"),
                html.Div(id="packet-result", className="module-result"),
                html.Div(id="receiver-status", className="receiver-box"),
            ]),

            html.Div(className="panel module-panel", children=[
                html.Div("DELAY TOLERANT NETWORKING", className="panel-kicker"),
                html.H2("DTN STORE & FORWARD"),
                dcc.Dropdown(
                    id="dtn-node",
                    options=[
                        {"label": "EARTH GROUND NODE", "value": "EARTH"},
                        {"label": "LUNAR RELAY NODE", "value": "MOON"},
                        {"label": "DEEP SPACE RELAY", "value": "DSR"},
                        {"label": "MARS RELAY NODE", "value": "MARS"},
                    ],
                    value="EARTH",
                    clearable=False,
                    className="satellite-dropdown",
                ),
                html.Div(className="two-buttons", children=[
                    html.Button("STORE", id="store-packet", className="module-button"),
                    html.Button("FORWARD", id="forward-packet", className="module-button"),
                ]),
                html.Div(id="dtn-result", className="module-result"),
            ]),

            html.Div(className="panel module-panel", children=[
                html.Div("MISSION AI", className="panel-kicker"),
                html.H2("AI COMMUNICATION ASSISTANT"),
                html.Div(id="chat-output", className="chat-output",
                         children="Mission AI online. Ask about the satellite catalog, SGP4, communication metrics, or DTN."),
                dcc.Input(id="chat-input", type="text", placeholder="Ask Mission AI...",
                          className="module-input"),
                html.Button("ASK AI", id="chat-button", className="module-button"),
                html.Div(id="ml-prediction", className="prediction-box"),
            ])
        ]),

        html.Div(className="panel history-panel", children=[
            html.Div(className="panel-head", children=[
                html.Div([
                    html.Div("MISSION LOG", className="panel-kicker"),
                    html.H2("MISSION HISTORY")
                ])
            ]),
            html.Div(id="history-output")
        ]),

        html.Div(className="footer-bar", children=[
            "DEEP SPACE COMMUNICATION SIMULATOR  •  CelesTrak GP/TLE  •  SGP4  •  JPL DE440s  •  Educational Simulation"
        ])
    ])
])


# ============================================================
# AUTH / ROUTING CALLBACK
# ============================================================

@app.callback(
    Output("session", "data"),
    Output("auth-mode", "data"),
    Output("auth-message", "children"),
    Output("login-form", "style"),
    Output("signup-form", "style"),
    Output("reset-form", "style"),
    Input("login-button", "n_clicks"),
    Input("signup-button", "n_clicks"),
    Input("reset-button", "n_clicks"),
    Input("show-signup", "n_clicks"),
    Input("show-reset", "n_clicks"),
    Input("back-login", "n_clicks"),
    Input("reset-back-login", "n_clicks"),
    State("login-username", "value"),
    State("login-password", "value"),
    State("signup-username", "value"),
    State("signup-password", "value"),
    State("signup-confirm", "value"),
    State("reset-username", "value"),
    State("reset-password", "value"),
    State("reset-confirm", "value"),
    prevent_initial_call=True
)
def auth_controller(login_n, signup_n, reset_n, show_signup, show_reset, back_login,
                    reset_back, lu, lp, su, sp, sc, ru, rp, rc):
    from dash import ctx
    trigger = ctx.triggered_id

    login_style = {"display": "block"}
    signup_style = {"display": "none"}
    reset_style = {"display": "none"}

    if trigger == "show-signup":
        return no_update, "signup", "", {"display": "none"}, {"display": "block"}, {"display": "none"}
    if trigger == "show-reset":
        return no_update, "reset", "", {"display": "none"}, {"display": "none"}, {"display": "block"}
    if trigger in ("back-login", "reset-back-login"):
        return no_update, "login", "", login_style, signup_style, reset_style

    if trigger == "login-button":
        if not lu or not lp:
            return no_update, "login", "Enter username and password.", login_style, signup_style, reset_style
        if authenticate(lu, lp):
            add_history("LOGIN", f"User {lu.strip()} authenticated")
            return {"logged_in": True, "username": lu.strip()}, "login", "", login_style, signup_style, reset_style
        return no_update, "login", "Invalid username or password.", login_style, signup_style, reset_style

    if trigger == "signup-button":
        su = (su or "").strip()
        if len(su) < 3:
            return no_update, "signup", "Username must contain at least 3 characters.", {"display":"none"}, {"display":"block"}, {"display":"none"}
        if not sp or len(sp) < 4:
            return no_update, "signup", "Password must contain at least 4 characters.", {"display":"none"}, {"display":"block"}, {"display":"none"}
        if sp != sc:
            return no_update, "signup", "Passwords do not match.", {"display":"none"}, {"display":"block"}, {"display":"none"}
        if su.lower() in get_users():
            return no_update, "signup", "Username already exists.", {"display":"none"}, {"display":"block"}, {"display":"none"}
        save_user(su, sp)
        add_history("SIGNUP", f"New mission-control user {su}")
        # Important: first signup immediately opens the dashboard.
        return {"logged_in": True, "username": su}, "login", "", login_style, signup_style, reset_style

    if trigger == "reset-button":
        ru = (ru or "").strip()
        if not ru or not rp:
            return no_update, "reset", "Enter all password-reset fields.", {"display":"none"}, {"display":"none"}, {"display":"block"}
        if rp != rc:
            return no_update, "reset", "Passwords do not match.", {"display":"none"}, {"display":"none"}, {"display":"block"}
        if reset_password(ru, rp):
            return no_update, "login", "Password changed successfully. Please log in.", login_style, signup_style, reset_style
        return no_update, "reset", "Username not found.", {"display":"none"}, {"display":"none"}, {"display":"block"}

    raise PreventUpdate


@app.callback(
    Output("auth-container", "style"),
    Output("dashboard-container", "style"),
    Input("session", "data")
)
def route_ui(session):
    if session and session.get("logged_in"):
        return {"display": "none"}, {"display": "block"}
    return {"display": "flex"}, {"display": "none"}


@app.callback(
    Output("session", "data", allow_duplicate=True),
    Input("logout-button", "n_clicks"),
    prevent_initial_call=True
)
def logout(n):
    if n:
        return {"logged_in": False, "username": ""}
    raise PreventUpdate


# ============================================================
# DASHBOARD LIVE CALLBACK
# ============================================================

@app.callback(
    Output("earth-graph", "figure"),
    Output("solar-graph", "figure"),
    Output("metrics-graph", "figure"),
    Output("tle-status", "children"),
    Output("prop-status", "children"),
    Output("planet-status", "children"),
    Output("catalog-count", "children"),
    Output("utc-clock", "children"),
    Output("satellite-live-badge", "children"),
    Output("satellite-details", "children"),
    Output("planet-detail", "children"),
    Output("metric-row", "children"),
    Output("telemetry-output", "children"),
    Output("ml-prediction", "children"),
    Input("live-interval", "n_intervals"),
    Input("satellite-select", "value"),
    Input("station-select", "value"),
    Input("refresh-catalog", "n_clicks"),
    Input("session", "data"),
)
def update_dashboard(n, selected, station, refresh, session):
    if not session or not session.get("logged_in"):
        return (
            empty_figure(), empty_figure(), empty_figure(),
            "OFFLINE", "WAITING", "WAITING", "0",
            "UTC --:--:--", "OFFLINE", "", "", [], "", ""
        )

    global SATELLITES, SATELLITE_MAP
    if refresh:
        SATELLITES = load_satellite_catalog(force=True)
        SATELLITE_MAP = {s["name"]: s for s in SATELLITES}

    now = datetime.now(timezone.utc)
    clock = now.strftime("UTC %H:%M:%S")
    catalog = f"{len(SATELLITE_MAP):,}"

    if not selected or selected not in SATELLITE_MAP:
        return (
            empty_figure("Select a satellite from the live CelesTrak catalog."),
            make_solar_figure(),
            empty_figure(),
            "CelesTrak catalog loaded" if SATELLITE_MAP else "CelesTrak unavailable",
            "No satellite selected",
            EPHEMERIS_STATUS,
            catalog,
            clock,
            "NO SATELLITE",
            "",
            "JPL planetary positions use ephemeris time.",
            [],
            "",
            "",
        )

    record = SATELLITE_MAP[selected]
    metrics = get_satellite_metrics(record, station)
    earth = make_earth_figure(record, station)
    solar = make_solar_figure()

    tle_status = "LIVE GP/TLE"
    prop_status = "SGP4 ACTIVE" if metrics else "SGP4 ERROR"
    planet_status = EPHEMERIS_STATUS

    if not metrics:
        return (
            earth, solar, empty_figure(),
            tle_status, prop_status, planet_status, catalog, clock,
            "PROPAGATION ERROR", selected, "JPL DE440s",
            [], "SGP4 could not propagate this record.", ""
        )

    m = metrics
    link = link_metrics(m["distance"])
    predicted = ml_prediction(m["distance"])

    details = html.Div([
        html.Span(f"NORAD {m['norad'] or '—'}"),
        html.Span(f"LAT {m['lat']:.3f}°"),
        html.Span(f"LON {m['lon']:.3f}°"),
        html.Span(f"ALT {m['alt']:.1f} km"),
        html.Span(f"VEL {m['velocity']:.3f} km/s"),
        html.Span(f"TLE EPOCH {m['epoch']}"),
    ], className="detail-line")

    em_distance = earth_mars_distance_km(now)
    em_text = (f"Earth–Mars distance {em_distance:,.0f} km • one-way light time {em_distance / C / 60:.2f} min"
               if em_distance is not None else "Earth–Mars distance unavailable: JPL ephemeris not loaded")
    planet_detail = html.Div([
        html.Span("CURRENT UTC EPHEMERIS"),
        html.Span(now.strftime("%Y-%m-%d %H:%M:%S UTC")),
        html.Span(em_text)
    ], className="detail-line planet-line")

    cards = [
        metric_card("ALTITUDE", f"{m['alt']:.1f}", "km"),
        metric_card("VELOCITY", f"{m['velocity']:.3f}", "km/s"),
        metric_card("RANGE", f"{m['distance']:,.0f}", "km"),
        metric_card("ONE-WAY DELAY", f"{m['delay']*1000:.2f}", "ms"),
        metric_card("SNR", f"{m['snr']:.2f}", "dB"),
        metric_card("PACKET LOSS", f"{m['packet_loss']*100:.4f}", "%"),
        metric_card("BER", f"{m['ber']:.3e}", ""),
        metric_card("THROUGHPUT", f"{m['throughput']/1e6:.3f}", "Mbps"),
    ]

    telemetry = (
        f"UTC              : {clock}\n"
        f"SATELLITE        : {selected}\n"
        f"NORAD            : {m['norad']}\n"
        f"LATITUDE         : {m['lat']:.6f} deg\n"
        f"LONGITUDE        : {m['lon']:.6f} deg\n"
        f"ALTITUDE         : {m['alt']:.3f} km\n"
        f"VELOCITY         : {m['velocity']:.6f} km/s\n"
        f"RANGE            : {m['distance']:.3f} km\n"
        f"ONE-WAY DELAY    : {m['delay']*1000:.6f} ms\n"
        f"FREE-SPACE LOSS  : {m['fspl']:.3f} dB\n"
        f"ATTENUATION      : {m['attenuation']:.3f} dB\n"
        f"NOISE FLOOR      : {m['noise']:.3f} dBm\n"
        f"RECEIVED POWER   : {m['rx']:.3f} dBm\n"
        f"SNR              : {m['snr']:.3f} dB\n"
        f"BER              : {m['ber']:.6e}\n"
        f"PACKET LOSS      : {m['packet_loss']*100:.6f} %\n"
        f"THROUGHPUT       : {m['throughput']/1e6:.6f} Mbps"
    )

    prediction = html.Div([
        html.Div("ML LINK-LOSS ESTIMATE", className="prediction-title"),
        html.Div(f"{predicted:.3f} %", className="prediction-value"),
        html.Div("RandomForest trained on model-derived samples; 80/20 train/test split.", className="prediction-note"),
        html.Div(
            f"Test MAE: {ML_EVALUATION['mae']:.4f} percentage points | Test R²: {ML_EVALUATION['r2']:.4f} | Samples: {ML_EVALUATION['samples']}"
            if ML_EVALUATION.get("mae") is not None and ML_EVALUATION.get("r2") is not None
            else "ML evaluation unavailable; analytical link model is used as fallback.",
            className="prediction-note"
        )
    ])

    return (
        earth, solar, make_metrics_figure(m["distance"]),
        tle_status, prop_status, planet_status, catalog, clock,
        "LIVE • SGP4", details, planet_detail, cards, telemetry, prediction
    )


@app.callback(
    Output("communication-analysis", "children"),
    Output("communication-windows", "children"),
    Input("live-interval", "n_intervals"),
    Input("satellite-select", "value"),
    Input("station-select", "value"),
    Input("session", "data"),
)
def communication_analysis_view(n, selected, station, session):
    if not session or not session.get("logged_in"):
        return "Sign in to analyze satellite links.", ""
    record = SATELLITE_MAP.get(selected) if selected else None
    if not record:
        return "No selected satellite record.", ""
    metrics = get_satellite_metrics(record, station)
    analysis = satellite_analysis_text(record, station, metrics)
    windows = predict_communication_windows(record, station, hours=6)
    if windows:
        rows = [html.Tr([html.Th("START (UTC)"), html.Th("END (UTC)"), html.Th("DURATION")])]
        rows.extend(html.Tr([html.Td(w["start"].strftime("%Y-%m-%d %H:%M:%S")),
                             html.Td(w["end"].strftime("%Y-%m-%d %H:%M:%S")),
                             html.Td(f"{w['duration_min']:.0f} min")]) for w in windows[:8])
        table = html.Table(rows, className="history-table")
    else:
        table = html.Div("No predicted pass found above 10° in the next 6 hours (or SGP4 is unavailable).", className="empty-history")
    return analysis, table


# ============================================================
# PACKET / DTN / CHAT / HISTORY
# ============================================================

@app.callback(
    Output("packet-result", "children"),
    Output("receiver-status", "children"),
    Input("send-packet", "n_clicks"),
    State("packet-input", "value"),
    State("satellite-select", "value"),
    State("station-select", "value"),
    prevent_initial_call=True
)
def send_packet(n, packet, selected, station):
    if not n:
        raise PreventUpdate
    if not packet:
        return "Enter packet data.", "RECEIVER: IDLE"
    rec = SATELLITE_MAP.get(selected)
    m = get_satellite_metrics(rec, station) if rec else None
    if not m:
        return "Packet not transmitted: no valid propagated link.", "RECEIVER: WAITING"
    loss = m["packet_loss"]
    ok = np.random.random() > loss
    add_history("PACKET TX", f"{selected} via {station}; {'DELIVERED' if ok else 'LOST'}")
    if ok:
        return f"TX SUCCESS • {len(packet.encode())} bytes • range {m['distance']:,.1f} km", "RECEIVER: PACKET ACCEPTED"
    return f"TX RESULT • packet loss model triggered at {loss*100:.4f}%", "RECEIVER: PACKET LOST"


@app.callback(
    Output("dtn-result", "children"),
    Input("store-packet", "n_clicks"),
    Input("forward-packet", "n_clicks"),
    State("packet-input", "value"),
    State("dtn-node", "value"),
    prevent_initial_call=True
)
def dtn_controller(store, forward, packet, node):
    from dash import ctx
    if ctx.triggered_id == "store-packet":
        if not packet:
            return "No packet to store."
        dtn_store(packet, node)
        add_history("DTN STORE", f"{len(packet)} chars at {node}")
        return f"STORED • {len(DTN_STORE)} packet(s) in DTN buffer • node {node}"
    if ctx.triggered_id == "forward-packet":
        if not DTN_STORE:
            return "DTN BUFFER EMPTY."
        item = DTN_STORE.pop(0)
        add_history("DTN FORWARD", f"Forwarded packet from {item['node']}")
        return f"FORWARDED • packet from {item['node']} • buffer remaining {len(DTN_STORE)}"
    raise PreventUpdate


@app.callback(
    Output("chat-output", "children"),
    Input("chat-button", "n_clicks"),
    State("chat-input", "value"),
    prevent_initial_call=True
)
def chat(n, text):
    if not n:
        raise PreventUpdate
    return chatbot_answer(text)


@app.callback(
    Output("history-output", "children"),
    Input("live-interval", "n_intervals")
)
def history_view(n):
    rows = read_history()[:12]
    if not rows:
        return html.Div("No mission events yet.", className="empty-history")
    return html.Table([
        html.Thead(html.Tr([html.Th("UTC TIME"), html.Th("EVENT"), html.Th("DETAILS")])),
        html.Tbody([
            html.Tr([html.Td(r.get("time","")), html.Td(r.get("event","")), html.Td(r.get("details",""))])
            for r in rows
        ])
    ], className="history-table")


if __name__ == "__main__":
    print("=" * 70)
    print("DEEP SPACE COMMUNICATION SIMULATOR")
    print("Dynamic CelesTrak GP/TLE + SGP4 propagation")
    print("URL: http://127.0.0.1:8051/")
    print("=" * 70)
    print(f"Loaded catalog records: {len(SATELLITE_MAP):,}")
    print(f"SGP4 available: {SGP4_OK}")
    print(f"Skyfield available: {SKYFIELD_OK}")
    print(f"JPL status: {EPHEMERIS_STATUS}")
    app.run(host="127.0.0.1", port=8051, debug=True)
