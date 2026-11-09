# ============================================================
# DEEP SPACE COMMUNICATION MISSION CONTROL
# COMPLETE SINGLE-FILE VERSION
# ============================================================

import sqlite3
import hashlib
import secrets
import random
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    ctx,
    no_update
)

from sgp4.api import Satrec, jday

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_PORT = 8051

DATABASE = "deep_space_users.db"

SPEED_OF_LIGHT = 299792.458

# Successful TLE is cached for 2 hours
TLE_CACHE_SECONDS = 7200

# Failed TLE requests are not retried for 10 minutes
FAILED_RETRY_SECONDS = 600


# ============================================================
# CelesTrak SATELLITE CATALOG
#
# IMPORTANT:
# STARLINK 44713 was producing 404.
# 44714 is used here instead.
# ============================================================

SATELLITE_CATALOG_NUMBERS = {

    "ISS": "25544",

    "HUBBLE": "20580",

    "STARLINK": "44714",

    "TIANGONG": "48274"

}


# ============================================================
# TLE CACHE
# ============================================================

TLE_CACHE = {}

TLE_CACHE_TIME = {}

TLE_FAILED_TIME = {}


# ============================================================
# DATABASE
# ============================================================

def initialize_database():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()

    connection.close()


initialize_database()


# ============================================================
# PASSWORD SECURITY
# ============================================================

def hash_password(password):

    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()

    return salt + ":" + password_hash


def verify_password(
    password,
    stored_password
):

    try:

        salt, stored_hash = (
            stored_password.split(":")
        )

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000
        ).hex()

        return secrets.compare_digest(
            password_hash,
            stored_hash
        )

    except Exception:

        return False


# ============================================================
# REGISTER USER
# ============================================================

def register_user(
    username,
    email,
    password
):

    if not username:

        return False, "Username is required."

    if not email:

        return False, "Email is required."

    if not password:

        return False, "Password is required."

    username = username.strip()

    email = email.strip().lower()

    if len(password) < 6:

        return False, (
            "Password must contain at least 6 characters."
        )

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (username, email, password, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                email,
                hash_password(password),
                datetime.now(
                    timezone.utc
                ).isoformat()
            )
        )

        connection.commit()

        return True, (
            "Account created successfully."
        )

    except sqlite3.IntegrityError:

        return False, (
            "Username or email already exists."
        )

    finally:

        connection.close()


# ============================================================
# LOGIN
# ============================================================

def login_user(
    username,
    password
):

    if not username or not password:

        return False

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT password
        FROM users
        WHERE username=?
        """,
        (
            username.strip(),
        )
    )

    result = cursor.fetchone()

    connection.close()

    if result:

        return verify_password(
            password,
            result[0]
        )

    return False


# ============================================================
# RESET PASSWORD
# ============================================================

def reset_password(
    username,
    email,
    new_password
):

    if (
        not username
        or not email
        or not new_password
    ):

        return False, (
            "All fields are required."
        )

    if len(new_password) < 6:

        return False, (
            "Password must contain at least 6 characters."
        )

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM users
        WHERE username=?
        AND email=?
        """,
        (
            username.strip(),
            email.strip().lower()
        )
    )

    result = cursor.fetchone()

    if not result:

        connection.close()

        return False, (
            "Username and email do not match."
        )

    cursor.execute(
        """
        UPDATE users
        SET password=?
        WHERE username=?
        """,
        (
            hash_password(new_password),
            username.strip()
        )
    )

    connection.commit()

    connection.close()

    return True, (
        "Password reset successfully."
    )


# ============================================================
# CHANGE PASSWORD
# ============================================================

def change_user_password(
    username,
    old_password,
    new_password
):

    if not old_password:

        return False, (
            "Enter current password."
        )

    if not new_password:

        return False, (
            "Enter new password."
        )

    if len(new_password) < 6:

        return False, (
            "New password must contain at least 6 characters."
        )

    if not login_user(
        username,
        old_password
    ):

        return False, (
            "Current password is incorrect."
        )

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE users
        SET password=?
        WHERE username=?
        """,
        (
            hash_password(new_password),
            username
        )
    )

    connection.commit()

    connection.close()

    return True, (
        "Password changed successfully."
    )


# ============================================================
# LIVE CELESTRAK TLE DOWNLOAD
# ============================================================

def download_single_tle(
    catalog_number
):

    now = time.time()

    # --------------------------------------------------------
    # SUCCESS CACHE
    # --------------------------------------------------------

    if catalog_number in TLE_CACHE:

        last_update = TLE_CACHE_TIME.get(
            catalog_number,
            0
        )

        if (
            now - last_update
            <
            TLE_CACHE_SECONDS
        ):

            return TLE_CACHE[
                catalog_number
            ]

    # --------------------------------------------------------
    # FAILED REQUEST PROTECTION
    # --------------------------------------------------------

    failed_time = TLE_FAILED_TIME.get(
        catalog_number,
        0
    )

    if (
        now - failed_time
        <
        FAILED_RETRY_SECONDS
    ):

        return TLE_CACHE.get(
            catalog_number
        )

    # --------------------------------------------------------
    # CELESTRAK URL
    # --------------------------------------------------------

    url = (
        "https://celestrak.org/"
        "NORAD/elements/gp.php"
    )

    params = {

        "CATNR": catalog_number,

        "FORMAT": "TLE"

    }

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "DeepSpaceMissionControl/1.0",

        "Accept":
            "text/plain"

    }

    try:

        response = requests.get(

            url,

            params=params,

            headers=headers,

            timeout=20

        )

        response.raise_for_status()

        lines = [

            line.strip()

            for line
            in response.text.splitlines()

            if line.strip()

        ]

        # ----------------------------------------------------
        # SEARCH FOR VALID TLE
        # ----------------------------------------------------

        for i in range(
            len(lines) - 2
        ):

            if (

                lines[i + 1].startswith("1 ")

                and

                lines[i + 2].startswith("2 ")

            ):

                name = lines[i]

                line1 = lines[i + 1]

                line2 = lines[i + 2]

                tle = (
                    line1,
                    line2
                )

                TLE_CACHE[
                    catalog_number
                ] = tle

                TLE_CACHE_TIME[
                    catalog_number
                ] = now

                TLE_FAILED_TIME.pop(
                    catalog_number,
                    None
                )

                print(
                    "LIVE TLE UPDATED:",
                    name
                )

                return tle

        print(
            "Invalid TLE response:",
            catalog_number
        )

        TLE_FAILED_TIME[
            catalog_number
        ] = now

        return TLE_CACHE.get(
            catalog_number
        )

    except requests.exceptions.HTTPError as error:

        print(
            "CelesTrak request failed:",
            catalog_number,
            error
        )

        TLE_FAILED_TIME[
            catalog_number
        ] = now

        return TLE_CACHE.get(
            catalog_number
        )

    except requests.exceptions.RequestException as error:

        print(
            "Network error:",
            catalog_number,
            error
        )

        TLE_FAILED_TIME[
            catalog_number
        ] = now

        return TLE_CACHE.get(
            catalog_number
        )

    except Exception as error:

        print(
            "TLE error:",
            catalog_number,
            error
        )

        TLE_FAILED_TIME[
            catalog_number
        ] = now

        return TLE_CACHE.get(
            catalog_number
        )


# ============================================================
# LOAD LIVE SATELLITES
# ============================================================

def load_satellites():

    satellites = {}

    for (
        name,
        catalog_number
    ) in SATELLITE_CATALOG_NUMBERS.items():

        tle = download_single_tle(
            catalog_number
        )

        if not tle:

            continue

        try:

            satellite = Satrec.twoline2rv(
                tle[0],
                tle[1]
            )

            satellites[name] = {

                "satellite":
                    satellite,

                "catalog_number":
                    catalog_number,

                "tle":
                    tle

            }

        except Exception as error:

            print(
                "SGP4 error:",
                name,
                error
            )

    return satellites


# ============================================================
# SATELLITE POSITION
# ============================================================

def satellite_position(
    satellite,
    offset=0
):

    now = (
        datetime.now(
            timezone.utc
        )
        +
        timedelta(
            seconds=offset
        )
    )

    jd, fr = jday(

        now.year,

        now.month,

        now.day,

        now.hour,

        now.minute,

        now.second

    )

    error, position, velocity = (
        satellite.sgp4(
            jd,
            fr
        )
    )

    if error == 0:

        return (

            position[0],

            position[1],

            position[2]

        )

    return 0, 0, 0


# ============================================================
# ORBIT TRAIL
# ============================================================

def orbit_trail(
    satellite
):

    x_values = []

    y_values = []

    z_values = []

    for seconds in range(
        0,
        5400,
        180
    ):

        x, y, z = satellite_position(
            satellite,
            seconds
        )

        x_values.append(x)

        y_values.append(y)

        z_values.append(z)

    return (
        x_values,
        y_values,
        z_values
    )


# ============================================================
# GROUND STATIONS
# ============================================================

GROUND_STATIONS = [

    {
        "name": "Goldstone",
        "latitude": 35.247,
        "longitude": -116.793
    },

    {
        "name": "Madrid",
        "latitude": 40.431,
        "longitude": -4.248
    },

    {
        "name": "Canberra",
        "latitude": -35.402,
        "longitude": 148.981
    }

]


# ============================================================
# LATITUDE / LONGITUDE TO XYZ
# ============================================================

def latlon_to_xyz(
    latitude,
    longitude,
    radius=10.4
):

    latitude = np.radians(
        latitude
    )

    longitude = np.radians(
        longitude
    )

    x = (
        radius
        *
        np.cos(latitude)
        *
        np.cos(longitude)
    )

    y = (
        radius
        *
        np.cos(latitude)
        *
        np.sin(longitude)
    )

    z = (
        radius
        *
        np.sin(latitude)
    )

    return (
        x,
        y,
        z
    )


# ============================================================
# EARTH
# ============================================================

def create_earth():

    u = np.linspace(
        0,
        2 * np.pi,
        80
    )

    v = np.linspace(
        0,
        np.pi,
        50
    )

    x = (
        10
        *
        np.outer(
            np.cos(u),
            np.sin(v)
        )
    )

    y = (
        10
        *
        np.outer(
            np.sin(u),
            np.sin(v)
        )
    )

    z = (
        10
        *
        np.outer(
            np.ones(
                len(u)
            ),
            np.cos(v)
        )
    )

    return go.Surface(

        x=x,

        y=y,

        z=z,

        colorscale="Blues",

        opacity=0.95,

        showscale=False,

        name="Earth"

    )


# ============================================================
# PLANETS
# ============================================================

PLANETS = [

    ("Mercury", 15, 14, 6),

    ("Venus", 25, 18, 8),

    ("Earth", 0, 10, 8),

    ("Mars", 35, 23, 7),

    ("Jupiter", 50, 30, 14),

    ("Saturn", 65, 37, 12),

    ("Uranus", 80, 44, 10),

    ("Neptune", 95, 51, 10)

]


PLANET_STYLES = {

    "Mercury": {
        "size": 6,
        "color": "#A8A8A8"
    },

    "Venus": {
        "size": 8,
        "color": "#D9B36C"
    },

    "Mars": {
        "size": 7,
        "color": "#C96B4B"
    },

    "Jupiter": {
        "size": 14,
        "color": "#C9A77A"
    },

    "Saturn": {
        "size": 12,
        "color": "#D8C58A"
    },

    "Uranus": {
        "size": 10,
        "color": "#8ED6D6"
    },

    "Neptune": {
        "size": 10,
        "color": "#6489D8"
    }

}


# ============================================================
# COMMUNICATION LINE
# ============================================================

def create_signal_line(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    name
):

    return go.Scatter3d(

        x=[
            x1,
            x2
        ],

        y=[
            y1,
            y2
        ],

        z=[
            z1,
            z2
        ],

        mode="lines",

        line=dict(

            color="#00E5FF",

            width=4

        ),

        name=name

    )


# ============================================================
# COMMUNICATION WAVE
# ============================================================

def create_wave_point(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    progress,
    name
):

    progress = (
        progress % 1.0
    )

    x = (
        x1
        +
        (x2 - x1)
        *
        progress
    )

    y = (
        y1
        +
        (y2 - y1)
        *
        progress
    )

    z = (
        z1
        +
        (z2 - z1)
        *
        progress
    )

    return go.Scatter3d(

        x=[x],

        y=[y],

        z=[z],

        mode="markers",

        marker=dict(

            size=11,

            color="#00FFFF"

        ),

        name=name

    )


# ============================================================
# MACHINE LEARNING
# ============================================================

def train_ml_model():

    np.random.seed(42)

    samples = 3000

    distance = np.random.uniform(
        10000,
        500000,
        samples
    )

    noise = np.random.uniform(
        0,
        20,
        samples
    )

    packet_loss = np.random.uniform(
        0,
        50,
        samples
    )

    risk_score = (

        distance / 500000

        +

        noise / 40

        +

        packet_loss / 100

    )

    target = (
        risk_score > 0.65
    ).astype(int)

    data = pd.DataFrame({

        "distance": distance,

        "noise": noise,

        "packet_loss": packet_loss

    })

    model = RandomForestClassifier(

        n_estimators=100,

        random_state=42

    )

    model.fit(
        data,
        target
    )

    return model


ML_MODEL = train_ml_model()


# ============================================================
# COMMUNICATION CALCULATIONS
# ============================================================

def calculate_delay(
    distance
):

    return (
        distance
        /
        SPEED_OF_LIGHT
    )


def generate_noise():

    return float(
        np.random.normal(
            0,
            5
        )
    )


def calculate_packet_loss(
    distance,
    noise
):

    loss = (

        0.5

        +

        distance / 10000

        +

        abs(noise) * 0.7

    )

    return min(
        max(
            loss,
            0
        ),
        95
    )


def calculate_signal_strength(
    packet_loss,
    noise
):

    strength = (

        100

        -

        packet_loss

        -

        abs(noise) * 2

    )

    return max(
        0,
        min(
            100,
            strength
        )
    )


def predict_signal_risk(
    distance,
    noise,
    packet_loss
):

    input_data = pd.DataFrame([{

        "distance": distance,

        "noise": abs(noise),

        "packet_loss": packet_loss

    }])

    probability = (
        ML_MODEL
        .predict_proba(
            input_data
        )[0][1]
    )

    return (
        probability * 100
    )


# ============================================================
# CHATBOT
# ============================================================

def chatbot(
    message
):

    if not message:

        return (
            "🚀 Please enter a question."
        )

    message = message.lower()

    if "iss" in message:

        return (
            "🛰️ ISS uses live TLE data "
            "and SGP4 orbital propagation."
        )

    if "hubble" in message:

        return (
            "🔭 Hubble orbital information "
            "is obtained from live TLE data."
        )

    if "starlink" in message:

        return (
            "🛰️ Starlink uses a live "
            "CelesTrak catalog object."
        )

    if "delay" in message:

        return (
            "⏱️ Propagation delay is "
            "distance divided by the speed of light."
        )

    if "packet" in message:

        return (
            "📦 Packet loss is simulated "
            "using distance and channel noise."
        )

    if "noise" in message:

        return (
            "📡 Noise is generated using "
            "a Gaussian communication-channel model."
        )

    if (
        "ml" in message
        or
        "ai" in message
        or
        "machine learning" in message
    ):

        return (
            "🤖 Random Forest predicts "
            "signal-loss risk."
        )

    if "station" in message:

        return (
            "📡 Ground stations: "
            "Goldstone, Madrid and Canberra."
        )

    if "planet" in message:

        return (
            "🪐 The visualization contains "
            "Mercury through Neptune."
        )

    if "security" in message:

        return (
            "🔐 Passwords are protected "
            "using salted PBKDF2-HMAC-SHA256."
        )

    return (
        "🚀 Ask about ISS, Hubble, Starlink, "
        "delay, packet loss, noise, ML, "
        "planets, stations or security."
    )


# ============================================================
# UI STYLES
# ============================================================

PAGE_STYLE = {

    "backgroundColor": "#020202",

    "color": "white",

    "minHeight": "100vh",

    "padding": "20px",

    "fontFamily": "Arial"

}


INPUT_STYLE = {

    "width": "90%",

    "padding": "12px",

    "margin": "7px",

    "backgroundColor": "#111111",

    "color": "white",

    "border": "1px solid #00FFFF",

    "borderRadius": "5px"

}


BUTTON_STYLE = {

    "padding": "11px 18px",

    "margin": "7px",

    "backgroundColor": "#111111",

    "color": "white",

    "border": "1px solid #00FFFF",

    "borderRadius": "5px",

    "cursor": "pointer"

}


CARD_STYLE = {

    "border": "1px solid #00FFFF",

    "borderRadius": "10px",

    "padding": "15px",

    "margin": "7px",

    "boxShadow": "0 0 12px #004F5A"

}


HIDDEN = {

    "display": "none"

}


VISIBLE = {

    "display": "block"

}


# ============================================================
# DASH APPLICATION
# ============================================================

app = Dash(
    __name__
)

app.title = (
    "Deep Space Mission Control"
)


# ============================================================
# COMPLETE STATIC LAYOUT
# ============================================================

app.layout = html.Div(

    [

        dcc.Store(

            id="session-store",

            storage_type="session",

            data={
                "logged_in": False,
                "username": ""
            }

        ),

        dcc.Store(

            id="page-store",

            data="login"

        ),

        # ====================================================
        # LOGIN PAGE
        # ====================================================

        html.Div(

            id="login-page",

            children=[

                html.H1(

                    "🚀 DEEP SPACE "
                    "MISSION CONTROL",

                    style={

                        "textAlign": "center",

                        "textShadow":
                            "0 0 20px #00FFFF"

                    }

                ),

                html.H3(

                    "Secure Mission Login",

                    style={

                        "textAlign": "center",

                        "color": "#00FFFF"

                    }

                ),

                html.Div(

                    [

                        dcc.Input(

                            id="login-username",

                            placeholder="Username",

                            style=INPUT_STYLE

                        ),

                        dcc.Input(

                            id="login-password",

                            placeholder="Password",

                            type="password",

                            style=INPUT_STYLE

                        ),

                        html.Button(

                            "🔐 LOGIN",

                            id="login-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Button(

                            "📝 SIGN UP",

                            id="signup-page-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Button(

                            "🔑 FORGOT PASSWORD",

                            id="forgot-page-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Div(

                            id="login-message"

                        )

                    ],

                    style={

                        "maxWidth": "500px",

                        "margin": "40px auto",

                        **CARD_STYLE

                    }

                )

            ],

            style=VISIBLE

        ),

        # ====================================================
        # SIGNUP PAGE
        # ====================================================

        html.Div(

            id="signup-page",

            children=[

                html.H1(

                    "📝 CREATE ACCOUNT",

                    style={

                        "textAlign": "center",

                        "color": "#00FFFF"

                    }

                ),

                html.Div(

                    [

                        dcc.Input(

                            id="signup-username",

                            placeholder="Username",

                            style=INPUT_STYLE

                        ),

                        dcc.Input(

                            id="signup-email",

                            placeholder="Email",

                            type="email",

                            style=INPUT_STYLE

                        ),

                        dcc.Input(

                            id="signup-password",

                            placeholder="Password",

                            type="password",

                            style=INPUT_STYLE

                        ),

                        dcc.Input(

                            id="signup-confirm",

                            placeholder="Confirm Password",

                            type="password",

                            style=INPUT_STYLE

                        ),

                        html.Button(

                            "CREATE ACCOUNT",

                            id="signup-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Button(

                            "BACK TO LOGIN",

                            id="signup-back-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Div(

                            id="signup-message"

                        )

                    ],

                    style={

                        "maxWidth": "500px",

                        "margin": "40px auto",

                        **CARD_STYLE

                    }

                )

            ],

            style=HIDDEN

        ),

        # ====================================================
        # FORGOT PASSWORD
        # ====================================================

        html.Div(

            id="forgot-page",

            children=[

                html.H1(

                    "🔑 PASSWORD RECOVERY",

                    style={

                        "textAlign": "center",

                        "color": "#00FFFF"

                    }

                ),

                html.Div(

                    [

                        dcc.Input(

                            id="forgot-username",

                            placeholder="Username",

                            style=INPUT_STYLE

                        ),

                        dcc.Input(

                            id="forgot-email",

                            placeholder="Registered Email",

                            type="email",

                            style=INPUT_STYLE

                        ),

                        dcc.Input(

                            id="forgot-new-password",

                            placeholder="New Password",

                            type="password",

                            style=INPUT_STYLE

                        ),

                        html.Button(

                            "RESET PASSWORD",

                            id="forgot-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Button(

                            "BACK TO LOGIN",

                            id="forgot-back-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Div(

                            id="forgot-message"

                        )

                    ],

                    style={

                        "maxWidth": "500px",

                        "margin": "40px auto",

                        **CARD_STYLE

                    }

                )

            ],

            style=HIDDEN

        ),

        # ====================================================
        # DASHBOARD
        # ====================================================

        html.Div(

            id="dashboard-page",

            children=[

                html.Div(

                    [

                        html.H1(

                            "🚀 NASA-STYLE "
                            "SPACE MISSION CONTROL",

                            style={

                                "display":
                                    "inline-block",

                                "textShadow":
                                    "0 0 20px #00FFFF"

                            }

                        ),

                        html.Button(

                            "🚪 LOGOUT",

                            id="logout-button",

                            n_clicks=0,

                            style={

                                **BUTTON_STYLE,

                                "float": "right"

                            }

                        )

                    ],

                    style={
                        "overflow": "hidden"
                    }

                ),

                html.Div(

                    id="operator-name",

                    style={

                        "color": "#00FFFF",

                        "fontSize": "18px"

                    }

                ),

                html.Div(

                    [

                        html.Button(

                            "🌍 SPACE VIEW",

                            id="space-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Button(

                            "📡 COMMUNICATION",

                            id="communication-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Button(

                            "🤖 AI CHATBOT",

                            id="chatbot-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Button(

                            "🔐 CHANGE PASSWORD",

                            id="password-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        )

                    ]

                ),

                # =================================================
                # SPACE PANEL
                # =================================================

                html.Div(

                    id="space-panel",

                    children=[

                        html.H2(

                            "🌌 3D SPACE "
                            "VISUALIZATION",

                            style={
                                "color":
                                    "#00FFFF"
                            }

                        ),

                        html.P(

                            "Light 3D coordinate "
                            "axes are independent "
                            "from the solar-system "
                            "objects.",

                            style={
                                "color":
                                    "#C7D2DA"
                            }

                        ),

                        dcc.Graph(

                            id="space-graph",

                            style={
                                "height":
                                    "70vh"
                            }

                        ),

                        html.H2(

                            "📊 LIVE TELEMETRY",

                            style={
                                "color":
                                    "#00FFFF"
                            }

                        ),

                        html.Div(

                            id="telemetry-output"

                        )

                    ],

                    style=VISIBLE

                ),

                # =================================================
                # COMMUNICATION PANEL
                # =================================================

                html.Div(

                    id="communication-panel",

                    children=[

                        html.H2(

                            "📡 COMMUNICATION "
                            "ANALYSIS",

                            style={
                                "color":
                                    "#00FFFF"
                            }

                        ),

                        html.Div(

                            id="communication-output",

                            style={

                                "display":
                                    "grid",

                                "gridTemplateColumns":
                                    "repeat(3, 1fr)",

                                "gap":
                                    "10px"

                            }

                        ),

                        dcc.Graph(

                            id="communication-graph"

                        ),

                        dcc.Graph(

                            id="packet-loss-graph"

                        ),

                        dcc.Graph(

                            id="signal-noise-graph"

                        ),

                        dcc.Graph(

                            id="ml-risk-graph"

                        )

                    ],

                    style=HIDDEN

                ),

                # =================================================
                # CHATBOT
                # =================================================

                html.Div(

                    id="chatbot-panel",

                    children=[

                        html.H2(

                            "🤖 AI SPACE ASSISTANT",

                            style={
                                "color":
                                    "#00FFFF"
                            }

                        ),

                        dcc.Input(

                            id="chat-input",

                            placeholder=
                                "Ask about ISS, "
                                "delay, packet loss, "
                                "noise, ML...",

                            style=INPUT_STYLE

                        ),

                        html.Button(

                            "SEND",

                            id="send-chat-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Div(

                            id="chat-output",

                            style=CARD_STYLE

                        )

                    ],

                    style=HIDDEN

                ),

                # =================================================
                # PASSWORD PANEL
                # =================================================

                html.Div(

                    id="password-panel",

                    children=[

                        html.H2(

                            "🔐 CHANGE PASSWORD",

                            style={
                                "color":
                                    "#00FFFF"
                            }

                        ),

                        dcc.Input(

                            id="old-password",

                            placeholder=
                                "Current Password",

                            type="password",

                            style=INPUT_STYLE

                        ),

                        dcc.Input(

                            id="new-password",

                            placeholder=
                                "New Password",

                            type="password",

                            style=INPUT_STYLE

                        ),

                        dcc.Input(

                            id="confirm-password",

                            placeholder=
                                "Confirm New Password",

                            type="password",

                            style=INPUT_STYLE

                        ),

                        html.Button(

                            "UPDATE PASSWORD",

                            id="change-password-button",

                            n_clicks=0,

                            style=BUTTON_STYLE

                        ),

                        html.Div(

                            id="password-message"

                        )

                    ],

                    style=HIDDEN

                ),

                # =================================================
                # LIVE REFRESH
                # =================================================

                dcc.Interval(

                    id="live-update",

                    interval=15000,

                    n_intervals=0

                )

            ],

            style=HIDDEN

        )

    ],

    style=PAGE_STYLE

)


# ============================================================
# LOGIN CALLBACK
# ============================================================

@app.callback(

    Output(
        "page-store",
        "data",
        allow_duplicate=True
    ),

    Output(
        "session-store",
        "data",
        allow_duplicate=True
    ),

    Output(
        "login-message",
        "children"
    ),

    Input(
        "login-button",
        "n_clicks"
    ),

    State(
        "login-username",
        "value"
    ),

    State(
        "login-password",
        "value"
    ),

    prevent_initial_call=True

)
def login_callback(
    n_clicks,
    username,
    password
):

    if login_user(
        username,
        password
    ):

        return (

            "dashboard",

            {
                "logged_in": True,
                "username": username
            },

            "✅ Login successful."

        )

    return (

        "login",

        {
            "logged_in": False,
            "username": ""
        },

        "❌ Invalid username or password."

    )


# ============================================================
# SIGNUP / FORGOT NAVIGATION
# ============================================================

@app.callback(

    Output(
        "page-store",
        "data",
        allow_duplicate=True
    ),

    Input(
        "signup-page-button",
        "n_clicks"
    ),

    Input(
        "forgot-page-button",
        "n_clicks"
    ),

    Input(
        "signup-back-button",
        "n_clicks"
    ),

    Input(
        "forgot-back-button",
        "n_clicks"
    ),

    prevent_initial_call=True

)
def auth_navigation(
    signup_click,
    forgot_click,
    signup_back,
    forgot_back
):

    button = ctx.triggered_id

    if button == "signup-page-button":

        return "signup"

    if button == "forgot-page-button":

        return "forgot"

    if button in [

        "signup-back-button",

        "forgot-back-button"

    ]:

        return "login"

    return no_update


# ============================================================
# SIGNUP CALLBACK
# ============================================================

@app.callback(

    Output(
        "signup-message",
        "children"
    ),

    Input(
        "signup-button",
        "n_clicks"
    ),

    State(
        "signup-username",
        "value"
    ),

    State(
        "signup-email",
        "value"
    ),

    State(
        "signup-password",
        "value"
    ),

    State(
        "signup-confirm",
        "value"
    ),

    prevent_initial_call=True

)
def signup_callback(
    n_clicks,
    username,
    email,
    password,
    confirm
):

    if password != confirm:

        return (
            "❌ Passwords do not match."
        )

    success, message = register_user(

        username,

        email,

        password

    )

    if success:

        return (
            "✅ Account created successfully. "
            "Click BACK TO LOGIN."
        )

    return (
        "❌ " + message
    )


# ============================================================
# FORGOT PASSWORD CALLBACK
# ============================================================

@app.callback(

    Output(
        "forgot-message",
        "children"
    ),

    Input(
        "forgot-button",
        "n_clicks"
    ),

    State(
        "forgot-username",
        "value"
    ),

    State(
        "forgot-email",
        "value"
    ),

    State(
        "forgot-new-password",
        "value"
    ),

    prevent_initial_call=True

)
def forgot_callback(
    n_clicks,
    username,
    email,
    password
):

    success, message = reset_password(

        username,

        email,

        password

    )

    if success:

        return (
            "✅ " + message
        )

    return (
        "❌ " + message
    )


# ============================================================
# LOGOUT
# ============================================================

@app.callback(

    Output(
        "page-store",
        "data",
        allow_duplicate=True
    ),

    Output(
        "session-store",
        "data",
        allow_duplicate=True
    ),

    Input(
        "logout-button",
        "n_clicks"
    ),

    prevent_initial_call=True

)
def logout_callback(
    n_clicks
):

    return (

        "login",

        {
            "logged_in": False,
            "username": ""
        }

    )


# ============================================================
# PAGE VISIBILITY
# ============================================================

@app.callback(

    Output(
        "login-page",
        "style"
    ),

    Output(
        "signup-page",
        "style"
    ),

    Output(
        "forgot-page",
        "style"
    ),

    Output(
        "dashboard-page",
        "style"
    ),

    Output(
        "operator-name",
        "children"
    ),

    Input(
        "page-store",
        "data"
    ),

    State(
        "session-store",
        "data"
    )

)
def page_visibility(
    page,
    session
):

    if page == "signup":

        return (

            HIDDEN,

            VISIBLE,

            HIDDEN,

            HIDDEN,

            ""

        )

    if page == "forgot":

        return (

            HIDDEN,

            HIDDEN,

            VISIBLE,

            HIDDEN,

            ""

        )

    if (

        page == "dashboard"

        and

        session.get(
            "logged_in",
            False
        )

    ):

        return (

            HIDDEN,

            HIDDEN,

            HIDDEN,

            VISIBLE,

            "👨‍🚀 Mission Operator: "
            +
            session.get(
                "username",
                ""
            )

        )

    return (

        VISIBLE,

        HIDDEN,

        HIDDEN,

        HIDDEN,

        ""

    )


# ============================================================
# DASHBOARD NAVIGATION
# ============================================================

@app.callback(

    Output(
        "space-panel",
        "style"
    ),

    Output(
        "communication-panel",
        "style"
    ),

    Output(
        "chatbot-panel",
        "style"
    ),

    Output(
        "password-panel",
        "style"
    ),

    Input(
        "space-button",
        "n_clicks"
    ),

    Input(
        "communication-button",
        "n_clicks"
    ),

    Input(
        "chatbot-button",
        "n_clicks"
    ),

    Input(
        "password-button",
        "n_clicks"
    ),

    prevent_initial_call=True

)
def dashboard_navigation(
    space_click,
    communication_click,
    chatbot_click,
    password_click
):

    button = ctx.triggered_id

    if button == "communication-button":

        return (

            HIDDEN,

            VISIBLE,

            HIDDEN,

            HIDDEN

        )

    if button == "chatbot-button":

        return (

            HIDDEN,

            HIDDEN,

            VISIBLE,

            HIDDEN

        )

    if button == "password-button":

        return (

            HIDDEN,

            HIDDEN,

            HIDDEN,

            VISIBLE

        )

    return (

        VISIBLE,

        HIDDEN,

        HIDDEN,

        HIDDEN

    )


# ============================================================
# 3D SPACE VISUALIZATION
# ============================================================

@app.callback(

    Output(
        "space-graph",
        "figure"
    ),

    Output(
        "telemetry-output",
        "children"
    ),

    Input(
        "live-update",
        "n_intervals"
    ),

    Input(
        "space-button",
        "n_clicks"
    )

)
def update_space(
    interval,
    space_click
):

    traces = []

    telemetry = []

    # ========================================================
    # EARTH
    # ========================================================

    traces.append(
        create_earth()
    )

    # ========================================================
    # PLANETS
    # ========================================================

    for (

        planet,

        angle,

        radius,

        size

    ) in PLANETS:

        if planet == "Earth":

            continue

        theta = np.radians(
            angle
        )

        x = (
            radius
            *
            np.cos(theta)
        )

        y = (
            radius
            *
            np.sin(theta)
        )

        style = PLANET_STYLES.get(

            planet,

            {
                "size": size,
                "color": "#FFFFFF"
            }

        )

        traces.append(

            go.Scatter3d(

                x=[x],

                y=[y],

                z=[0],

                mode="markers+text",

                marker=dict(

                    size=style["size"],

                    color=style["color"],

                    opacity=1.0

                ),

                text=[planet],

                textposition=
                    "top center",

                textfont=dict(

                    color="white",

                    size=11

                ),

                name=planet

            )

        )

    # ========================================================
    # GROUND STATIONS
    # ========================================================

    for station in GROUND_STATIONS:

        gx, gy, gz = (
            latlon_to_xyz(

                station["latitude"],

                station["longitude"]

            )
        )

        traces.append(

            go.Scatter3d(

                x=[gx],

                y=[gy],

                z=[gz],

                mode="markers+text",

                marker=dict(

                    size=8,

                    color="#FFD166"

                ),

                text=[

                    "📡 "

                    +

                    station["name"]

                ],

                textposition=
                    "top center",

                textfont=dict(

                    color="#FFD166",

                    size=10

                ),

                name=station["name"]

            )

        )

    # ========================================================
    # LIVE SATELLITES
    # ========================================================

    satellites = load_satellites()

    progress = (

        interval % 100

    ) / 100.0

    for (

        name,

        data

    ) in satellites.items():

        satellite = (
            data["satellite"]
        )

        sx, sy, sz = (
            satellite_position(
                satellite
            )
        )

        # ----------------------------------------------------
        # SATELLITE
        # ----------------------------------------------------

        traces.append(

            go.Scatter3d(

                x=[sx],

                y=[sy],

                z=[sz],

                mode="markers+text",

                marker=dict(

                    size=10,

                    color="#F5F7FA",

                    line=dict(

                        color="#B8C4CE",

                        width=2

                    )

                ),

                text=[

                    "🛰️ "

                    +

                    name

                ],

                textposition=
                    "top center",

                textfont=dict(

                    color="#FFFFFF",

                    size=11

                ),

                name=name

            )

        )

        # ----------------------------------------------------
        # ORBIT
        # ----------------------------------------------------

        ox, oy, oz = (
            orbit_trail(
                satellite
            )
        )

        traces.append(

            go.Scatter3d(

                x=ox,

                y=oy,

                z=oz,

                mode="lines",

                line=dict(

                    color="#AEB8C2",

                    width=2

                ),

                name=name +
                    " Orbit"

            )

        )

        # ----------------------------------------------------
        # EARTH -> SATELLITE
        # ----------------------------------------------------

        earth_x = 10

        earth_y = 0

        earth_z = 0

        traces.append(

            create_signal_line(

                earth_x,

                earth_y,

                earth_z,

                sx,

                sy,

                sz,

                "🌍 → 🛰️ "
                +
                name

            )

        )

        # ----------------------------------------------------
        # COMMUNICATION WAVE
        # ----------------------------------------------------

        traces.append(

            create_wave_point(

                earth_x,

                earth_y,

                earth_z,

                sx,

                sy,

                sz,

                progress,

                "〰️ Communication Wave"

            )

        )

        # ----------------------------------------------------
        # SATELLITE -> GROUND
        # ----------------------------------------------------

        station = (
            GROUND_STATIONS[0]
        )

        gx, gy, gz = (
            latlon_to_xyz(

                station["latitude"],

                station["longitude"]

            )
        )

        traces.append(

            create_signal_line(

                sx,

                sy,

                sz,

                gx,

                gy,

                gz,

                "🛰️ → 📡 "
                +
                station["name"]

            )

        )

        # ----------------------------------------------------
        # RETURN WAVE
        # ----------------------------------------------------

        traces.append(

            create_wave_point(

                sx,

                sy,

                sz,

                gx,

                gy,

                gz,

                1 - progress,

                "〰️ Return Wave"

            )

        )

        # ----------------------------------------------------
        # TELEMETRY CARD
        # ----------------------------------------------------

        telemetry.append(

            html.Div(

                [

                    html.H3(

                        "🛰️ "
                        +
                        name

                    ),

                    html.P(

                        "Catalog ID: "

                        +

                        data[
                            "catalog_number"
                        ]

                    ),

                    html.P(

                        f"X: {sx:.2f} km"

                    ),

                    html.P(

                        f"Y: {sy:.2f} km"

                    ),

                    html.P(

                        f"Z: {sz:.2f} km"

                    ),

                    html.P(

                        "📡 Communication: ACTIVE"

                    ),

                    html.P(

                        "〰️ Wave: ACTIVE"

                    )

                ],

                style=CARD_STYLE

            )

        )

    # ========================================================
    # FIGURE
    # ========================================================

    figure = go.Figure(
        data=traces
    )

    figure.update_layout(

        paper_bgcolor="#020202",

        plot_bgcolor="#020202",

        font=dict(

            color="white"

        ),

        scene=dict(

            bgcolor="#020202",

            # ================================================
            # LIGHT X AXIS
            # ================================================

            xaxis=dict(

                visible=True,

                title=dict(

                    text="X (km)",

                    font=dict(

                        color="#D9E2EC",

                        size=12

                    )

                ),

                color="#D9E2EC",

                gridcolor="#45505C",

                zerolinecolor="#6B7280",

                showbackground=False,

                showgrid=True,

                showline=True,

                linecolor="#D9E2EC",

                tickfont=dict(

                    color="#C7D2DA"

                )

            ),

            # ================================================
            # LIGHT Y AXIS
            # ================================================

            yaxis=dict(

                visible=True,

                title=dict(

                    text="Y (km)",

                    font=dict(

                        color="#D9E2EC",

                        size=12

                    )

                ),

                color="#D9E2EC",

                gridcolor="#45505C",

                zerolinecolor="#6B7280",

                showbackground=False,

                showgrid=True,

                showline=True,

                linecolor="#D9E2EC",

                tickfont=dict(

                    color="#C7D2DA"

                )

            ),

            # ================================================
            # LIGHT Z AXIS
            # ================================================

            zaxis=dict(

                visible=True,

                title=dict(

                    text="Z (km)",

                    font=dict(

                        color="#D9E2EC",

                        size=12

                    )

                ),

                color="#D9E2EC",

                gridcolor="#45505C",

                zerolinecolor="#6B7280",

                showbackground=False,

                showgrid=True,

                showline=True,

                linecolor="#D9E2EC",

                tickfont=dict(

                    color="#C7D2DA"

                )

            )

        ),

        margin=dict(

            l=0,

            r=0,

            t=0,

            b=0

        )

    )

    # --------------------------------------------------------
    # NO LIVE DATA MESSAGE
    # --------------------------------------------------------

    if not satellites:

        telemetry.append(

            html.Div(

                [

                    html.H3(
                        "⚠️ Live TLE unavailable"
                    ),

                    html.P(

                        "CelesTrak could not be "
                        "reached currently. "
                        "The dashboard remains "
                        "operational."

                    )

                ],

                style={

                    **CARD_STYLE,

                    "color":
                        "#FFD166"

                }

            )

        )

    return (

        figure,

        telemetry

    )


# ============================================================
# COMMUNICATION ANALYSIS
# ============================================================

@app.callback(

    Output(
        "communication-output",
        "children"
    ),

    Output(
        "communication-graph",
        "figure"
    ),

    Output(
        "packet-loss-graph",
        "figure"
    ),

    Output(
        "signal-noise-graph",
        "figure"
    ),

    Output(
        "ml-risk-graph",
        "figure"
    ),

    Input(
        "live-update",
        "n_intervals"
    ),

    Input(
        "communication-button",
        "n_clicks"
    )

)
def communication_analysis(
    interval,
    communication_click
):

    distance = random.uniform(

        10000,

        500000

    )

    delay = calculate_delay(
        distance
    )

    noise = generate_noise()

    packet_loss = (
        calculate_packet_loss(
            distance,
            noise
        )
    )

    signal_strength = (
        calculate_signal_strength(
            packet_loss,
            noise
        )
    )

    ml_risk = (
        predict_signal_risk(

            distance,

            noise,

            packet_loss

        )
    )

    packet_received = (

        random.random()

        >

        packet_loss / 100

    )

    if packet_loss < 5:

        condition = (
            "🟢 EXCELLENT"
        )

    elif packet_loss < 15:

        condition = (
            "🟡 GOOD"
        )

    elif packet_loss < 30:

        condition = (
            "🟠 MODERATE"
        )

    else:

        condition = (
            "🔴 POOR"
        )

    packet_status = (

        "✅ PACKET RECEIVED"

        if packet_received

        else

        "❌ PACKET LOST"

    )

    # ========================================================
    # CARDS
    # ========================================================

    cards = [

        html.Div(

            [

                html.H4(
                    "📏 Distance"
                ),

                html.H2(
                    f"{distance:,.0f} km"
                )

            ],

            style=CARD_STYLE

        ),

        html.Div(

            [

                html.H4(
                    "⏱️ Delay"
                ),

                html.H2(
                    f"{delay:.4f} s"
                )

            ],

            style=CARD_STYLE

        ),

        html.Div(

            [

                html.H4(
                    "📦 Packet Loss"
                ),

                html.H2(
                    f"{packet_loss:.2f}%"
                ),

                html.P(
                    condition
                )

            ],

            style=CARD_STYLE

        ),

        html.Div(

            [

                html.H4(
                    "📡 Noise"
                ),

                html.H2(
                    f"{noise:.2f}"
                )

            ],

            style=CARD_STYLE

        ),

        html.Div(

            [

                html.H4(
                    "📶 Signal"
                ),

                html.H2(
                    f"{signal_strength:.2f}%"
                )

            ],

            style=CARD_STYLE

        ),

        html.Div(

            [

                html.H4(
                    "🤖 ML Risk"
                ),

                html.H2(
                    f"{ml_risk:.2f}%"
                )

            ],

            style=CARD_STYLE

        ),

        html.Div(

            [

                html.H4(
                    "📡 Transmission"
                ),

                html.H2(
                    packet_status
                )

            ],

            style=CARD_STYLE

        )

    ]

    # ========================================================
    # DATA FOR GRAPHS
    # ========================================================

    np.random.seed(
        int(interval) + 100
    )

    distances = np.linspace(

        10000,

        500000,

        60

    )

    noises = np.random.normal(

        5,

        2,

        60

    )

    packet_losses = []

    delays = []

    signal_strengths = []

    ml_risks = []

    for (

        d,

        n

    ) in zip(

        distances,

        noises

    ):

        p = calculate_packet_loss(

            d,

            n

        )

        packet_losses.append(
            p
        )

        delays.append(

            calculate_delay(
                d
            )

        )

        signal_strengths.append(

            calculate_signal_strength(

                p,

                n

            )

        )

        ml_risks.append(

            predict_signal_risk(

                d,

                n,

                p

            )

        )

    # ========================================================
    # DELAY GRAPH
    # ========================================================

    delay_fig = go.Figure()

    delay_fig.add_trace(

        go.Scatter(

            x=distances,

            y=delays,

            mode="lines+markers",

            name="Propagation Delay"

        )

    )

    delay_fig.update_layout(

        title=
            "⏱️ Propagation Delay vs Distance",

        xaxis_title=
            "Distance (km)",

        yaxis_title=
            "Delay (seconds)",

        paper_bgcolor="#020202",

        plot_bgcolor="#020202",

        font=dict(
            color="white"
        )

    )

    # ========================================================
    # PACKET LOSS
    # ========================================================

    packet_fig = go.Figure()

    packet_fig.add_trace(

        go.Scatter(

            x=distances,

            y=packet_losses,

            mode="lines+markers",

            name="Packet Loss %"

        )

    )

    packet_fig.update_layout(

        title=
            "📦 Packet Loss Percentage",

        xaxis_title=
            "Distance (km)",

        yaxis_title=
            "Packet Loss (%)",

        paper_bgcolor="#020202",

        plot_bgcolor="#020202",

        font=dict(
            color="white"
        )

    )

    # ========================================================
    # SIGNAL AND NOISE
    # ========================================================

    signal_fig = go.Figure()

    signal_fig.add_trace(

        go.Scatter(

            x=distances,

            y=signal_strengths,

            mode="lines",

            name="Signal Strength (%)"

        )

    )

    signal_fig.add_trace(

        go.Scatter(

            x=distances,

            y=noises,

            mode="lines",

            name="Noise"

        )

    )

    signal_fig.update_layout(

        title=
            "📡 Signal Strength and Noise",

        xaxis_title=
            "Distance (km)",

        yaxis_title=
            "Signal / Noise",

        paper_bgcolor="#020202",

        plot_bgcolor="#020202",

        font=dict(
            color="white"
        )

    )

    # ========================================================
    # ML RISK
    # ========================================================

    ml_fig = go.Figure()

    ml_fig.add_trace(

        go.Scatter(

            x=distances,

            y=ml_risks,

            mode="lines+markers",

            name="ML Signal Loss Risk"

        )

    )

    ml_fig.update_layout(

        title=
            "🤖 ML Predicted Signal Loss Risk",

        xaxis_title=
            "Distance (km)",

        yaxis_title=
            "Risk (%)",

        paper_bgcolor="#020202",

        plot_bgcolor="#020202",

        font=dict(
            color="white"
        )

    )

    return (

        cards,

        delay_fig,

        packet_fig,

        signal_fig,

        ml_fig

    )


# ============================================================
# CHATBOT CALLBACK
# ============================================================

@app.callback(

    Output(
        "chat-output",
        "children"
    ),

    Input(
        "send-chat-button",
        "n_clicks"
    ),

    State(
        "chat-input",
        "value"
    ),

    prevent_initial_call=True

)
def chatbot_callback(
    n_clicks,
    message
):

    return chatbot(
        message
    )


# ============================================================
# CHANGE PASSWORD CALLBACK
# ============================================================

@app.callback(

    Output(
        "password-message",
        "children"
    ),

    Input(
        "change-password-button",
        "n_clicks"
    ),

    State(
        "old-password",
        "value"
    ),

    State(
        "new-password",
        "value"
    ),

    State(
        "confirm-password",
        "value"
    ),

    State(
        "session-store",
        "data"
    ),

    prevent_initial_call=True

)
def password_callback(

    n_clicks,

    old_password,

    new_password,

    confirm_password,

    session

):

    if new_password != confirm_password:

        return (
            "❌ New passwords do not match."
        )

    username = session.get(
        "username",
        ""
    )

    success, message = (
        change_user_password(

            username,

            old_password,

            new_password

        )
    )

    if success:

        return (
            "✅ " + message
        )

    return (
        "❌ " + message
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)

    print(
        "🚀 DEEP SPACE COMMUNICATION "
        "MISSION CONTROL"
    )

    print("=" * 70)

    print(
        "🔐 Login"
    )

    print(
        "📝 Sign Up"
    )

    print(
        "🔑 Forgot Password"
    )

    print(
        "🔒 Change Password"
    )

    print(
        "🚪 Logout"
    )

    print(
        "🛰️ Live TLE Satellite Tracking"
    )

    print(
        "🌍 3D Earth"
    )

    print(
        "🪐 Planets"
    )

    print(
        "📡 Ground Stations"
    )

    print(
        "〰️ Communication Waves"
    )

    print(
        "⏱️ Communication Delay"
    )

    print(
        "📦 Packet Loss"
    )

    print(
        "📡 Noise"
    )

    print(
        "📶 Signal Strength"
    )

    print(
        "🤖 ML Signal Prediction"
    )

    print(
        "💬 AI Chatbot"
    )

    print("=" * 70)

    print(
        "🌐 Open:"
    )

    print(
        f"http://127.0.0.1:{APP_PORT}"
    )

    print("=" * 70)

    app.run(
        debug=True,
        port=APP_PORT
    )
    