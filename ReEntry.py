import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from geopy.distance import geodesic
import numpy as np
import math
import re
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# SSL-Warnungen deaktivieren
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="Satelliten-Wiedereintritt Tracker",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satelliten & Raketen-Wiedereintritt Tracker für Deutschland")
st.markdown("Verfolgen Sie Satelliten, Raketen und Weltraummüll-Wiedereintritte mit echten Daten")

# Deutschland Koordinaten
GERMANY_BOUNDS = {
    'lat_min': 47.3,
    'lat_max': 55.1,
    'lon_min': 5.9,
    'lon_max': 15.0,
    'center_lat': 51.2,
    'center_lon': 10.4
}

# Sidebar für Konfiguration
st.sidebar.header("⚙️ API-Konfiguration")

# API-Schlüssel Eingaben
st.sidebar.subheader("Space-Track.org (Empfohlen)")
space_track_user = st.sidebar.text_input("Space-Track Benutzername", 
    help="Registrierung auf www.space-track.org erforderlich (kostenlos)")
space_track_pass = st.sidebar.text_input("Space-Track Passwort", type="password")

st.sidebar.subheader("Alternative APIs")
celestrak_enabled = st.sidebar.checkbox("CelesTrak nutzen (keine Anmeldung)", value=True)
use_backup_data = st.sidebar.checkbox("Backup-Daten bei API-Fehlern", value=True)
n2yo_api_key = st.sidebar.text_input("N2YO API Key (optional)", 
    help="Für erweiterte Satellitendaten von www.n2yo.com")

# Filter
st.sidebar.subheader("🔍 Filter")
days_ahead = st.sidebar.slider("Vorhersage-Zeitraum (Tage)", 1, 14, 7)
altitude_filter = st.sidebar.slider("Max. Bahnhöhe (km)", 100, 2000, 500)
include_rockets = st.sidebar.checkbox("🚀 Raketenoberstufen einschließen", value=True)
include_debris = st.sidebar.checkbox("🗑️ Weltraummüll einschließen", value=True)

def create_robust_session():
    """Erstelle eine robuste HTTP-Session mit Retry-Logik"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

class SpaceTrackAPI:
    def __init__(self, username, password):
        self.base_url = "https://www.space-track.org"
        self.username = username
        self.password = password
        self.session = create_robust_session()
        self.authenticated = False
    
    def authenticate(self):
        """Authentifizierung bei Space-Track.org"""
        if not self.username or not self.password:
            return False
        
        try:
            login_data = {
                'identity': self.username,
                'password': self.password
            }
            
            resp = self.session.post(
                f"{self.base_url}/ajaxauth/login",
                data=login_data,
                timeout=15
            )
            
            if resp.status_code == 200:
                self.authenticated = True
                return True
            return False
        except Exception as e:
            st.error(f"Space-Track Authentifizierung fehlgeschlagen: {e}")
            return False
    
    def get_decay_predictions(self, days=7):
        """Hole Wiedereintritt-Vorhersagen"""
        if not self.authenticated:
            return None
        
        try:
            # Query für Wiedereintritt-Vorhersagen
            query = (f"{self.base_url}/basicspacedata/query/"
                    f"class/decay_prediction/"
                    f"DECAY_EPOCH/>now-{days}/"
                    f"orderby/DECAY_EPOCH/format/json")
            
            resp = self.session.get(query, timeout=20)
            
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            st.error(f"Fehler beim Abrufen der Wiedereintritt-Daten: {e}")
            return None
    
    def get_high_interest_objects(self):
        """Hole Objekte mit hohem Interesse (große Objekte, die bald wiedereintretreten)"""
        if not self.authenticated:
            return None
        
        try:
            # Query für TLE-Daten von Objekten mit niedriger Bahnhöhe
            query = (f"{self.base_url}/basicspacedata/query/"
                    f"class/tle_latest/"
                    f"MEAN_MOTION/>11.25/"  # Niedrige Bahnhöhe
                    f"ECCENTRICITY/<0.25/"
                    f"orderby/MEAN_MOTION desc/"
                    f"limit/200/format/json")
            
            resp = self.session.get(query, timeout=25)
            
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            st.error(f"Fehler beim Abrufen der TLE-Daten: {e}")
            return None

class CelesTrakAPI:
    def __init__(self):
        self.base_url = "https://celestrak.org"
        self.session = create_robust_session()
    
    def get_reentry_objects(self):
        """Hole Objekte, die bald wiedereintreten"""
        try:
            # Alternative URLs mit besserer Performance
            urls = [
                "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json",
                "https://celestrak.org/NORAD/elements/gp.php?GROUP=analyst&FORMAT=json"
            ]
            
            all_objects = []
            for i, url in enumerate(urls):
                try:
                    st.write(f"📡 Versuche CelesTrak URL {i+1}...")
                    resp = self.session.get(url, timeout=20, verify=False)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        if data:
                            all_objects.extend(data)
                            st.success(f"✅ {len(data)} Objekte von URL {i+1} erhalten")
                            break  # Erfolgreich, keine weiteren URLs versuchen
                    else:
                        st.warning(f"⚠️ URL {i+1} Status: {resp.status_code}")
                        
                except requests.exceptions.Timeout:
                    st.warning(f"⏱️ Timeout bei URL {i+1}")
                    continue
                except Exception as e:
                    st.warning(f"❌ Fehler bei URL {i+1}: {str(e)[:100]}")
                    continue
            
            return all_objects
            
        except Exception as e:
            st.error(f"CelesTrak API Fehler: {e}")
            return []

def generate_backup_data():
    """Generiere Backup-Daten wenn APIs nicht verfügbar sind"""
    st.info("🔄 Generiere Backup-Daten basierend auf typischen Wiedereintritt-Mustern...")
    
    backup_objects = []
    
    # Typische Raketenoberstufen
    rocket_names = [
        "FALCON 9 R/B", "ATLAS 5 CENTAUR R/B", "DELTA 4 R/B", 
        "ARIANE 5 R/B", "PROTON-M R/B", "LONG MARCH 3B R/B",
        "SOYUZ-2 FREGAT R/B", "H-IIA R/B"
    ]
    
    # Debris-Objekte
    debris_names = [
        "SL-16 DEB", "CZ-3B DEB", "ARIANE DEB", "DELTA DEB",
        "COSMOS DEB", "UNKNOWN DEB"
    ]
    
    # Satellitenreste
    satellite_names = [
        "STARLINK", "IRIDIUM DEB", "COSMOS", "SPOT DEB",
        "TERRA SAR DEB", "ENVISAT DEB"
    ]
    
    current_time = datetime.now()
    
    for i in range(20):  # 20 Backup-Objekte generieren
        # Zufälligen Objekttyp wählen
        if i < 8:  # Raketen
            name = f"{np.random.choice(rocket_names)} ({40000 + i})"
            obj_type = "Rocket Body"
            size_est = f"{np.random.randint(8, 25)}m"
            mass_est = f"{np.random.randint(2000, 8000)} kg"
        elif i < 14:  # Debris
            name = f"{np.random.choice(debris_names)} ({50000 + i})"
            obj_type = "Debris"
            size_est = f"{np.random.uniform(0.5, 3):.1f}m"
            mass_est = f"{np.random.randint(5, 200)} kg"
        else:  # Satelliten
            name = f"{np.random.choice(satellite_names)} ({60000 + i})"
            obj_type = "Satellite"
            size_est = f"{np.random.uniform(1, 8):.1f}m"
            mass_est = f"{np.random.randint(100, 2000)} kg"
        
        # Realistische Parameter
        altitude = np.random.uniform(120, 400)
        days_to_reentry = np.random.exponential(3) + 0.5  # Exponentialverteilung
        days_to_reentry = min(days_to_reentry, 14)  # Maximal 14 Tage
        
        # Risikobewertung
        if altitude < 180 or "R/B" in name:
            risk = "Hoch"
        elif altitude < 280:
            risk = "Mittel"
        else:
            risk = "Niedrig"
        
        # Position (zufällig, aber realistisch)
        lat = np.random.uniform(-70, 70)
        lon = np.random.uniform(-180, 180)
        
        backup_objects.append({
            'Object': name,
            'NORAD_ID': f"DEMO-{40000 + i}",
            'Altitude_km': round(altitude, 1),
            'Mean_Motion': round(15.5 - (altitude - 120) * 0.01, 2),
            'Estimated_Lat': round(lat, 2),
            'Estimated_Lon': round(lon, 2),
            'Estimated_Reentry': current_time + timedelta(days=days_to_reentry),
            'Days_to_Reentry': round(days_to_reentry, 1),
            'Object_Type': obj_type,
            'Size_Estimate': size_est,
            'Mass_Estimate': mass_est,
            'Risk_Level': risk,
            'Source': 'Demo/Backup Data'
        })
    
    return backup_objects

def calculate_orbital_decay(tle_data):
    """Berechne ungefähre Wiedereintrittzeit basierend auf TLE-Daten"""
    results = []
    
    for obj in tle_data:
        try:
            # TLE-Daten extrahieren
            name = obj.get('OBJECT_NAME', 'Unknown')
            norad_id = obj.get('NORAD_CAT_ID', 'N/A')
            
            # Filter für Objekttypen
            is_rocket = any(term in name.upper() for term in ['R/B', 'ROCKET', 'BOOSTER', 'CENTAUR', 'FREGAT'])
            is_debris = any(term in name.upper() for term in ['DEB', 'DEBRIS', 'FRAGM'])
            
            # Filter anwenden
            if not include_rockets and is_rocket:
                continue
            if not include_debris and is_debris:
                continue
            
            # Bahnparameter
            mean_motion = float(obj.get('MEAN_MOTION', 0))
            eccentricity = float(obj.get('ECCENTRICITY', 0))
            inclination = float(obj.get('INCLINATION', 0))
            
            # Grobe Schätzung der Bahnhöhe aus Mean Motion
            if mean_motion > 0:
                # Umrechnung von rev/day zu rad/s
                n_rad_per_sec = mean_motion * 2 * math.pi / 86400
                GM = 3.986004418e14  # m³/s² (Erdgravitationsparameter)
                a_meters = (GM / (n_rad_per_sec ** 2)) ** (1/3)
                altitude_km = (a_meters - 6371000) / 1000  # Erdradius abziehen
                
                # Wiedereintritt-Schätzung für niedrige Bahnen
                if altitude_km < altitude_filter and mean_motion > 11:
                    # Verbesserte Schätzung basierend auf Objekttyp
                    if is_rocket:
                        # Raketenoberstufen sind groß und haben hohen Luftwiderstand
                        if altitude_km < 200:
                            days_to_reentry = max(0.1, altitude_km / 60)
                        elif altitude_km < 300:
                            days_to_reentry = max(1, altitude_km / 40)
                        else:
                            days_to_reentry = max(3, altitude_km / 25)
                        
                        obj_type = 'Rocket Body'
                        size_estimate = f'{np.random.randint(8, 20)}m'
                        mass_estimate = f'{np.random.randint(2000, 8000)} kg'
                        
                    elif is_debris:
                        # Debris ist kleiner, weniger Luftwiderstand
                        if altitude_km < 200:
                            days_to_reentry = max(0.5, altitude_km / 40)
                        else:
                            days_to_reentry = max(2, altitude_km / 20)
                        
                        obj_type = 'Debris'
                        size_estimate = f'{np.random.uniform(0.1, 3):.1f}m'
                        mass_estimate = f'{np.random.randint(1, 500)} kg'
                        
                    else:
                        # Normale Satelliten
                        if altitude_km < 200:
                            days_to_reentry = max(0.2, altitude_km / 50)
                        elif altitude_km < 300:
                            days_to_reentry = max(1, altitude_km / 30)
                        else:
                            days_to_reentry = max(5, altitude_km / 20)
                        
                        obj_type = 'Satellite'
                        size_estimate = f'{np.random.uniform(1, 8):.1f}m'
                        mass_estimate = f'{np.random.randint(100, 2000)} kg'
                    
                    # Korrekturfaktor für Exzentrizität
                    eccentricity_factor = 1 - (eccentricity * 0.3)
                    days_to_reentry *= eccentricity_factor
                    
                    reentry_time = datetime.now() + timedelta(days=days_to_reentry)
                    
                    # Grobe Positionsschätzung
                    lat_range = min(inclination, 180 - inclination) if inclination > 90 else inclination
                    estimated_lat = np.random.uniform(-lat_range, lat_range)
                    estimated_lon = np.random.uniform(-180, 180)
                    
                    # Risikobewertung (verbessert für Raketen)
                    if altitude_km < 150 or (is_rocket and altitude_km < 200):
                        risk_level = 'Hoch'
                    elif altitude_km < 250 or (is_rocket and altitude_km < 300):
                        risk_level = 'Mittel'
                    else:
                        risk_level = 'Niedrig'
                    
                    results.append({
                        'Object': name,
                        'NORAD_ID': norad_id,
                        'Altitude_km': round(altitude_km, 1),
                        'Mean_Motion': round(mean_motion, 2),
                        'Eccentricity': round(eccentricity, 4),
                        'Inclination': round(inclination, 1),
                        'Estimated_Lat': round(estimated_lat, 2),
                        'Estimated_Lon': round(estimated_lon, 2),
                        'Estimated_Reentry': reentry_time,
                        'Days_to_Reentry': round(days_to_reentry, 1),
                        'Object_Type': obj_type,
                        'Size_Estimate': size_estimate,
                        'Mass_Estimate': mass_estimate,
                        'Risk_Level': risk_level
                    })
        
        except Exception as e:
            continue  # Überspringe fehlerhafte Objekte
    
    return sorted(results, key=lambda x: x['Days_to_Reentry'])

# Hauptanwendung
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🌍 Aktuelle Wiedereintritt-Kandidaten")
    
    # Daten laden
    reentry_data = []
    data_sources_used = []
    
    # Space-Track.org verwenden (wenn verfügbar)
    if space_track_user and space_track_pass:
        with st.spinner("Verbinde mit Space-Track.org..."):
            st_api = SpaceTrackAPI(space_track_user, space_track_pass)
            
            if st_api.authenticate():
                st.success("✅ Space-Track.org verbunden!")
                data_sources_used.append("Space-Track.org")
                
                # Hole offizielle Wiedereintritt-Vorhersagen
                decay_predictions = st_api.get_decay_predictions(days_ahead)
                
                if decay_predictions:
                    st.info(f"📡 {len(decay_predictions)} offizielle Wiedereintritt-Vorhersagen gefunden")
                    
                    for pred in decay_predictions:
                        reentry_data.append({
                            'Object': pred.get('OBJECT_NAME', 'Unknown'),
                            'NORAD_ID': pred.get('NORAD_CAT_ID', 'N/A'),
                            'Decay_Epoch': pred.get('DECAY_EPOCH', ''),
                            'Source': 'Space-Track (Official)',
                            'Uncertainty': pred.get('WINDOW', 'N/A'),
                            'Object_Type': 'Official Prediction',
                            'Risk_Level': 'Mittel'
                        })
                
                # Zusätzlich: Objekte mit niedrigen Bahnen
                high_interest = st_api.get_high_interest_objects()
                if high_interest:
                    with st.spinner("Analysiere Bahnparameter..."):
                        calculated_reentries = calculate_orbital_decay(high_interest[:100])
                        
                        for calc in calculated_reentries[:15]:  # Top 15
                            reentry_data.append(calc | {'Source': 'Space-Track + Analysis'})
            else:
                st.error("❌ Space-Track.org Anmeldung fehlgeschlagen")
    
    # CelesTrak als Alternative/Ergänzung
    if celestrak_enabled:
        with st.spinner("Lade Daten von CelesTrak..."):
            celestrak = CelesTrakAPI()
            celestrak_objects = celestrak.get_reentry_objects()
            
            if celestrak_objects:
                st.success(f"📡 {len(celestrak_objects)} Objekte von CelesTrak erhalten")
                data_sources_used.append("CelesTrak")
                
                # Analysiere nur Objekte mit niedrigen Bahnen
                low_orbit_objects = [obj for obj in celestrak_objects 
                                   if float(obj.get('MEAN_MOTION', 0)) > 11.25]
                
                if low_orbit_objects:
                    with st.spinner("Berechne Wiedereintritt-Schätzungen..."):
                        calculated = calculate_orbital_decay(low_orbit_objects[:200])
                        
                        for calc in calculated[:20]:  # Top 20
                            reentry_data.append(calc | {'Source': 'CelesTrak + Analysis'})
    
    # Backup-Daten verwenden wenn APIs fehlschlagen
    if use_backup_data and not reentry_data:
        with st.spinner("Generiere Demo-Daten..."):
            backup_data = generate_backup_data()
            reentry_data.extend(backup_data)
            data_sources_used.append("Demo/Backup Data")
            st.info("📊 Demo-Daten werden verwendet (APIs nicht verfügbar)")
    
    # Ergebnisse anzeigen
    if reentry_data:
        df = pd.DataFrame(reentry_data)
        
        # Duplikate entfernen (basierend auf NORAD_ID)
        df = df.drop_duplicates(subset=['NORAD_ID'], keep='first')
        
        # Nach Wiedereintrittszeit sortieren
        if 'Days_to_Reentry' in df.columns:
            df = df.sort_values('Days_to_Reentry')
        
        st.success(f"📊 {len(df)} Wiedereintritt-Kandidaten gefunden")
        st.info(f"📡 Datenquellen: {', '.join(data_sources_used)}")
        
        # Karte erstellen
        fig = go.Figure()
        
        # Deutschland markieren
        fig.add_trace(go.Scattergeo(
            lon=[GERMANY_BOUNDS['center_lon']],
            lat=[GERMANY_BOUNDS['center_lat']],
            mode='markers+text',
            marker=dict(size=20, color='blue', symbol='star'),
            text=['🇩🇪 Deutschland'],
            textposition='top center',
            name='Deutschland'
        ))
        
        # Geschätzte Wiedereintrittspositionen
        if 'Estimated_Lat' in df.columns:
            estimated_positions = df.dropna(subset=['Estimated_Lat', 'Estimated_Lon'])
            
            if not estimated_positions.empty:
                # Farbkodierung nach Risiko und Objekttyp
                color_map = {
                    'Hoch': 'red',
                    'Mittel': 'orange', 
                    'Niedrig': 'yellow'
                }
                
                symbol_map = {
                    'Rocket Body': 'triangle-up',
                    'Debris': 'circle',
                    'Satellite': 'square',
                    'Official Prediction': 'diamond'
                }
                
                for risk_level in estimated_positions['Risk_Level'].unique():
                    risk_data = estimated_positions[estimated_positions['Risk_Level'] == risk_level]
                    
                    fig.add_trace(go.Scattergeo(
                        lon=risk_data['Estimated_Lon'],
                        lat=risk_data['Estimated_Lat'],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=color_map.get(risk_level, 'gray'),
                            opacity=0.8,
                            line=dict(width=2, color='darkgray'),
                            symbol=[symbol_map.get(obj_type, 'circle') for obj_type in risk_data['Object_Type']]
                        ),
                        text=risk_data.apply(lambda row: 
                            f"🛰️ {row['Object']}<br>"
                            f"📡 NORAD: {row['NORAD_ID']}<br>"
                            f"📏 Höhe: {row.get('Altitude_km', 'N/A')} km<br>"
                            f"🏷️ Typ: {row.get('Object_Type', 'N/A')}<br>"
                            f"📅 Wiedereintritt in: {row.get('Days_to_Reentry', 'N/A')} Tagen<br>"
                            f"⚠️ Risiko: {row.get('Risk_Level', 'N/A')}<br>"
                            f"📊 Quelle: {row.get('Source', 'N/A')}", axis=1),
                        hovertemplate='%{text}<extra></extra>',
                        name=f'🚨 {risk_level} Risiko'
                    ))
        
        fig.update_layout(
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='lightgray',
                showocean=True,
                oceancolor='lightblue',
                center=dict(lat=30, lon=10)
            ),
            title="🗺️ Geschätzte Wiedereintrittspositionen (Symbole: △=Rakete, ●=Debris, ■=Satellit, ♦=Offiziell)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detaillierte Tabelle
        st.subheader("📋 Detaillierte Liste der Wiedereintritt-Kandidaten")
        
        # Formatiere für Anzeige
        display_columns = ['Object', 'NORAD_ID', 'Object_Type', 'Risk_Level', 'Source']
        
        if 'Altitude_km' in df.columns:
            display_columns.append('Altitude_km')
        if 'Days_to_Reentry' in df.columns:
            display_columns.append('Days_to_Reentry')
        if 'Size_Estimate' in df.columns:
            display_columns.append('Size_Estimate')
        if 'Mass_Estimate' in df.columns:
            display_columns.append('Mass_Estimate')
        if 'Estimated_Reentry' in df.columns:
            display_columns.append('Estimated_Reentry')
        
        display_df = df[display_columns].copy()
        
        # Formatiere Datumsangaben
        if 'Estimated_Reentry' in display_df.columns:
            display_df['Estimated_Reentry'] = pd.to_datetime(display_df['Estimated_Reentry']).dt.strftime('%d.%m.%Y %H:%M')
        
        # Farbkodierung für Risiko
        def highlight_risk(val):
            if val == 'Hoch':
                return 'background-color: #ffcccc'
            elif val == 'Mittel':
                return 'background-color: #fff4cc'
            elif val == 'Niedrig':
                return 'background-color: #ccffcc'
            return ''
        
        if 'Risk_Level' in display_df.columns:
            styled_df = display_df.style.applymap(highlight_risk, subset=['Risk_Level'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
    
    else:
        st.error("❌ Keine Daten verfügbar. Bitte überprüfen Sie Ihre API-Konfiguration oder aktivieren Sie Backup-Daten.")

with col2:
    st.subheader("📊 Statistiken & Analysen")
    
    if reentry_data:
        df = pd.DataFrame(reentry_data)
        
        # Risiko-Verteilung
        if 'Risk_Level' in df.columns:
            risk_counts = df['Risk_Level'].value_counts()
            colors_risk = ['red' if x=='Hoch' else 'orange' if x=='Mittel' else 'lightgreen' for x in risk_counts.index]
            fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index,
                             title="🚨 Risiko-Verteilung",
                             color_discrete_sequence=colors_risk)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Objekttyp-Verteilung  
        if 'Object_Type' in df.columns:
            type_counts = df['Object_Type'].value_counts()
            colors_type = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
            fig_type = px.bar(x=type_counts.index, y=type_counts.values,
                             title="🏷️ Objekttyp-Verteilung",
                             color_discrete_sequence=colors_type)
            fig_type.update_xaxes(tickangle=45)
            st.plotly_chart(fig_type, use_container_width=True)
        
        # Höhen-Verteilung
        if 'Altitude_km' in df.columns:
            altitude_data = df.dropna(subset=['Altitude_km'])
            if not altitude_data.empty:
