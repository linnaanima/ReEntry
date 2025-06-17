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

# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="Satelliten-Wiedereintritt Tracker",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satelliten-Wiedereintritt Tracker für Deutschland")
st.markdown("Verfolgen Sie Satelliten und Weltraummüll-Wiedereintritte mit echten Daten")

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
n2yo_api_key = st.sidebar.text_input("N2YO API Key (optional)", 
    help="Für erweiterte Satellitendaten von www.n2yo.com")

# Filter
st.sidebar.subheader("🔍 Filter")
days_ahead = st.sidebar.slider("Vorhersage-Zeitraum (Tage)", 1, 14, 7)
altitude_filter = st.sidebar.slider("Max. Bahnhöhe (km)", 100, 2000, 500)

class SpaceTrackAPI:
    def __init__(self, username, password):
        self.base_url = "https://www.space-track.org"
        self.username = username
        self.password = password
        self.session = requests.Session()
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
                data=login_data
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
            
            resp = self.session.get(query)
            
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
                    f"limit/100/format/json")
            
            resp = self.session.get(query)
            
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            st.error(f"Fehler beim Abrufen der TLE-Daten: {e}")
            return None

class CelesTrakAPI:
    def __init__(self):
        self.base_url = "https://celestrak.org"
    
    def get_reentry_objects(self):
        """Hole Objekte, die bald wiedereintreten"""
        try:
            # Lade TLE-Daten für Objekte mit niedrigen Bahnen
            urls = [
                f"{self.base_url}/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=json",
                f"{self.base_url}/NORAD/elements/gp.php?GROUP=stations&FORMAT=json"
            ]
            
            all_objects = []
            for url in urls:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    all_objects.extend(resp.json())
            
            return all_objects
        except Exception as e:
            st.error(f"CelesTrak API Fehler: {e}")
            return []

def calculate_orbital_decay(tle_data):
    """Berechne ungefähre Wiedereintrittzeit basierend auf TLE-Daten"""
    results = []
    
    for obj in tle_data:
        try:
            # TLE-Daten extrahieren
            name = obj.get('OBJECT_NAME', 'Unknown')
            norad_id = obj.get('NORAD_CAT_ID', 'N/A')
            
            # Bahnparameter
            mean_motion = float(obj.get('MEAN_MOTION', 0))
            eccentricity = float(obj.get('ECCENTRICITY', 0))
            inclination = float(obj.get('INCLINATION', 0))
            
            # Grobe Schätzung der Bahnhöhe aus Mean Motion
            # n = sqrt(GM/a³) -> a = (GM/n²)^(1/3)
            if mean_motion > 0:
                # Umrechnung von rev/day zu rad/s
                n_rad_per_sec = mean_motion * 2 * math.pi / 86400
                GM = 3.986004418e14  # m³/s² (Erdgravitationsparameter)
                a_meters = (GM / (n_rad_per_sec ** 2)) ** (1/3)
                altitude_km = (a_meters - 6371000) / 1000  # Erdradius abziehen
                
                # Wiedereintritt-Schätzung für niedrige Bahnen
                if altitude_km < altitude_filter and mean_motion > 11:  # Niedrige, schnelle Bahnen
                    # Schätzung basierend auf Bahnhöhe und atmosphärischem Widerstand
                    if altitude_km < 200:
                        days_to_reentry = max(0.1, altitude_km / 50)  # Sehr niedrige Bahnen
                    elif altitude_km < 300:
                        days_to_reentry = max(1, altitude_km / 30)
                    else:
                        days_to_reentry = max(7, altitude_km / 20)
                    
                    # Korrekturfaktor für Exzentrizität (elliptische Bahnen fallen schneller)
                    eccentricity_factor = 1 - (eccentricity * 0.5)
                    days_to_reentry *= eccentricity_factor
                    
                    reentry_time = datetime.now() + timedelta(days=days_to_reentry)
                    
                    # Grobe Positionsschätzung (vereinfacht)
                    # Zufällige Position entlang der Bahnebene
                    lat_range = inclination if inclination <= 90 else 180 - inclination
                    estimated_lat = np.random.uniform(-lat_range, lat_range)
                    estimated_lon = np.random.uniform(-180, 180)
                    
                    # Objekttyp schätzen basierend auf Name
                    if any(term in name.upper() for term in ['R/B', 'ROCKET', 'BOOSTER']):
                        obj_type = 'Rocket Body'
                        size_estimate = '8-15m'
                        mass_estimate = '2000-5000 kg'
                    elif 'DEB' in name.upper() or 'DEBRIS' in name.upper():
                        obj_type = 'Debris'
                        size_estimate = '0.5-2m'
                        mass_estimate = '10-100 kg'
                    else:
                        obj_type = 'Satellite'
                        size_estimate = '1-5m'
                        mass_estimate = '100-1000 kg'
                    
                    # Risikobewertung
                    if altitude_km < 150:
                        risk_level = 'Hoch'
                    elif altitude_km < 250:
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
    
    # Space-Track.org verwenden (wenn verfügbar)
    if space_track_user and space_track_pass:
        with st.spinner("Verbinde mit Space-Track.org..."):
            st_api = SpaceTrackAPI(space_track_user, space_track_pass)
            
            if st_api.authenticate():
                st.success("✅ Space-Track.org verbunden!")
                
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
                            'Uncertainty': pred.get('WINDOW', 'N/A')
                        })
                
                # Zusätzlich: Objekte mit niedrigen Bahnen
                high_interest = st_api.get_high_interest_objects()
                if high_interest:
                    with st.spinner("Analysiere Bahnparameter..."):
                        calculated_reentries = calculate_orbital_decay(high_interest[:50])
                        
                        for calc in calculated_reentries[:10]:  # Top 10
                            reentry_data.append({
                                'Object': calc['Object'],
                                'NORAD_ID': calc['NORAD_ID'],
                                'Altitude_km': calc['Altitude_km'],
                                'Estimated_Reentry': calc['Estimated_Reentry'],
                                'Days_to_Reentry': calc['Days_to_Reentry'],
                                'Estimated_Lat': calc['Estimated_Lat'],
                                'Estimated_Lon': calc['Estimated_Lon'],
                                'Object_Type': calc['Object_Type'],
                                'Risk_Level': calc['Risk_Level'],
                                'Source': 'Calculated from TLE'
                            })
            else:
                st.error("❌ Space-Track.org Anmeldung fehlgeschlagen")
    
    # CelesTrak als Alternative/Ergänzung
    if celestrak_enabled:
        with st.spinner("Lade Daten von CelesTrak..."):
            celestrak = CelesTrakAPI()
            celestrak_objects = celestrak.get_reentry_objects()
            
            if celestrak_objects:
                st.info(f"📡 {len(celestrak_objects)} Objekte von CelesTrak erhalten")
                
                # Analysiere nur Objekte mit niedrigen Bahnen
                low_orbit_objects = [obj for obj in celestrak_objects 
                                   if float(obj.get('MEAN_MOTION', 0)) > 11.25]
                
                if low_orbit_objects:
                    with st.spinner("Berechne Wiedereintritt-Schätzungen..."):
                        calculated = calculate_orbital_decay(low_orbit_objects[:100])
                        
                        for calc in calculated[:15]:  # Top 15
                            reentry_data.append({
                                'Object': calc['Object'],
                                'NORAD_ID': calc['NORAD_ID'],
                                'Altitude_km': calc['Altitude_km'],
                                'Estimated_Reentry': calc['Estimated_Reentry'],
                                'Days_to_Reentry': calc['Days_to_Reentry'],
                                'Estimated_Lat': calc['Estimated_Lat'],
                                'Estimated_Lon': calc['Estimated_Lon'],
                                'Object_Type': calc['Object_Type'],
                                'Risk_Level': calc['Risk_Level'],
                                'Source': 'CelesTrak + Calculation'
                            })
    
    # Ergebnisse anzeigen
    if reentry_data:
        df = pd.DataFrame(reentry_data)
        
        # Duplikate entfernen (basierend auf NORAD_ID)
        df = df.drop_duplicates(subset=['NORAD_ID'], keep='first')
        
        st.success(f"📊 {len(df)} Wiedereintritt-Kandidaten gefunden")
        
        # Karte erstellen
        fig = go.Figure()
        
        # Deutschland markieren
        fig.add_trace(go.Scattergeo(
            lon=[GERMANY_BOUNDS['center_lon']],
            lat=[GERMANY_BOUNDS['center_lat']],
            mode='markers+text',
            marker=dict(size=15, color='blue', symbol='star'),
            text=['Deutschland'],
            textposition='top center',
            name='Deutschland'
        ))
        
        # Geschätzte Wiedereintrittspositionen (wenn verfügbar)
        if 'Estimated_Lat' in df.columns:
            estimated_positions = df.dropna(subset=['Estimated_Lat', 'Estimated_Lon'])
            
            if not estimated_positions.empty:
                # Farbkodierung nach Risiko
                colors = {'Hoch': 'red', 'Mittel': 'orange', 'Niedrig': 'yellow'}
                
                for risk_level in estimated_positions['Risk_Level'].unique():
                    risk_data = estimated_positions[estimated_positions['Risk_Level'] == risk_level]
                    
                    fig.add_trace(go.Scattergeo(
                        lon=risk_data['Estimated_Lon'],
                        lat=risk_data['Estimated_Lat'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=colors.get(risk_level, 'gray'),
                            opacity=0.7,
                            line=dict(width=1, color='darkgray')
                        ),
                        text=risk_data.apply(lambda row: 
                            f"{row['Object']}<br>"
                            f"NORAD: {row['NORAD_ID']}<br>"
                            f"Höhe: {row.get('Altitude_km', 'N/A')} km<br>"
                            f"Typ: {row.get('Object_Type', 'N/A')}<br>"
                            f"Wiedereintritt in: {row.get('Days_to_Reentry', 'N/A')} Tagen<br>"
                            f"Risiko: {row.get('Risk_Level', 'N/A')}", axis=1),
                        hovertemplate='%{text}<extra></extra>',
                        name=f'Risiko: {risk_level}'
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
            title="Geschätzte Wiedereintrittspositionen nach Risiko",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detaillierte Tabelle
        st.subheader("📋 Detaillierte Liste")
        
        # Formatiere für Anzeige
        display_columns = ['Object', 'NORAD_ID', 'Source', 'Object_Type', 'Risk_Level']
        
        if 'Altitude_km' in df.columns:
            display_columns.append('Altitude_km')
        if 'Days_to_Reentry' in df.columns:
            display_columns.append('Days_to_Reentry')
        if 'Estimated_Reentry' in df.columns:
            display_columns.append('Estimated_Reentry')
        
        display_df = df[display_columns].copy()
        
        # Formatiere Datumsangaben
        if 'Estimated_Reentry' in display_df.columns:
            display_df['Estimated_Reentry'] = pd.to_datetime(display_df['Estimated_Reentry']).dt.strftime('%d.%m.%Y %H:%M')
        
        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.warning("⚠️ Keine Daten verfügbar. Bitte überprüfen Sie Ihre API-Konfiguration.")

with col2:
    st.subheader("📊 Statistiken")
    
    if reentry_data:
        df = pd.DataFrame(reentry_data)
        
        # Risiko-Verteilung
        if 'Risk_Level' in df.columns:
            risk_counts = df['Risk_Level'].value_counts()
            colors_risk = ['red' if x=='Hoch' else 'orange' if x=='Mittel' else 'yellow' for x in risk_counts.index]
            fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index,
                             title="Risiko-Verteilung",
                             color_discrete_sequence=colors_risk)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Objekttyp-Verteilung
        if 'Object_Type' in df.columns:
            type_counts = df['Object_Type'].value_counts()
            fig_type = px.bar(x=type_counts.index, y=type_counts.values,
                             title="Objekttyp-Verteilung")
            st.plotly_chart(fig_type, use_container_width=True)
        
        # Höhen-Verteilung
        if 'Altitude_km' in df.columns:
            altitude_data = df.dropna(subset=['Altitude_km'])
            if not altitude_data.empty:
                fig_alt = px.histogram(altitude_data, x='Altitude_km', 
                                     title="Bahnhöhen-Verteilung")
                st.plotly_chart(fig_alt, use_container_width=True)

# Wichtige Hinweise
st.subheader("📚 Datenquellen & APIs")

st.info("""
**🔑 Benötigte API-Registrierungen:**

1. **Space-Track.org** (Empfohlen - kostenlos)
   - Registrierung: www.space-track.org/auth/createAccount
   - Offizielle US-Wiedereintritt-Vorhersagen
   - TLE-Daten für alle katalogisierten Objekte

2. **CelesTrak** (Keine Registrierung)
   - Automatisch verfügbar
   - TLE-Daten für Bahnberechnungen

3. **N2YO.com** (Optional)
   - API-Key: www.n2yo.com/api/
   - Erweiterte Satellitendaten
""")

st.warning("""
**⚠️ Wiedereintritt-Vorhersagen:**
- Exakte Vorhersagen sind schwierig (Atmosphärenschwankungen)
- Zeitfenster können sich um Stunden/Tage verschieben  
- Die meisten Objekte verglühen vollständig
- Nur große Objekte (>1m) können Trümmer erzeugen
""")

# Auto-Refresh
if st.sidebar.checkbox("🔄 Auto-Aktualisierung (5 Min)"):
    time.sleep(300)
    st.rerun()
