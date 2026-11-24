import io
import os
import duckdb
import pandas as pd
import requests

DB = 'data/gridsight.duckdb'
START = '2021-01-01'

ECO2MIX = ("https://odre.opendatasoft.com/api/explore/v2.1/catalog/"
           "datasets/eco2mix-national-cons-def/exports/csv")
METEO = "https://archive-api.open-meteo.com/v1/archive"
CITIES = {"paris": (48.85, 2.35), "lyon": (45.76, 4.84), "marseille": (43.30, 5.37),
          "lille": (50.63, 3.06), "toulouse": (43.60, 1.44)}

def fetch_load(end: str) -> pd.DataFrame:
    params = {
        "select" : "date_heure,consommation,taux_co2",
        "where" : f"date_heure >= '{START}' AND date_heure <= '{end}'",
        "delimiter": ",",
    }

    r=requests.get(ECO2MIX, params=params, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # offsets mixtes (+01:00/+02:00) -> on normalise en UTC pour la jointure météo
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df = df.dropna(subset=["consommation"])
    #pas de 30 min ->horaire

    return (df.set_index("date_heure").resample("1h").mean().reset_index() #pour obtenir une ligne par heure
            .rename(columns={"date_heure": "ts"}))




def fetch_weather(end:str) -> pd.DataFrame:
    frames = []
    for name, (lat, lon) in CITIES.items() :
        p = {"latitude": lat, "longitude": lon, "start_date": START, "end_date": end,
             "hourly": "temperature_2m,wind_speed_10m", "timezone": "UTC"}
        r = requests.get(METEO, params=p, timeout=120)
        r.raise_for_status()
        h = r.json()["hourly"]
        frames.append(pd.DataFrame({"ts": pd.to_datetime(h["time"], utc=True),
                                    "temp": h["temperature_2m"], "wind": h["wind_speed_10m"]}))
    # moyenne des 5 villes = proxy national
    return pd.concat(frames).groupby("ts", as_index=False).mean()

def main():
    end = (pd.Timestamp.utcnow() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    load, weather = fetch_load(end), fetch_weather(end)
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = duckdb.connect(DB)
    con.register("load_df", load)
    con.register("weather_df", weather)
    con.execute("CREATE OR REPLACE TABLE raw_load AS SELECT * FROM load_df")
    con.execute("CREATE OR REPLACE TABLE raw_weather AS SELECT * FROM weather_df")
    con.execute("""CREATE OR REPLACE TABLE dataset AS
                   SELECT l.ts, l.consommation, l.taux_co2, w.temp, w.wind
                   FROM raw_load l JOIN raw_weather w USING (ts) ORDER BY l.ts""")
    print(con.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM dataset").fetchall())


if __name__ == "__main__":
    main()