import requests

# --- CONFIG ---
API_KEY = "6533aeb5-593a-4312-ad1b-6adecebc9b0f"  # Get one free: http://bustime.mta.info/wiki/Developers
MONITORING_REF = "307582"     # Stop ID (MonitoringRef) - change for your location

# --- FUNCTION TO FETCH LIVE BUS LOCATIONS ---
def get_nyc_buses(api_key, stop_id):
    url = "http://bustime.mta.info/api/siri/stop-monitoring.json"
    params = {
        "key": api_key,
        "MonitoringRef": stop_id
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    buses = []
    visits = data["Siri"]["ServiceDelivery"]["StopMonitoringDelivery"][0]["MonitoredStopVisit"]
    for visit in visits:
        mvj = visit["MonitoredVehicleJourney"]
        location = mvj["VehicleLocation"]
        buses.append({
            "line": mvj["PublishedLineName"],
            "lat": location["Latitude"],
            "lon": location["Longitude"],
            "destination": mvj.get("DestinationName", "Unknown")
        })
    return buses

# --- MAIN ---
if __name__ == "__main__":
    buses = get_nyc_buses(API_KEY, MONITORING_REF)
    print(f"Active buses near stop {MONITORING_REF}:")
    for bus in buses:
        print(f"Route {bus['line']} to {bus['destination']}: ({bus['lat']}, {bus['lon']})")