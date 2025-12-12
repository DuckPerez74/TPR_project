import geoip2.database
from math import radians, cos, sin, asin, sqrt
import os

class GeoIPHelper:
    def __init__(self, db_path: str = "data/GeoLite2-City.mmdb"):
        self.db_path = db_path
        self.reader = None
        
    def _get_reader(self):
        if self.reader is None:
            try:
                self.reader = geoip2.database.Reader(self.db_path)
            except FileNotFoundError:
                return None
        return self.reader

    def get_location(self, ip: str):
        reader = self._get_reader()
        if not reader:
            return None
            
        # Ignore private IPs
        if ip.startswith(('192.168.', '10.', '172.16.', '127.')):
            return None

        try:
            response = reader.city(ip)
            return {
                'lat': response.location.latitude,
                'lon': response.location.longitude,
                'country': response.country.name
            }
        except Exception:
            return None

    def close(self):
        if self.reader:
            self.reader.close()
            self.reader = None

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2]) 
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r
