class GPSSimulator {
    constructor() {
        this.zones = {
            home: { lat: 30.0444, lng: 31.2357, name: "Home Zone", radius: 0.005 },
            downtown: { lat: 30.0580, lng: 31.2425, name: "Downtown Cairo", radius: 0.008 },
            park: { lat: 30.0350, lng: 31.2200, name: "Park Zone", radius: 0.006 },
            market: { lat: 30.0650, lng: 31.2500, name: "Market Zone", radius: 0.004 }
        };
        this.currentZone = 'home';
    }

    getCurrentCoordinates() {
        const zone = this.zones[this.currentZone];
        // Add random offset within zone radius
        const offsetLat = (Math.random() - 0.5) * zone.radius * 2;
        const offsetLng = (Math.random() - 0.5) * zone.radius * 2;
        return {
            lat: zone.lat + offsetLat,
            lng: zone.lng + offsetLng,
            zone: this.currentZone,
            zoneName: zone.name
        };
    }

    setZone(zoneName) {
        if (this.zones[zoneName]) {
            this.currentZone = zoneName;
            console.log(`📍 Zone changed to: ${this.zones[zoneName].name}`);
            return true;
        }
        return false;
    }

    getZones() {
        return Object.keys(this.zones).map(key => ({
            id: key,
            name: this.zones[key].name,
            lat: this.zones[key].lat,
            lng: this.zones[key].lng
        }));
    }
}

window.gpsSimulator = new GPSSimulator();