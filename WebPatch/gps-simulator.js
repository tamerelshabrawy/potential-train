/**
 * gps-simulator.js
 *
 * Simulates GPS positioning across named soundwalk zones and bridges each
 * zone change into the pd4web patch by sending a float on the [r zone]
 * receiver (values 1–35 as defined in Zone_Harness.pd).
 *
 * Dispatches a 'gps-zone-change' CustomEvent on window with:
 *   { zoneKey, zoneName, pdZone, lat, lon }
 *
 * Exposed global: window.gpsSimulator
 *   .setZone(zoneKey)              → Boolean
 *   .getCurrentCoordinates()       → { lat, lon, zoneKey, zoneName, pdZone }
 *   .getZones()                    → Array<{ id, name, pdZone, lat, lon }>
 *   .startWalk(intervalMs = DEFAULT_WALK_INTERVAL_MS)  → void
 *   .stopWalk()                    → void
 *   .isWalking                     → Boolean (read-only)
 */
class GPSSimulator {
    constructor() {
        // pdZone maps each named location to a numeric zone ID (1–35)
        // matching the preset message values in Zone_Harness.pd
        this.zones = {
            home:     { lat: 30.0444, lon: 31.2357, name: "Home Zone",      radius: 0.005, pdZone: 1  },
            park:     { lat: 30.0350, lon: 31.2200, name: "Park Zone",       radius: 0.006, pdZone: 8  },
            downtown: { lat: 30.0580, lon: 31.2425, name: "Downtown Cairo",  radius: 0.008, pdZone: 17 },
            market:   { lat: 30.0650, lon: 31.2500, name: "Market Zone",     radius: 0.004, pdZone: 31 }
        };

        this._zoneOrder  = ['home', 'park', 'downtown', 'market'];
        this._walkIndex  = 0;
        this._walkTimer  = null;
        this.currentZone = 'home';
        this.isWalking   = false;
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Returns the current position with a small random offset inside the zone.
     */
    getCurrentCoordinates() {
        const zone = this.zones[this.currentZone];
        const offsetLat = (Math.random() - 0.5) * zone.radius * 2;
        const offsetLng = (Math.random() - 0.5) * zone.radius * 2;
        return {
            lat:      zone.lat + offsetLat,
            lon:      zone.lon + offsetLng,
            zoneKey:  this.currentZone,
            zoneName: zone.name,
            pdZone:   zone.pdZone
        };
    }

    /**
     * Changes the active zone, sends its pdZone float to Pd, and fires the
     * 'gps-zone-change' event.  Returns true on success, false if the key is
     * unknown.
     */
    setZone(zoneKey) {
        if (!this.zones[zoneKey]) {
            console.warn('[GPSSimulator] Unknown zone key:', zoneKey);
            return false;
        }
        this.currentZone = zoneKey;
        this._walkIndex  = this._zoneOrder.indexOf(zoneKey);
        this._notify(zoneKey);
        return true;
    }

    /**
     * Returns metadata for all registered zones.
     */
    getZones() {
        return Object.keys(this.zones).map(key => ({
            id:     key,
            name:   this.zones[key].name,
            pdZone: this.zones[key].pdZone,
            lat:    this.zones[key].lat,
            lon:    this.zones[key].lon
        }));
    }

    /**
     * Starts an automatic walk that cycles through every zone on a timer.
     * @param {number} intervalMs – ms between zone transitions (default 30000, min 500)
     */
    startWalk(intervalMs) {
        if (this.isWalking) {
            this.stopWalk();
        }
        var ms = (typeof intervalMs === 'number' && intervalMs >= GPSSimulator.MIN_WALK_INTERVAL_MS)
            ? intervalMs
            : GPSSimulator.DEFAULT_WALK_INTERVAL_MS;
        var self = this;
        this.isWalking = true;
        // Send the current zone immediately, then advance on each tick
        this._notify(this.currentZone);
        this._walkTimer = window.setInterval(function () {
            self._walkIndex = (self._walkIndex + 1) % self._zoneOrder.length;
            var nextKey = self._zoneOrder[self._walkIndex];
            self.currentZone = nextKey;
            self._notify(nextKey);
        }, ms);
        console.log('[GPSSimulator] Walk started — interval:', ms, 'ms');
    }

    /**
     * Stops the automatic walk.
     */
    stopWalk() {
        if (!this.isWalking) return;
        window.clearInterval(this._walkTimer);
        this._walkTimer = null;
        this.isWalking  = false;
        console.log('[GPSSimulator] Walk stopped at zone:', this.currentZone);
    }

    // -------------------------------------------------------------------------
    // Internal
    // -------------------------------------------------------------------------

    /**
     * Sends pdZone to the Pd patch and dispatches the 'gps-zone-change' event.
     */
    _notify(zoneKey) {
        const zone = this.zones[zoneKey];
        const coords = this.getCurrentCoordinates();

        console.log('[GPSSimulator] Zone →', zone.name, '(pdZone:', zone.pdZone + ')');

        // Send numeric zone ID to [r zone] in the Pd patch
        if (window.Pd && typeof window.Pd.sendFloat === 'function') {
            window.Pd.sendFloat('zone', zone.pdZone);
        }

        window.dispatchEvent(new CustomEvent('gps-zone-change', {
            detail: {
                zoneKey:  zoneKey,
                zoneName: zone.name,
                pdZone:   zone.pdZone,
                lat:      coords.lat,
                lon:      coords.lon
            }
        }));
    }
}

// Walk timing constants
GPSSimulator.MIN_WALK_INTERVAL_MS     = 500;
GPSSimulator.DEFAULT_WALK_INTERVAL_MS = 30000;

window.gpsSimulator = new GPSSimulator();