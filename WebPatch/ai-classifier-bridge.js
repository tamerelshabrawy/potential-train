/**
 * ai-classifier-bridge.js
 *
 * Classifies live microphone input into one of four urban sound classes
 * (horn, motor, chatter, silence) derived from urban-sound-classifier.json
 * and dispatches 'ai-classification' CustomEvents carrying granular-synthesis
 * parameter presets for the pd4web patch.
 *
 * Exposed global: window.aiClassifier
 *   .startMicrophoneClassification() → Promise<void>
 *   .stopClassification()
 */
(function () {
    "use strict";

    /* -----------------------------------------------------------------------
     * Parameter presets per sound class
     * aiGrain  : 1–100   → street06GrainRate_idlework  (grain density)
     * aiStr    : 0–100%  → street06StretchProb_idlework (stretch probability)
     * aiWet    : 0–100%  → auraAmt_idlework (/100 → 0–1 in Pd)
     * aiPitHi  : –12–12  → street06PitchHi_idlework    (semitones)
     * ---------------------------------------------------------------------- */
    var CLASS_PARAMS = {
        horn:    { aiGrain: 15,  aiStr: 20,  aiWet: 70, aiPitHi:  7 },
        motor:   { aiGrain: 55,  aiStr: 85,  aiWet: 40, aiPitHi:  0 },
        chatter: { aiGrain: 30,  aiStr: 45,  aiWet: 60, aiPitHi:  4 },
        silence: { aiGrain:  5,  aiStr: 100, aiWet: 20, aiPitHi:  0 },
    };

    var EPSILON = 1e-12;

    function clamp(v, lo, hi) {
        return Math.min(hi, Math.max(lo, v));
    }

    /* -----------------------------------------------------------------------
     * Classifier
     * ---------------------------------------------------------------------- */
    function UrbanSoundClassifier() {
        this._audioContext  = null;
        this._mediaStream   = null;
        this._analyser      = null;
        this._silentSink    = null;
        this._timerId       = null;
        this._freqData      = null;
        this._timeData      = null;
        this._prevSpectrum  = null;

        /* Exponential-smoothed per-class scores */
        this._scores = { horn: 0, motor: 0, chatter: 0, silence: 1 };

        /* Current state */
        this.currentClass  = "silence";
        this.confidence    = 1;
        this.isRunning     = false;
    }

    /**
     * Requests microphone access, builds an analysis chain, and starts the
     * classification loop.  Returns a Promise that resolves once the mic is
     * active or rejects with a DOMException if access is denied.
     */
    UrbanSoundClassifier.prototype.startMicrophoneClassification = async function () {
        if (this.isRunning) return;

        if (!this._audioContext) {
            this._audioContext = new AudioContext();
        }
        await this._audioContext.resume();

        this._mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation:  false,
                noiseSuppression:  false,
                autoGainControl:   false,
                channelCount:      1,
            },
            video: false,
        });

        var source = this._audioContext.createMediaStreamSource(this._mediaStream);

        this._analyser = this._audioContext.createAnalyser();
        this._analyser.fftSize              = 1024;
        this._analyser.minDecibels          = -90;
        this._analyser.maxDecibels          = -12;
        this._analyser.smoothingTimeConstant = 0;

        /* Silence the analysis tap so it does not route to speakers */
        this._silentSink = this._audioContext.createGain();
        this._silentSink.gain.value = 0;

        source.connect(this._analyser);
        this._analyser.connect(this._silentSink);
        this._silentSink.connect(this._audioContext.destination);

        this._freqData    = new Float32Array(this._analyser.frequencyBinCount);
        this._timeData    = new Float32Array(this._analyser.fftSize);
        this._prevSpectrum = null;

        var self = this;
        this._timerId  = window.setInterval(function () { self._tick(); }, 100);
        this.isRunning = true;
    };

    /**
     * Stops classification and releases the microphone.
     */
    UrbanSoundClassifier.prototype.stopClassification = function () {
        if (!this.isRunning) return;

        if (this._timerId !== null) {
            window.clearInterval(this._timerId);
            this._timerId = null;
        }

        if (this._mediaStream) {
            this._mediaStream.getTracks().forEach(function (t) { t.stop(); });
            this._mediaStream = null;
        }

        if (this._audioContext) {
            this._audioContext.close();
            this._audioContext = null;
        }

        this._analyser     = null;
        this._silentSink   = null;
        this._freqData     = null;
        this._timeData     = null;
        this._prevSpectrum = null;
        this.isRunning     = false;
    };

    /* -----------------------------------------------------------------------
     * Internal: called every 100 ms
     * ---------------------------------------------------------------------- */
    UrbanSoundClassifier.prototype._tick = function () {
        if (!this._analyser || !this._audioContext) return;

        this._analyser.getFloatTimeDomainData(this._timeData);
        this._analyser.getFloatFrequencyData(this._freqData);

        /* --- RMS level ---------------------------------------------------- */
        var rmsAcc = 0;
        for (var i = 0; i < this._timeData.length; i++) {
            rmsAcc += this._timeData[i] * this._timeData[i];
        }
        var rms   = Math.sqrt(rmsAcc / this._timeData.length);
        var rmsDb = 20 * Math.log10(rms + EPSILON);

        /* --- Silence fast-path -------------------------------------------- */
        if (rmsDb < -52) {
            this._smooth({ horn: 0, motor: 0, chatter: 0, silence: 1 });
            this._dispatch("silence");
            return;
        }

        /* --- Spectral features -------------------------------------------- */
        var nyquist   = this._audioContext.sampleRate * 0.5;
        var hzPerBin  = nyquist / this._freqData.length;

        var totalPower  = 0;
        var centroidAcc = 0;
        var logPow      = 0;
        var usedBins    = 0;
        var speechBand  = 0; /* 200–3200 Hz */
        var highBand    = 0; /* 3200–8000 Hz */
        var currentSpec = new Float32Array(this._freqData.length);

        for (var b = 0; b < this._freqData.length; b++) {
            var amp   = Math.pow(10, this._freqData[b] / 20);
            var power = amp * amp;
            var hz    = (b + 0.5) * hzPerBin;

            currentSpec[b] = power;

            if (hz < 80 || hz > 9000) continue;

            totalPower  += power;
            centroidAcc += power * hz;
            logPow      += Math.log(power + EPSILON);
            usedBins    += 1;

            if (hz >= 200 && hz < 3200) speechBand += power;
            if (hz >= 3200 && hz < 8000) highBand   += power;
        }

        var centroidHz  = totalPower > EPSILON ? centroidAcc / totalPower : 0;
        var meanPow     = totalPower / Math.max(1, usedBins);
        var flatness    = totalPower > EPSILON
            ? Math.exp(logPow / Math.max(1, usedBins)) / (meanPow + EPSILON)
            : 0;
        flatness        = clamp(flatness, 0, 1);

        /* --- Spectral flux ------------------------------------------------ */
        var flux = 0;
        if (this._prevSpectrum) {
            var norm = totalPower + EPSILON;
            for (var k = 0; k < currentSpec.length; k++) {
                var cur  = currentSpec[k] / norm;
                var prev = this._prevSpectrum[k];
                flux    += Math.max(0, cur - prev);
                this._prevSpectrum[k] = cur;
            }
            flux = clamp(flux * 4.5, 0, 1);
        } else {
            this._prevSpectrum = new Float32Array(currentSpec.length);
            var norm0 = totalPower + EPSILON;
            for (var j = 0; j < currentSpec.length; j++) {
                this._prevSpectrum[j] = currentSpec[j] / norm0;
            }
        }

        /* --- Normalised helpers ------------------------------------------- */
        var speechRatio  = clamp(speechBand  / (totalPower + EPSILON), 0, 1);
        var highRatio    = clamp(highBand     / (totalPower + EPSILON), 0, 1);
        var levelNorm    = clamp((rmsDb + 60) / 42, 0, 1); /* –60 dBFS → 0, –18 dBFS → 1 */
        var centroidNorm = clamp((centroidHz - 500) / 3500, 0, 1);

        /* --- Raw class scores --------------------------------------------- */
        var raw = {
            /*
             * Horn: loud burst, high spectral centroid, tonal (low flatness),
             *       concentrated high-frequency energy, transient (high flux)
             */
            horn: clamp(
                levelNorm    * 0.20 +
                centroidNorm * 0.30 +
                highRatio    * 0.25 +
                (1 - flatness) * 0.15 +
                flux         * 0.10,
                0, 1
            ),

            /*
             * Motor: sustained, spectrally broad (high flatness), loud,
             *        not speech-dominated
             */
            motor: clamp(
                levelNorm         * 0.20 +
                flatness          * 0.40 +
                (1 - speechRatio) * 0.25 +
                centroidNorm      * 0.15,
                0, 1
            ),

            /*
             * Chatter: speech band dominant, tonal (low flatness in speech
             *          range), moderate level, low high-frequency ratio
             */
            chatter: clamp(
                speechRatio      * 0.50 +
                (1 - flatness)   * 0.20 +
                (1 - highRatio)  * 0.15 +
                levelNorm        * 0.15,
                0, 1
            ),

            silence: 0, /* handled above */
        };

        this._smooth(raw);
        this._dispatch(this._winner());
    };

    /* Smooth scores towards raw values with alpha=0.25 */
    UrbanSoundClassifier.prototype._smooth = function (raw) {
        var alpha = 0.25;
        var classes = Object.keys(raw);
        for (var c = 0; c < classes.length; c++) {
            var cls = classes[c];
            this._scores[cls] =
                this._scores[cls] + (raw[cls] - this._scores[cls]) * alpha;
        }
    };

    /* Return [winnerClass, confidence] */
    UrbanSoundClassifier.prototype._winner = function () {
        var best = "silence", bestScore = -Infinity, second = -Infinity;
        var classes = Object.keys(this._scores);
        for (var c = 0; c < classes.length; c++) {
            var cls   = classes[c];
            var score = this._scores[cls];
            if (score > bestScore) {
                second    = bestScore;
                bestScore = score;
                best      = cls;
            } else if (score > second) {
                second = score;
            }
        }
        return [best, clamp(bestScore - second, 0, 1)];
    };

    /* Dispatch the 'ai-classification' event consumed by index.html */
    UrbanSoundClassifier.prototype._dispatch = function (winnerOrPair) {
        var soundClass, confidence;
        if (Array.isArray(winnerOrPair)) {
            soundClass = winnerOrPair[0];
            confidence = winnerOrPair[1];
        } else {
            soundClass = winnerOrPair;
            confidence = 0.9;
        }

        this.currentClass = soundClass;
        this.confidence   = confidence;

        var params = Object.assign({}, CLASS_PARAMS[soundClass] || CLASS_PARAMS.silence);

        window.dispatchEvent(
            new CustomEvent("ai-classification", {
                detail: { soundClass: soundClass, confidence: confidence, params: params },
            })
        );
    };

    /* ---------------------------------------------------------------------- */

    window.aiClassifier = new UrbanSoundClassifier();
})();
