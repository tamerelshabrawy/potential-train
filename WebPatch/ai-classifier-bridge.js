/**
 * ai-classifier-bridge.js
 *
 * Real-time urban sound classifier for the WebPatch.
 * Loads the trained Gaussian Naive Bayes model from urban-sound-classifier.json,
 * processes live microphone input via the Web Audio API, classifies each frame
 * as one of: silence | chatter | motor | horn, and sends the results to Pure Data
 * via the pd4web JavaScript API.
 *
 * Pd receive names written to every analysis frame:
 *   aiClass    — integer index  0=silence  1=chatter  2=motor  3=horn
 *   aiSilence  — silence  confidence  0–1
 *   aiChatter  — chatter  confidence  0–1
 *   aiMotor    — motor    confidence  0–1
 *   aiHorn     — horn     confidence  0–1
 *   aiRmsDb    — current RMS level in dB  (reference / debug)
 *
 * Integration:
 *   1. Add  <script src="./ai-classifier-bridge.js"></script>  in index.html
 *      (after pd4web.js, before </body>).
 *   2. Expose the Pd4Web instance globally in the pd4web setup block:
 *        window.Pd4WebInstance = Pd4Web;
 *   3. Call  window.AiClassifierBridge.start()  when you want the mic to begin.
 *      The bridge auto-starts once Pd4WebInstance is available if
 *      autoStartMs is set (see CONFIG below).
 *
 * Global API:
 *   window.AiClassifierBridge.start()         — request mic + begin classifying
 *   window.AiClassifierBridge.stop()          — pause analysis (mic stays open)
 *   window.AiClassifierBridge.dispose()       — stop + release mic + free resources
 *   window.AiClassifierBridge.getSnapshot()   — return latest classification object
 *
 * Events (dispatched on window):
 *   ai-classifier-update  — every analysis frame  (detail = snapshot)
 *   ai-classifier-class   — only when the stable class changes  (detail = snapshot)
 */

(function () {
    "use strict";

    /* ─── configuration ────────────────────────────────────────────────────── */
    const CONFIG = {
        modelUrl:            "./urban-sound-classifier.json",
        fftSize:             1024,
        updateIntervalMs:    100,       // analysis frame rate
        inputAnalysisGain:   3,         // mic pre-gain for analysis chain
        inputHighpassHz:     80,        // remove sub-bass rumble
        inputLowpassHz:      9000,      // remove ultrasonic content
        smoothingAlpha:      0.30,      // exponential smoothing on log-probs (0=no smooth)
        minStateHoldMs:      420,       // minimum ms before stable class can change
        silenceDbThreshold:  -60,       // rmsDb below this → force silence class
        silenceHoldMs:       1200,      // ms of silence before class resets to silence
        minDb:               -90,
        maxDb:               -12,
        autoStartMs:         0,         // ms to wait after script load then auto-start
                                        // set to 0 to disable auto-start
    };

    const TWO_PI      = 2 * Math.PI;
    const LOG_TWO_PI  = Math.log(TWO_PI);
    const EPSILON     = 1e-12;

    /* ─── helpers ───────────────────────────────────────────────────────────── */
    function clamp(v, lo = 0, hi = 1) {
        return v < lo ? lo : v > hi ? hi : v;
    }

    function linearFromDb(db) {
        return Math.pow(10, db / 20);
    }

    /** Log-probability of x under N(mean, variance). */
    function logGaussian(x, mean, variance) {
        const diff = x - mean;
        return -0.5 * (LOG_TWO_PI + Math.log(variance + EPSILON) + diff * diff / (variance + EPSILON));
    }

    /** Numerically-stable softmax over an array of log-probabilities. */
    function softmaxFromLog(logProbs) {
        const maxLog = Math.max(...logProbs);
        const exps   = logProbs.map(lp => Math.exp(lp - maxLog));
        const sumExp = exps.reduce((a, b) => a + b, 0) + EPSILON;
        return exps.map(e => e / sumExp);
    }

    /* ─── Gaussian Naive Bayes inference ────────────────────────────────────── */
    function buildClassifier(model) {
        const classes  = model.classes;          // ["silence","chatter","motor","horn"]
        const features = model.features;         // ["rmsDb","centroidHz",...]
        const profiles = model.classProfiles;

        return function classify(featureVector) {
            const logProbs = classes.map(cls => {
                const profile  = profiles[cls];
                let   logProb  = Math.log(profile.prior + EPSILON);
                for (const feat of features) {
                    logProb += logGaussian(
                        featureVector[feat],
                        profile.mean[feat],
                        profile.variance[feat]
                    );
                }
                return logProb;
            });
            const probs     = softmaxFromLog(logProbs);
            let   bestIdx   = 0;
            for (let i = 1; i < probs.length; i++) {
                if (probs[i] > probs[bestIdx]) bestIdx = i;
            }
            const result = { classIndex: bestIdx, className: classes[bestIdx] };
            for (let i = 0; i < classes.length; i++) {
                result[classes[i]] = probs[i];
            }
            return result;
        };
    }

    /* ─── Audio feature extraction (mirrors urban-perceptual-state.js) ──────── */
    function extractFeatures(analyser, timeDomain, frequencyDb, previousSpectrum, sampleRate, opts) {
        analyser.getFloatTimeDomainData(timeDomain);
        analyser.getFloatFrequencyData(frequencyDb);

        const nyquist    = sampleRate * 0.5;
        const hzPerBin   = nyquist / frequencyDb.length;
        const lowpassHz  = opts.inputLowpassHz;

        // RMS
        let rmsAcc = 0;
        for (let i = 0; i < timeDomain.length; i++) {
            const s = timeDomain[i];
            rmsAcc += s * s;
        }
        const rms   = Math.sqrt(rmsAcc / timeDomain.length);
        const rmsDb = 20 * Math.log10(rms + EPSILON);

        // Spectral features
        const currentSpectrum = new Float32Array(frequencyDb.length);
        let totalPower     = 0;
        let centroidAcc    = 0;
        let logPowerAcc    = 0;
        let bandLow        = 0;    // 140–1100 Hz
        let bandMid        = 0;    // 1100–3200 Hz
        let bandHarsh      = 0;    // 3200–6200 Hz
        let bandAir        = 0;    // 6200–9000 Hz
        let fluxAcc        = 0;
        let usedBins       = 0;

        for (let i = 0; i < frequencyDb.length; i++) {
            const amplitude    = linearFromDb(frequencyDb[i]);
            const power        = amplitude * amplitude;
            currentSpectrum[i] = power;

            const hz = (i + 0.5) * hzPerBin;
            if (hz < 80 || hz > lowpassHz) continue;

            totalPower  += power;
            centroidAcc += power * hz;
            logPowerAcc += Math.log(power + EPSILON);
            usedBins    += 1;

            if      (hz >= 140  && hz < 1100) bandLow   += power;
            else if (hz >= 1100 && hz < 3200) bandMid   += power;
            else if (hz >= 3200 && hz < 6200) bandHarsh += power;
            else if (hz >= 6200 && hz < 9000) bandAir   += power;
        }

        const centroidHz = totalPower > EPSILON ? centroidAcc / totalPower : 0;
        const meanPower  = totalPower / Math.max(1, usedBins);
        const flatness   = totalPower > EPSILON
            ? Math.exp(logPowerAcc / Math.max(1, usedBins)) / (meanPower + EPSILON)
            : 0;

        if (previousSpectrum.data) {
            const norm = totalPower + EPSILON;
            for (let i = 0; i < currentSpectrum.length; i++) {
                const n   = currentSpectrum[i] / norm;
                const p   = previousSpectrum.data[i];
                fluxAcc  += Math.max(0, n - p);
                previousSpectrum.data[i] = n;
            }
        } else {
            previousSpectrum.data = new Float32Array(currentSpectrum.length);
            const norm = totalPower + EPSILON;
            for (let i = 0; i < currentSpectrum.length; i++) {
                previousSpectrum.data[i] = currentSpectrum[i] / norm;
            }
        }

        return {
            rmsDb,
            centroidHz,
            flatness:        clamp(flatness),
            flux:            clamp(fluxAcc * 4.5),
            speechLowRatio:  clamp(bandLow   / (totalPower + EPSILON)),
            speechMidRatio:  clamp(bandMid   / (totalPower + EPSILON)),
            harshRatio:      clamp(bandHarsh / (totalPower + EPSILON)),
            airRatio:        clamp(bandAir   / (totalPower + EPSILON)),
        };
    }

    /* ─── Pd bridge ─────────────────────────────────────────────────────────── */
    function sendToPd(name, value) {
        // Support both naming conventions used across PRs:
        //   window.Pd4WebInstance  (set by index.html-update snippet)
        //   window.Pd              (set by main branch's index.html Pd4WebModule block)
        const pd = window.Pd4WebInstance || window.Pd;
        if (!pd) return;
        // pd4web embind API
        if (typeof pd.sendFloat === "function") {
            pd.sendFloat(name, value);
        } else if (typeof pd.sendMessage === "function") {
            pd.sendMessage(name, "float", [value]);
        }
    }

    /* ─── Main classifier class ──────────────────────────────────────────────── */
    class AiClassifier {
        constructor(options = {}) {
            this.opts          = Object.assign({}, CONFIG, options);
            this.model         = null;
            this.classify      = null;
            this.audioCtx      = null;
            this.mediaStream   = null;
            this.sourceNode    = null;
            this.inputGain     = null;
            this.highpass      = null;
            this.lowpass       = null;
            this.analyser      = null;
            this.silentSink    = null;
            this.timeDomain    = null;
            this.frequencyDb   = null;
            this.prevSpectrum  = {};
            this.timerId       = null;
            this.running       = false;

            // Temporal smoothing buffers (log-space)
            this._smoothedLogProbs = null;

            // Stable-class state machine
            this._stableClass     = "silence";
            this._stableIdx       = 0;
            this._stableSinceMs   = performance.now();
            this._pendingClass    = null;
            this._pendingSinceMs  = null;
            this._quietSinceMs    = null;
            this._primed          = false;

            this._snapshot = this._blankSnapshot();
        }

        /* ── public ─────────────────────────────────────────────────────────── */

        /** Load the model JSON, then optionally start immediately. */
        async loadModel(url = this.opts.modelUrl) {
            const resp  = await fetch(url);
            if (!resp.ok) throw new Error(`[AiClassifier] Failed to load model: ${resp.status}`);
            this.model    = await resp.json();
            this.classify = buildClassifier(this.model);
            return this.model;
        }

        /** Request microphone permission and open the audio pipeline. */
        async attachMicrophone(constraints = {}) {
            if (!this.audioCtx) {
                this.audioCtx = new AudioContext();
            }
            await this.audioCtx.resume();
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation:  false,
                    noiseSuppression:  false,
                    autoGainControl:   false,
                    channelCount:      1,
                    ...constraints,
                },
                video: false,
            });
            this._buildPipeline(stream);
            return stream;
        }

        /** Begin analysis frames. Requires attachMicrophone() first. */
        start() {
            if (this.timerId !== null || !this.analyser) return;
            this.running  = true;
            this.timerId  = window.setInterval(() => this._analyzeFrame(), this.opts.updateIntervalMs);
        }

        /** Pause analysis without releasing the microphone. */
        stop() {
            if (this.timerId !== null) {
                window.clearInterval(this.timerId);
                this.timerId = null;
            }
            this.running = false;
        }

        /** Stop analysis and release all audio resources. */
        dispose() {
            this.stop();
            if (this.mediaStream) {
                for (const track of this.mediaStream.getTracks()) track.stop();
                this.mediaStream = null;
            }
            if (this.sourceNode) {
                try { this.sourceNode.disconnect(); } catch (_) { /* ignore */ }
            }
            this._resetState();
        }

        /** Return the most recent snapshot without triggering analysis. */
        getSnapshot() {
            return this._snapshot;
        }

        /* ── lifecycle helpers ──────────────────────────────────────────────── */

        /** Start analysis: load model (if not yet loaded) then open the mic. */
        async startAnalysis() {
            if (!this.classify) {
                await this.loadModel();
            }
            await this.attachMicrophone();
            AiClassifier.prototype.start.call(this);
        }

        /* ── private ────────────────────────────────────────────────────────── */

        _buildPipeline(stream) {
            this._resetState();
            this.mediaStream = stream;

            if (!this.audioCtx) this.audioCtx = new AudioContext();
            const ctx = this.audioCtx;

            this.inputGain = ctx.createGain();
            this.inputGain.gain.value = this.opts.inputAnalysisGain;

            this.highpass = ctx.createBiquadFilter();
            this.highpass.type = "highpass";
            this.highpass.frequency.value = this.opts.inputHighpassHz;

            this.lowpass = ctx.createBiquadFilter();
            this.lowpass.type = "lowpass";
            this.lowpass.frequency.value = this.opts.inputLowpassHz;

            this.analyser = ctx.createAnalyser();
            this.analyser.fftSize              = this.opts.fftSize;
            this.analyser.minDecibels          = this.opts.minDb;
            this.analyser.maxDecibels          = this.opts.maxDb;
            this.analyser.smoothingTimeConstant = 0;

            this.silentSink = ctx.createGain();
            this.silentSink.gain.value = 0;

            this.sourceNode = ctx.createMediaStreamSource(stream);
            this.sourceNode.connect(this.inputGain);
            this.inputGain.connect(this.highpass);
            this.highpass.connect(this.lowpass);
            this.lowpass.connect(this.analyser);
            this.analyser.connect(this.silentSink);
            this.silentSink.connect(ctx.destination);

            this.timeDomain  = new Float32Array(this.analyser.fftSize);
            this.frequencyDb = new Float32Array(this.analyser.frequencyBinCount);
            this.prevSpectrum = {};
        }

        _analyzeFrame() {
            if (!this.analyser || !this.classify) return;

            const features = extractFeatures(
                this.analyser,
                this.timeDomain,
                this.frequencyDb,
                this.prevSpectrum,
                this.audioCtx.sampleRate,
                this.opts,
            );

            // Raw GNB classification
            const raw = this.classify(features);

            // Smooth log-probabilities across frames
            const classes  = this.model.classes;
            const alpha    = this.opts.smoothingAlpha;
            if (!this._smoothedLogProbs) {
                this._smoothedLogProbs = {};
                for (const cls of classes) this._smoothedLogProbs[cls] = Math.log(raw[cls] + EPSILON);
            } else {
                for (const cls of classes) {
                    const logRaw = Math.log(raw[cls] + EPSILON);
                    this._smoothedLogProbs[cls] += alpha * (logRaw - this._smoothedLogProbs[cls]);
                }
            }

            // Convert smoothed log-probs back to probabilities
            const logArr    = classes.map(c => this._smoothedLogProbs[c]);
            const smoothed  = softmaxFromLog(logArr);
            const probs     = {};
            let   bestIdx   = 0;
            for (let i = 0; i < classes.length; i++) {
                probs[classes[i]] = smoothed[i];
                if (smoothed[i] > smoothed[bestIdx]) bestIdx = i;
            }
            const candidateClass = classes[bestIdx];

            // Silence override: if audio is below threshold, force silence
            const now = performance.now();
            if (features.rmsDb <= this.opts.silenceDbThreshold) {
                if (this._quietSinceMs === null) this._quietSinceMs = now;
                if (now - this._quietSinceMs >= this.opts.silenceHoldMs && this._stableClass !== "silence") {
                    this._commitClass("silence", now);
                }
            } else {
                this._quietSinceMs = null;
            }

            // State machine: require hold time before committing a class change
            if (!this._primed) {
                this._primed = true;
                this._commitClass(candidateClass, now);
            } else if (candidateClass !== this._stableClass) {
                if (this._pendingClass !== candidateClass) {
                    this._pendingClass   = candidateClass;
                    this._pendingSinceMs = now;
                } else if (now - this._pendingSinceMs >= this.opts.minStateHoldMs) {
                    const prevClass = this._stableClass;
                    this._commitClass(candidateClass, now);
                    this._pendingClass = null;
                    if (prevClass !== this._stableClass) {
                        this._emitEvent("ai-classifier-class", this._snapshot);
                    }
                }
            } else {
                this._pendingClass = null;
            }

            // Build snapshot
            this._snapshot = {
                timestampMs:    now,
                stableClass:    this._stableClass,
                stableIndex:    this._stableIdx,
                candidateClass,
                stableForMs:    now - this._stableSinceMs,
                probabilities:  probs,
                features,
            };

            // Broadcast
            this._emitEvent("ai-classifier-update", this._snapshot);
            this._sendAllToPd(this._snapshot);
        }

        _commitClass(cls, now) {
            const idx       = this.model ? this.model.classIndices[cls] : 0;
            this._stableClass   = cls;
            this._stableIdx     = idx !== undefined ? idx : 0;
            this._stableSinceMs = now;
        }

        _sendAllToPd(snap) {
            sendToPd("aiClass",   snap.stableIndex);
            sendToPd("aiSilence", snap.probabilities.silence  || 0);
            sendToPd("aiChatter", snap.probabilities.chatter  || 0);
            sendToPd("aiMotor",   snap.probabilities.motor    || 0);
            sendToPd("aiHorn",    snap.probabilities.horn     || 0);
            sendToPd("aiRmsDb",   snap.features.rmsDb);
        }

        _emitEvent(name, detail) {
            window.dispatchEvent(new CustomEvent(name, { detail }));
        }

        _resetState() {
            this.prevSpectrum      = {};
            this._smoothedLogProbs = null;
            this._stableClass      = "silence";
            this._stableIdx        = 0;
            this._stableSinceMs    = performance.now();
            this._pendingClass     = null;
            this._pendingSinceMs   = null;
            this._quietSinceMs     = null;
            this._primed           = false;
        }

        _blankSnapshot() {
            return {
                timestampMs:    performance.now(),
                stableClass:    "silence",
                stableIndex:    0,
                candidateClass: "silence",
                stableForMs:    0,
                probabilities:  { silence: 1, chatter: 0, motor: 0, horn: 0 },
                features: {
                    rmsDb: -90, centroidHz: 0, flatness: 0,
                    flux: 0, speechLowRatio: 0, speechMidRatio: 0,
                    harshRatio: 0, airRatio: 0,
                },
            };
        }
    }

    /* ─── Singleton bridge ──────────────────────────────────────────────────── */
    const bridge = new AiClassifier();

    /** Call this after Pd4WebInstance is available to begin classifying. */
    bridge.start = async function () {
        try {
            await bridge.startAnalysis();
        } catch (err) {
            console.error("[AiClassifierBridge] start() failed:", err);
        }
    };

    window.AiClassifierBridge = bridge;
    window.AiClassifier       = AiClassifier;       // expose class for custom instantiation

    // Optional: auto-start when Pd4WebInstance (or window.Pd) becomes available
    if (CONFIG.autoStartMs > 0) {
        setTimeout(() => {
            if (window.Pd4WebInstance || window.Pd) {
                bridge.start();
            }
        }, CONFIG.autoStartMs);
    }

    if (location.hostname === "127.0.0.1" || location.hostname === "localhost") {
        console.log("[AiClassifierBridge] loaded. Call window.AiClassifierBridge.start() to begin.");
    }

    /* ─── Backward-compatibility shim ────────────────────────────────────────
     * Main branch's index.html (merged from PR#4) references:
     *   window.aiClassifier.startMicrophoneClassification()
     *   window.aiClassifier.stopClassification()
     *   CustomEvent 'ai-classification' with detail { soundClass, confidence, params }
     *     where params = { aiGrain, aiStr, aiWet, aiPitHi }
     * This shim adapts the new GNB bridge to satisfy those existing bindings.
     * ────────────────────────────────────────────────────────────────────────── */
    const _CLASS_PARAMS_COMPAT = {
        horn:    { aiGrain: 15,  aiStr: 20,  aiWet: 70, aiPitHi:  7 },
        motor:   { aiGrain: 55,  aiStr: 85,  aiWet: 40, aiPitHi:  0 },
        chatter: { aiGrain: 30,  aiStr: 45,  aiWet: 60, aiPitHi:  4 },
        silence: { aiGrain:  5,  aiStr: 100, aiWet: 20, aiPitHi:  0 },
    };

    window.aiClassifier = {
        get isRunning()   { return bridge.running; },
        get currentClass(){ return bridge.getSnapshot().stableClass; },
        get confidence()  { return bridge.getSnapshot().probabilities[bridge.getSnapshot().stableClass] || 0; },

        startMicrophoneClassification: function () {
            return bridge.start();
        },
        stopClassification: function () {
            bridge.dispose();
        },
    };

    // Translate ai-classifier-update → ai-classification for main's index.html UI
    window.addEventListener("ai-classifier-update", function (e) {
        var snap   = e.detail;
        var cls    = snap.stableClass;
        var params = Object.assign({}, _CLASS_PARAMS_COMPAT[cls] || _CLASS_PARAMS_COMPAT.silence);
        window.dispatchEvent(new CustomEvent("ai-classification", {
            detail: {
                soundClass: cls,
                confidence: snap.probabilities[cls] || 0,
                params:     params,
            },
        }));
    });

})();
