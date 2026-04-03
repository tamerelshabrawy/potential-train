(function () {
    const DEFAULTS = {
        fftSize: 1024,
        updateIntervalMs: 90,
        inputAnalysisGain: 3,
        inputHighpassHz: 80,
        inputLowpassHz: 9000,
        featureSmoothingAlpha: 0.28,
        scoreSmoothingAlpha: 0.24,
        minStateHoldMs: 520,
        switchMargin: 0.04,
        silenceDbThreshold: -57,
        silenceHoldMs: 1400,
        minDb: -90,
        maxDb: -12,
    };

    const STATE_INDEX = {
        calm_open: 0,
        human_active: 1,
        harsh_mechanical: 2,
    };

    const EPSILON = 1e-12;

    function clamp(value, min = 0, max = 1) {
        return Math.min(max, Math.max(min, value));
    }

    function smooth(previous, next, alpha) {
        return previous + (next - previous) * alpha;
    }

    function linearFromDecibels(db) {
        return Math.pow(10, db / 20);
    }

    function normalizeRange(value, min, max) {
        if (max <= min) {
            return 0;
        }
        return clamp((value - min) / (max - min));
    }

    function getWinningState(scores) {
        const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
        return [entries[0][0], entries[0][1], entries[1][1]];
    }

    class UrbanPerceptualStateAnalyzer {
        constructor(options = {}) {
            this.options = { ...DEFAULTS, ...options };
            this.audioContext = null;
            this.mediaStream = null;
            this.sourceNode = null;
            this.inputGain = null;
            this.inputHighpass = null;
            this.inputLowpass = null;
            this.analyser = null;
            this.silentSink = null;
            this.timeDomain = null;
            this.frequencyDb = null;
            this.previousSpectrum = null;
            this.timerId = null;
            this.hasPrimedState = false;
            this.stableState = "calm_open";
            this.stableSinceMs = performance.now();
            this.quietSinceMs = null;
            this.pendingState = null;
            this.listeners = {
                update: new Set(),
                state: new Set(),
            };
            this.smoothedMetrics = {
                streetPressure: 0,
                humanPresence: 0,
                harshness: 0,
                openness: 1,
            };
            this.smoothedScores = {
                calm_open: 1,
                human_active: 0,
                harsh_mechanical: 0,
            };
            this.latestSnapshot = {
                timestampMs: performance.now(),
                state: "calm_open",
                stateIndex: 0,
                candidateState: "calm_open",
                confidence: 1,
                stableForMs: 0,
                features: {
                    rmsDb: -90,
                    centroidHz: 0,
                    flatness: 0,
                    flux: 0,
                    speechLowRatio: 0,
                    speechMidRatio: 0,
                    harshRatio: 0,
                    airRatio: 0,
                },
                metrics: {
                    streetPressure: 0,
                    humanPresence: 0,
                    harshness: 0,
                    openness: 1,
                },
                scores: {
                    calm_open: 1,
                    human_active: 0,
                    harsh_mechanical: 0,
                },
            };
        }

        async attachMicrophone(constraints = {}) {
            if (!this.audioContext) {
                this.audioContext = new AudioContext();
            }
            await this.audioContext.resume();
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    channelCount: 1,
                    ...constraints,
                },
                video: false,
            });
            this.attachStream(stream);
            return stream;
        }

        attachStream(stream) {
            this.mediaStream = stream;
            this.resetAnalysisState();
            if (!this.audioContext) {
                this.audioContext = new AudioContext();
            }

            this.inputGain = this.audioContext.createGain();
            this.inputGain.gain.value = this.options.inputAnalysisGain;

            this.inputHighpass = this.audioContext.createBiquadFilter();
            this.inputHighpass.type = "highpass";
            this.inputHighpass.frequency.value = this.options.inputHighpassHz;

            this.inputLowpass = this.audioContext.createBiquadFilter();
            this.inputLowpass.type = "lowpass";
            this.inputLowpass.frequency.value = this.options.inputLowpassHz;

            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.options.fftSize;
            this.analyser.minDecibels = this.options.minDb;
            this.analyser.maxDecibels = this.options.maxDb;
            this.analyser.smoothingTimeConstant = 0;

            this.silentSink = this.audioContext.createGain();
            this.silentSink.gain.value = 0;

            this.sourceNode = this.audioContext.createMediaStreamSource(stream);
            this.sourceNode.connect(this.inputGain);
            this.inputGain.connect(this.inputHighpass);
            this.inputHighpass.connect(this.inputLowpass);
            this.inputLowpass.connect(this.analyser);
            this.analyser.connect(this.silentSink);
            this.silentSink.connect(this.audioContext.destination);

            this.timeDomain = new Float32Array(this.analyser.fftSize);
            this.frequencyDb = new Float32Array(this.analyser.frequencyBinCount);
        }

        start() {
            if (this.timerId !== null || !this.analyser) {
                return;
            }
            this.timerId = window.setInterval(() => this.analyzeFrame(), this.options.updateIntervalMs);
        }

        stop() {
            if (this.timerId !== null) {
                window.clearInterval(this.timerId);
                this.timerId = null;
            }
        }

        dispose() {
            this.stop();
            if (this.mediaStream) {
                for (const track of this.mediaStream.getTracks()) {
                    track.stop();
                }
                this.mediaStream = null;
            }
            if (this.sourceNode && this.inputHighpass) {
                try {
                    this.sourceNode.disconnect(this.inputGain || this.inputHighpass);
                } catch (_) {
                    /* ignore */
                }
            }
            this.resetAnalysisState();
        }

        getLatestSnapshot() {
            return this.latestSnapshot;
        }

        on(eventName, listener) {
            this.listeners[eventName].add(listener);
            return () => {
                this.listeners[eventName].delete(listener);
            };
        }

        emit(eventName, payload) {
            for (const listener of this.listeners[eventName]) {
                listener(payload);
            }
        }

        analyzeFrame() {
            if (!this.analyser || !this.timeDomain || !this.frequencyDb || !this.audioContext) {
                return;
            }
            this.analyser.getFloatTimeDomainData(this.timeDomain);
            this.analyser.getFloatFrequencyData(this.frequencyDb);

            const features = this.computeRawFeatures();
            const metrics = this.computeMetrics(features);
            const scores = this.computeScores(metrics, features);
            let [candidateState, candidateScore, secondScore] = getWinningState(scores);
            const speechRatio = clamp(features.speechLowRatio + features.speechMidRatio, 0, 1);
            const directHumanCue =
                features.rmsDb > -64 &&
                speechRatio > 0.29 &&
                features.speechMidRatio > 0.1 &&
                features.flatness < 0.74 &&
                features.harshRatio < 0.34;
            const directHarshCue =
                features.rmsDb > -50 &&
                features.harshRatio > 0.24 &&
                features.flux > 0.16 &&
                speechRatio < 0.34;

            if (directHumanCue) {
                candidateState = "human_active";
                candidateScore = Math.max(candidateScore, scores.human_active + 0.2, 0.66);
            } else if (directHarshCue) {
                candidateState = "harsh_mechanical";
                candidateScore = Math.max(candidateScore, scores.harsh_mechanical + 0.1, 0.54);
            }

            const confidence = clamp(candidateScore - secondScore, 0, 1);
            const now = performance.now();

            let stateChanged = false;
            if (!this.hasPrimedState) {
                this.hasPrimedState = true;
                this.stableState = candidateState;
                this.stableSinceMs = now;
            } else if (
                features.rmsDb <= this.options.silenceDbThreshold &&
                metrics.streetPressure < 0.18 &&
                metrics.harshness < 0.24
            ) {
                if (this.quietSinceMs === null) {
                    this.quietSinceMs = now;
                } else if (
                    this.stableState !== "calm_open" &&
                    now - this.quietSinceMs >= this.options.silenceHoldMs
                ) {
                    this.stableState = "calm_open";
                    this.stableSinceMs = now;
                    this.pendingState = null;
                    stateChanged = true;
                }
            } else {
                this.quietSinceMs = null;
            }

            const stableScore = scores[this.stableState];
            if (candidateState !== this.stableState) {
                const dynamicMargin =
                    directHumanCue || directHarshCue
                        ? this.options.switchMargin * 0.18
                        : confidence > 0.22 ? this.options.switchMargin * 0.55 : this.options.switchMargin;
                const strongEnough =
                    candidateScore >= stableScore + dynamicMargin ||
                    (stableScore < 0.28 && candidateScore > 0.34);
                const requiredHoldMs = directHumanCue || directHarshCue
                    ? this.options.minStateHoldMs * 0.18
                    : candidateScore > 0.62
                      ? this.options.minStateHoldMs * 0.5
                      : candidateScore > 0.46
                        ? this.options.minStateHoldMs * 0.7
                        : this.options.minStateHoldMs;
                if (strongEnough) {
                    if (!this.pendingState || this.pendingState.state !== candidateState) {
                        this.pendingState = { state: candidateState, sinceMs: now };
                    } else if (now - this.pendingState.sinceMs >= requiredHoldMs) {
                        this.stableState = candidateState;
                        this.stableSinceMs = now;
                        this.pendingState = null;
                        stateChanged = true;
                    }
                } else {
                    this.pendingState = null;
                }
            } else {
                this.pendingState = null;
            }

            this.latestSnapshot = {
                timestampMs: now,
                state: this.stableState,
                stateIndex: STATE_INDEX[this.stableState],
                candidateState,
                confidence,
                stableForMs: now - this.stableSinceMs,
                features,
                metrics,
                scores,
            };

            this.emit("update", this.latestSnapshot);
            if (stateChanged) {
                this.emit("state", this.latestSnapshot);
            }
        }

        computeRawFeatures() {
            const nyquist = this.audioContext.sampleRate * 0.5;
            const hzPerBin = nyquist / this.frequencyDb.length;

            let rmsAccumulator = 0;
            for (let i = 0; i < this.timeDomain.length; i += 1) {
                const sample = this.timeDomain[i];
                rmsAccumulator += sample * sample;
            }

            const rms = Math.sqrt(rmsAccumulator / this.timeDomain.length);
            const rmsDb = 20 * Math.log10(rms + EPSILON);

            const currentSpectrum = new Float32Array(this.frequencyDb.length);
            let totalPower = 0;
            let centroidAccumulator = 0;
            let logPowerAccumulator = 0;
            let bandSpeechLow = 0;
            let bandSpeechMid = 0;
            let bandHarsh = 0;
            let bandAir = 0;
            let fluxAccumulator = 0;
            let usedBins = 0;

            for (let i = 0; i < this.frequencyDb.length; i += 1) {
                const db = this.frequencyDb[i];
                const amplitude = linearFromDecibels(db);
                const power = amplitude * amplitude;
                currentSpectrum[i] = power;

                const frequencyHz = (i + 0.5) * hzPerBin;
                if (frequencyHz < 80 || frequencyHz > this.options.inputLowpassHz) {
                    continue;
                }

                totalPower += power;
                centroidAccumulator += power * frequencyHz;
                logPowerAccumulator += Math.log(power + EPSILON);
                usedBins += 1;

                if (frequencyHz >= 140 && frequencyHz < 1100) {
                    bandSpeechLow += power;
                } else if (frequencyHz >= 1100 && frequencyHz < 3200) {
                    bandSpeechMid += power;
                } else if (frequencyHz >= 3200 && frequencyHz < 6200) {
                    bandHarsh += power;
                } else if (frequencyHz >= 6200 && frequencyHz < 9000) {
                    bandAir += power;
                }
            }

            const centroidHz = totalPower > EPSILON ? centroidAccumulator / totalPower : 0;
            const meanPower = totalPower / Math.max(1, usedBins);
            const flatness =
                totalPower > EPSILON
                    ? Math.exp(logPowerAccumulator / Math.max(1, usedBins)) / (meanPower + EPSILON)
                    : 0;

            if (this.previousSpectrum) {
                const normalizer = totalPower + EPSILON;
                for (let i = 0; i < currentSpectrum.length; i += 1) {
                    const normalized = currentSpectrum[i] / normalizer;
                    const previous = this.previousSpectrum[i];
                    fluxAccumulator += Math.max(0, normalized - previous);
                    this.previousSpectrum[i] = normalized;
                }
            } else {
                this.previousSpectrum = new Float32Array(currentSpectrum.length);
                const normalizer = totalPower + EPSILON;
                for (let i = 0; i < currentSpectrum.length; i += 1) {
                    this.previousSpectrum[i] = currentSpectrum[i] / normalizer;
                }
            }

            return {
                rmsDb,
                centroidHz,
                flatness: clamp(flatness, 0, 1),
                flux: clamp(fluxAccumulator * 4.5, 0, 1),
                speechLowRatio: clamp(bandSpeechLow / (totalPower + EPSILON)),
                speechMidRatio: clamp(bandSpeechMid / (totalPower + EPSILON)),
                harshRatio: clamp(bandHarsh / (totalPower + EPSILON)),
                airRatio: clamp(bandAir / (totalPower + EPSILON)),
            };
        }

        computeMetrics(features) {
            const levelNorm = normalizeRange(features.rmsDb, -60, -18);
            const centroidNorm = normalizeRange(features.centroidHz, 700, 4200);
            const flatnessNorm = normalizeRange(features.flatness, 0.08, 0.55);
            const fluxNorm = normalizeRange(features.flux, 0.04, 0.45);
            const speechRatio = clamp(features.speechLowRatio * 0.45 + features.speechMidRatio * 0.55);
            const voiceProminence = clamp(
                features.speechLowRatio * 1.15 +
                    features.speechMidRatio * 1.8 -
                    features.harshRatio * 0.48 -
                    features.airRatio * 0.12 -
                    flatnessNorm * 0.08,
            );
            const voiceFocus = clamp(
                features.speechMidRatio * 1.35 +
                    features.speechLowRatio * 0.7 -
                    flatnessNorm * 0.16 -
                    features.harshRatio * 0.1 -
                    features.airRatio * 0.05,
            );
            const presenceFocus = clamp(
                speechRatio * 0.86 +
                    voiceProminence * 0.58 +
                    voiceFocus * 0.36 -
                    features.harshRatio * 0.22 +
                    features.airRatio * 0.04,
            );

            const nextMetrics = {
                streetPressure: clamp(
                    levelNorm * 0.34 +
                        fluxNorm * 0.26 +
                        flatnessNorm * 0.2 +
                        centroidNorm * 0.1 +
                        features.harshRatio * 0.1,
                ),
                humanPresence: clamp(
                    presenceFocus * 0.48 +
                        voiceProminence * 0.24 +
                        voiceFocus * 0.14 +
                        speechRatio * 0.08 +
                        levelNorm * 0.04 +
                        (1 - features.harshRatio) * 0.02,
                ),
                harshness: clamp(
                    features.harshRatio * 0.42 +
                        flatnessNorm * 0.22 +
                        fluxNorm * 0.2 +
                        centroidNorm * 0.1 -
                        speechRatio * 0.12,
                ),
                openness: clamp(
                    features.airRatio * 0.34 +
                        (1 - levelNorm) * 0.22 +
                        (1 - flatnessNorm) * 0.16 +
                        (1 - fluxNorm) * 0.14 +
                        (1 - features.harshRatio) * 0.14,
                ),
            };

            if (!this.hasPrimedState) {
                this.smoothedMetrics = nextMetrics;
                return this.smoothedMetrics;
            }

            this.smoothedMetrics = {
                streetPressure: smooth(this.smoothedMetrics.streetPressure, nextMetrics.streetPressure, this.options.featureSmoothingAlpha),
                humanPresence: smooth(this.smoothedMetrics.humanPresence, nextMetrics.humanPresence, this.options.featureSmoothingAlpha),
                harshness: smooth(this.smoothedMetrics.harshness, nextMetrics.harshness, this.options.featureSmoothingAlpha),
                openness: smooth(this.smoothedMetrics.openness, nextMetrics.openness, this.options.featureSmoothingAlpha),
            };

            return this.smoothedMetrics;
        }

        computeScores(metrics, features) {
            const speechRatio = clamp(features.speechLowRatio * 0.45 + features.speechMidRatio * 0.55);
            const voiceProminence = clamp(
                features.speechLowRatio * 1.15 +
                    features.speechMidRatio * 1.8 -
                    features.harshRatio * 0.48 -
                    features.airRatio * 0.12,
            );
            const nextScores = {
                calm_open: clamp(
                    metrics.openness * 0.42 +
                        (1 - metrics.streetPressure) * 0.16 +
                        (1 - metrics.harshness) * 0.16 +
                        features.airRatio * 0.08 +
                        (1 - metrics.humanPresence) * 0.22 -
                        voiceProminence * 0.08,
                ),
                human_active: clamp(
                    metrics.humanPresence * 0.62 +
                        voiceProminence * 0.2 +
                        speechRatio * 0.12 +
                        metrics.streetPressure * 0.04 +
                        (1 - metrics.harshness) * 0.02,
                ),
                harsh_mechanical: clamp(
                    metrics.harshness * 0.62 +
                        metrics.streetPressure * 0.18 +
                        features.harshRatio * 0.14 +
                        features.flux * 0.08 -
                        metrics.humanPresence * 0.16 -
                        voiceProminence * 0.08,
                ),
            };

            if (!this.hasPrimedState) {
                this.smoothedScores = nextScores;
                return this.smoothedScores;
            }

            this.smoothedScores = {
                calm_open: smooth(this.smoothedScores.calm_open, nextScores.calm_open, this.options.scoreSmoothingAlpha),
                human_active: smooth(this.smoothedScores.human_active, nextScores.human_active, this.options.scoreSmoothingAlpha),
                harsh_mechanical: smooth(this.smoothedScores.harsh_mechanical, nextScores.harsh_mechanical, this.options.scoreSmoothingAlpha),
            };

            return this.smoothedScores;
        }

        resetAnalysisState() {
            this.previousSpectrum = null;
            this.pendingState = null;
            this.quietSinceMs = null;
            this.hasPrimedState = false;
            this.stableState = "calm_open";
            this.stableSinceMs = performance.now();
            this.smoothedMetrics = {
                streetPressure: 0,
                humanPresence: 0,
                harshness: 0,
                openness: 1,
            };
            this.smoothedScores = {
                calm_open: 1,
                human_active: 0,
                harsh_mechanical: 0,
            };
        }
    }

    function emitIntegrationState(snapshot) {
        window.__urbanPerceptualSnapshot = snapshot;
        window.dispatchEvent(
            new CustomEvent("urban-perceptual-state-update", {
                detail: snapshot,
            }),
        );
        if (typeof window.__pd4webUrbanStateSink === "function") {
            window.__pd4webUrbanStateSink(snapshot);
        }
    }

    function emitIntegrationStateChange(snapshot) {
        window.dispatchEvent(
            new CustomEvent("urban-perceptual-state-change", {
                detail: snapshot,
            }),
        );
    }

    function mountPanel(container, analyzer) {
        const startButton = container.querySelector("[data-role='urban-start']");
        const stopButton = container.querySelector("[data-role='urban-stop']");
        const stateNode = container.querySelector("[data-role='urban-state']");
        const confidenceNode = container.querySelector("[data-role='urban-confidence']");
        const pressureBar = container.querySelector("[data-role='urban-pressure']");
        const humanBar = container.querySelector("[data-role='urban-human']");
        const harshBar = container.querySelector("[data-role='urban-harsh']");
        const openBar = container.querySelector("[data-role='urban-open']");
        const errorNode = container.querySelector("[data-role='urban-error']");

        const setRunning = (running) => {
            startButton.disabled = running;
            stopButton.disabled = !running;
        };

        const setError = (message) => {
            errorNode.textContent = message || "";
        };

        const renderSnapshot = (snapshot) => {
            stateNode.textContent = snapshot.state;
            confidenceNode.textContent = `${Math.round(snapshot.confidence * 100)}%`;
            pressureBar.style.width = `${Math.round(snapshot.metrics.streetPressure * 100)}%`;
            humanBar.style.width = `${Math.round(snapshot.metrics.humanPresence * 100)}%`;
            harshBar.style.width = `${Math.round(snapshot.metrics.harshness * 100)}%`;
            openBar.style.width = `${Math.round(snapshot.metrics.openness * 100)}%`;
            container.dataset.state = snapshot.state;
            emitIntegrationState(snapshot);
        };

        analyzer.on("update", renderSnapshot);
        analyzer.on("state", emitIntegrationStateChange);

        startButton.addEventListener("click", async () => {
            setError("");
            try {
                await analyzer.attachMicrophone();
                analyzer.start();
                setRunning(true);
                renderSnapshot(analyzer.getLatestSnapshot());
            } catch (error) {
                setRunning(false);
                setError(error && error.message ? error.message : "Mic access failed");
            }
        });

        stopButton.addEventListener("click", () => {
            analyzer.stop();
            analyzer.dispose();
            setRunning(false);
        });

        setRunning(false);
        renderSnapshot(analyzer.getLatestSnapshot());
    }

    function createPanelMarkup() {
        const panel = document.createElement("section");
        panel.className = "urban-state-panel";
        panel.id = "UrbanPerceptualPanel";
        panel.innerHTML = `
            <div class="urban-state-header">
                <div class="urban-state-title">Urban Perceptual State</div>
                <div class="urban-state-pill" data-role="urban-state">calm_open</div>
            </div>
            <div class="urban-state-actions">
                <button type="button" data-role="urban-start">Start Mic Analysis</button>
                <button type="button" data-role="urban-stop">Stop</button>
                <span class="urban-state-confidence">confidence <strong data-role="urban-confidence">100%</strong></span>
            </div>
            <div class="urban-state-metrics">
                <div class="urban-state-row">
                    <span>streetPressure</span>
                    <div class="urban-state-meter"><div data-role="urban-pressure"></div></div>
                </div>
                <div class="urban-state-row">
                    <span>humanPresence</span>
                    <div class="urban-state-meter"><div data-role="urban-human"></div></div>
                </div>
                <div class="urban-state-row">
                    <span>harshness</span>
                    <div class="urban-state-meter"><div data-role="urban-harsh"></div></div>
                </div>
                <div class="urban-state-row">
                    <span>openness</span>
                    <div class="urban-state-meter"><div data-role="urban-open"></div></div>
                </div>
            </div>
            <div class="urban-state-error" data-role="urban-error"></div>
        `;
        return panel;
    }

    function setupUrbanPerceptualStatePrototype(options = {}) {
        const host = document.getElementById(options.containerId || "UrbanPerceptualHost") || document.body;
        const panel = createPanelMarkup();
        host.appendChild(panel);
        const analyzer = new UrbanPerceptualStateAnalyzer(options.analyzer || {});
        window.urbanPerceptualAnalyzer = analyzer;
        mountPanel(panel, analyzer);
        if (location.hostname === "127.0.0.1" || location.hostname === "localhost") {
            const beacon = new Image();
            beacon.src = `./icon-192.png?urban-ready=${Date.now()}`;
        }
        return analyzer;
    }

    window.UrbanPerceptualStateAnalyzer = UrbanPerceptualStateAnalyzer;
    window.setupUrbanPerceptualStatePrototype = setupUrbanPerceptualStatePrototype;
})();
