/* ============================================================
   PhotonEdge Web Application — Frontend Logic
   Copyright (c) 2024-2026 FRAGMERGENT TECHNOLOGY S.R.L.
   ============================================================ */

(function () {
    "use strict";

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    let socket = null;
    let currentSource = "demo";      // "webcam" | "upload" | "demo"
    let currentView = "overlay";
    let webcamStream = null;
    let webcamInterval = null;
    let capabilities = {};

    // ------------------------------------------------------------------
    // DOM references
    // ------------------------------------------------------------------
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const btnWebcam = $("#btn-webcam");
    const btnUpload = $("#btn-upload");
    const btnDemo = $("#btn-demo");
    const btnRunDemo = $("#btn-run-demo");
    const fileInput = $("#file-input");
    const demoShape = $("#demo-shape");
    const demoSnr = $("#demo-snr");
    const demoSection = $("#demo-section");

    const mainImage = $("#main-image");
    const noImageMsg = $("#no-image-msg");
    const alertBanner = $("#alert-banner");
    const alertText = $("#alert-text");
    const simMetrics = $("#sim-metrics");

    const statusIndicator = $("#status-indicator");
    const statusText = $(".status-text");

    const video = $("#webcam-video");
    const canvas = $("#webcam-canvas");
    const ctx = canvas.getContext("2d");

    // ------------------------------------------------------------------
    // Socket.IO connection
    // ------------------------------------------------------------------
    function connectSocket() {
        socket = io({ transports: ["websocket", "polling"] });

        socket.on("connect", () => {
            statusIndicator.className = "status connected";
            statusText.textContent = "Connected";
            loadCapabilities();
        });

        socket.on("disconnect", () => {
            statusIndicator.className = "status disconnected";
            statusText.textContent = "Disconnected";
        });

        socket.on("status", (data) => {
            console.log("Server:", data.msg);
        });

        socket.on("mode_changed", (data) => {
            syncToggles(data.params);
            syncSliders(data.params);
        });

        socket.on("result", handleWebcamResult);
    }

    // ------------------------------------------------------------------
    // Capabilities
    // ------------------------------------------------------------------
    async function loadCapabilities() {
        try {
            const resp = await fetch("/api/capabilities");
            capabilities = await resp.json();

            // Mode selector
            const modeContainer = $("#mode-selector");
            modeContainer.innerHTML = "";
            for (const [key, info] of Object.entries(capabilities.modes)) {
                const btn = document.createElement("button");
                btn.className = "mode-btn" + (key === "edge_detection" ? " active" : "");
                btn.dataset.mode = key;
                btn.innerHTML = `<span class="mode-icon">${modeIcon(info.icon)}</span>${info.name}`;
                btn.title = info.description;
                btn.addEventListener("click", () => selectMode(key));
                modeContainer.appendChild(btn);
            }

            // Capability badges
            setBadge("cap-yolo", capabilities.yolo);
            setBadge("cap-torch", capabilities.torch);
            setBadge("cap-cuda", capabilities.cuda);
        } catch (e) {
            console.error("Failed to load capabilities:", e);
        }
    }

    function modeIcon(name) {
        const icons = {
            scan: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 7 4 4 7 4"/><polyline points="20 7 20 4 17 4"/><polyline points="4 17 4 20 7 20"/><polyline points="20 17 20 20 17 20"/><line x1="4" y1="12" x2="20" y2="12"/></svg>',
            moon: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>',
            box: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/></svg>',
            settings: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>',
            cpu: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        };
        return icons[name] || "";
    }

    function setBadge(id, available) {
        const el = document.getElementById(id);
        if (available) {
            el.className = "cap-badge cap-on";
            el.textContent = "OK";
        } else {
            el.className = "cap-badge cap-off";
            el.textContent = "N/A";
        }
    }

    // ------------------------------------------------------------------
    // Mode selection
    // ------------------------------------------------------------------
    function selectMode(mode) {
        $$(".mode-btn").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
        if (socket && socket.connected) {
            socket.emit("set_mode", { mode });
        }
    }

    function syncToggles(params) {
        if (!params) return;
        const map = {
            "tog-denoise": "denoise",
            "tog-contrast": "contrast",
            "tog-pseudo": "pseudocolor",
            "tog-shapes": "detect_shapes",
            "tog-objects": "detect_objects",
            "tog-depth": "depth",
            "tog-tracking": "tracking",
        };
        for (const [elemId, paramKey] of Object.entries(map)) {
            const el = document.getElementById(elemId);
            if (el) el.checked = !!params[paramKey];
        }
    }

    function syncSliders(params) {
        if (!params) return;
        if (params.edge_t !== undefined) {
            $("#param-edge-t").value = params.edge_t;
            $("#val-edge-t").textContent = params.edge_t;
        }
        if (params.smooth_sigma !== undefined) {
            $("#param-smooth").value = params.smooth_sigma;
            $("#val-smooth").textContent = params.smooth_sigma;
        }
        if (params.coverage_radius !== undefined) {
            $("#param-coverage").value = params.coverage_radius;
            $("#val-coverage").textContent = params.coverage_radius;
        }
        if (params.fusion_mode !== undefined) {
            $("#param-fusion").value = params.fusion_mode;
        }
    }

    // ------------------------------------------------------------------
    // Source selection
    // ------------------------------------------------------------------
    function setSource(src) {
        currentSource = src;
        [btnWebcam, btnUpload, btnDemo].forEach(b => b.classList.remove("active"));
        if (src === "webcam") { btnWebcam.classList.add("active"); startWebcam(); }
        else { stopWebcam(); }
        if (src === "upload") { btnUpload.classList.add("active"); fileInput.click(); }
        if (src === "demo") { btnDemo.classList.add("active"); }

        demoSection.style.display = src === "demo" ? "" : "none";
        simMetrics.classList.toggle("hidden", src !== "demo");
    }

    btnWebcam.addEventListener("click", () => setSource("webcam"));
    btnUpload.addEventListener("click", () => setSource("upload"));
    btnDemo.addEventListener("click", () => setSource("demo"));

    // ------------------------------------------------------------------
    // View tabs
    // ------------------------------------------------------------------
    $$(".view-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            $$(".view-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            currentView = tab.dataset.view;
        });
    });

    // ------------------------------------------------------------------
    // Webcam
    // ------------------------------------------------------------------
    async function startWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            video.srcObject = webcamStream;
            video.play();
            canvas.width = 640;
            canvas.height = 480;

            webcamInterval = setInterval(() => {
                if (!socket || !socket.connected) return;
                ctx.drawImage(video, 0, 0, 640, 480);
                const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
                socket.emit("frame", { image: dataUrl });
            }, 100); // ~10 FPS send rate
        } catch (e) {
            console.error("Webcam error:", e);
            alert("Could not access webcam. Please check permissions.");
            setSource("demo");
        }
    }

    function stopWebcam() {
        if (webcamInterval) {
            clearInterval(webcamInterval);
            webcamInterval = null;
        }
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
        }
    }

    function handleWebcamResult(data) {
        // Display selected view
        const imgKey = currentView;
        const imgData = (data.images || {})[imgKey];
        if (imgData) {
            showImage(imgData);
        } else if (data.images && data.images.overlay) {
            showImage(data.images.overlay);
        }

        updateMetrics(data.metrics);
        updateDetections(data.shapes, data.objects);
        updateTracked(data.tracked);
        updateAlert(data.alert);

        // Scale images
        for (let i = 0; i < 3; i++) {
            const key = `edges_scale_${i}`;
            if (data.images && data.images[key]) {
                const scaleImg = $(`#img-scale-${i}`);
                if (scaleImg) scaleImg.src = "data:image/jpeg;base64," + data.images[key];
            }
        }
        if (data.metrics && data.metrics.edge_counts) {
            data.metrics.edge_counts.forEach((c, i) => {
                const el = $(`#count-scale-${i}`);
                if (el) el.textContent = c.toLocaleString() + " px";
            });
        }
    }

    // ------------------------------------------------------------------
    // Upload
    // ------------------------------------------------------------------
    fileInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("image", file);

        try {
            const resp = await fetch("/api/process", { method: "POST", body: formData });
            const data = await resp.json();
            if (data.error) { alert(data.error); return; }
            displayResult(data);
        } catch (err) {
            console.error("Upload error:", err);
        }
        fileInput.value = "";
    });

    // ------------------------------------------------------------------
    // Demo shapes
    // ------------------------------------------------------------------
    demoSnr.addEventListener("input", () => {
        $("#snr-val").textContent = demoSnr.value;
    });

    btnRunDemo.addEventListener("click", async () => {
        const shape = demoShape.value;
        const snr = parseFloat(demoSnr.value);
        btnRunDemo.textContent = "Processing...";
        btnRunDemo.disabled = true;

        try {
            const resp = await fetch(`/api/demo/${shape}?snr=${snr}`);
            const data = await resp.json();
            displayResult(data);

            // Show sim metrics
            if (data.sim_metrics) {
                simMetrics.classList.remove("hidden");
                $("#sim-p").textContent = data.sim_metrics.precision;
                $("#sim-r").textContent = data.sim_metrics.recall;
                $("#sim-f1").textContent = data.sim_metrics.f1;
            }
            // Sim images
            if (data.images) {
                if (data.images.sim_input) $("#sim-img-input").src = "data:image/jpeg;base64," + data.images.sim_input;
                if (data.images.sim_detector) $("#sim-img-detector").src = "data:image/jpeg;base64," + data.images.sim_detector;
                if (data.images.gt_edges) $("#sim-img-gt").src = "data:image/jpeg;base64," + data.images.gt_edges;
                if (data.images.sim_edges) $("#sim-img-edges").src = "data:image/jpeg;base64," + data.images.sim_edges;
            }
        } catch (err) {
            console.error("Demo error:", err);
        }
        btnRunDemo.textContent = "Run Simulation";
        btnRunDemo.disabled = false;
    });

    // ------------------------------------------------------------------
    // Display results
    // ------------------------------------------------------------------
    function displayResult(data) {
        // Main image
        const imgKey = currentView;
        const imgData = (data.images || {})[imgKey];
        if (imgData) {
            showImage(imgData);
        } else if (data.images && data.images.overlay) {
            showImage(data.images.overlay);
        }

        updateMetrics(data.metrics);
        updateDetections(data.shapes, data.objects);
        updateTracked(data.tracked);
        updateAlert(data.alert);

        // Scale images
        if (data.images) {
            for (let i = 0; i < 3; i++) {
                const key = `edges_scale_${i}`;
                if (data.images[key]) {
                    const scaleImg = $(`#img-scale-${i}`);
                    if (scaleImg) scaleImg.src = "data:image/jpeg;base64," + data.images[key];
                }
            }
        }

        if (data.metrics && data.metrics.edge_counts) {
            data.metrics.edge_counts.forEach((c, i) => {
                const el = $(`#count-scale-${i}`);
                if (el) el.textContent = c.toLocaleString() + " px";
            });
        }
    }

    function showImage(b64) {
        mainImage.src = "data:image/jpeg;base64," + b64;
        mainImage.style.display = "block";
        noImageMsg.style.display = "none";
    }

    function updateMetrics(m) {
        if (!m) return;
        if (m.snr_db !== undefined) $("#m-snr").textContent = m.snr_db;
        if (m.edge_threshold !== undefined) $("#m-thresh").textContent = m.edge_threshold;
        if (m.fused_edge_count !== undefined) $("#m-edges").textContent = m.fused_edge_count.toLocaleString();
        if (m.latency_ms !== undefined) $("#m-latency").textContent = m.latency_ms;
        if (m.fps !== undefined) $("#m-fps").textContent = m.fps;
        if (m.frame_size) $("#m-size").textContent = m.frame_size.join("x");
    }

    function updateDetections(shapes, objects) {
        const list = $("#detections-list");
        const items = [...(shapes || []), ...(objects || [])];
        if (!items.length) {
            list.innerHTML = '<p class="empty-msg">No detections</p>';
            return;
        }
        list.innerHTML = items.slice(0, 12).map(d => `
            <div class="det-item">
                <span class="det-label">${d.label}</span>
                <span>
                    <span class="det-conf">${(d.confidence * 100).toFixed(0)}%</span>
                    ${d.priority ? `<span class="det-prio">P:${d.priority}</span>` : ""}
                </span>
            </div>
        `).join("");
    }

    function updateTracked(tracked) {
        const list = $("#tracked-list");
        if (!tracked || !tracked.length) {
            list.innerHTML = '<p class="empty-msg">No tracked objects</p>';
            return;
        }
        list.innerHTML = tracked.slice(0, 10).map(t => `
            <div class="trk-item">
                <span><span class="trk-id">#${t.id}</span> <span class="trk-label">${t.label}</span></span>
                <span>v=(${t.velocity[0]}, ${t.velocity[1]})</span>
            </div>
        `).join("");
    }

    function updateAlert(alert) {
        if (alert) {
            alertBanner.classList.remove("hidden");
            alertText.textContent = `${alert.target} — Priority: ${alert.priority}`;
        } else {
            alertBanner.classList.add("hidden");
        }
    }

    // ------------------------------------------------------------------
    // Parameter changes
    // ------------------------------------------------------------------
    function sendParams(key, value) {
        if (socket && socket.connected) {
            const update = {};
            update[key] = value;
            socket.emit("update_params", update);
        }
    }

    // Sliders
    $("#param-edge-t").addEventListener("input", function () {
        $("#val-edge-t").textContent = this.value;
        sendParams("edge_t", parseFloat(this.value));
    });
    $("#param-smooth").addEventListener("input", function () {
        $("#val-smooth").textContent = this.value;
        sendParams("smooth_sigma", parseFloat(this.value));
    });
    $("#param-coverage").addEventListener("input", function () {
        $("#val-coverage").textContent = this.value;
        sendParams("coverage_radius", parseInt(this.value));
    });
    $("#param-fusion").addEventListener("change", function () {
        sendParams("fusion_mode", this.value);
    });

    // Toggles
    const toggleMap = {
        "tog-denoise": "denoise",
        "tog-contrast": "contrast",
        "tog-pseudo": "pseudocolor",
        "tog-shapes": "detect_shapes",
        "tog-objects": "detect_objects",
        "tog-depth": "depth",
        "tog-tracking": "tracking",
    };
    for (const [elemId, paramKey] of Object.entries(toggleMap)) {
        document.getElementById(elemId).addEventListener("change", function () {
            sendParams(paramKey, this.checked);
        });
    }

    // ------------------------------------------------------------------
    // Keyboard shortcuts
    // ------------------------------------------------------------------
    document.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
        switch (e.key) {
            case "1": setSource("webcam"); break;
            case "2": setSource("upload"); break;
            case "3": setSource("demo"); break;
            case "d": document.getElementById("tog-denoise").click(); break;
            case "c": document.getElementById("tog-contrast").click(); break;
            case "p": document.getElementById("tog-pseudo").click(); break;
            case "s": document.getElementById("tog-shapes").click(); break;
            case "o": document.getElementById("tog-objects").click(); break;
            case "Enter":
                if (currentSource === "demo") btnRunDemo.click();
                break;
        }
    });

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------
    connectSocket();
    setSource("demo");

})();
