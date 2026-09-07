/* ==========================================================================
   Advanced AI Biometrics, Head Pose, Eye Gaze & Clothing Color JS Controller
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    let currentMode = 1;
    let isCameraActive = true;
    let isRoiEnabled = false;
    let isAudioAlertsEnabled = true;
    let spokenTags = new Set();
    let chartInstance = null;

    // DOM Elements
    const metricsOverlay = document.getElementById('metricsOverlay');
    const tagsWrapper = document.getElementById('tagsWrapper');
    const distanceBadge = document.getElementById('distanceBadge');
    const confValue = document.getElementById('confValue');
    const confSlider = document.getElementById('confSlider');
    const galleryGrid = document.getElementById('galleryGrid');
    const toggleCamBtn = document.getElementById('toggleCamBtn');
    const totalCountBadge = document.getElementById('totalCountBadge');
    const roiSwitch = document.getElementById('roiSwitch');
    const audioSwitch = document.getElementById('audioSwitch');
    const headerAlertBadge = document.getElementById('headerAlertBadge');

    // Web Speech Synthesis (Text-to-Speech Voice Alerts)
    function speakText(text) {
        if (!isAudioAlertsEnabled || !('speechSynthesis' in window)) return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }

    // Initialize Chart.js Telemetry Graph
    function initChart() {
        const ctx = document.getElementById('telemetryChart');
        if (!ctx) return;

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Detected Targets',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'FPS',
                        data: [],
                        borderColor: '#38bdf8',
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, labels: { color: '#94a3b8', font: { size: 10 } } } },
                scales: {
                    x: { display: false },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8', font: { size: 10 } } }
                }
            }
        });
    }

    // Mode Switcher
    window.selectMode = function(modeId) {
        currentMode = modeId;
        document.querySelectorAll('.mode-btn').forEach(btn => {
            const btnMode = parseInt(btn.getAttribute('data-mode') || btn.dataset.mode);
            if (btnMode === modeId) btn.classList.add('active');
            else btn.classList.remove('active');
        });
        sendSettings({ mode: modeId });
    };

    // Confidence Slider Update
    window.onConfidenceChange = function(val) {
        confValue.innerText = `${val}%`;
        sendSettings({ confidence: parseFloat(val) / 100.0 });
    };

    // Toggle Camera Stream
    window.toggleCameraStream = function() {
        isCameraActive = !isCameraActive;
        toggleCamBtn.innerHTML = isCameraActive ? '⏸️ Pause Stream' : '▶️ Resume Stream';
        sendSettings({ camera_active: isCameraActive });
    };

    // Toggle ROI Security Zone
    if (roiSwitch) {
        roiSwitch.addEventListener('change', (e) => {
            isRoiEnabled = e.target.checked;
            sendSettings({ roi_enabled: isRoiEnabled });
        });
    }

    // Toggle Audio Voice Alerts
    if (audioSwitch) {
        audioSwitch.addEventListener('change', (e) => {
            isAudioAlertsEnabled = e.target.checked;
            sendSettings({ audio_alerts: isAudioAlertsEnabled });
            if (isAudioAlertsEnabled) speakText("AI Voice Alert system active.");
        });
    }

    // Send Settings to Backend API
    function sendSettings(data) {
        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        }).catch(err => console.error("Settings error:", err));
    }

    // Capture Snapshot
    window.captureSnapshot = function() {
        fetch('/api/capture', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    speakText("Snapshot captured.");
                    loadGallery();
                }
            }).catch(err => console.error("Capture error:", err));
    };

    // Delete Snapshot
    window.deleteSnapshot = function(filename, event) {
        if (event) event.stopPropagation();
        fetch('/api/delete_snapshot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: filename })
        }).then(res => res.json())
          .then(data => {
              if (data.status === 'success') loadGallery();
          });
    };

    // Export CSV Audit Log
    window.exportCSV = function() {
        window.location.href = '/api/export_csv';
    };

    // Load Image Gallery
    function loadGallery() {
        fetch('/api/gallery')
            .then(res => res.json())
            .then(data => {
                galleryGrid.innerHTML = '';
                if (data.images.length === 0) {
                    galleryGrid.innerHTML = '<p style="color: var(--text-muted); font-size: 0.8rem; grid-column: span 3;">No snapshots captured yet.</p>';
                    return;
                }
                data.images.forEach(imgUrl => {
                    const filename = imgUrl.split('/').pop();
                    const card = document.createElement('div');
                    card.className = 'gallery-card';
                    card.innerHTML = `
                        <img src="${imgUrl}" class="gallery-thumbnail" onclick="window.open('${imgUrl}', '_blank')">
                        <button class="gallery-delete-btn" onclick="deleteSnapshot('${filename}', event)">✕</button>
                    `;
                    galleryGrid.appendChild(card);
                });
            }).catch(err => console.error("Gallery error:", err));
    }

    // Poll Backend Stats (FPS, Latency, Distance, Heart Rate, Drowsiness, Posture Angle, Emotion, Gaze, Dress)
    setInterval(() => {
        fetch('/api/stats')
            .then(res => res.json())
            .then(stats => {
                // Update Overlay Metrics & Distance
                metricsOverlay.innerText = `FPS: ${stats.fps} | Latency: ${stats.latency} ms`;
                totalCountBadge.innerText = `${stats.total_detections} Active Targets`;
                distanceBadge.innerText = `📏 Dist: ${stats.closest_distance}`;

                // Update Vitals Cards
                const bpmEl = document.getElementById('bpmVal');
                const drowsyEl = document.getElementById('drowsinessVal');
                const kneeEl = document.getElementById('kneeAngleVal');
                const attentiveEl = document.getElementById('attentiveVal');

                if (bpmEl) bpmEl.innerText = `${stats.heart_rate_bpm || 72} BPM`;
                if (drowsyEl) drowsyEl.innerText = stats.drowsiness_index || '0% Alert';
                if (kneeEl) kneeEl.innerText = stats.knee_angle || '172° Standing';
                if (attentiveEl) attentiveEl.innerText = stats.attentiveness_score || '98%';

                // Handle Security Intrusion Alert
                if (stats.intrusion_detected) {
                    headerAlertBadge.style.display = 'flex';
                    speakText("Security Alert! Intrusion detected in security zone!");
                } else {
                    headerAlertBadge.style.display = 'none';
                }

                // Update Object & Biometric Tags
                tagsWrapper.innerHTML = '';
                const objects = stats.detected_summary || {};
                const keys = Object.keys(objects);
                const analytics = stats.facial_analytics || [];

                if (keys.length === 0 && analytics.length === 0) {
                    tagsWrapper.innerHTML = '<span class="tag-item" style="opacity: 0.6;">Scanning for targets, gaze & dress colors...</span>';
                    spokenTags.clear();
                } else {
                    // Object & Skeleton Limb tags
                    keys.forEach(key => {
                        const count = objects[key];
                        const tag = document.createElement('span');
                        tag.className = stats.intrusion_detected ? 'tag-item intrusion' : 'tag-item';
                        
                        let displayLabel = `🔍 <strong>${key.replace('_', ' ').toUpperCase()}:</strong> ${count}`;
                        if (key === 'left_hand') displayLabel = `🖐️ <strong>LEFT HAND:</strong> ${count}`;
                        else if (key === 'right_hand') displayLabel = `🖐️ <strong>RIGHT HAND:</strong> ${count}`;
                        else if (key === 'left_leg') displayLabel = `🦵 <strong>LEFT LEG:</strong> ${count}`;
                        else if (key === 'right_leg') displayLabel = `🦵 <strong>RIGHT LEG:</strong> ${count}`;
                        
                        tag.innerHTML = displayLabel;
                        tagsWrapper.appendChild(tag);

                        // Voice alert for new object detections
                        if (!spokenTags.has(key) && isAudioAlertsEnabled) {
                            speakText(`Detected ${key.replace('_', ' ')}`);
                            spokenTags.add(key);
                        }
                    });

                    // Detailed Facial Biometric Analytics (Pose, Gaze, Emotion, Dress Color)
                    analytics.forEach(item => {
                        // Pose Tag
                        const poseTag = document.createElement('span');
                        poseTag.className = 'tag-item pose';
                        poseTag.innerHTML = `👤 <strong>POSE:</strong> ${item.pose}`;
                        tagsWrapper.appendChild(poseTag);

                        // Gaze Tag
                        const gazeTag = document.createElement('span');
                        gazeTag.className = 'tag-item gaze';
                        gazeTag.innerHTML = `👀 <strong>GAZE:</strong> ${item.gaze}`;
                        tagsWrapper.appendChild(gazeTag);

                        // Emotion Tag
                        const emoTag = document.createElement('span');
                        emoTag.className = 'tag-item emotion';
                        emoTag.innerHTML = `🎭 <strong>MOOD:</strong> ${item.emotion}`;
                        tagsWrapper.appendChild(emoTag);

                        // Dress Color Tag
                        const dressTag = document.createElement('span');
                        dressTag.className = 'tag-item dress';
                        dressTag.innerHTML = `👔 <strong>DRESS:</strong> ${item.dress}`;
                        tagsWrapper.appendChild(dressTag);

                        // Voice alert for clothing color detection
                        const dressKey = `dress_${item.dress}`;
                        if (!spokenTags.has(dressKey) && isAudioAlertsEnabled) {
                            speakText(`Target wearing ${item.dress} dress`);
                            spokenTags.add(dressKey);
                        }
                    });
                }

                // Update Telemetry Chart
                if (chartInstance) {
                    const timeLabel = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                    chartInstance.data.labels.push(timeLabel);
                    chartInstance.data.datasets[0].data.push(stats.total_detections);
                    chartInstance.data.datasets[1].data.push(stats.fps);

                    if (chartInstance.data.labels.length > 15) {
                        chartInstance.data.labels.shift();
                        chartInstance.data.datasets[0].data.shift();
                        chartInstance.data.datasets[1].data.shift();
                    }
                    chartInstance.update('none');
                }
            }).catch(err => console.error("Telemetry sync error:", err));
    }, 700);

    initChart();
    loadGallery();
});
