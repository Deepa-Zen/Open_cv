/* ==========================================================================
   Python & OpenCV Master Slide Deck - Logic & Interactive Demos
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    let currentSlide = 1;
    const totalSlides = 10;

    // Speaker Notes Database
    const speakerNotes = {
        1: "Slide 1: Introduce Python's dominant position in data science and computer vision. Emphasize that while Python syntax is easy to write, libraries like OpenCV and NumPy execute at high speeds using C/C++ compiled binaries under the hood.",
        2: "Slide 2: Overview of industry applications. Highlight how face detection, autonomous driving, and medical image segmentation all share common foundational algorithms like filtering, edge detection, and matrix operations.",
        3: "Slide 3: Introduce OpenCV's history and architecture. Point out the core modules (imgproc, highgui, video, dnn) and mention its hardware optimization capabilities (CUDA, SIMD).",
        4: "Slide 4: Explain digital image representation. Emphasize the BGR color space in OpenCV (Blue, Green, Red) instead of RGB, and explain that an image is simply a 3D matrix array of uint8 values (0-255).",
        5: "Slide 5: Walk through the classical computer vision pipeline: Grayscale conversion -> Gaussian smoothing -> Canny edge detection -> Contour extraction.",
        6: "Slide 6: Explain real-time frame loop architecture using cv2.VideoCapture. Mention key frame manipulation routines like cv2.flip and waitKey keyboard polling.",
        7: "Slide 7: Compare classical Haar Cascades with modern Deep Learning Object Detectors (SSD / YOLO). Discuss speed vs accuracy trade-offs.",
        8: "Slide 8: Highlight OpenCV's built-in cv2.dnn module. Explain how blobFromImage performs mean subtraction, scaling, and channel swapping without needing PyTorch or TensorFlow installed.",
        9: "Slide 9: Case Study Demonstration of our active project. Show how confidence thresholding filters bounding boxes and overlaying real-time detection metrics.",
        10: "Slide 10: Wrap up with future horizons in Spatial AI, MediaPipe, ONNX runtime, and WebAssembly deployment."
    };

    // DOM Elements
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const currentSlideNum = document.getElementById('currentSlideNum');
    const progressBar = document.getElementById('progressBar');
    const currentTopicBadge = document.getElementById('currentTopicBadge');
    const slideItems = document.querySelectorAll('.slide-item');
    const slideCards = document.querySelectorAll('.slide-card');
    const toggleNotesBtn = document.getElementById('toggleNotesBtn');
    const closeNotesBtn = document.getElementById('closeNotesBtn');
    const notesDrawer = document.getElementById('notesDrawer');
    const notesContent = document.getElementById('notesContent');
    const notesSlideNum = document.getElementById('notesSlideNum');
    const fullscreenBtn = document.getElementById('fullscreenBtn');

    // Slide Switching Logic
    function updateSlide(targetSlide) {
        if (targetSlide < 1 || targetSlide > totalSlides) return;
        
        currentSlide = targetSlide;

        // Update Slide Cards
        slideCards.forEach(card => card.classList.remove('active'));
        const activeCard = document.getElementById(`slide-${currentSlide}`);
        if (activeCard) activeCard.classList.add('active');

        // Update Sidebar items
        slideItems.forEach(item => {
            const slideNum = parseInt(item.getAttribute('data-slide'));
            if (slideNum === currentSlide) {
                item.classList.add('active');
                item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                item.classList.remove('active');
            }
        });

        // Update Controls & Header
        currentSlideNum.textContent = currentSlide;
        currentTopicBadge.textContent = `Topic ${currentSlide} of ${totalSlides}`;
        const progress = (currentSlide / totalSlides) * 100;
        progressBar.style.width = `${progress}%`;

        // Update Speaker Notes
        if (notesSlideNum) notesSlideNum.textContent = currentSlide;
        if (notesContent) notesContent.textContent = speakerNotes[currentSlide] || "No notes available for this slide.";

        // Run slide-specific initializations
        if (currentSlide === 4) initPixelInspector();
        if (currentSlide === 9) initDemoCanvas();
    }

    // Event Listeners for Navigation
    if (prevBtn) prevBtn.addEventListener('click', () => updateSlide(currentSlide - 1));
    if (nextBtn) nextBtn.addEventListener('click', () => updateSlide(currentSlide + 1));

    slideItems.forEach(item => {
        item.addEventListener('click', () => {
            const slideNum = parseInt(item.getAttribute('data-slide'));
            updateSlide(slideNum);
        });
    });

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === ' ') {
            e.preventDefault();
            updateSlide(currentSlide + 1);
        } else if (e.key === 'ArrowLeft') {
            e.preventDefault();
            updateSlide(currentSlide - 1);
        } else if (e.key === 'f' || e.key === 'F') {
            toggleFullscreen();
        }
    });

    // Speaker Notes Toggle
    if (toggleNotesBtn) {
        toggleNotesBtn.addEventListener('click', () => {
            notesDrawer.classList.toggle('open');
        });
    }

    if (closeNotesBtn) {
        closeNotesBtn.addEventListener('click', () => {
            notesDrawer.classList.remove('open');
        });
    }

    // Fullscreen Toggle
    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.log(`Fullscreen request error: ${err.message}`);
            });
        } else {
            if (document.exitFullscreen) document.exitFullscreen();
        }
    }

    if (fullscreenBtn) fullscreenBtn.addEventListener('click', toggleFullscreen);

    // ==========================================================================
    // Slide 4: Interactive Pixel Inspector
    // ==========================================================================
    function initPixelInspector() {
        const grid = document.getElementById('pixelGrid');
        if (!grid || grid.children.length > 0) return;

        const pixelCoord = document.getElementById('pixelCoord');
        const pixelBGR = document.getElementById('pixelBGR');

        for (let y = 0; y < 8; y++) {
            for (let x = 0; x < 8; x++) {
                const cell = document.createElement('div');
                cell.className = 'pixel-cell';
                
                // Generate color gradient
                const r = Math.floor((x / 7) * 255);
                const g = Math.floor((y / 7) * 200);
                const b = Math.floor(((x + y) / 14) * 255);

                cell.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
                
                cell.addEventListener('mouseenter', () => {
                    if (pixelCoord) pixelCoord.textContent = `(x: ${x}, y: ${y})`;
                    if (pixelBGR) pixelBGR.textContent = `[B: ${b}, G: ${g}, R: ${r}]`;
                });

                grid.appendChild(cell);
            }
        }
    }

    // ==========================================================================
    // Slide 9: Face Detection Canvas Simulation
    // ==========================================================================
    let simAnimId = null;
    let showBoundingBox = true;

    function initDemoCanvas() {
        const canvas = document.getElementById('demoCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const confSlider = document.getElementById('confSlider');
        const confValue = document.getElementById('confValue');
        const simDetectBtn = document.getElementById('simDetectBtn');
        const simToggleBox = document.getElementById('simToggleBox');

        let threshold = 70;

        if (confSlider) {
            confSlider.addEventListener('input', (e) => {
                threshold = parseInt(e.target.value);
                if (confValue) confValue.textContent = `${threshold}%`;
            });
        }

        if (simToggleBox) {
            simToggleBox.addEventListener('click', () => {
                showBoundingBox = !showBoundingBox;
            });
        }

        if (simDetectBtn) {
            simDetectBtn.addEventListener('click', () => {
                drawFrame();
            });
        }

        function drawFrame() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw simulated video background grid
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Grid lines
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.1)';
            ctx.lineWidth = 1;
            for (let x = 0; x < canvas.width; x += 40) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }

            // Draw stylized face avatar
            const faceX = canvas.width / 2;
            const faceY = canvas.height / 2;
            
            // Head
            ctx.fillStyle = '#1e293b';
            ctx.beginPath();
            ctx.arc(faceX, faceY, 60, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Eyes
            ctx.fillStyle = '#38bdf8';
            ctx.beginPath();
            ctx.arc(faceX - 20, faceY - 10, 6, 0, Math.PI * 2);
            ctx.arc(faceX + 20, faceY - 10, 6, 0, Math.PI * 2);
            ctx.fill();

            // Simulated Detection (Confidence score = 94.2%)
            const currentConf = 94;
            if (showBoundingBox && currentConf >= threshold) {
                const boxX = faceX - 80;
                const boxY = faceY - 80;
                const boxW = 160;
                const boxH = 160;

                // Bounding box
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 3;
                ctx.strokeRect(boxX, boxY, boxW, boxH);

                // Label tag
                ctx.fillStyle = '#10b981';
                ctx.fillRect(boxX, boxY - 24, 130, 24);

                ctx.fillStyle = '#000000';
                ctx.font = 'bold 12px Fira Code';
                ctx.fillText(`Face: ${(currentConf / 100).toFixed(2)}`, boxX + 8, boxY - 8);
            }
        }

        drawFrame();
    }

    // Initialize first slide
    updateSlide(1);
});
