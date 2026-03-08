(() => {
    'use strict';

    /* ============================
       SLIDE MANAGEMENT
       ============================ */
    const slides = document.querySelectorAll('.slide');
    const totalSlides = slides.length;
    let currentSlide = 0;

    const progressBar = document.getElementById('progressBar');
    const slideCounter = document.getElementById('slideCounter');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');

    function goToSlide(index) {
        if (index < 0 || index >= totalSlides) return;

        const prev = slides[currentSlide];
        const next = slides[index];

        // Determine direction
        if (index > currentSlide) {
            prev.classList.remove('active');
            prev.classList.add('exit-left');
            setTimeout(() => prev.classList.remove('exit-left'), 560);
        } else {
            prev.classList.remove('active');
            prev.style.transform = 'translateX(60px)';
            setTimeout(() => { prev.style.transform = ''; }, 560);
        }

        next.classList.add('active');
        currentSlide = index;
        updateUI();
    }

    function nextSlide() { goToSlide(currentSlide + 1); }
    function prevSlide() { goToSlide(currentSlide - 1); }

    function updateUI() {
        // Progress bar
        const pct = ((currentSlide + 1) / totalSlides) * 100;
        progressBar.style.width = pct + '%';

        // Slide counter
        slideCounter.textContent = `${currentSlide + 1} / ${totalSlides}`;

        // Nav button visibility
        prevBtn.style.opacity = currentSlide === 0 ? '0.2' : '';
        prevBtn.style.pointerEvents = currentSlide === 0 ? 'none' : '';
        nextBtn.style.opacity = currentSlide === totalSlides - 1 ? '0.2' : '';
        nextBtn.style.pointerEvents = currentSlide === totalSlides - 1 ? 'none' : '';
    }

    // Initialize first slide
    slides[0].classList.add('active');
    updateUI();

    // Button listeners
    prevBtn.addEventListener('click', prevSlide);
    nextBtn.addEventListener('click', nextSlide);

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
            e.preventDefault();
            nextSlide();
        } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
            e.preventDefault();
            prevSlide();
        } else if (e.key === 'Home') {
            e.preventDefault();
            goToSlide(0);
        } else if (e.key === 'End') {
            e.preventDefault();
            goToSlide(totalSlides - 1);
        }
    });

    // Touch / swipe support
    let touchStartX = 0;
    let touchStartY = 0;
    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
        touchStartY = e.changedTouches[0].screenY;
    }, { passive: true });

    document.addEventListener('touchend', (e) => {
        const dx = e.changedTouches[0].screenX - touchStartX;
        const dy = e.changedTouches[0].screenY - touchStartY;
        if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 50) {
            if (dx < 0) nextSlide();
            else prevSlide();
        }
    }, { passive: true });


    /* ============================
       10-MINUTE COUNTDOWN TIMER
       ============================ */
    const timerEl = document.getElementById('timer');
    let totalSeconds = 10 * 60; // 10 minutes
    let timerRunning = false;
    let timerInterval = null;

    function formatTime(sec) {
        const m = Math.floor(sec / 60);
        const s = sec % 60;
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    function updateTimerDisplay() {
        timerEl.textContent = formatTime(totalSeconds);

        // Warning states
        timerEl.classList.remove('warning', 'danger');
        if (totalSeconds <= 60) {
            timerEl.classList.add('danger');
        } else if (totalSeconds <= 120) {
            timerEl.classList.add('warning');
        }
    }

    function startTimer() {
        if (timerRunning) return;
        timerRunning = true;
        timerInterval = setInterval(() => {
            if (totalSeconds > 0) {
                totalSeconds--;
                updateTimerDisplay();
            } else {
                clearInterval(timerInterval);
                timerRunning = false;
            }
        }, 1000);
    }

    function pauseTimer() {
        clearInterval(timerInterval);
        timerRunning = false;
    }

    function resetTimer() {
        pauseTimer();
        totalSeconds = 10 * 60;
        updateTimerDisplay();
    }

    // Click to start/pause, double-click to reset
    timerEl.addEventListener('click', () => {
        if (timerRunning) pauseTimer();
        else startTimer();
    });
    timerEl.addEventListener('dblclick', (e) => {
        e.preventDefault();
        resetTimer();
    });

    updateTimerDisplay();

})();
