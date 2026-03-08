// =============================================================================
// MLSYSIM LANDING PAGE — Carousel & Copy Button
// =============================================================================

(function () {
  "use strict";

  // ── Carousel ──────────────────────────────────────────────────────────────
  const track = document.querySelector(".im-carousel-track");
  if (!track) return;

  const slides = track.querySelectorAll(".im-slide");
  const dots = document.querySelectorAll(".im-dot");
  const prevBtn = track.querySelector(".im-arrow-prev");
  const nextBtn = track.querySelector(".im-arrow-next");
  let current = 0;
  let autoTimer = null;
  const AUTO_INTERVAL = 6000; // ms between auto-advance

  function goTo(index) {
    // Wrap around
    if (index < 0) index = slides.length - 1;
    if (index >= slides.length) index = 0;

    slides[current].classList.remove("im-slide-active");
    if (dots[current]) dots[current].classList.remove("im-dot-active");

    current = index;

    slides[current].classList.add("im-slide-active");
    if (dots[current]) dots[current].classList.add("im-dot-active");
  }

  function startAuto() {
    stopAuto();
    autoTimer = setInterval(function () {
      goTo(current + 1);
    }, AUTO_INTERVAL);
  }

  function stopAuto() {
    if (autoTimer) {
      clearInterval(autoTimer);
      autoTimer = null;
    }
  }

  // Arrow buttons
  if (prevBtn) {
    prevBtn.addEventListener("click", function () {
      goTo(current - 1);
      startAuto(); // reset timer on manual interaction
    });
  }
  if (nextBtn) {
    nextBtn.addEventListener("click", function () {
      goTo(current + 1);
      startAuto();
    });
  }

  // Dot buttons
  dots.forEach(function (dot) {
    dot.addEventListener("click", function () {
      var idx = parseInt(dot.getAttribute("data-slide"), 10);
      if (!isNaN(idx)) {
        goTo(idx);
        startAuto();
      }
    });
  });

  // Keyboard navigation (when carousel area is in view)
  document.addEventListener("keydown", function (e) {
    if (e.key === "ArrowLeft") { goTo(current - 1); startAuto(); }
    if (e.key === "ArrowRight") { goTo(current + 1); startAuto(); }
  });

  // Start auto-rotation
  startAuto();

  // ── Copy Button ───────────────────────────────────────────────────────────
  window.copyInstall = function () {
    var cmd = document.querySelector("code.im-cmd");
    if (!cmd) return;
    var text = cmd.textContent.trim();
    navigator.clipboard.writeText(text).then(function () {
      var btn = document.getElementById("copy-btn");
      if (btn) {
        btn.textContent = "Copied!";
        setTimeout(function () { btn.textContent = "Copy"; }, 2000);
      }
    });
  };
})();
