// =============================================================================
// MLSYSIM LANDING PAGE — Copy Button + Carousel
// =============================================================================

(function () {
  "use strict";

  // ---- Copy install command ----
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

  // ---- Carousel ----
  document.addEventListener("DOMContentLoaded", function () {
    var track = document.querySelector(".im-carousel-track");
    if (!track) return;

    var slides = track.querySelectorAll(".im-slide");
    var dots = document.querySelectorAll(".im-dot");
    var prevBtn = track.querySelector(".im-arrow-prev");
    var nextBtn = track.querySelector(".im-arrow-next");
    var current = 0;
    var total = slides.length;
    var autoTimer = null;
    var AUTO_INTERVAL = 6000;

    function goTo(idx) {
      idx = ((idx % total) + total) % total;  // wrap around
      slides[current].classList.remove("im-slide-active");
      dots[current].classList.remove("im-dot-active");
      current = idx;
      slides[current].classList.add("im-slide-active");
      dots[current].classList.add("im-dot-active");
    }

    function next() { goTo(current + 1); }
    function prev() { goTo(current - 1); }

    function resetAuto() {
      clearInterval(autoTimer);
      autoTimer = setInterval(next, AUTO_INTERVAL);
    }

    // Arrow buttons
    if (prevBtn) {
      prevBtn.addEventListener("click", function (e) {
        e.preventDefault();
        prev();
        resetAuto();
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener("click", function (e) {
        e.preventDefault();
        next();
        resetAuto();
      });
    }

    // Dot buttons
    dots.forEach(function (dot) {
      dot.addEventListener("click", function () {
        var idx = parseInt(this.getAttribute("data-slide"), 10);
        goTo(idx);
        resetAuto();
      });
    });

    // Keyboard navigation
    document.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { prev(); resetAuto(); }
      if (e.key === "ArrowRight") { next(); resetAuto(); }
    });

    // Start auto-rotation
    autoTimer = setInterval(next, AUTO_INTERVAL);
  });
})();
