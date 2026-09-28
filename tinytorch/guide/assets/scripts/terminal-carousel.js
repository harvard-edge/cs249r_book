<script>
/**
 * TinyTorch Interactive Terminal Showcase Carousel
 * Dot-paginated carousel with click-to-zoom lightbox modal
 */

(function () {
  let currentSlide = 0;
  const slidesData = [
    {
      index: 0,
      title: "Tito CLI Companion & Environment Diagnostics",
      tagline: "First-time setup, 100% green health checks, and module workflow companion",
      command: "tito && tito system health",
      gif: "assets/images/demos/tinytorch-06-tito-companion.gif",
      mp4: "assets/images/demos/tinytorch-06-tito-companion.mp4",
      alt: "Tito CLI companion welcome banner and environment health diagnostic verification"
    },
    {
      index: 1,
      title: "One-Line Install & Milestone 01 Out of the Box",
      tagline: "From curl to running Frank Rosenblatt's 1958 Perceptron in seconds",
      command: "curl -fsSL https://mlsysbook.ai/tinytorch/install.sh | bash",
      gif: "assets/images/demos/tinytorch-01-install-and-run.gif",
      mp4: "assets/images/demos/tinytorch-01-install-and-run.mp4",
      alt: "TinyTorch one-line install and running milestone 01 Perceptron out of the box"
    },
    {
      index: 2,
      title: "Module Mastery & Milestone Progress Grid",
      tagline: "20/20 modules completed, autograd verified, 7 milestones unlocked",
      command: "tito module complete 06 && tito module status",
      gif: "assets/images/demos/tinytorch-02-modules-mastery.gif",
      mp4: "assets/images/demos/tinytorch-02-modules-mastery.mp4",
      alt: "Completing autograd module and viewing 20/20 modules mastery progress grid"
    },
    {
      index: 3,
      title: "TinyGPT Conversational Streaming Chat",
      tagline: "Decoder transformer streaming token responses character-by-character on CPU",
      command: "tito milestone run 05 --part 4",
      gif: "assets/images/demos/tinytorch-03-tinygpt-chat.gif",
      mp4: "assets/images/demos/tinytorch-03-tinygpt-chat.mp4",
      alt: "TinyGPT conversational interactive terminal chat with live streaming response"
    },
    {
      index: 4,
      title: "TinyCopilot: Code Generation on TinyPy",
      tagline: "Autoregressive code synthesis with 100% AST compiler verification",
      command: "tito milestone run 05 --part 3",
      gif: "assets/images/demos/tinytorch-04-tinycopilot.gif",
      mp4: "assets/images/demos/tinytorch-04-tinycopilot.mp4",
      alt: "TinyCopilot code generation on TinyPy dataset with zero syntax errors verified by AST"
    },
    {
      index: 5,
      title: "MLPerf Benchmarking",
      tagline: "INT8 quantization (4x), im2col (>70x), and KV-cache Pareto frontier",
      command: "tito milestone run 06 --part 1",
      gif: "assets/images/demos/tinytorch-05-mlperf-opt.gif",
      mp4: "assets/images/demos/tinytorch-05-mlperf-opt.mp4",
      alt: "MLPerf optimization triad evaluating dense, spatial, and autoregressive models"
    },
    {
      index: 6,
      title: "TinyTorch Olympics & Competition Events",
      tagline: "Speed, compression, and accuracy leaderboards with interlocking Olympic rings",
      command: "tito olympics",
      gif: "assets/images/demos/tinytorch-07-tito-olympics.gif",
      mp4: "assets/images/demos/tinytorch-07-tito-olympics.mp4",
      alt: "TinyTorch Olympics competition events board and ASCII Olympic rings logo"
    }
  ];

  function updateCarousel(newIndex) {
    const total = slidesData.length;
    currentSlide = (newIndex + total) % total;

    // Update slides
    const slides = document.querySelectorAll(".tt-carousel-slide");
    slides.forEach((slide, idx) => {
      const vid = slide.querySelector("video");
      if (idx === currentSlide) {
        slide.classList.add("active");
        slide.setAttribute("aria-hidden", "false");
        if (vid) {
          vid.currentTime = 0;
          vid.play().catch(() => {});
        }
      } else {
        slide.classList.remove("active");
        slide.setAttribute("aria-hidden", "true");
        if (vid) {
          vid.pause();
        }
      }
    });

    // Update dots
    const dots = document.querySelectorAll(".tt-dot-btn");
    dots.forEach((dot, idx) => {
      if (idx === currentSlide) {
        dot.classList.add("active");
        dot.setAttribute("aria-current", "true");
      } else {
        dot.classList.remove("active");
        dot.removeAttribute("aria-current");
      }
    });

    // Update caption
    const captionEl = document.getElementById("tt-carousel-caption");
    if (captionEl && slidesData[currentSlide]) {
      const data = slidesData[currentSlide];
      captionEl.innerHTML = `<strong>${data.title}</strong>`;
    }
  }

  // Global lightbox handlers
  window.openTerminalLightbox = function (idx) {
    const slideIdx = typeof idx === "number" ? idx : currentSlide;
    const data = slidesData[slideIdx];
    if (!data) return;

    const modal = document.getElementById("tt-lightbox-modal");
    const video = document.getElementById("tt-lightbox-video");
    const videoSrc = document.getElementById("tt-lightbox-video-src");
    const img = document.getElementById("tt-lightbox-img");
    const title = document.getElementById("tt-lightbox-title");
    const caption = document.getElementById("tt-lightbox-caption");

    if (modal) {
      if (img) {
        img.src = data.gif;
        img.alt = data.alt;
      }
      if (video && videoSrc) {
        videoSrc.src = data.mp4;
        video.load();
        video.play().catch(() => {});
      }
      if (title) title.textContent = `Tiny🔥Torch · ${data.title}`;
      if (caption) caption.textContent = data.tagline;
      modal.classList.add("show");
      document.body.style.overflow = "hidden";
    }
  };

  window.closeTerminalLightbox = function (event) {
    if (event && event.target && event.target.closest(".tt-lightbox-content") && !event.target.classList.contains("tt-lightbox-close")) {
      return;
    }
    const modal = document.getElementById("tt-lightbox-modal");
    const video = document.getElementById("tt-lightbox-video");
    if (modal) {
      if (video) video.pause();
      modal.classList.remove("show");
      document.body.style.overflow = "";
    }
  };

  function initCarousel() {
    const container = document.getElementById("terminal-showcase");
    if (!container) return;

    // Prev / Next button listeners
    const prevBtn = container.querySelector(".tt-prev");
    const nextBtn = container.querySelector(".tt-next");

    if (prevBtn) {
      prevBtn.addEventListener("click", () => updateCarousel(currentSlide - 1));
    }
    if (nextBtn) {
      nextBtn.addEventListener("click", () => updateCarousel(currentSlide + 1));
    }

    // Dot navigation listeners
    const dots = container.querySelectorAll(".tt-dot-btn");
    dots.forEach((dot, idx) => {
      dot.addEventListener("click", () => updateCarousel(idx));
    });

    // Keyboard navigation
    document.addEventListener("keydown", (e) => {
      const modal = document.getElementById("tt-lightbox-modal");
      const isModalOpen = modal && modal.classList.contains("show");

      if (e.key === "Escape" && isModalOpen) {
        window.closeTerminalLightbox();
      } else if (!isModalOpen && container.matches(":hover")) {
        if (e.key === "ArrowLeft") updateCarousel(currentSlide - 1);
        if (e.key === "ArrowRight") updateCarousel(currentSlide + 1);
      }
    });

    updateCarousel(0);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initCarousel);
  } else {
    initCarousel();
  }
})();
</script>
