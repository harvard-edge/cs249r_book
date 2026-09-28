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
      title: "One-Line Install & Milestone 01 Out of the Box",
      tagline: "Step 1 of 5: From curl to running Frank Rosenblatt's 1958 Perceptron in seconds",
      command: "curl -fsSL https://mlsysbook.ai/tinytorch/install.sh | bash",
      gif: "assets/images/demos/tinytorch-01-install-and-run.gif",
      alt: "TinyTorch one-line install and running milestone 01 Perceptron out of the box"
    },
    {
      index: 1,
      title: "Module Mastery & Milestone Progress Grid",
      tagline: "Step 2 of 5: 20/20 modules completed, autograd verified, 7 milestones unlocked",
      command: "tito module complete 06 && tito module status",
      gif: "assets/images/demos/tinytorch-02-modules-mastery.gif",
      alt: "Completing autograd module and viewing 20/20 modules mastery progress grid"
    },
    {
      index: 2,
      title: "TinyGPT Conversational Chat (Typewriter Streaming)",
      tagline: "Step 3 of 5: Decoder transformer streaming token responses character-by-character on CPU",
      command: "tito milestone run 05 --part 4",
      gif: "assets/images/demos/tinytorch-03-tinygpt-chat.gif",
      alt: "TinyGPT conversational interactive terminal chat with live streaming response"
    },
    {
      index: 3,
      title: "TinyGPT Code Completion on TinyPy",
      tagline: "Step 4 of 5: Autoregressive code generation with 100% AST compiler verification",
      command: "tito milestone run 05 --part 3",
      gif: "assets/images/demos/tinytorch-04-tinygpt-code.gif",
      alt: "TinyGPT code completion on TinyPy dataset with zero syntax errors verified by AST"
    },
    {
      index: 4,
      title: "MLPerf Hardware Optimization Olympics",
      tagline: "Step 5 of 5: INT8 quantization (4x), im2col (>70x), and KV-cache Pareto frontier",
      command: "tito milestone run 06 --part 1",
      gif: "assets/images/demos/tinytorch-05-mlperf-opt.gif",
      alt: "MLPerf optimization triad evaluating dense, spatial, and autoregressive models"
    }
  ];

  function updateCarousel(newIndex) {
    const total = slidesData.length;
    currentSlide = (newIndex + total) % total;

    // Update slides
    const slides = document.querySelectorAll(".tt-carousel-slide");
    slides.forEach((slide, idx) => {
      if (idx === currentSlide) {
        slide.classList.add("active");
        slide.setAttribute("aria-hidden", "false");
      } else {
        slide.classList.remove("active");
        slide.setAttribute("aria-hidden", "true");
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
      captionEl.innerHTML = `<strong>${data.title}</strong> &middot; <span class="tt-caption-tag">${data.tagline}</span>`;
    }
  }

  // Global lightbox handlers
  window.openTerminalLightbox = function (idx) {
    const slideIdx = typeof idx === "number" ? idx : currentSlide;
    const data = slidesData[slideIdx];
    if (!data) return;

    const modal = document.getElementById("tt-lightbox-modal");
    const img = document.getElementById("tt-lightbox-img");
    const title = document.getElementById("tt-lightbox-title");
    const caption = document.getElementById("tt-lightbox-caption");

    if (modal && img) {
      img.src = data.gif;
      img.alt = data.alt;
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
    if (modal) {
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
