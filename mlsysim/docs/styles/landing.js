// =============================================================================
// MLSYSIM LANDING PAGE — Copy Button
// =============================================================================

(function () {
  "use strict";

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
