document.addEventListener("DOMContentLoaded", function () {
  // Fetch CIF file and render with 3Dmol.js
  fetch("data/relaxed_sample.cif")
    .then(function (response) { return response.text(); })
    .then(function (cifData) {
      var viewer = $3Dmol.createViewer("viewport-sample", {
        backgroundColor: "white",
      });

      viewer.addModel(cifData, "cif");
      viewer.setStyle({}, {
        sphere: { scale: 1.0 },
      });
      viewer.zoomTo();
      viewer.render();

      // Spin toggle
      var toggleSpinBtn = document.getElementById("toggleSpin-sample");
      var isSpinning = false;
      toggleSpinBtn.addEventListener("click", function () {
        if (!isSpinning) {
          viewer.spin("y");
          isSpinning = true;
          toggleSpinBtn.textContent = "Stop Spin";
        } else {
          viewer.spin(false);
          isSpinning = false;
          toggleSpinBtn.textContent = "Spin";
        }
      });

      // Reset view
      var resetViewBtn = document.getElementById("resetView-sample");
      resetViewBtn.addEventListener("click", function () {
        viewer.zoomTo();
        viewer.render();
      });
    })
    .catch(function (err) {
      console.error("Failed to load structure:", err);
    });
});
