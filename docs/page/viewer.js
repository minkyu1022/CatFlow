document.addEventListener("DOMContentLoaded", function () {
  var stage = new NGL.Stage("viewport-sample", { backgroundColor: "white" });

  stage.loadFile("data/relaxed_sample.pdb", {
    defaultRepresentation: false,
    ext: "pdb",
  }).then(function (comp) {
    comp.setName("catalyst-sample");
    comp.addRepresentation("spacefill", {
      radiusScale: 1.5,
    });
    comp.autoView();
  }).catch(function (err) {
    console.error("Failed to load structure:", err);
  });

  // Spin toggle
  var toggleSpinBtn = document.getElementById("toggleSpin-sample");
  var isSpinning = false;
  toggleSpinBtn.addEventListener("click", function () {
    if (!isSpinning) {
      stage.setSpin([0, 1, 0], 0.01);
      isSpinning = true;
      toggleSpinBtn.textContent = "Stop Spin";
    } else {
      stage.setSpin(null, null);
      isSpinning = false;
      toggleSpinBtn.textContent = "Spin";
    }
  });

  // Reset view
  var resetViewBtn = document.getElementById("resetView-sample");
  resetViewBtn.addEventListener("click", function () {
    var comp = stage.getComponentsByName("catalyst-sample").list[0];
    if (comp) comp.autoView(500);
  });

  // Handle window resize
  window.addEventListener("resize", function () {
    stage.handleResize();
  });
});
