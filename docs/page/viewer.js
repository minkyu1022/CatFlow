document.addEventListener("DOMContentLoaded", function () {
  var stage = new NGL.Stage("viewport-sample", { backgroundColor: "white" });

  stage.loadFile("data/relaxed_sample.pdb", {
    defaultRepresentation: false,
    ext: "pdb",
  }).then(function (comp) {
    comp.setName("catalyst-sample");

    comp.removeAllRepresentations();

    comp.addRepresentation("spacefill", {
      colorScheme: "element",
      radiusScale: 0.5
    });

    comp.autoView();
  });

  var toggleSpinBtn = document.getElementById("toggleSpin-sample");
  var isSpinning = false;
  if (toggleSpinBtn) {
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
  }

  var resetViewBtn = document.getElementById("resetView-sample");
  if (resetViewBtn) {
    resetViewBtn.addEventListener("click", function () {
      var comp = stage.getComponentsByName("catalyst-sample").list[0];
      if (comp) comp.autoView(500);
    });
  }

  window.addEventListener("resize", function () {
    stage.handleResize();
  });
});