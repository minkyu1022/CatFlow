document.addEventListener("DOMContentLoaded", function () {
  var stage = new NGL.Stage("viewport-sample", { backgroundColor: "white" });

  stage.loadFile("data/relaxed_sample.pdb", {
    defaultRepresentation: false, // 1차 방어
    ext: "pdb",
  }).then(function (comp) {
    comp.setName("catalyst-sample");

    // [핵심 추가] 혹시 모를 기본 그림(큰 공, 본드)을 강제로 싹 지웁니다.
    comp.removeAllRepresentations();

    // 이제 우리가 원하는 작은 공만 새로 그립니다.
    comp.addRepresentation("spacefill", {
      colorScheme: "element",
      radiusScale: 0.4  // 0.4 정도면 확실히 작아집니다.
    });

    comp.autoView();
  });

  // --- 버튼 이벤트 (기존과 동일) ---
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