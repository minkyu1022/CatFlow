document.addEventListener("DOMContentLoaded", function () {
  var stage = new NGL.Stage("viewport-sample", { backgroundColor: "white" });

  var mainComp = null;
  var supercellComps = [];
  var isSupercellVisible = false;

  var vecA = new NGL.Vector3(-8.94147, 0.0, -0.00871);
  var vecB = new NGL.Vector3(0.0, -10.32218, 0.05818);

  stage.loadFile("data/relaxed_sample.pdb", {
    defaultRepresentation: false,
    ext: "pdb",
  }).then(function (comp) {
    mainComp = comp;
    comp.setName("catalyst-sample");

    comp.addRepresentation("spacefill", {
      colorScheme: "element",
      radiusScale: 0.5,
    });

    comp.autoView();

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

    var toggleCellBtn = document.getElementById("toggleCell-sample");
    if (toggleCellBtn) {
      toggleCellBtn.addEventListener("click", function () {
        alert("현재 데이터의 격자 벡터가 회전되어 있어, 표준 Unit Cell 박스와 겹치지 않습니다. Supercell 기능을 이용해 전체 구조를 확인하세요.");
      });
    }

    var toggleSupercellBtn = document.getElementById("toggleSupercell-sample");
    if (toggleSupercellBtn) {
      toggleSupercellBtn.addEventListener("click", function () {
        if (!isSupercellVisible) {
          if (supercellComps.length === 0) {
            for (var i = -1; i <= 1; i++) {
              for (var j = -1; j <= 1; j++) {
                if (i === 0 && j === 0) continue;

                var shift = new NGL.Vector3()
                  .copy(vecA).multiplyScalar(i)
                  .add(new NGL.Vector3().copy(vecB).multiplyScalar(j));

                stage.loadFile("data/relaxed_sample.pdb", {
                  defaultRepresentation: false,
                  ext: "pdb"
                }).then(function (cloneComp) {
                  cloneComp.setPosition([shift.x, shift.y, shift.z]);

                  cloneComp.addRepresentation("spacefill", {
                    colorScheme: "element",
                    radiusScale: 0.5,
                    opacity: 1.0
                  });

                  supercellComps.push(cloneComp);
                });
              }
            }
          } else {
            supercellComps.forEach(function (c) { c.setVisibility(true); });
          }

          isSupercellVisible = true;
          toggleSupercellBtn.textContent = "Hide Supercell";
          toggleSupercellBtn.style.color = "blue";
          toggleSupercellBtn.style.fontWeight = "bold";

          setTimeout(function () { mainComp.autoView(1000); }, 500);

        } else {
          supercellComps.forEach(function (c) { c.setVisibility(false); });
          isSupercellVisible = false;
          toggleSupercellBtn.textContent = "Supercell";
          toggleSupercellBtn.style.color = "black";
          toggleSupercellBtn.style.fontWeight = "normal";

          mainComp.autoView(500);
        }
      });
    }

    var resetViewBtn = document.getElementById("resetView-sample");
    if (resetViewBtn) {
      resetViewBtn.addEventListener("click", function () {
        mainComp.autoView(500);
      });
    }

  });

  window.addEventListener("resize", function () {
    stage.handleResize();
  });
});