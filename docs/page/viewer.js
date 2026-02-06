document.addEventListener("DOMContentLoaded", function () {
  var stage = new NGL.Stage("viewport-sample", { backgroundColor: "white" });

  var pdbData = `CRYST1    8.941   10.322   29.755  89.88  90.11  89.94 P 1           1
MODEL     1
ATOM      1  Ir  MOL     1       3.750   4.332  14.105  1.00  0.00          IR
ATOM      2  Ir  MOL     1       3.750   0.891  14.105  1.00  0.00          IR
ATOM      3  Ir  MOL     1       3.750   7.772  14.105  1.00  0.00          IR
ATOM      4  Ir  MOL     1       3.750   4.332  11.130  1.00  0.00          IR
ATOM      5  Ir  MOL     1       3.750   0.891  11.130  1.00  0.00          IR
ATOM      6  Ir  MOL     1       3.750   7.772  11.130  1.00  0.00          IR
ATOM      7  Ir  MOL     1       3.795   4.412  16.931  1.00  0.00          IR
ATOM      8  Ir  MOL     1       3.808   0.924  16.951  1.00  0.00          IR
ATOM      9  Ir  MOL     1       3.742   7.756  16.926  1.00  0.00          IR
ATOM     10  Ir  MOL     1       0.769   4.332  14.105  1.00  0.00          IR
ATOM     11  Ir  MOL     1       0.769   0.891  14.105  1.00  0.00          IR
ATOM     12  Ir  MOL     1       0.769   7.772  14.105  1.00  0.00          IR
ATOM     13  Ir  MOL     1       0.769   4.332  11.130  1.00  0.00          IR
ATOM     14  Ir  MOL     1       0.769   0.891  11.130  1.00  0.00          IR
ATOM     15  Ir  MOL     1       0.769   7.772  11.130  1.00  0.00          IR
ATOM     16  Ir  MOL     1       0.752   4.393  16.925  1.00  0.00          IR
ATOM     17  Ir  MOL     1       0.709   0.913  16.949  1.00  0.00          IR
ATOM     18  Ir  MOL     1       0.784   7.746  16.927  1.00  0.00          IR
ATOM     19  Ir  MOL     1       6.730   4.332  14.105  1.00  0.00          IR
ATOM     20  Ir  MOL     1       6.730   0.891  14.105  1.00  0.00          IR
ATOM     21  Ir  MOL     1       6.730   7.772  14.105  1.00  0.00          IR
ATOM     22  Ir  MOL     1       6.730   4.332  11.130  1.00  0.00          IR
ATOM     23  Ir  MOL     1       6.730   0.891  11.130  1.00  0.00          IR
ATOM     24  Ir  MOL     1       6.730   7.772  11.130  1.00  0.00          IR
ATOM     25  Ir  MOL     1       6.737   4.340  16.908  1.00  0.00          IR
ATOM     26  Ir  MOL     1       6.739   0.834  16.853  1.00  0.00          IR
ATOM     27  Ir  MOL     1       6.730   7.731  16.949  1.00  0.00          IR
ATOM     28  Ti  MOL     1       5.228   6.048  12.617  1.00  0.00          TI
ATOM     29  Ti  MOL     1       5.228   2.607  12.617  1.00  0.00          TI
ATOM     30  Ti  MOL     1       5.228   9.488  12.617  1.00  0.00          TI
ATOM     31  Ti  MOL     1       5.228   6.048   9.642  1.00  0.00          TI
ATOM     32  Ti  MOL     1       5.228   2.607   9.642  1.00  0.00          TI
ATOM     33  Ti  MOL     1       5.228   9.488   9.642  1.00  0.00          TI
ATOM     34  Ti  MOL     1       5.236   6.050  15.731  1.00  0.00          TI
ATOM     35  Ti  MOL     1       5.240   2.621  15.709  1.00  0.00          TI
ATOM     36  Ti  MOL     1       5.240   9.430  15.729  1.00  0.00          TI
ATOM     37  Ti  MOL     1       2.248   6.048  12.617  1.00  0.00          TI
ATOM     38  Ti  MOL     1       2.248   2.607  12.617  1.00  0.00          TI
ATOM     39  Ti  MOL     1       2.248   9.488  12.617  1.00  0.00          TI
ATOM     40  Ti  MOL     1       2.248   6.048   9.642  1.00  0.00          TI
ATOM     41  Ti  MOL     1       2.248   2.607   9.642  1.00  0.00          TI
ATOM     42  Ti  MOL     1       2.248   9.488   9.642  1.00  0.00          TI
ATOM     43  Ti  MOL     1       2.261   6.047  15.723  1.00  0.00          TI
ATOM     44  Ti  MOL     1       2.258   2.663  15.836  1.00  0.00          TI
ATOM     45  Ti  MOL     1       2.257   9.459  15.724  1.00  0.00          TI
ATOM     46  Ti  MOL     1       8.209   6.048  12.617  1.00  0.00          TI
ATOM     47  Ti  MOL     1       8.209   2.607  12.617  1.00  0.00          TI
ATOM     48  Ti  MOL     1       8.209   9.488  12.617  1.00  0.00          TI
ATOM     49  Ti  MOL     1       8.209   6.048   9.642  1.00  0.00          TI
ATOM     50  Ti  MOL     1       8.209   2.607   9.642  1.00  0.00          TI
ATOM     51  Ti  MOL     1       8.209   9.488   9.642  1.00  0.00          TI
ATOM     52  Ti  MOL     1       8.232   6.046  15.735  1.00  0.00          TI
ATOM     53  Ti  MOL     1       8.213   2.621  15.697  1.00  0.00          TI
ATOM     54  Ti  MOL     1       8.220   9.415  15.729  1.00  0.00          TI
ATOM     55  C   MOL     1       2.274   1.926  19.360  1.00  0.00          C 
ATOM     56  C   MOL     1       2.258   1.894  17.973  1.00  0.00          C 
ATOM     57  H   MOL     1       2.305   1.026  19.991  1.00  0.00          H 
ATOM     58  H   MOL     1       2.222   3.822  19.488  1.00  0.00          H 
ATOM     59  O   MOL     1       2.253   3.051  20.112  1.00  0.00          O 
ENDMDL`;

  var blob = new Blob([pdbData], { type: 'text/plain' });
  var mainComp = null;
  var supercellComps = [];
  var unitcellRep = null;
  var isSupercellOn = false;

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
      stage.autoView(500);
    });
  }

  var toggleCellBtn = document.getElementById("toggleCell-sample");
  if (toggleCellBtn) {
    toggleCellBtn.addEventListener("click", function () {
      if (unitcellRep) {
        var isVisible = !unitcellRep.visible;
        unitcellRep.setVisibility(isVisible);
        toggleCellBtn.style.fontWeight = isVisible ? "bold" : "normal";
        toggleCellBtn.style.color = isVisible ? "blue" : "black";
      }
    });
  }

  var toggleSupercellBtn = document.getElementById("toggleSupercell-sample");
  if (toggleSupercellBtn) {
    toggleSupercellBtn.addEventListener("click", function () {
      if (!isSupercellOn) {
        if (!mainComp) return;

        if (supercellComps.length === 0) {
          var vA = new NGL.Vector3(8.941, 0, 0);
          var vB = new NGL.Vector3(0, 10.322, 0);
          var offsets = [{ i: 1, j: 0 }, { i: 0, j: 1 }, { i: 1, j: 1 }];

          offsets.forEach(function (off) {
            var shift = new NGL.Vector3()
              .add(new NGL.Vector3().copy(vA).multiplyScalar(off.i))
              .add(new NGL.Vector3().copy(vB).multiplyScalar(off.j));

            stage.loadFile(blob, { ext: "pdb", defaultRepresentation: false })
              .then(function (clone) {
                clone.setPosition([shift.x, shift.y, shift.z]);
                clone.addRepresentation("spacefill", {
                  colorScheme: "element",
                  radiusScale: 0.5
                });
                supercellComps.push(clone);
              });
          });
        } else {
          supercellComps.forEach(function (c) { c.setVisibility(true); });
        }

        isSupercellOn = true;
        toggleSupercellBtn.textContent = "Hide Supercell";
        toggleSupercellBtn.style.color = "blue";
        toggleSupercellBtn.style.fontWeight = "bold";

        setTimeout(function () { stage.autoView(500); }, 300);

      } else {
        supercellComps.forEach(function (c) { c.setVisibility(false); });
        isSupercellOn = false;
        toggleSupercellBtn.textContent = "Supercell";
        toggleSupercellBtn.style.color = "black";
        toggleSupercellBtn.style.fontWeight = "normal";
        mainComp.autoView(500);
      }
    });
  }

  stage.loadFile(blob, {
    defaultRepresentation: false,
    ext: "pdb",
  }).then(function (comp) {
    mainComp = comp;
    comp.setName("catalyst-sample");

    comp.addRepresentation("spacefill", {
      colorScheme: "element",
      radiusScale: 0.5,
    });

    unitcellRep = comp.addRepresentation("unitcell", {
      colorValue: "gray",
      radiusScale: 1.0,
      visible: false
    });

    comp.autoView();
  });

  window.addEventListener("resize", function () {
    stage.handleResize();
  });
});