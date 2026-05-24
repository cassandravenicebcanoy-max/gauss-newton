/* ═══════════════════════════════════════════
   Gauss–Newton Calculator  ·  main.js
═══════════════════════════════════════════ */

// ── Preset configurations ──────────────────
const PRESETS = {
  exp: {
    model:       "a0 * exp(a1 * x)",
    x_data:      "0, 1, 2, 3",
    y_data:      "2.0, 0.74, 0.27, 0.10",
    init_params: "1.0, -1.0",
  },
  mm: {
    model:       "(a0 * x) / (a1 + x)",
    x_data:      "0.5, 1.0, 2.0, 4.0, 8.0",
    y_data:      "1.10, 1.60, 2.30, 2.90, 3.40",
    init_params: "4.0, 1.0",
  },
  power: {
    model:       "a0 * x**a1",
    x_data:      "1, 2, 3, 4, 5, 6",
    y_data:      "2.1, 5.8, 11.5, 20.0, 31.0, 44.0",
    init_params: "2.0, 1.5",
  },
  logistic: {
    model:       "a0 / (1 + exp(-a1 * (x - a2)))",
    x_data:      "0, 1, 2, 3, 4, 5, 6, 7, 8",
    y_data:      "0.05, 0.10, 0.26, 0.50, 0.74, 0.88, 0.95, 0.98, 0.99",
    init_params: "1.0, 1.0, 4.0",
  },
};

// ── DOM references ─────────────────────────
const modelInput  = document.getElementById("model");
const xInput      = document.getElementById("x_data");
const yInput      = document.getElementById("y_data");
const initInput   = document.getElementById("init_params");
const maxIterInp  = document.getElementById("max_iter");
const tolInput    = document.getElementById("tol");
const calcBtn     = document.getElementById("calc-btn");
const btnText     = calcBtn.querySelector(".btn-text");
const btnSpinner  = calcBtn.querySelector(".btn-spinner");
const errorBox    = document.getElementById("error-box");
const resultsPanel= document.getElementById("results-panel");

let fitChart = null;

// ── Preset buttons ─────────────────────────
document.querySelectorAll(".preset-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const p = PRESETS[btn.dataset.preset];
    if (!p) return;
    modelInput.value  = p.model;
    xInput.value      = p.x_data;
    yInput.value      = p.y_data;
    initInput.value   = p.init_params;
    resultsPanel.style.display = "none";
    errorBox.classList.add("hidden");
  });
});

// ── Tab switching ──────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
  });
});

// ── Main calculation ───────────────────────
calcBtn.addEventListener("click", async () => {
  errorBox.classList.add("hidden");
  setLoading(true);

  const payload = {
    model:       modelInput.value.trim(),
    x_data:      xInput.value.trim(),
    y_data:      yInput.value.trim(),
    init_params: initInput.value.trim(),
    max_iter:    parseInt(maxIterInp.value) || 100,
    tol:         parseFloat(tolInput.value) || 1e-8,
  };

  try {
    const res  = await fetch("/calculate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || "Unexpected server error.");
      return;
    }

    renderResults(data, payload.model);
  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    setLoading(false);
  }
});

// ── Render results ─────────────────────────
function renderResults(data, modelExpr) {
  resultsPanel.style.display = "block";
  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });

  // Summary
  const status = data.converged
    ? `<span class="ok">✓ Converged</span>`
    : `<span class="warn">⚠ Did not converge (max iterations reached)</span>`;

  const paramStrs = data.params.map((v, i) =>
    `a${i} = <span class="val">${v.toFixed(8)}</span>`
  ).join("  &nbsp;|&nbsp;  ");

  document.getElementById("result-summary").innerHTML = `
    Status: ${status}<br>
    Iterations: <span class="val">${data.num_iterations}</span><br>
    Final RSS: <span class="val">${data.rss.toExponential(4)}</span><br>
    ${paramStrs}
  `;

  // Chart
  renderChart(data);

  // Parameters table
  buildParamsTable(data.params);

  // Iterations table
  buildIterationsTable(data.iterations);

  // Residuals table
  buildResidualsTable(data.x_data, data.y_data, data.residuals);

  // Show first tab
  document.querySelectorAll(".tab-btn").forEach((b,i) => b.classList.toggle("active", i===0));
  document.querySelectorAll(".tab-content").forEach((c,i) => c.classList.toggle("hidden", i!==0));
}

function renderChart(data) {
  const ctx = document.getElementById("fitChart").getContext("2d");
  if (fitChart) fitChart.destroy();

  fitChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Observed data",
          data: data.x_data.map((x,i) => ({ x, y: data.y_data[i] })),
          backgroundColor: "#c8500a",
          pointRadius: 7,
          pointHoverRadius: 9,
        },
        {
          label: "Fitted curve",
          data: data.curve_x.map((x,i) => ({ x, y: data.curve_y[i] })),
          type: "line",
          borderColor: "#1a6b8a",
          borderWidth: 2.5,
          pointRadius: 0,
          fill: false,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { font: { family: "'Outfit', sans-serif", size: 12 }, color: "#333" }
        },
        tooltip: {
          callbacks: {
            label: ctx => `(${ctx.parsed.x.toFixed(4)},  ${ctx.parsed.y.toFixed(6)})`,
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "x", font: { size: 13 }, color: "#555" },
          grid: { color: "rgba(0,0,0,0.05)" },
        },
        y: {
          title: { display: true, text: "y", font: { size: 13 }, color: "#555" },
          grid: { color: "rgba(0,0,0,0.05)" },
        },
      },
    },
  });
}

function buildParamsTable(params) {
  const tbl = document.getElementById("params-table");
  let html = `<thead><tr><th>Parameter</th><th>Symbol</th><th>Value</th></tr></thead><tbody>`;
  params.forEach((v, i) => {
    html += `<tr>
      <td class="highlight">a${i}</td>
      <td>p<sub>${i+1}</sub></td>
      <td>${v.toFixed(10)}</td>
    </tr>`;
  });
  html += `</tbody>`;
  tbl.innerHTML = html;
}

function buildIterationsTable(iterations) {
  if (!iterations || iterations.length === 0) {
    document.getElementById("iters-table").innerHTML = "<tr><td>No iteration data.</td></tr>";
    return;
  }
  const nParams = iterations[0].params.length;
  let header = `<thead><tr><th>Iter</th>`;
  for (let j = 0; j < nParams; j++) header += `<th>a${j}</th>`;
  header += `<th>RSS</th><th>‖δ‖</th></tr></thead>`;

  let rows = "<tbody>";
  iterations.forEach(it => {
    rows += `<tr><td class="highlight">${it.iteration}</td>`;
    it.params.forEach(p => { rows += `<td>${p.toFixed(6)}</td>`; });
    rows += `<td>${it.rss.toExponential(4)}</td>`;
    rows += `<td>${it.delta_norm.toExponential(4)}</td>`;
    rows += `</tr>`;
  });
  rows += "</tbody>";
  document.getElementById("iters-table").innerHTML = header + rows;
}

function buildResidualsTable(xData, yData, residuals) {
  let html = `<thead><tr><th>i</th><th>xᵢ</th><th>yᵢ (observed)</th><th>rᵢ (residual)</th><th>rᵢ²</th></tr></thead><tbody>`;
  xData.forEach((x, i) => {
    const r = residuals[i];
    const cls = r >= 0 ? "positive" : "negative";
    html += `<tr>
      <td class="highlight">${i+1}</td>
      <td>${x}</td>
      <td>${yData[i]}</td>
      <td class="${cls}">${r.toFixed(8)}</td>
      <td>${(r*r).toExponential(4)}</td>
    </tr>`;
  });
  html += `</tbody>`;
  document.getElementById("residuals-table").innerHTML = html;
}

// ── Helpers ────────────────────────────────
function setLoading(state) {
  calcBtn.disabled = state;
  btnText.classList.toggle("hidden", state);
  btnSpinner.classList.toggle("hidden", !state);
}

function showError(msg) {
  errorBox.textContent = "⚠ " + msg;
  errorBox.classList.remove("hidden");
}
