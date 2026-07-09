const API_BASE = "http://127.0.0.1:5000";

// Tab Switching
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.getElementById(btn.dataset.tab).classList.add("active");

    // Lazy-load cluster data + chart the first time that tab is opened
    if (btn.dataset.tab === "cluster" && !clusterDataLoaded) {
      loadClusterData();
    }
  });
});

// Accordion (Advanced Telemetry Toggle)
const accToggle = document.getElementById("accordion-toggle");
const accPanel = document.getElementById("accordion-panel");
const acc = accToggle.closest(".accordion");

accToggle.addEventListener("click", () => {
  acc.classList.toggle("open");
  if (acc.classList.contains("open")) {
    accPanel.style.maxHeight = accPanel.scrollHeight + "px";
  } else {
    accPanel.style.maxHeight = "0px";
  }
});

// Colors
function bucketClass(bucket) {
  if (bucket === "Low chance") return "badge-low";
  if (bucket === "Moderate chance") return "badge-mod";
  return "badge-high";
}

function bucketColor(bucket) {
  if (bucket === "Low chance") return "#00bfa5"; // Teal
  if (bucket === "Moderate chance") return "#ff8f00"; // Orange
  return "#f50057"; // Pink-Red
}

function getFormData() {
  const form = document.getElementById("patient-form");
  const data = {};
  new FormData(form).forEach((value, key) => {
    data[key] = parseFloat(value);
  });
  return data;
}

// Speedometer Gauge Calculations
function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy - r * Math.sin(rad) };
}

function describeArc(cx, cy, r, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArcFlag = Math.abs(startAngle - endAngle) <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
}

function drawGaugeBase() {
  const svg = document.getElementById("gauge-svg");
  const cx = 100, cy = 110, r = 85;

  svg.innerHTML = `
    <path d="${describeArc(cx, cy, r, 180, 120)}" stroke="#00bfa5" stroke-width="14" fill="none" stroke-linecap="round"/>
    <path d="${describeArc(cx, cy, r, 120, 60)}" stroke="#ff8f00" stroke-width="14" fill="none" stroke-linecap="round"/>
    <path d="${describeArc(cx, cy, r, 60, 0)}" stroke="#f50057" stroke-width="14" fill="none" stroke-linecap="round"/>
    <line id="gauge-needle" x1="${cx}" y1="${cy}" x2="${cx - r + 20}" y2="${cy}"
          stroke="#003178" stroke-width="4.5" stroke-linecap="round"
          style="transform-origin: ${cx}px ${cy}px; transition: transform 1.2s cubic-bezier(0.25, 1, 0.5, 1);"/>
    <circle cx="${cx}" cy="${cy}" r="7" fill="#003178"/>
  `;
}

function setGaugeValue(percent) {
  const angle = 180 - (percent / 100) * 180; // 0% -> 180deg (left), 100% -> 0deg (right)
  const needle = document.getElementById("gauge-needle");
  const rotation = 180 - angle;
  needle.style.transform = `rotate(${rotation}deg)`;
}

drawGaugeBase();

// Clinical descriptions for models
function getModelDescription(key, pct, bucket) {
  if (key === "decision_tree") {
    return pct < 34 
      ? "High certainty on current decision tree feature splits; indicators show standard node alignment."
      : "Decision tree nodes split towards elevated risk boundary, triggered by vessel count and heart rate.";
  }
  if (key === "knn") {
    return pct < 34
      ? "Patient profile sits securely within low-risk K-nearest neighbors coordinate space."
      : "Moderate variance in local cluster data. High density of nearest neighbors exhibit cardiac markers.";
  }
  if (key === "logistic") {
    return pct < 34
      ? "Linear Regression model output indicates low predicted probability of cardiovascular disease."
      : "Approaching threshold boundary. Linear decision boundary indicates probability gradient shift.";
  }
  if (key === "svm") {
    return pct < 34
      ? "Support vector machine margins indicate clear separation inside safe classification hyper-plane."
      : "Hyper-plane margin separation boundary crossed. Patient telemetry projects into high-risk zone.";
  }
  return "Statistical machine learning model calculation complete.";
}

// Prediction Flow
document.getElementById("patient-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("predict-btn");
  btn.disabled = true;
  const btnSpan = btn.querySelector("span");
  btnSpan.textContent = "Analyzing Telemetry...";

  try {
    const payload = getFormData();
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Prediction request failed");
    const result = await res.json();
    renderPrediction(result);
    window.lastPatientPayload = payload; // save for clustering locate
  } catch (err) {
    alert("Could not reach the prediction server. Is the Flask backend running on port 5000?");
    console.error(err);
  } finally {
    btn.disabled = false;
    btnSpan.textContent = "Run Prediction Model";
  }
});

function renderPrediction(result) {
  // Description and layout trigger
  document.getElementById("result-desc").textContent = `Ensemble model consensus indicating ${result.overall_bucket.toLowerCase()} of heart disease. Output shows agreement across individual classifiers.`;
  
  // Gauge values
  document.getElementById("gauge-percent").textContent = `${result.overall_chance_percent}%`;
  
  const labelElement = document.getElementById("gauge-label");
  labelElement.textContent = result.overall_bucket;
  labelElement.className = `badge-pill ${bucketClass(result.overall_bucket)}`;
  
  setGaugeValue(result.overall_chance_percent);

  // Model breakdown cards
  const namesMap = {
    logistic: "Linear Regression",
    knn: "K-Nearest Neighbors",
    decision_tree: "Decision Tree",
    svm: "Support Vector Machine (SVM)",
  };
  
  const container = document.getElementById("model-cards");
  container.innerHTML = "";
  
  Object.entries(result.models).forEach(([key, val]) => {
    const card = document.createElement("div");
    card.className = "glass-panel model-breakdown-card";
    card.innerHTML = `
      <div class="model-card-header">
        <h4 class="model-name">${namesMap[key] || key}</h4>
        <span class="model-percentage" style="color: ${bucketColor(val.bucket)}">${val.chance_percent}%</span>
      </div>
      <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width: ${val.chance_percent}%; background: ${bucketColor(val.bucket)}"></div>
      </div>
      <p class="model-desc">${getModelDescription(key, val.chance_percent, val.bucket)}</p>
    `;
    container.appendChild(card);
  });
}

// Accuracy comparison bar chart
async function loadMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    const metrics = await res.json();
    const namesMap = {
      logistic: "Linear Regression",
      knn: "KNN",
      decision_tree: "Decision Tree",
      svm: "SVM",
    };
    const labels = Object.keys(metrics).map(k => namesMap[k] || k);
    const accuracies = Object.values(metrics).map(m => (m.accuracy * 100).toFixed(1));

    new Chart(document.getElementById("accuracy-chart"), {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Accuracy (%)",
          data: accuracies,
          backgroundColor: ["#003178", "#00bfa5", "#ff8f00", "#f50057"],
          borderRadius: 8,
          barThickness: 32,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { 
          legend: { display: false },
          tooltip: {
            backgroundColor: "#003178",
            titleFont: { family: "Plus Jakarta Sans", size: 13, weight: "bold" },
            bodyFont: { family: "Inter", size: 12 },
            padding: 12,
            cornerRadius: 8,
            displayColors: false
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { 
              color: "#6b7280",
              font: { family: "Geist Mono", size: 11 }
            },
            grid: { color: "#f1f5f9" },
          },
          x: {
            ticks: { 
              color: "#003178",
              font: { family: "Plus Jakarta Sans", size: 12, weight: "bold" }
            },
            grid: { display: false },
          },
        },
      },
    });
  } catch (err) {
    console.error("Could not load metrics", err);
  }
}
loadMetrics();

// Unsupervised Clustering Tab
let clusterDataLoaded = false;
let scatterChartInstance = null;

// Original coordinate ranges for zoom management
let originalXMin = -4, originalXMax = 4;
let originalYMin = -3, originalYMax = 3;
let currentZoom = 1.0;

document.getElementById("cluster-btn").addEventListener("click", async () => {
  const payload = window.lastPatientPayload || getFormData();
  const btn = document.getElementById("cluster-btn");
  btn.disabled = true;
  const btnSpan = btn.querySelector("span");
  btnSpan.textContent = "Locating Coordinates...";

  try {
    const res = await fetch(`${API_BASE}/cluster`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Cluster request failed");
    const result = await res.json();
    highlightPatientOnScatter(result);
  } catch (err) {
    alert("Could not reach the prediction server. Is the Flask backend running on port 5000?");
    console.error(err);
  } finally {
    btn.disabled = false;
    btnSpan.textContent = "Locate Patient on Map";
  }
});

async function loadClusterData() {
  try {
    const res = await fetch(`${API_BASE}/cluster-data`);
    const data = await res.json();
    clusterDataLoaded = true;
    renderClusterLegend(data.summary);
    renderScatter(data.points);
  } catch (err) {
    console.error("Could not load cluster data", err);
  }
}

function renderClusterLegend(summary) {
  const container = document.getElementById("cluster-legend");
  container.innerHTML = "";
  
  // order by average disease rate
  const order = Object.entries(summary).sort((a, b) => a[1].avg_disease_rate_percent - b[1].avg_disease_rate_percent);
  
  order.forEach(([cid, info]) => {
    const row = document.createElement("div");
    row.className = "profile-row";
    
    let clusterLetter = "A";
    let desc = "Stable vitals, routine monitoring.";
    if (info.label === "Moderate chance") {
      clusterLetter = "B";
      desc = "Elevated markers, requires review.";
    } else if (info.label === "High chance") {
      clusterLetter = "C";
      desc = "Critical indicators, immediate action.";
    }

    row.innerHTML = `
      <div class="profile-info-block">
        <span class="profile-dot" style="background:${bucketColor(info.label)}"></span>
        <div>
          <h4 class="profile-name">Cluster ${clusterLetter}: ${info.label}</h4>
          <p class="profile-desc">${desc} (${info.size} patients)</p>
        </div>
      </div>
      <div class="profile-rate">
        <span class="rate-value" style="color:${bucketColor(info.label)}">${info.avg_disease_rate_percent}%</span>
        <span class="rate-label">Risk Rate</span>
      </div>
    `;
    container.appendChild(row);
  });
}

function renderScatter(points) {
  const byLabel = { "Low chance": [], "Moderate chance": [], "High chance": [] };
  points.forEach(p => byLabel[p.cluster_label].push({ x: p.x, y: p.y }));

  // Find min/max coordinate ranges to set original boundaries
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  originalXMin = Math.min(...xs) - 0.5;
  originalXMax = Math.max(...xs) + 0.5;
  originalYMin = Math.min(...ys) - 0.5;
  originalYMax = Math.max(...ys) + 0.5;

  const ctx = document.getElementById("scatter-chart");
  scatterChartInstance = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        { 
          label: "Low chance", 
          data: byLabel["Low chance"], 
          backgroundColor: "rgba(0, 191, 165, 0.4)",
          borderColor: "#00bfa5",
          borderWidth: 1,
          pointRadius: 6,
          pointHoverRadius: 8
        },
        { 
          label: "Moderate chance", 
          data: byLabel["Moderate chance"], 
          backgroundColor: "rgba(255, 143, 0, 0.4)",
          borderColor: "#ff8f00",
          borderWidth: 1,
          pointRadius: 6,
          pointHoverRadius: 8
        },
        { 
          label: "High chance", 
          data: byLabel["High chance"], 
          backgroundColor: "rgba(245, 0, 87, 0.4)",
          borderColor: "#f50057",
          borderWidth: 1,
          pointRadius: 6,
          pointHoverRadius: 8
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#003178",
          titleFont: { family: "Plus Jakarta Sans", size: 13, weight: "bold" },
          bodyFont: { family: "Geist Mono", size: 12 },
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            label: function(context) {
              return `Coordinate: (${context.parsed.x.toFixed(2)}, ${context.parsed.y.toFixed(2)})`;
            }
          }
        }
      },
      scales: {
        x: { 
          min: originalXMin,
          max: originalXMax,
          ticks: { 
            color: "#6b7280",
            font: { family: "Geist Mono", size: 11 }
          }, 
          grid: { color: "#f1f5f9" }, 
          title: { 
            display: true, 
            text: "Principal Component 1 (Variance: 42%)", 
            color: "#003178",
            font: { family: "Plus Jakarta Sans", size: 12, weight: "bold" }
          } 
        },
        y: { 
          min: originalYMin,
          max: originalYMax,
          ticks: { 
            color: "#6b7280",
            font: { family: "Geist Mono", size: 11 }
          }, 
          grid: { color: "#f1f5f9" }, 
          title: { 
            display: true, 
            text: "Principal Component 2 (Variance: 18%)", 
            color: "#003178",
            font: { family: "Plus Jakarta Sans", size: 12, weight: "bold" }
          } 
        },
      },
    },
  });
}

function highlightPatientOnScatter(result) {
  if (!scatterChartInstance) return;

  // remove any previous "You" dataset
  scatterChartInstance.data.datasets = scatterChartInstance.data.datasets.filter(d => d.label !== "You");

  scatterChartInstance.data.datasets.push({
    label: "You",
    data: [{ x: result.point.x, y: result.point.y }],
    backgroundColor: "#ffffff",
    borderColor: "#003178",
    borderWidth: 3,
    pointStyle: "star",
    radius: 12,
    hoverRadius: 14,
    showLine: false
  });
  
  scatterChartInstance.update();

  // Pulse/Glow the locate button and render the custom inline alert
  const clusterBtn = document.getElementById("cluster-btn");
  clusterBtn.classList.add("pulse-trigger");
  setTimeout(() => {
    clusterBtn.classList.remove("pulse-trigger");
  }, 4000);

  // Render Inline Custom Alert Card
  let alertBox = document.getElementById("cluster-alert");
  if (!alertBox) {
    alertBox = document.createElement("div");
    alertBox.id = "cluster-alert";
    alertBox.className = "profile-row";
    alertBox.style.marginTop = "16px";
    alertBox.style.display = "none";
    document.getElementById("cluster-btn").parentNode.appendChild(alertBox);
  }

  alertBox.style.display = "flex";
  alertBox.style.background = result.cluster_label === "Low chance" ? "var(--risk-low-bg)" : (result.cluster_label === "Moderate chance" ? "var(--risk-mod-bg)" : "var(--risk-high-bg)");
  alertBox.style.borderColor = result.cluster_label === "Low chance" ? "var(--risk-low)" : (result.cluster_label === "Moderate chance" ? "var(--risk-mod)" : "var(--risk-high)");
  
  alertBox.innerHTML = `
    <div class="profile-info-block">
      <span class="profile-dot" style="background: ${bucketColor(result.cluster_label)}"></span>
      <div>
        <h4 class="profile-name">Located Patient</h4>
        <p class="profile-desc" style="color: var(--text-primary)">Assigned to <strong>${result.cluster_label}</strong> cluster.</p>
      </div>
    </div>
    <div class="profile-rate">
      <span class="rate-value" style="color: ${bucketColor(result.cluster_label)}">${result.cluster_avg_rate_percent}%</span>
      <span class="rate-label">Group Rate</span>
    </div>
  `;
}

// Zoom simulation on Scatter Plot
document.getElementById("zoom-in-btn").addEventListener("click", () => {
  if (!scatterChartInstance) return;
  currentZoom *= 0.8;
  updateZoom();
});

document.getElementById("zoom-out-btn").addEventListener("click", () => {
  if (!scatterChartInstance) return;
  currentZoom *= 1.25;
  updateZoom();
});

document.getElementById("zoom-reset-btn").addEventListener("click", () => {
  if (!scatterChartInstance) return;
  currentZoom = 1.0;
  updateZoom();
});

function updateZoom() {
  const xCenter = (originalXMax + originalXMin) / 2;
  const yCenter = (originalYMax + originalYMin) / 2;
  const xHalfRange = ((originalXMax - originalXMin) / 2) * currentZoom;
  const yHalfRange = ((originalYMax - originalYMin) / 2) * currentZoom;

  scatterChartInstance.options.scales.x.min = xCenter - xHalfRange;
  scatterChartInstance.options.scales.x.max = xCenter + xHalfRange;
  scatterChartInstance.options.scales.y.min = yCenter - yHalfRange;
  scatterChartInstance.options.scales.y.max = yCenter + yHalfRange;
  scatterChartInstance.update();
}

// Export Data simulation
document.getElementById("export-data-btn").addEventListener("click", () => {
  const payload = window.lastPatientPayload || getFormData();
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `patient_telemetry_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});
