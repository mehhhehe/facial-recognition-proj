const video = document.getElementById("camera");
const percent = document.getElementById("percent");
const banner = document.getElementById("result-banner");
const bannerText = document.getElementById("result-text");
let authorized = false;

// 🔥 Start the camera
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    alert("Camera access failed: " + err);
    console.error("Camera error:", err);
  });

/* ---- CONFIG ---- */
const INTERVAL_MS = 1200;

/* ---- Capture helper ---- */
const canvas   = document.createElement("canvas");
const ctx      = canvas.getContext("2d");

function grabFrame() {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg");   // tiny base64 string
}

async function tick() {
  if (authorized) return;

  const frame = grabFrame();

  const res = await fetch("/api/compare", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({frame})
  });
  const j = await res.json();
  if (j.error) return;   // no face, etc.

  percent.textContent   = j.similarity + "%";
  document.getElementById("match-name").textContent = j.name || "_Unknown_";

  if (j.similarity > 90) {
      authorized = true;
      banner.className  = "authorized";
      bannerText.textContent = "AUTHORIZED";
      setTimeout(() => window.location.href = "/dashboard", 20000);
  } else {
      banner.className  = "unauthorized";
      bannerText.textContent = "NOT AUTHORIZED";
  }
}

setInterval(tick, INTERVAL_MS);
