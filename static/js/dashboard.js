/* live clock */
function tick() {
    const now = new Date();
    document.getElementById("clock").textContent =
      now.toLocaleTimeString("en-GB", {hour:'2-digit', minute:'2-digit', second:'2-digit'}) +
      " " +
      now.toLocaleDateString("en-GB");
  }
  setInterval(tick, 1000);
  tick();
  