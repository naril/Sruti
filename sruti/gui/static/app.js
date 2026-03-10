document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-poll-url]").forEach((node) => {
    if (node.getAttribute("data-poll-scope") === "disabled") {
      return;
    }
    const url = node.getAttribute("data-poll-url");
    const interval = Number(node.getAttribute("data-poll-interval") || "5000");
    if (!url) {
      return;
    }
    const tick = async () => {
      try {
        const response = await fetch(url, { headers: { "X-Requested-With": "fetch" } });
        if (!response.ok) {
          return;
        }
        node.innerHTML = await response.text();
      } catch (error) {
        console.error("poll failed", error);
      }
    };
    window.setInterval(tick, interval);
  });
});
