/* FAKE ACCOUNT DETECTOR - front-end behaviour
   Handles: animated counters, scroll reveal, form validation, example buttons,
   loading state, progress bars and Chart.js charts. */

(function () {
  "use strict";

  document.documentElement.classList.add("js");

  var reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ------------------------------------------------------------------ */
  /* Navbar shadow when scrolling                                        */
  /* ------------------------------------------------------------------ */
  var nav = document.getElementById("mainNav");
  function updateNav() {
    if (nav) { nav.classList.toggle("scrolled", window.scrollY > 10); }
  }
  window.addEventListener("scroll", updateNav, { passive: true });
  updateNav();

  /* ------------------------------------------------------------------ */
  /* Animated counters:  <span data-count="94.5" data-decimals="1" data-suffix="%">  */
  /* ------------------------------------------------------------------ */
  function formatNumber(value, decimals) {
    return value.toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }

  function animateCounter(element) {
    var target = parseFloat(element.getAttribute("data-count"));
    if (isNaN(target)) { return; }
    var decimals = parseInt(element.getAttribute("data-decimals") || "0", 10);
    var suffix = element.getAttribute("data-suffix") || "";

    if (reducedMotion) {
      element.textContent = formatNumber(target, decimals) + suffix;
      return;
    }
    var duration = 1400;
    var start = null;
    function step(timestamp) {
      if (start === null) { start = timestamp; }
      var progress = Math.min((timestamp - start) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      element.textContent = formatNumber(target * eased, decimals) + suffix;
      if (progress < 1) { window.requestAnimationFrame(step); }
    }
    window.requestAnimationFrame(step);
  }

  var counters = document.querySelectorAll("[data-count]");
  if ("IntersectionObserver" in window) {
    var counterObserver = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.3 });
    counters.forEach(function (counter) { counterObserver.observe(counter); });
  } else {
    counters.forEach(animateCounter);
  }

  /* ------------------------------------------------------------------ */
  /* Scroll reveal                                                       */
  /* ------------------------------------------------------------------ */
  var revealItems = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var revealObserver = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    revealItems.forEach(function (item) { revealObserver.observe(item); });
  } else {
    revealItems.forEach(function (item) { item.classList.add("visible"); });
  }

  /* ------------------------------------------------------------------ */
  /* Probability / importance bars                                       */
  /* ------------------------------------------------------------------ */
  window.setTimeout(function () {
    document.querySelectorAll(".progress-bar[data-width]").forEach(function (bar) {
      var width = Math.max(0, Math.min(100, parseFloat(bar.getAttribute("data-width")) || 0));
      bar.style.width = width + "%";
    });
  }, 150);

  /* ------------------------------------------------------------------ */
  /* Analysis form                                                       */
  /* ------------------------------------------------------------------ */
  var SAMPLES = {
    genuine: {
      username: "sarah.jones", followers: "850", following: "420", posts: "310",
      account_age_days: "1500", has_profile_pic: "yes", has_bio: "yes",
      avg_likes_per_post: "95", avg_comments_per_post: "8"
    },
    fake: {
      username: "user83749201", followers: "12", following: "4800", posts: "0",
      account_age_days: "9", has_profile_pic: "no", has_bio: "no",
      avg_likes_per_post: "0", avg_comments_per_post: "0"
    }
  };

  var form = document.getElementById("analyzeForm");
  if (form) {
    function clearValidation() {
      form.classList.remove("was-validated");
      form.querySelectorAll(".is-invalid").forEach(function (field) { field.classList.remove("is-invalid"); });
    }

    // Example buttons
    document.querySelectorAll("[data-sample]").forEach(function (button) {
      button.addEventListener("click", function () {
        var sample = SAMPLES[button.getAttribute("data-sample")];
        if (!sample) { return; }
        clearValidation();
        Object.keys(sample).forEach(function (name) {
          var field = form.elements[name];
          if (field) { field.value = sample[name]; }
        });
      });
    });

    // Reset button
    var resetButton = document.getElementById("resetButton");
    if (resetButton) {
      resetButton.addEventListener("click", function () {
        Array.prototype.forEach.call(form.elements, function (field) {
          if (field.tagName === "SELECT") { field.selectedIndex = 0; }
          else if (field.type !== "submit" && field.type !== "button") { field.value = ""; }
        });
        clearValidation();
      });
    }

    // Validation + loading state
    var submitButton = document.getElementById("analyzeButton");
    form.addEventListener("submit", function (event) {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
        form.classList.add("was-validated");
        var firstInvalid = form.querySelector(":invalid");
        if (firstInvalid) { firstInvalid.focus(); }
        return;
      }
      if (submitButton) {
        submitButton.disabled = true;
        var label = submitButton.querySelector(".button-label");
        var loading = submitButton.querySelector(".button-loading");
        if (label) { label.classList.add("d-none"); }
        if (loading) { loading.classList.remove("d-none"); }
      }
    });

    // Re-enable the button if the user comes back with the browser "Back" button.
    window.addEventListener("pageshow", function (event) {
      if (event.persisted && submitButton) {
        submitButton.disabled = false;
        var label = submitButton.querySelector(".button-label");
        var loading = submitButton.querySelector(".button-loading");
        if (label) { label.classList.remove("d-none"); }
        if (loading) { loading.classList.add("d-none"); }
      }
    });
  }

  // If the server found errors, jump back to the form.
  var errorForm = document.querySelector('.form-card[data-scroll="true"]');
  if (errorForm) {
    window.setTimeout(function () { errorForm.scrollIntoView({ behavior: "auto", block: "start" }); }, 50);
  }

  /* ------------------------------------------------------------------ */
  /* Charts (Chart.js is loaded from a CDN on the pages that need it)    */
  /* ------------------------------------------------------------------ */
  function chartUnavailable(canvas) {
    if (!canvas) { return; }
    var note = document.createElement("p");
    note.className = "text-muted small mb-0";
    note.textContent = "The chart could not load. Please check your internet connection and refresh the page.";
    canvas.parentNode.replaceChild(note, canvas);
  }

  var chartCanvases = ["probabilityChart", "metricsChart", "sessionChart"]
    .map(function (id) { return document.getElementById(id); })
    .filter(Boolean);

  if (chartCanvases.length && typeof window.Chart === "undefined") {
    chartCanvases.forEach(chartUnavailable);
    return;
  }

  if (typeof window.Chart !== "undefined") {
    Chart.defaults.color = "#9aa4c4";
    Chart.defaults.font.family = "Inter, system-ui, sans-serif";
    Chart.defaults.borderColor = "rgba(148, 163, 184, 0.15)";
    if (reducedMotion) { Chart.defaults.animation = false; }
  }

  // Result page: fake vs genuine doughnut
  var probabilityCanvas = document.getElementById("probabilityChart");
  if (probabilityCanvas) {
    var fake = parseFloat(probabilityCanvas.getAttribute("data-fake")) || 0;
    var genuine = parseFloat(probabilityCanvas.getAttribute("data-genuine")) || 0;
    new Chart(probabilityCanvas, {
      type: "doughnut",
      data: {
        labels: ["Fake", "Genuine"],
        datasets: [{ data: [fake, genuine], backgroundColor: ["#ef4444", "#22c55e"], borderColor: "#0f1733", borderWidth: 3 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false, cutout: "68%",
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: { label: function (ctx) { return " " + ctx.label + ": " + ctx.parsed + "%"; } } }
        }
      }
    });
  }

  // Dashboard charts
  var dashboardData = window.DASHBOARD_DATA;
  if (dashboardData) {
    var metricsCanvas = document.getElementById("metricsChart");
    if (metricsCanvas && dashboardData.metrics) {
      new Chart(metricsCanvas, {
        type: "bar",
        data: {
          labels: dashboardData.metrics.labels,
          datasets: [{
            label: "Score (%)",
            data: dashboardData.metrics.values,
            backgroundColor: ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7"],
            borderRadius: 10, maxBarThickness: 70
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: { y: { min: 0, max: 100, ticks: { callback: function (v) { return v + "%"; } } } },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: function (ctx) { return " " + ctx.parsed.y + "%"; } } }
          }
        }
      });
    }

    var sessionCanvas = document.getElementById("sessionChart");
    if (sessionCanvas) {
      new Chart(sessionCanvas, {
        type: "doughnut",
        data: {
          labels: dashboardData.session.labels,
          datasets: [{ data: dashboardData.session.values, backgroundColor: ["#22c55e", "#ef4444"], borderColor: "#0f1733", borderWidth: 3 }]
        },
        options: { responsive: true, maintainAspectRatio: false, cutout: "62%", plugins: { legend: { position: "bottom" } } }
      });
    }
  }
})();
