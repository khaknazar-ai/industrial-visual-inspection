(() => {
  "use strict";

  const fileInput = document.getElementById("file-input");
  const chooseButton = document.getElementById("choose-button");
  const dropZone = document.getElementById("drop-zone");
  const emptyPanel = document.getElementById("results-empty");
  const loadingPanel = document.getElementById("results-loading");
  const resultPanel = document.getElementById("results-content");
  const errorBox = document.getElementById("error-box");
  const systemState = document.getElementById("system-state");
  const systemStateText = document.getElementById("system-state-text");
  const resultImage = document.getElementById("result-image");
  const imagePlaceholder = document.getElementById("image-placeholder");
  const findings = document.getElementById("findings");
  const profileLabel = document.getElementById("profile-label");
  const modelProfile = document.getElementById("model-profile");

  let maxBytes = 0;
  let allowedTypes = new Set();
  let profileReady = false;
  let scoreSpace = "raw_score";
  let loadedModelId = "";
  let currentResult = null;
  let currentView = "overlay";
  let requestSerial = 0;
  let activeRequest = null;
  let imageSerial = 0;
  let configuredCategory = "";
  let reportRestoreAttempted = false;

  function reportUrl(requestId) {
    if (!window.location?.href || !window.history?.replaceState) return;
    const url = new URL(window.location.href);
    if (requestId) url.searchParams.set("report", requestId);
    else url.searchParams.delete("report");
    window.history.replaceState(null, "", url.pathname + url.search + url.hash);
  }

  async function restoreSavedReport() {
    if (reportRestoreAttempted) return;
    reportRestoreAttempted = true;
    if (!window.location?.href) return;
    const requestId = new URL(window.location.href).searchParams.get("report");
    if (!requestId) return;
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(requestId)) {
      showError("The saved report link is invalid.");
      return;
    }
    const serial = ++requestSerial;
    const controller = new AbortController();
    activeRequest = controller;
    showPanel("loading");
    try {
      const response = await fetch("/reports/" + encodeURIComponent(requestId), {
        cache: "no-store", signal: controller.signal,
      });
      const raw = await response.text();
      if (serial !== requestSerial) return;
      const result = parseInspectionResponse(raw, response.ok);
      if (result.request_id !== requestId || result.category !== configuredCategory ||
          (loadedModelId && result.model_id !== loadedModelId)) {
        throw new Error("This saved report does not match the current model profile.");
      }
      renderResult(result);
    } catch (error) {
      if (serial !== requestSerial || error?.name === "AbortError") return;
      showError(error instanceof Error ? error.message : "The saved report could not be opened.");
    } finally {
      if (serial === requestSerial) activeRequest = null;
    }
  }

  function enableUpload(enabled) {
    profileReady = enabled;
    chooseButton.disabled = !enabled;
    fileInput.disabled = !enabled;
    dropZone.setAttribute("aria-disabled", String(!enabled));
  }

  function setSystemStatus(kind, message) {
    systemState.dataset.state = kind;
    systemStateText.textContent = message;
  }

  async function checkReady() {
    try {
      const response = await fetch("/ready", { cache: "no-store" });
      const body = await response.json();
      const ready = response.ok && body.ready !== false;
      setSystemStatus(ready ? "ready" : "offline", ready ? "System ready" : "System needs attention");
      if (!ready) { enableUpload(false); return; }
      const profileResponse = await fetch("/models", { cache: "no-store" });
      const profileBody = profileResponse.ok ? await profileResponse.json() : {};
      const profile = profileBody.profile || {};
      const limits = profile.upload_limits || {};
      if (!profileResponse.ok || !profile.category || !Number.isInteger(limits.max_image_bytes) || limits.max_image_bytes < 1 || !Array.isArray(limits.allowed_content_types)) {
        throw new Error("Profile metadata is unavailable");
      }
      maxBytes = limits.max_image_bytes;
      allowedTypes = new Set(limits.allowed_content_types);
      scoreSpace = profile.score_space || "raw_score";
      loadedModelId = profileBody.loaded_model_id || "";
      fileInput.accept = [...allowedTypes].join(",");
      document.getElementById("upload-hint").textContent =
        [...allowedTypes].map((value) => value.replace("image/", "").toUpperCase()).join(", ") +
        " · up to " + (maxBytes / (1024 * 1024)).toLocaleString(undefined, { maximumFractionDigits: 2 }) + " MiB";
      document.getElementById("calibration-status").textContent = "Calibration: " + (profile.calibration_status || "unavailable").replaceAll("_", " ");
      if (typeof profile.category === "string" && profile.category) {
        configuredCategory = profile.category;
        profileLabel.textContent = configuredCategory.toUpperCase() + " INSPECTION";
        document.getElementById("detail-category").textContent = configuredCategory;
      }
      if (typeof profile.model_name === "string" && profile.model_name) {
        modelProfile.textContent =
          profile.model_name.replaceAll("_", " ") + " · " + configuredCategory;
      }
      enableUpload(true);
      await restoreSavedReport();
    } catch {
      enableUpload(false);
      setSystemStatus("offline", "System unavailable");
    }
  }

  function showPanel(name) {
    emptyPanel.classList.toggle("hidden", name !== "empty");
    loadingPanel.classList.toggle("hidden", name !== "loading");
    resultPanel.classList.toggle("hidden", name !== "result");
  }

  function showError(message) {
    showPanel("empty");
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  }

  function responseError(body) {
    if (body && typeof body.detail === "string") return body.detail;
    if (body && body.detail && typeof body.detail.message === "string") {
      return body.detail.message;
    }
    return "The image could not be inspected. Please try another file.";
  }

  function encodePath(rawPath) {
    return String(rawPath)
      .split("/")
      .filter(Boolean)
      .map((part) => encodeURIComponent(part))
      .join("/");
  }

  function evidenceUrl(rawPath) {
    if (!rawPath) return "";

    let value = String(rawPath).replace(/\\/g, "/");
    const marker = "reports/runtime/";
    const markerIndex = value.toLowerCase().indexOf(marker);
    if (markerIndex >= 0) {
      value = value.slice(markerIndex + marker.length);
    }

    value = value.replace(/^\/+/, "").replace(/^\.\//, "");
    return "/evidence/" + encodePath(value);
  }

  function evidencePath(result, view) {
    const evidence = result.evidence || {};
    const names = {
      original: ["original_path", "original"],
      heatmap: ["heatmap_path", "heatmap", "anomaly_map_path"],
      overlay: ["overlay_path", "overlay", "visualization_path"],
    };

    const fields = names[view] || names.overlay;
    const field = fields.find((name) => evidence[name]);
    return field ? evidence[field] : "";
  }

  function renderImage(view) {
    currentView = view;
    const path = currentResult ? evidencePath(currentResult, view) : "";
    const renderSerial = ++imageSerial;

    resultImage.onerror = () => {
      if (renderSerial !== imageSerial) return;
      resultImage.classList.add("hidden");
      imagePlaceholder.textContent = "Preview is not available";
      imagePlaceholder.classList.remove("hidden");
    };
    resultImage.onload = () => {
      if (renderSerial !== imageSerial) return;
      resultImage.classList.remove("hidden");
      imagePlaceholder.classList.add("hidden");
    };

    if (!path) {
      resultImage.removeAttribute("src");
      resultImage.classList.add("hidden");
      imagePlaceholder.textContent = "No preview available";
      imagePlaceholder.classList.remove("hidden");
    } else {
      imagePlaceholder.textContent = "Loading preview…";
      imagePlaceholder.classList.remove("hidden");
      resultImage.classList.remove("hidden");
      resultImage.src = evidenceUrl(path);
    }

    document.querySelectorAll(".view-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.view === view);
      tab.setAttribute("aria-pressed", String(tab.dataset.view === view));
    });
  }

  function verdictFor(decision) {
    const verdicts = {
      fail: {
        title: "Anomaly detected",
        subtitle: "The anomaly score reached the fail threshold.",
        icon: "!",
      },
      review: {
        title: "Needs review",
        subtitle: "This image requires an operator decision.",
        icon: "?",
      },
      pass: {
        title: "Looks normal",
        subtitle: "No clear visual difference was found.",
        icon: "✓",
      },
    };
    return verdicts[decision] || verdicts.pass;
  }

  function addFinding(message, tone) {
    const item = document.createElement("div");
    item.className = "finding " + tone;

    const dot = document.createElement("span");
    dot.className = "finding-dot";

    const text = document.createElement("span");
    text.textContent = message;

    item.append(dot, text);
    findings.append(item);
  }

  function renderFindings(decision, regions, reason = "") {
    findings.replaceChildren();

    if (decision === "fail") {
      if (regions.length) {
        regions.slice(0, 6).forEach((_, index) => {
          addFinding("Unusual visual area " + (index + 1) + " needs a closer look.", "fail");
        });
      } else {
        addFinding("Anomaly detected, but no localised region was returned.", "fail");
      }
      return;
    }

    if (decision === "review") {
      addFinding(
        reason || "Please compare this part with your normal reference before accepting it.",
        "review",
      );
      return;
    }

    addFinding("The score is below the review threshold for this model and category.", "pass");
  }

  function usesNativeBinaryBoundary(result) {
    const objective = result.metadata?.identity?.policy?.image?.objective || result.thresholds?.objective;
    return objective === "native_binary_boundary_with_equality_review";
  }

  function reviewExplanation(result) {
    const messages = {
      resolution_too_small: "The photo is too small for a reliable inspection.",
      brightness_out_of_range: "The photo is too dark or too bright for this inspection.",
      blurred: "The photo is blurred. Use a sharper image.",
      invalid_model_output: "The model did not return a usable result. Review this image manually.",
      score_in_review_band: usesNativeBinaryBoundary(result)
        ? "The score is exactly at the decision threshold."
        : "The score is between the pass and fail thresholds.",
    };
    const metadata = result.metadata || {};
    const reasons = metadata.review_reasons || metadata.quality?.reasons || [];
    return reasons.map((reason) => messages[reason] || "Review required: " + String(reason).replaceAll("_", " ")).join(" ");
  }

  function renderScore(result) {
    const space = result.metadata?.identity?.score_space || result.metadata?.score_space || scoreSpace;
    const probability = space === "probability";
    const value = result.anomaly_score;
    const format = (number) => probability
      ? (number * 100).toLocaleString(undefined, { maximumFractionDigits: 2 }) + "%"
      : number.toLocaleString(undefined, { maximumSignificantDigits: 6 });
    document.getElementById("score-label").textContent = probability ? "Calibrated probability" :
      space === "raw_distance" ? "Anomaly distance" : "Anomaly score";
    document.getElementById("score-note").textContent = probability ? "Calibrated model output" : "Not a probability";
    document.getElementById("score-value").textContent = value === null ? "Unavailable" : format(value);
    const thresholds = result.thresholds || {};
    const pass = thresholds.pass ?? thresholds.pass_threshold;
    const fail = thresholds.fail ?? thresholds.fail_threshold;
    const validThresholds = Number.isFinite(pass) && Number.isFinite(fail) && pass < fail;
    document.getElementById("threshold-value").textContent = !validThresholds ? "Thresholds unavailable"
      : usesNativeBinaryBoundary(result)
        ? "Decision threshold " + format(pass) + " · Review at the boundary"
        : "Review from " + format(pass) + " · Fail from " + format(fail);
    const low = Math.min(0, value ?? 0, validThresholds ? pass : 0);
    const high = Math.max(probability ? 1 : 0, value ?? 0, validThresholds ? fail : 0, low + 1e-9);
    const position = (number) => 100 * (number - low) / (high - low);
    document.getElementById("score-fill").style.width = value === null ? "0%" : position(value) + "%";
    const marker = document.getElementById("score-marker");
    marker.classList.toggle("hidden", !validThresholds);
    if (validThresholds) marker.style.left = position(fail) + "%";
    const track = document.getElementById("score-track");
    track.style.background = validThresholds
      ? `linear-gradient(90deg, #52b887 0% ${position(pass)}%, #e3a745 ${position(pass)}% ${position(fail)}%, #df757a ${position(fail)}% 100%)`
      : "#dfe7ef";
    document.getElementById("score-scale").textContent = "Displayed scale: " + format(low) + " – " + format(high);
  }

  function renderResult(result) {
    currentResult = result;
    if (typeof result.request_id === "string") reportUrl(result.request_id);
    showPanel("result");
    clearError();

    const requestedDecision = result.decision.trim().toLowerCase();
    const knownDecision = ["pass", "review", "fail"].includes(requestedDecision);
    const scoreAvailable = result.anomaly_score !== null;
    const decision = knownDecision && scoreAvailable ? requestedDecision : "review";
    const reviewReason = !knownDecision
      ? "The service returned an unknown decision. Please review this image manually."
      : !scoreAvailable
        ? "The inspection score is unavailable. Please review this image manually."
        : decision === "review" ? reviewExplanation(result) : "";
    const verdict = verdictFor(decision);

    const verdictHeader = document.getElementById("verdict-header");
    verdictHeader.className = "verdict-header " + decision;
    document.getElementById("verdict-icon").textContent = verdict.icon;
    document.getElementById("verdict-title").textContent = verdict.title;
    document.getElementById("verdict-subtitle").textContent = reviewReason || verdict.subtitle;

    renderScore(result);

    const timings = result.timing_ms || result.timings || {};
    const elapsed = timings.total ?? timings.total_ms ?? timings.model ?? timings.model_ms;
    document.getElementById("detail-time").textContent =
      typeof elapsed === "number" && Number.isFinite(elapsed) ? Math.round(elapsed) + " ms" : "—";
    document.getElementById("detail-category").textContent = result.category || configuredCategory;
    document.getElementById("detail-request").textContent = result.request_id || result.id || "—";

    renderFindings(decision, Array.isArray(result.regions) ? result.regions : [], reviewReason);
    renderImage(currentView);
  }

  function parseInspectionResponse(raw, responseOk) {
    let body;
    try {
      body = JSON.parse(raw);
    } catch {
      throw new Error(
        responseOk
          ? "The inspection response was invalid."
          : "The image could not be inspected. Please try again.",
      );
    }
    if (!responseOk) throw new Error(responseError(body));
    if (!body || typeof body !== "object" || Array.isArray(body)) {
      throw new Error("The inspection response was invalid.");
    }
    if (typeof body.decision !== "string" || !body.decision.trim()) {
      throw new Error("The inspection response did not include a valid decision.");
    }
    if (
      body.anomaly_score !== null &&
      (typeof body.anomaly_score !== "number" || !Number.isFinite(body.anomaly_score))
    ) {
      throw new Error("The inspection response did not include a valid score.");
    }
    return body;
  }

  async function inspect(file) {
    clearError();
    const requestId = ++requestSerial;
    if (activeRequest) activeRequest.abort();
    activeRequest = null;
    currentResult = null;
    reportUrl(null);
    if (!file) return;
    if (!profileReady) { showError("The model profile is not ready. Please wait for the system status."); return; }

    if (!allowedTypes.has(file.type)) {
      showError("Please choose an allowed image type: " + [...allowedTypes].map((type) => type.replace("image/", "").toUpperCase()).join(", ") + ".");
      return;
    }
    if (file.size > maxBytes) {
      showError("This file exceeds the " + (maxBytes / (1024 * 1024)).toLocaleString(undefined, { maximumFractionDigits: 2 }) + " MiB limit. Please choose a smaller image.");
      return;
    }

    showPanel("loading");
    const controller = new AbortController();
    activeRequest = controller;
    try {
      const response = await fetch("/inspect?category=" + encodeURIComponent(configuredCategory) + (loadedModelId ? "&model_id=" + encodeURIComponent(loadedModelId) : ""), {
        method: "POST",
        headers: { "Content-Type": file.type },
        body: file,
        signal: controller.signal,
      });
      const raw = await response.text();
      if (requestId !== requestSerial) return;
      const body = parseInspectionResponse(raw, response.ok);
      renderResult(body);
    } catch (error) {
      if (requestId !== requestSerial || error?.name === "AbortError") return;
      showError(error instanceof Error ? error.message : "The image could not be inspected.");
    } finally {
      if (requestId === requestSerial) activeRequest = null;
    }
  }

  chooseButton.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("click", (event) => {
    if (!event.target.closest("button")) fileInput.click();
  });
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    inspect(fileInput.files[0]);
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragging");
    });
  });
  dropZone.addEventListener("drop", (event) => inspect(event.dataTransfer.files[0]));

  document.querySelectorAll(".view-tab").forEach((tab) => {
    tab.addEventListener("click", () => renderImage(tab.dataset.view));
  });
  document.getElementById("new-check-button").addEventListener("click", () => {
    requestSerial += 1;
    if (activeRequest) activeRequest.abort();
    activeRequest = null;
    currentResult = null;
    reportUrl(null);
    showPanel("empty");
    clearError();
    fileInput.click();
  });

  enableUpload(false);
  checkReady();
  window.setInterval(checkReady, 30000);
})();
