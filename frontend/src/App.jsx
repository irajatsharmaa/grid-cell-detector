import React, { useState, useEffect, useRef } from "react";
import "./index.css";

function App() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [zoomedImage, setZoomedImage] = useState(null);
  const [isZoomModalOpen, setIsZoomModalOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  // Refs for calculating layout height
  const headerRef = useRef(null);
  const controlsRef = useRef(null);
  const resultsRef = useRef(null);
  const [nonImageHeight, setNonImageHeight] = useState(250);

  // Update height dynamically based on content
  useEffect(() => {
    const calculateHeight = () => {
      const headerH = headerRef.current?.offsetHeight || 0;
      const controlsH = controlsRef.current?.offsetHeight || 0;
      const resultsH = resultsRef.current?.offsetHeight || 0;
      const paddingAndMargins = 80;
      setNonImageHeight(headerH + controlsH + resultsH + paddingAndMargins);
    };
    calculateHeight();
    window.addEventListener("resize", calculateHeight);
    return () => window.removeEventListener("resize", calculateHeight);
  }, [results]);

  // Handle image file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file.");
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
  };

  // Upload image to backend
  const handleUpload = async (mode) => {
    if (!image) return alert("Please select an image first.");

    const formData = new FormData();
    formData.append("file", image);
    formData.append("mode", mode);

    setLoading(true);
    setResults(null);

    try {
      const BACKEND_URL = process.env.NODE_ENV === "production"
        ? "https://your-backend.onrender.com"  // ← Change this to your real backend URL after deployment
        : "http://localhost:8000";

      const res = await fetch(`${BACKEND_URL}/upload-image`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setResults({ ...data, mode });
      } else {
        const errorData = await res.json().catch(() => ({
          error: "Server error processing the request.",
        }));
        console.error("Server error:", res.status, errorData);
        setResults({ error: errorData.error || "Server error." });
      }
    } catch (err) {
      console.error("Upload error:", err);
      setResults({ error: "Network error or server unavailable." });
    } finally {
      setLoading(false);
    }
  };

  // Zoom clicked image
  const handleZoomClick = (imageUrl) => {
    setZoomedImage(imageUrl);
    setIsZoomModalOpen(true);
  };

  // Close zoom modal
  const closeZoomModal = () => {
    setIsZoomModalOpen(false);
    setZoomedImage(null);
  };

  // Export coordinates as JSON
  const handleExport = () => {
    const dataToExport = results.mode === "scanned"
      ? { tables: results.tables, rows: results.rows, columns: results.columns }
      : { cell_coordinates: results.cell_coordinates };

    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "coordinates.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle("dark-mode");
  };

  return (
    <div className="container">
      {/* Dark Mode Toggle */}
      <button className="toggle-theme-btn" onClick={toggleDarkMode}>
        {darkMode ? "☀️ Light Mode" : "🌙 Dark Mode"}
      </button>

      {/* Header */}
      <header ref={headerRef}>
        <h1>🖼️ Grid Cell Detector</h1>
        <p>Detect table cells in images — Scanned or Camera</p>
      </header>

      {/* Image Preview */}
      <div className="image-preview">
        {previewUrl ? (
          <img src={previewUrl} alt="Preview" className="preview" />
        ) : (
          <p>Select an image to preview</p>
        )}
      </div>

      {/* Controls */}
      <div className="controls" ref={controlsRef}>
        <div className="text-center my-4">
          <input type="file" accept="image/*" onChange={handleFileChange} />
        </div>

        <div className="buttons">
          <button
            className="btn-scanned"
            onClick={() => handleUpload("scanned")}
            disabled={!image || loading}
          >
            Scanned
          </button>
          <button
            className="btn-camera"
            onClick={() => handleUpload("camera")}
            disabled={!image || loading}
          >
            Camera
          </button>
        </div>

        {loading && <p className="loading">Processing image...</p>}
        {results?.error && <p className="error">{results.error}</p>}
      </div>

      {/* Results */}
      {results?.images?.length > 0 && (
        <div className="results" ref={resultsRef}>
          <h2>📸 Detected Output:</h2>
          <div className="result-grid">
            {results.images.map((url, idx) => (
              <div key={idx} className="result-card">
                <img
                  src={url}
                  alt={`Detected Image ${idx + 1}`}
                  onClick={() => handleZoomClick(url)}
                />
                <p>Output {idx + 1}</p>
              </div>
            ))}
          </div>
          <div className="text-center">
            <button className="btn-export" onClick={handleExport}>
              📤 Export Coordinates
            </button>
          </div>
        </div>
      )}

      {/* Zoom Modal */}
      {isZoomModalOpen && (
        <div className="zoom-modal" onClick={closeZoomModal}>
          <span className="zoom-close" onClick={closeZoomModal}>
            &times;
          </span>
          <img src={zoomedImage} alt="Zoomed" />
        </div>
      )}
    </div>
  );
}

export default App;