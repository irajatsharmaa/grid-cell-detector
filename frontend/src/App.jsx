import { useState } from 'react';

function App() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
  };

  const handleUpload = async () => {
    if (!image) return alert("Please select an image first.");
    const formData = new FormData();
    formData.append("file", image);
    setLoading(true);

    try {
      const res = await fetch("https://grid-backend-846l.onrender.com/upload-image", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResults(data);
    } catch (err) {
      setResults({ error: "Server error." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 font-sans">
      <h1 className="text-2xl font-bold mb-4">🖼️ Grid Cell Detector</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />
      {previewUrl && (
        <div className="my-4">
          <img src={previewUrl} alt="Preview" className="max-h-64 rounded border" />
        </div>
      )}

      <button
        onClick={handleUpload}
        className="mt-2 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Run Script
      </button>

      {loading && <p className="mt-4 text-gray-500">Processing...</p>}

      {results && results.error && (
        <p className="mt-4 text-red-600">{results.error}</p>
      )}

      {results && results.images && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">📸 Detected Cells:</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {results.images.map((url, idx) => (
              <div key={idx} className="text-center">
                <img src={url} alt={`Cell ${idx + 1}`} className="w-full rounded shadow" />
                <p className="mt-2 font-semibold text-sm">Cell {idx + 1}</p>
              </div>
            ))}
          </div>
          <div className="mt-4">
            <a
              href={results.pdf_url}
              download
              className="inline-block bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
            >
              📄 Download All as PDF
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
