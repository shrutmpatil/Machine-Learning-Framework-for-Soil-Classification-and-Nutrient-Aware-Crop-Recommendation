// Home.jsx
import React, { useState, useEffect } from "react";
import { createIcons } from "lucide";

export default function Home() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [weather, setWeather] = useState({ temp: "--°C", desc: "Loading..." });
  const [sensor, setSensor] = useState({ moisture: "--", temp: "--", humidity: "--" });
  const [manualData, setManualData] = useState({ nitrogen: "", phosphorus: "", potassium: "", ph: "" });
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState("");
  const [history, setHistory] = useState([]);

  useEffect(() => {
    createIcons(); // render lucide icons
    fetchWeather();
    fetchThingSpeakData();
    fetchHistory();
  }, []);

  // ✅ Weather API
  const fetchWeather = async () => {
    try {
      const res = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?q=Palghar&units=metric&appid=YOUR_API_KEY`
      );
      const data = await res.json();
      setWeather({
        temp: `${data.main.temp}°C`,
        desc: data.weather[0].description,
      });
    } catch {
      setWeather({ temp: "--°C", desc: "Failed to load weather" });
    }
  };

  // ✅ ThingSpeak Sensor Data
  const fetchThingSpeakData = async () => {
    try {
      const res = await fetch(
        `https://api.thingspeak.com/channels/YOUR_CHANNEL_ID/feeds/last.json?api_key=YOUR_READ_API_KEY`
      );
      const data = await res.json();
      setSensor({
        moisture: `${data.field1}%`,
        temp: `${data.field2}°C`,
        humidity: `${data.field3}%`,
      });
    } catch {
      setSensor({ moisture: "--", temp: "--", humidity: "--" });
    }
  };

  // ✅ Manual Submit
  const submitManualData = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:5000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(manualData),
      });
      const result = await res.json();
      setOutput(`🌱 Recommended Crop: ${result.crop}`);
      fetchHistory();
    } catch {
      setOutput("❌ Failed to fetch recommendation.");
    }
    setLoading(false);
  };

  // ✅ Fetch Prediction History
  const fetchHistory = async () => {
    try {
      const res = await fetch("http://localhost:5000/history");
      const data = await res.json();
      setHistory(data);
    } catch {
      setHistory([{ time: "Error", result: "Failed to load history" }]);
    }
  };

  return (
    <div className="flex flex-col md:flex-row min-h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="bg-green-700 text-white w-full md:w-64 p-6 space-y-6">
        <div className="text-2xl font-bold mb-6 flex items-center gap-2">
          <i data-lucide="leaf" className="w-6 h-6"></i> AgriSense
        </div>
        <nav className="space-y-4 text-sm">
          {[
            { id: "dashboard", icon: "layout-dashboard", label: "Dashboard" },
            { id: "sensor", icon: "signal", label: "Sensor Data" },
            { id: "crop-recommendation", icon: "bar-chart-3", label: "Crop Recommendation" },
            { id: "history", icon: "history", label: "History & Analysis" },
            { id: "location", icon: "map-pin", label: "Farm Location" },
            { id: "news", icon: "newspaper", label: "Agriculture News" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 w-full text-left ${
                activeTab === tab.id ? "text-green-200 font-bold" : "hover:text-green-200"
              }`}
            >
              <i data-lucide={tab.icon} className="w-5 h-5"></i>
              {tab.label}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8">
        {/* Dashboard */}
        {activeTab === "dashboard" && (
          <div>
            <h1 className="text-3xl font-bold text-gray-800 mb-1">Welcome back, Farmer! 👨‍🌾</h1>
            <p className="text-gray-600 mb-6">
              Monitor your farm's health and get personalized crop recommendations
            </p>

            <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
              <Card title="Soil Temperature" value="24°C" note="+2°C from yesterday" icon="thermometer" />
              <Card title="Soil Moisture" value="68%" note="Optimal range" icon="droplets" />
              <Card title="NPK Levels" value="Good" note="N:40 P:35 K:42" icon="leaf" />
              <Card title="Weather" value={weather.temp} note={weather.desc} icon="sun" />
            </div>
          </div>
        )}

        {/* Sensor Data */}
        {activeTab === "sensor" && (
          <div>
            <h2 className="text-xl font-bold mb-4">📡 Live Sensor Data</h2>
            <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
              <Card title="Soil Moisture" value={sensor.moisture} />
              <Card title="Soil Temperature" value={sensor.temp} />
              <Card title="Humidity" value={sensor.humidity} />
            </div>
          </div>
        )}

        {/* Crop Recommendation */}
        {activeTab === "crop-recommendation" && (
          <div>
            <h2 className="text-2xl font-bold mb-4">🌾 Crop Recommendation</h2>
            <form className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <input
                type="number"
                placeholder="Nitrogen (N)"
                className="p-2 border rounded"
                value={manualData.nitrogen}
                onChange={(e) => setManualData({ ...manualData, nitrogen: e.target.value })}
              />
              <input
                type="number"
                placeholder="Phosphorus (P)"
                className="p-2 border rounded"
                value={manualData.phosphorus}
                onChange={(e) => setManualData({ ...manualData, phosphorus: e.target.value })}
              />
              <input
                type="number"
                placeholder="Potassium (K)"
                className="p-2 border rounded"
                value={manualData.potassium}
                onChange={(e) => setManualData({ ...manualData, potassium: e.target.value })}
              />
              <input
                type="number"
                placeholder="pH Value"
                className="p-2 border rounded"
                step="0.1"
                value={manualData.ph}
                onChange={(e) => setManualData({ ...manualData, ph: e.target.value })}
              />
            </form>
            <div className="flex gap-2 mb-4">
              <button
                onClick={submitManualData}
                className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded"
              >
                📤 Submit
              </button>
              <button
                onClick={fetchThingSpeakData}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
              >
                🔌 From Sensor
              </button>
            </div>
            {loading && <p className="text-yellow-500">⏳ Loading...</p>}
            {output && <p className="text-gray-800 bg-gray-100 p-2 rounded">{output}</p>}

            <div className="mt-6">
              <h3 className="text-lg font-bold">📜 Prediction History</h3>
              <ul className="list-disc ml-5">
                {history.map((h, i) => (
                  <li key={i}>
                    {h.time}: 🌾 {h.result}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Other Tabs */}
        {activeTab === "history" && <Placeholder title="History & Analysis" />}
        {activeTab === "location" && <Placeholder title="Farm Location" />}
        {activeTab === "news" && <Placeholder title="Agriculture News" />}
      </main>
    </div>
  );
}

// ✅ Reusable Card
function Card({ title, value, note, icon }) {
  return (
    <div className="bg-white rounded-xl shadow p-4 flex gap-3">
      {icon && <i data-lucide={icon} className="w-6 h-6 text-green-600"></i>}
      <div>
        <h2 className="text-sm font-semibold text-gray-600">{title}</h2>
        <div className="text-xl font-bold">{value}</div>
        {note && <div className="text-xs text-gray-500">{note}</div>}
      </div>
    </div>
  );
}

// ✅ Placeholder Component
function Placeholder({ title }) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">{title}</h2>
      <p className="text-gray-600">Coming soon...</p>
    </div>
  );
}


