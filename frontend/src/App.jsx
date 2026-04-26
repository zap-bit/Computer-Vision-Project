import { useEffect, useState } from 'react'
import './App.css'

function App() {
  const [backend, setBackend] = useState('checking...')
  const [latest, setLatest] = useState(null)

  async function loadBackendData() {
    try {
      const healthRes = await fetch('/api/health')
      const health = await healthRes.json()
      setBackend(health.status || 'ok')

      const countsRes = await fetch('/api/latest-counts')
      if (countsRes.ok) {
        const counts = await countsRes.json()
        setLatest(counts)
      }
    } catch {
      setBackend('offline')
    }
  }

  useEffect(() => {
    loadBackendData()
    const timer = setInterval(loadBackendData, 2000)
    return () => clearInterval(timer)
  }, [])

  const counts = latest?.counts || {
    'Water Bottles': 12,
    Oranges: 18,
    Apples: 9,
    Bananas: 15,
  }

  return (
    <div className="page">
      <aside className="sidebar">
        <h1>STOCKSYNC</h1>
        <p>Automated Retail Inventory Tracking</p>

        <nav>
          <a className="active">Dashboard</a>
          <a>Inventory</a>
          <a>Analytics</a>
          <a>Alerts</a>
          <a>Settings</a>
        </nav>
      </aside>

      <main className="main">
        <header className="header">
          <div>
            <h2>Inventory Dashboard</h2>
            <p>Real-time computer vision stock monitoring</p>
          </div>

          <span className={backend === 'offline' ? 'offline' : 'online'}>
            Backend: {backend}
          </span>
        </header>

        <section className="stats">
          <div className="stat">
            <h3>94.8%</h3>
            <p>Detection Accuracy</p>
          </div>

          <div className="stat">
            <h3>{Object.values(counts).reduce((a, b) => a + b, 0)}</h3>
            <p>Total Items Detected</p>
          </div>

          <div className="stat">
            <h3>{Object.keys(counts).length}</h3>
            <p>Product Categories</p>
          </div>

          <div className="stat">
            <h3>Live</h3>
            <p>Camera Status</p>
          </div>
        </section>

        <section className="grid">
          <div className="panel video">
            <h3>Live Detection Feed</h3>

            <div className="camera-box">
              <img
                src="/api/video-feed"
                alt="Live Camera Feed"
                className="camera-feed"
              />
            </div>
          </div>

          <div className="panel">
            <h3>Inventory Counts</h3>

            {Object.entries(counts).map(([item, count]) => (
              <div className="item" key={item}>
                <span>{item}</span>
                <b>{count}</b>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <h3>System Activity</h3>
          <p>Last updated from backend every 2 seconds.</p>
          <p>Run ID: {latest?.run_id || 'demo'}</p>
          <p>Frame: {latest?.frame_idx ?? 'waiting for detection data'}</p>
        </section>
      </main>
    </div>
  )
}

export default App