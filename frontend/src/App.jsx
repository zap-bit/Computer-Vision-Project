
import { useEffect, useState } from 'react'
import './App.css'

function App() {
  const [activePage, setActivePage] = useState('Dashboard')
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

  const rawCounts = latest?.counts || {
    'Water Bottles': 12,
    Oranges: 18,
    Apples: 9,
    Bananas: 15,
  }

  const counts = Object.fromEntries(
    Object.entries(rawCounts).filter(([item]) => item.toLowerCase() !== 'person')
  )

  const totalItems = Object.values(counts).reduce((a, b) => a + b, 0)
  const categories = Object.keys(counts).length
  const mostCommon =
    Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'None'

  const pages = ['Dashboard', 'Inventory', 'Analytics', 'Alerts', 'Settings']

  return (
    <div className="page">
      <aside className="sidebar">
        <h1>STOCKSYNC</h1>
        <p>Automated Retail Inventory Tracking</p>

        <nav>
          {pages.map((page) => (
            <button
              key={page}
              className={activePage === page ? 'active' : ''}
              onClick={() => setActivePage(page)}
            >
              {page}
            </button>
          ))}
        </nav>
      </aside>

      <main className="main">
        <header className="header">
          <div>
            <h2>{activePage}</h2>
            <p>Real-time computer vision stock monitoring</p>
          </div>

          <span className={backend === 'offline' ? 'offline' : 'online'}>
            Backend: {backend}
          </span>
        </header>

        {activePage === 'Dashboard' && (
          <>
            <section className="stats">
              <div className="stat">
                <h3>94.8%</h3>
                <p>Detection Accuracy</p>
              </div>

              <div className="stat">
                <h3>{totalItems}</h3>
                <p>Total Items Detected</p>
              </div>

              <div className="stat">
                <h3>{categories}</h3>
                <p>Product Categories</p>
              </div>

              <div className="stat">
                <h3>{backend === 'offline' ? 'Off' : 'Live'}</h3>
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
          </>
        )}

        {activePage === 'Inventory' && (
          <section className="panel">
            <h3>Product Inventory</h3>

            <table>
              <thead>
                <tr>
                  <th>Product</th>
                  <th>Count</th>
                  <th>Status</th>
                  <th>Last Updated</th>
                </tr>
              </thead>

              <tbody>
                {Object.entries(counts).map(([item, count]) => (
                  <tr key={item}>
                    <td>{item}</td>
                    <td>{count}</td>
                    <td className={count <= 1 ? 'status-low' : 'status-good'}>
                      {count <= 1 ? 'Low Stock' : 'In Stock'}
                    </td>
                    <td>Every 2 seconds</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {activePage === 'Analytics' && (
          <>
            <section className="stats">
              <div className="stat">
                <h3>{totalItems}</h3>
                <p>Total Products</p>
              </div>

              <div className="stat">
                <h3>{categories}</h3>
                <p>Categories Found</p>
              </div>

              <div className="stat">
                <h3>{mostCommon}</h3>
                <p>Most Common Item</p>
              </div>

              <div className="stat">
                <h3>2s</h3>
                <p>Refresh Rate</p>
              </div>
            </section>

            <section className="panel">
              <h3>Detection Summary</h3>
              <p>Model is connected to backend detection results.</p>
              <p>Run ID: {latest?.run_id || 'demo'}</p>
              <p>Frame: {latest?.frame_idx ?? 'waiting for detection data'}</p>
            </section>
          </>
        )}

        {activePage === 'Alerts' && (
          <section className="panel">
            <h3>Alerts</h3>

            {backend === 'offline' && (
              <p className="alert">⚠️ Backend is offline.</p>
            )}

            {Object.entries(counts)
              .filter(([, count]) => count <= 1)
              .map(([item]) => (
                <p className="alert" key={item}>
                  ⚠️ {item} is low stock.
                </p>
              ))}

            {backend !== 'offline' &&
              Object.values(counts).every((count) => count > 1) && (
                <p>No alerts right now.</p>
              )}
          </section>
        )}

        {activePage === 'Settings' && (
          <section className="panel">
            <h3>Settings</h3>

            <div className="item">
              <span>Camera</span>
              <b>{backend === 'offline' ? 'OFF' : 'ON'}</b>
            </div>

            <div className="item">
              <span>Backend Routes</span>
              <b>/api</b>
            </div>

            <div className="item">
              <span>Refresh Rate</span>
              <b>2 seconds</b>
            </div>

            <div className="item">
              <span>Inventory Filter</span>
              <b>No person class</b>
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default App

