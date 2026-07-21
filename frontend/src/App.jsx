import { useState, useEffect, useRef } from 'react'

function App() {
  const API_BASE = import.meta.env.VITE_API_URL || '';
  const [page, setPage] = useState('upload') // 'upload', 'results', 'history'
  const [activeModel, setActiveModel] = useState('None set')
  
  // File state
  const [baselineFile, setBaselineFile] = useState(null)
  const [candidateFile, setCandidateFile] = useState(null)

  // Selection Mode State (preloaded vs upload)
  const [baselineMode, setBaselineMode] = useState('preloaded') // 'preloaded', 'upload'
  const [candidateMode, setCandidateMode] = useState('preloaded') // 'preloaded', 'upload'
  
  const [preloadedBaseline, setPreloadedBaseline] = useState('model_v0')
  const [preloadedCandidate, setPreloadedCandidate] = useState('model_v1')
  
  // Loading & error states
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  
  // Results & History state
  const [results, setResults] = useState(null)
  const [history, setHistory] = useState([])
  const [showGraph, setShowGraph] = useState(false)
  const [reviewActed, setReviewActed] = useState(false)
  const [expandedHistoryRun, setExpandedHistoryRun] = useState(null)

  // Refs for file inputs
  const baselineInputRef = useRef(null)
  const candidateInputRef = useRef(null)

  // Fetch active model on mount
  useEffect(() => {
    fetchActiveModel()
  }, [])

  const fetchActiveModel = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/active-model`)
      if (res.ok) {
        const data = await res.json()
        setActiveModel(data.active_model)
        // If no model set, activate the default preloaded model
        if (data.active_model === 'None set' || data.active_model === 'Active Model: None' || data.active_model === 'None') {
          await handlePreloadedBaselineChange('model_v0')
        }
      }
    } catch (err) {
      console.error('Error fetching active model:', err)
    }
  }

  // Handle immediate baseline activation
  const handleBaselineFileChange = async (file) => {
    if (!file) return
    if (!file.name.endsWith('.pkl')) {
      setError('Please upload a .pkl file.')
      return
    }
    
    setBaselineFile(file)
    setError(null)
    
    // Immediate activation of baseline model in the backend
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      setIsLoading(true)
      const res = await fetch(`${API_BASE}/api/upload-baseline`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to activate baseline model.')
      }
      const data = await res.json()
      setActiveModel(data.active_model)
    } catch (err) {
      setError(err.message)
      setBaselineFile(null)
    } finally {
      setIsLoading(false)
    }
  }

  // Handle immediate preloaded baseline activation
  const handlePreloadedBaselineChange = async (modelName) => {
    if (!modelName) return
    setPreloadedBaseline(modelName)
    setError(null)
    
    const formData = new FormData()
    formData.append('baseline_model_name', modelName)
    
    try {
      setIsLoading(true)
      const res = await fetch(`${API_BASE}/api/upload-baseline`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to activate baseline model.')
      }
      const data = await res.json()
      setActiveModel(data.active_model)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const changeBaselineMode = async (mode) => {
    setBaselineMode(mode)
    if (mode === 'preloaded') {
      await handlePreloadedBaselineChange(preloadedBaseline)
    } else {
      if (baselineFile) {
        await handleBaselineFileChange(baselineFile)
      } else {
        // Reset active model
        try {
          const res = await fetch(`${API_BASE}/api/reset-model`, { method: 'POST' })
          if (res.ok) {
            const data = await res.json()
            setActiveModel(data.active_model)
          }
        } catch (err) {
          console.error(err)
        }
      }
    }
  }

  const handleCandidateFileChange = (file) => {
    if (!file) return
    if (!file.name.endsWith('.pkl')) {
      setError('Please upload a .pkl file.')
      return
    }
    setCandidateFile(file)
    setError(null)
  }

  const handleRunAnalysis = async (e) => {
    e.preventDefault()
    
    const isBaselineReady = baselineMode === 'preloaded' ? !!preloadedBaseline : !!baselineFile
    const isCandidateReady = candidateMode === 'preloaded' ? !!preloadedCandidate : !!candidateFile
    
    if (!isBaselineReady || !isCandidateReady) {
      setError('Please provide both models.')
      return
    }

    setIsLoading(true)
    setError(null)

    const formData = new FormData()
    if (baselineMode === 'preloaded') {
      formData.append('baseline_model_name', preloadedBaseline)
    } else {
      formData.append('baseline_file', baselineFile)
    }

    if (candidateMode === 'preloaded') {
      formData.append('candidate_model_name', preloadedCandidate)
    } else {
      formData.append('candidate_file', candidateFile)
    }

    try {
      const res = await fetch(`${API_BASE}/api/run-analysis`, {
        method: 'POST',
        body: formData,
      })
      
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'An error occurred during analysis.')
      }

      const data = await res.json()
      setResults(data)
      setActiveModel(data.active_model)
      setReviewActed(false)
      setShowGraph(false)
      setPage('results')
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReviewAction = async (action) => {
    if (!results) return
    
    setIsLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/api/review-action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action,
          candidate_name: results.candidate_name,
          baseline_name: results.baseline_name
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to submit review action.')
      }

      const data = await res.json()
      setActiveModel(data.active_model)
      setReviewActed(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleBackToUpload = async () => {
    setIsLoading(true)
    try {
      // If we are in preloaded mode, make sure model_v0 is active
      if (baselineMode === 'preloaded') {
        await handlePreloadedBaselineChange('model_v0')
      } else {
        const res = await fetch(`${API_BASE}/api/reset-model`, { method: 'POST' })
        if (res.ok) {
          const data = await res.json()
          setActiveModel(data.active_model)
        }
      }
      // Reset local states
      setBaselineFile(null)
      setCandidateFile(null)
      setResults(null)
      setPage('upload')
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const fetchHistory = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/api/history`)
      if (!res.ok) throw new Error('Failed to load deployment history.')
      const data = await res.json()
      setHistory(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleNavClick = (targetPage) => {
    if (targetPage === 'history') {
      fetchHistory()
    }
    setPage(targetPage)
  }

  const toggleHistoryItem = (id) => {
    if (expandedHistoryRun === id) {
      setExpandedHistoryRun(null)
    } else {
      setExpandedHistoryRun(id)
    }
  }

  // Drag and Drop helpers
  const [dragActiveBaseline, setDragActiveBaseline] = useState(false)
  const [dragActiveCandidate, setDragActiveCandidate] = useState(false)

  const handleDrag = (e, type) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      if (type === 'baseline') setDragActiveBaseline(true)
      else setDragActiveCandidate(true)
    } else if (e.type === "dragleave") {
      if (type === 'baseline') setDragActiveBaseline(false)
      else setDragActiveCandidate(false)
    }
  }

  const handleDrop = (e, type) => {
    e.preventDefault()
    e.stopPropagation()
    if (type === 'baseline') {
      setDragActiveBaseline(false)
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleBaselineFileChange(e.dataTransfer.files[0])
      }
    } else {
      setDragActiveCandidate(false)
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleCandidateFileChange(e.dataTransfer.files[0])
      }
    }
  }

  // SVG Bar Chart helper constants
  const renderSVGChart = () => {
    if (!results) return null
    const metrics = [
      { name: 'Flip Rate', val: results.flip_rate, label: 'Flip' },
      { name: 'Confidence Shift', val: results.conf_shift, label: 'Conf' },
      { name: 'Feature Drift', val: results.feature_drift, label: 'Drift' },
      { name: 'Subgroup Risk', val: results.subgroup_risk, label: 'Subgroup' },
      { name: 'Bias Severity', val: results.bias_severity, label: 'Bias' },
    ]

    const maxVal = Math.max(...metrics.map(m => m.val), 0.001)
    const yMax = maxVal * 1.1 // Give some padding at the top
    
    // Chart geometry
    const chartHeight = 180
    const chartWidth = 500
    const paddingLeft = 45
    const paddingRight = 20
    const paddingTop = 20
    const paddingBottom = 30
    
    const plotWidth = chartWidth - paddingLeft - paddingRight
    const plotHeight = chartHeight - paddingTop - paddingBottom
    
    const barWidth = 40
    const barGap = (plotWidth - (barWidth * metrics.length)) / (metrics.length + 1)

    return (
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="svg-chart">
        {/* Y Grid Lines & Labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const val = ratio * yMax
          const y = paddingTop + plotHeight - (ratio * plotHeight)
          return (
            <g key={ratio}>
              <line 
                x1={paddingLeft} 
                y1={y} 
                x2={chartWidth - paddingRight} 
                y2={y} 
                className="chart-grid-line" 
              />
              <text 
                x={paddingLeft - 10} 
                y={y + 4} 
                textAnchor="end" 
                className="chart-text"
              >
                {val.toFixed(3)}
              </text>
            </g>
          )
        })}

        {/* X Axis Line */}
        <line 
          x1={paddingLeft} 
          y1={paddingTop + plotHeight} 
          x2={chartWidth - paddingRight} 
          y2={paddingTop + plotHeight} 
          className="chart-axis-line" 
        />

        {/* Bars */}
        {metrics.map((m, idx) => {
          const barHeight = (m.val / yMax) * plotHeight
          const x = paddingLeft + barGap + (idx * (barWidth + barGap))
          const y = paddingTop + plotHeight - barHeight

          return (
            <g key={m.name}>
              {/* Bar Rect */}
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                rx="4"
                className="chart-bar"
              >
                <title>{`${m.name}: ${m.val.toFixed(4)}`}</title>
              </rect>
              
              {/* Bar Value Label */}
              <text
                x={x + barWidth / 2}
                y={y - 6}
                textAnchor="middle"
                className="chart-bar-value"
              >
                {m.val.toFixed(3)}
              </text>
              
              {/* X Axis Label */}
              <text
                x={x + barWidth / 2}
                y={paddingTop + plotHeight + 18}
                textAnchor="middle"
                className="chart-text"
              >
                {m.label}
              </text>
            </g>
          )
        })}
      </svg>
    )
  }

  // Radial progress calculations
  const radius = 50
  const circumference = 2 * Math.PI * radius
  const riskPercent = results ? Math.min(results.final_risk * 100, 100) : 0
  const strokeDashoffset = circumference - (riskPercent / 100) * circumference

  return (
    <div id="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <h1 className="sidebar-title">ModelGuard AI</h1>
        <div className="sidebar-status">
          <span className="pulse-dot"></span>
          <span>{activeModel}</span>
        </div>
        <nav>
          <ul className="nav-menu">
            <li className="nav-item">
              <button 
                className={`nav-button ${page === 'upload' ? 'active' : ''}`}
                onClick={() => handleNavClick('upload')}
              >
                📤 Upload
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-button ${page === 'results' ? 'active' : ''}`}
                onClick={() => handleNavClick('results')}
                disabled={!results}
              >
                📊 Results
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-button ${page === 'history' ? 'active' : ''}`}
                onClick={() => handleNavClick('history')}
              >
                🕓 Deployment History
              </button>
            </li>
          </ul>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Error notification banner */}
        {error && (
          <div className="error-banner">
            <span>⚠️ {error}</span>
            <button className="error-close" onClick={() => setError(null)}>×</button>
          </div>
        )}

        {/* Page 1: Upload */}
        {page === 'upload' && (
          <div>
            <h2 className="page-title">Model Risk Assessment Engine</h2>
            
            <form onSubmit={handleRunAnalysis} className="section-card">
              <div className="upload-grid">
                
                {/* Baseline Card */}
                <div className="upload-card">
                  <label className="upload-label">Baseline Model</label>
                  <div className="segment-control">
                    <button 
                      type="button" 
                      className={`segment-btn ${baselineMode === 'preloaded' ? 'active' : ''}`}
                      onClick={() => changeBaselineMode('preloaded')}
                    >
                      ⚡ Preloaded
                    </button>
                    <button 
                      type="button" 
                      className={`segment-btn ${baselineMode === 'upload' ? 'active' : ''}`}
                      onClick={() => changeBaselineMode('upload')}
                    >
                      📁 Upload File
                    </button>
                  </div>

                  {baselineMode === 'preloaded' ? (
                    <div className="select-container">
                      <select 
                        className="model-select"
                        value={preloadedBaseline}
                        onChange={(e) => handlePreloadedBaselineChange(e.target.value)}
                      >
                        <option value="model_v0">Model v0 (Baseline / Deployed)</option>
                        <option value="model_v1">Model v1 (Small Drift)</option>
                        <option value="model_v2">Model v2 (Moderate Bias)</option>
                        <option value="model_v3">Model v3 (Severe Bias)</option>
                      </select>
                      <p className="select-helper-text">
                        {preloadedBaseline === 'model_v0' && '🏆 Standard trained baseline model. 0% bias injected.'}
                        {preloadedBaseline === 'model_v1' && '📈 Small drift model (slightly different tree depth, minor noise).'}
                        {preloadedBaseline === 'model_v2' && '⚠️ Moderately biased candidate (trained without High radius subgroup).'}
                        {preloadedBaseline === 'model_v3' && '🚨 Severely biased candidate (trained without Low radius subgroup + noise).'}
                      </p>
                    </div>
                  ) : (
                    <div 
                      className={`dropzone ${dragActiveBaseline ? 'active' : ''} ${baselineFile ? 'file-selected' : ''}`}
                      onDragEnter={(e) => handleDrag(e, 'baseline')}
                      onDragOver={(e) => handleDrag(e, 'baseline')}
                      onDragLeave={(e) => handleDrag(e, 'baseline')}
                      onDrop={(e) => handleDrop(e, 'baseline')}
                      onClick={() => baselineInputRef.current.click()}
                    >
                      <input 
                        type="file" 
                        ref={baselineInputRef}
                        style={{ display: 'none' }}
                        accept=".pkl"
                        onChange={(e) => handleBaselineFileChange(e.target.files[0])}
                      />
                      <svg className="dropzone-icon" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      {baselineFile ? (
                        <div className="file-pill" onClick={(e) => e.stopPropagation()}>
                          <span>{baselineFile.name}</span>
                          <button className="file-clear-btn" onClick={() => setBaselineFile(null)}>×</button>
                        </div>
                      ) : (
                        <>
                          <p className="dropzone-text">Drag & drop or browse</p>
                          <p className="dropzone-subtext">Baseline pickled model file (.pkl)</p>
                        </>
                      )}
                    </div>
                  )}
                </div>

                {/* Candidate Card */}
                <div className="upload-card">
                  <label className="upload-label">Candidate Model</label>
                  <div className="segment-control">
                    <button 
                      type="button" 
                      className={`segment-btn ${candidateMode === 'preloaded' ? 'active' : ''}`}
                      onClick={() => setCandidateMode('preloaded')}
                    >
                      ⚡ Preloaded
                    </button>
                    <button 
                      type="button" 
                      className={`segment-btn ${candidateMode === 'upload' ? 'active' : ''}`}
                      onClick={() => setCandidateMode('upload')}
                    >
                      📁 Upload File
                    </button>
                  </div>

                  {candidateMode === 'preloaded' ? (
                    <div className="select-container">
                      <select 
                        className="model-select"
                        value={preloadedCandidate}
                        onChange={(e) => setPreloadedCandidate(e.target.value)}
                      >
                        <option value="model_v0">Model v0 (Baseline / Deployed)</option>
                        <option value="model_v1">Model v1 (Small Drift)</option>
                        <option value="model_v2">Model v2 (Moderate Bias)</option>
                        <option value="model_v3">Model v3 (Severe Bias)</option>
                      </select>
                      <p className="select-helper-text">
                        {preloadedCandidate === 'model_v0' && '🏆 Standard trained baseline model. 0% bias injected.'}
                        {preloadedCandidate === 'model_v1' && '📈 Small drift model (slightly different tree depth, minor noise).'}
                        {preloadedCandidate === 'model_v2' && '⚠️ Moderately biased candidate (trained without High radius subgroup).'}
                        {preloadedCandidate === 'model_v3' && '🚨 Severely biased candidate (trained without Low radius subgroup + noise).'}
                      </p>
                    </div>
                  ) : (
                    <div 
                      className={`dropzone ${dragActiveCandidate ? 'active' : ''} ${candidateFile ? 'file-selected' : ''}`}
                      onDragEnter={(e) => handleDrag(e, 'candidate')}
                      onDragOver={(e) => handleDrag(e, 'candidate')}
                      onDragLeave={(e) => handleDrag(e, 'candidate')}
                      onDrop={(e) => handleDrop(e, 'candidate')}
                      onClick={() => candidateInputRef.current.click()}
                    >
                      <input 
                        type="file" 
                        ref={candidateInputRef}
                        style={{ display: 'none' }}
                        accept=".pkl"
                        onChange={(e) => handleCandidateFileChange(e.target.files[0])}
                      />
                      <svg className="dropzone-icon" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      {candidateFile ? (
                        <div className="file-pill" onClick={(e) => e.stopPropagation()}>
                          <span>{candidateFile.name}</span>
                          <button className="file-clear-btn" onClick={() => setCandidateFile(null)}>×</button>
                        </div>
                      ) : (
                        <>
                          <p className="dropzone-text">Drag & drop or browse</p>
                          <p className="dropzone-subtext">Candidate pickled model file (.pkl)</p>
                        </>
                      )}
                    </div>
                  )}
                </div>

              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                <button 
                  type="submit" 
                  className="btn btn-primary"
                  disabled={
                    isLoading || 
                    (baselineMode === 'upload' && !baselineFile) || 
                    (candidateMode === 'upload' && !candidateFile)
                  }
                >
                  {isLoading ? 'Processing Models...' : 'Run Risk Analysis'}
                </button>
                {isLoading && <div className="spinner"></div>}
              </div>
            </form>
          </div>
        )}

        {/* Page 2: Results */}
        {page === 'results' && results && (
          <div>
            <div className="results-header">
              <h2 className="page-title">AI Risk Intelligence Report</h2>
              <button className="btn btn-outline" onClick={handleBackToUpload}>
                ⬅ Back
              </button>
            </div>

            {/* Metrics Dashboard */}
            <div className="section-card">
              <h3 style={{ fontSize: '1.2rem', marginBottom: '1.25rem' }}>Risk Metrics Summary</h3>
              <div className="results-grid">
                <div className="metric-card">
                  <span className="metric-label">Flip Rate</span>
                  <span className="metric-value">{results.flip_rate.toFixed(4)}</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Confidence Shift</span>
                  <span className="metric-value">{results.conf_shift.toFixed(4)}</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Feature Drift</span>
                  <span className="metric-value">{results.feature_drift.toFixed(4)}</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Subgroup Risk</span>
                  <span className="metric-value">{results.subgroup_risk.toFixed(4)}</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Bias Severity</span>
                  <span className="metric-value">{results.bias_severity.toFixed(4)}</span>
                </div>
              </div>

              <div className="overall-risk-container">
                <div className="radial-progress">
                  <svg width="120" height="120">
                    <circle cx="60" cy="60" r={radius} className="radial-bg" />
                    <circle 
                      cx="60" 
                      cy="60" 
                      r={radius} 
                      className="radial-fill"
                      strokeDasharray={circumference}
                      strokeDashoffset={strokeDashoffset}
                    />
                  </svg>
                  <div className="radial-text">
                    <span className="radial-percent">{results.final_risk.toFixed(4)}</span>
                    <span className="radial-label">Score</span>
                  </div>
                </div>

                <div className="risk-desc-card">
                  <h4 className="risk-desc-title">Weighted Final Risk Index</h4>
                  <p className="risk-desc-text">
                    This score is synthesized from prediction shifts (30%), calibration changes (25%), 
                    SHAP drift (20%), worst-case subgroup performance drops (15%), and bias indicators (10%).
                  </p>
                </div>
              </div>
            </div>

            {/* Decision Recommendation Banner */}
            <div className="section-card">
              <h3 style={{ fontSize: '1.2rem', marginBottom: '1.25rem' }}>Deployment Recommendation</h3>
              
              {results.decision === 'DEPLOY' && (
                <div className="decision-alert deploy">
                  <div className="decision-alert-title">
                    <span>✅ SAFE TO DEPLOY</span>
                  </div>
                  <p className="decision-alert-body">
                    The overall risk score of <strong>{results.final_risk.toFixed(4)}</strong> is within the safe threshold (&lt; 0.02). 
                    The candidate model has been automatically promoted to active.
                  </p>
                </div>
              )}

              {results.decision === 'ROLLBACK' && (
                <div className="decision-alert rollback">
                  <div className="decision-alert-title">
                    <span>🚨 ROLLBACK TRIGGERED</span>
                  </div>
                  <p className="decision-alert-body">
                    The risk index of <strong>{results.final_risk.toFixed(4)}</strong> exceeds the critical ceiling (&ge; 0.07). 
                    The baseline model has been preserved/restored as the active model.
                  </p>
                </div>
              )}

              {results.decision === 'REVIEW' && (
                <div className="decision-alert review">
                  <div className="decision-alert-title">
                    <span>⚠️ DEPLOY WITH CAUTION (REVIEW REQUIRED)</span>
                  </div>
                  <p className="decision-alert-body">
                    The calculated risk is <strong>{results.final_risk.toFixed(4)}</strong>. This is in the caution zone (0.02 - 0.07). 
                    Manual review is required before updating active production instances.
                  </p>
                  
                  {!reviewActed ? (
                    <div className="decision-alert-actions">
                      <button 
                        className="btn btn-success" 
                        onClick={() => handleReviewAction('accept')}
                        disabled={isLoading}
                      >
                        Accept Candidate
                      </button>
                      <button 
                        className="btn btn-danger" 
                        onClick={() => handleReviewAction('rollback')}
                        disabled={isLoading}
                      >
                        Rollback to Baseline
                      </button>
                    </div>
                  ) : (
                    <p style={{ fontStyle: 'italic', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                      ✓ Decision submitted. Active model status: <strong>{activeModel}</strong>
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Toggleable Visualization Graph */}
            <div className="section-card chart-container">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
                <h3 style={{ fontSize: '1.2rem' }}>Metric Breakdown Chart</h3>
                <button className="btn btn-outline" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }} onClick={() => setShowGraph(!showGraph)}>
                  {showGraph ? 'Hide Graph' : 'Show Graph'}
                </button>
              </div>

              {showGraph && (
                <div className="chart-wrapper">
                  {renderSVGChart()}
                  <div className="chart-legend">
                    <div className="legend-item">
                      <span className="legend-dot"></span>
                      <span>Metrics Scale (0.0 - {Math.max(results.flip_rate, results.conf_shift, results.feature_drift, results.subgroup_risk, results.bias_severity, 0.1).toFixed(2)})</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
              <button className="btn btn-outline" onClick={handleBackToUpload}>
                ⬅ Back to Upload
              </button>
            </div>
          </div>
        )}

        {/* Page 3: History */}
        {page === 'history' && (
          <div>
            <div className="results-header">
              <h2 className="page-title">Deployment History</h2>
              <button className="btn btn-outline" onClick={() => setPage('upload')}>
                ⬅ Back
              </button>
            </div>

            {isLoading && history.length === 0 ? (
              <div className="loading-container">
                <div className="spinner"></div>
                <p className="loading-text">Fetching historical records...</p>
              </div>
            ) : history.length === 0 ? (
              <div className="empty-placeholder">
                No past runs found in the comparisons database.
              </div>
            ) : (
              <div className="history-list">
                {history.map((h, idx) => {
                  const isExpanded = expandedHistoryRun === h.id
                  const runNumber = history.length - idx
                  
                  return (
                    <div key={h.id} className={`history-item ${isExpanded ? 'expanded' : ''}`}>
                      <div className="history-item-header" onClick={() => toggleHistoryItem(h.id)}>
                        <div className="history-item-title">
                          <span className="history-run-number">Run #{runNumber}</span>
                          <span className="history-run-models">{h.candidate_name} vs {h.baseline_name}</span>
                        </div>
                        <div className="history-item-meta">
                          <span className="history-timestamp">{new Date(h.timestamp).toLocaleString()}</span>
                          <span className={`badge ${
                            h.decision === 'DEPLOY' ? 'badge-deploy' : 
                            h.decision === 'ROLLBACK' ? 'badge-rollback' : 'badge-review'
                          }`}>
                            {h.decision}
                          </span>
                          <svg className="history-expand-icon" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>
                      </div>
                      
                      {isExpanded && (
                        <div className="history-item-body">
                          <div className="history-metrics-grid">
                            <div className="history-metric">
                              <div className="history-metric-label">Flip Rate</div>
                              <div className="history-metric-value">{h.flip_rate.toFixed(4)}</div>
                            </div>
                            <div className="history-metric">
                              <div className="history-metric-label">Conf Shift</div>
                              <div className="history-metric-value">{h.confidence_shift.toFixed(4)}</div>
                            </div>
                            <div className="history-metric">
                              <div className="history-metric-label">Drift</div>
                              <div className="history-metric-value">{h.feature_drift.toFixed(4)}</div>
                            </div>
                            <div className="history-metric">
                              <div className="history-metric-label">Subgroup</div>
                              <div className="history-metric-value">{h.subgroup_risk.toFixed(4)}</div>
                            </div>
                            <div className="history-metric">
                              <div className="history-metric-label">Bias</div>
                              <div className="history-metric-value">{h.bias_severity.toFixed(4)}</div>
                            </div>
                          </div>
                          
                          <div className="history-summary">
                            <div className="history-summary-models">
                              Baseline Model: <span>{h.baseline_model}</span> | Candidate Model: <span>{h.updated_model}</span>
                            </div>
                            <div>
                              Overall Risk: <strong style={{ color: 'var(--text-primary)' }}>{h.final_risk.toFixed(4)}</strong>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
