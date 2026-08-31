import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, ImageOverlay, GeoJSON, ZoomControl, useMap, useMapEvents } from 'react-leaflet';
import { 
  Menu,
  X,
  Search, 
  Eye, 
  Download, 
  FileJson, 
  FileSpreadsheet, 
  ZoomIn, 
  CheckCircle2, 
  AlertCircle, 
  Crosshair, 
  ChevronDown,
  ChevronUp,
  Layers,
  Check,
  Ban,
  Copy,
  CheckCheck,
  Sparkles,
  Info,
  ArrowLeft,
  MapPin
} from 'lucide-react';
import LandingPage from './LandingPage';

const API_BASE = 'http://127.0.0.1:8000';
const CONCEPT_DEMO_ID = 'concept_demo';
const DEMO_WIDTH = 320;
const DEMO_HEIGHT = 240;
const DEFAULT_BOUNDS = [[26.14665, 91.7241], [26.19837, 91.77582]];

/* ──────────────────────────────────────────────
   Top-Level Error Boundary
─────────────────────────────────────────────── */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('DRISHTI Workstation Error Boundary caught an exception:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: 40,
          color: '#ffffff',
          textAlign: 'center',
          background: '#030712',
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: 'system-ui, sans-serif'
        }}>
          <AlertCircle size={36} style={{ color: '#38bdf8', marginBottom: 12 }} />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 8, letterSpacing: '0.02em' }}>
            Workstation Display Restored
          </h3>
          <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.8rem', maxWidth: 400, marginBottom: 20, lineHeight: 1.5 }}>
            An unexpected error occurred during map rendering. Please select a valid target location or reload the workstation.
          </p>
          <button 
            onClick={() => { this.setState({ hasError: false }); window.location.reload(); }}
            style={{
              padding: '9px 20px',
              background: '#38bdf8',
              color: '#030712',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontWeight: 700,
              fontSize: '0.8rem',
              letterSpacing: '0.03em'
            }}
          >
            RELOAD WORKSTATION
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const CATEGORY_COLORS = {
  'ROAD CHANGE': '#ffffff',             // White
  'NEW CONSTRUCTION': '#d97706',        // Dark Amber/Gold
  'BUILDING CHANGE': '#d97706',         // Dark Amber/Gold
  'WATER EXTENT CHANGE': '#1d4ed8',     // Dark Blue
  'RIVER CHANGE': '#1d4ed8',            // Dark Blue
  'WATER CHANGE': '#1d4ed8',            // Dark Blue
  'VEGETATION LOSS': '#15803d',         // Dark Green
  'FOREST CHANGE': '#15803d',           // Dark Green
};

const getCategoryColor = (type) => {
  return CATEGORY_COLORS[type] || '#ffffff';
};

const ALL_CAT_IDS = ['roads', 'construction', 'water', 'vegetation'];

function MoDLogo() {
  const [imgError, setImgError] = useState(false);

  if (imgError) {
    return (
      <img 
        src="/Ministry_of_Defence_India.png" 
        alt="Ministry of Defence India" 
        className="mod-header-logo-img png-fallback" 
      />
    );
  }

  return (
    <img 
      src="/Ministry_of_Defence_India.svg" 
      alt="Ministry of Defence India" 
      className="mod-header-logo-img" 
      onError={() => setImgError(true)}
    />
  );
}

// Leaflet Map Resizer Component
function MapResizer({ isSidebarOpen, selectedChange }) {
  const map = useMap();
  useEffect(() => {
    const timer = setTimeout(() => {
      map.invalidateSize();
    }, 300);
    return () => clearTimeout(timer);
  }, [isSidebarOpen, selectedChange, map]);
  return null;
}

// Leaflet Bounds Fitter Component
function MapBoundsFitter({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds && Array.isArray(bounds) && bounds.length === 2 && Array.isArray(bounds[0]) && Array.isArray(bounds[1])) {
      try {
        map.fitBounds(bounds, { padding: [20, 20], maxZoom: 16 });
      } catch (err) {
        // Safe catch for Leaflet bounds invalidation
      }
    }
  }, [bounds, map]);
  return null;
}

// Map Dual-Pane Synchronizer Component
function MapSynchronizer({ center, zoom, onMapMoved }) {
  const map = useMap();
  const isUpdatingFromProps = useRef(false);

  useEffect(() => {
    if (center && Array.isArray(center) && center.length === 2 && typeof center[0] === 'number' && !isNaN(center[0])) {
      isUpdatingFromProps.current = true;
      map.setView(center, zoom, { animate: false });
      setTimeout(() => { isUpdatingFromProps.current = false; }, 50);
    }
  }, [center, zoom, map]);

  useMapEvents({
    moveend: () => {
      if (isUpdatingFromProps.current) return;
      const c = map.getCenter();
      const z = map.getZoom();
      if (onMapMoved) {
        onMapMoved([c.lat, c.lng], z);
      }
    }
  });

  return null;
}

function getChangeCoordinates(change, locBounds) {
  if (!change) return { latStr: 'N/A', lonStr: 'N/A', latVal: null, lonVal: null };

  let lat = null;
  let lon = null;

  if (Array.isArray(change.centroid) && change.centroid.length === 2 && typeof change.centroid[0] === 'number') {
    lat = change.centroid[0];
    lon = change.centroid[1];
  } else if (Array.isArray(change.center) && change.center.length === 2 && typeof change.center[0] === 'number') {
    lat = change.center[0];
    lon = change.center[1];
  } else if (Array.isArray(change.centroid_lonlat) && change.centroid_lonlat.length === 2 && typeof change.centroid_lonlat[0] === 'number') {
    lon = change.centroid_lonlat[0];
    lat = change.centroid_lonlat[1];
  } else if (change.geometry && change.geometry.coordinates && change.geometry.coordinates[0]) {
    const coordsArr = change.geometry.coordinates[0];
    let sumLon = 0, sumLat = 0;
    coordsArr.forEach(([cLon, cLat]) => {
      sumLon += cLon;
      sumLat += cLat;
    });
    lon = sumLon / coordsArr.length;
    lat = sumLat / coordsArr.length;
  } else if (change.pixel_bbox) {
    const [minX, minY, maxX, maxY] = change.pixel_bbox;
    const b = locBounds || DEFAULT_BOUNDS;
    const latMin = b[0][0], lonMin = b[0][1], latMax = b[1][0], lonMax = b[1][1];
    const avgX = (minX + maxX) / 2 / DEMO_WIDTH;
    const avgY = (minY + maxY) / 2 / DEMO_HEIGHT;
    lon = lonMin + avgX * (lonMax - lonMin);
    lat = latMax - avgY * (latMax - latMin);
  } else {
    lat = 26.1624;
    lon = 91.7422;
  }

  const latStr = `${Math.abs(lat).toFixed(5)}° ${lat >= 0 ? 'N' : 'S'}`;
  const lonStr = `${Math.abs(lon).toFixed(5)}° ${lon >= 0 ? 'E' : 'W'}`;

  return { latStr, lonStr, latVal: lat, lonVal: lon };
}

const CONCEPT_DEMO_LOCATION = {
  location_id: CONCEPT_DEMO_ID,
  name: 'Concept Demo',
  category: 'CHANGE DETECTION DEMO',
  description: 'Lightweight synthetic dataset for demonstrating the change-detection workflow.',
  reference_scene: { date: 'DEMO BEFORE (2024-02-10)' },
  target_scene: { date: 'DEMO AFTER (2024-10-22)' },
};

function drawDemoScene(ctx, after = false) {
  ctx.clearRect(0, 0, DEMO_WIDTH, DEMO_HEIGHT);
  ctx.fillStyle = '#b8956f'; ctx.fillRect(0, 0, DEMO_WIDTH, DEMO_HEIGHT);
  ctx.fillStyle = '#3d633b'; ctx.fillRect(150, 0, 170, 240);
  ctx.fillStyle = '#2b4d29'; ctx.fillRect(20, 20, 90, 80);
  ctx.fillStyle = '#7a7672'; ctx.beginPath(); ctx.arc(70, 170, 35, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#1e3a5f'; ctx.beginPath(); ctx.moveTo(0, 110); ctx.bezierCurveTo(100, 100, 220, 140, 320, 120); ctx.lineTo(320, 150); ctx.bezierCurveTo(220, 170, 100, 130, 0, 140); ctx.closePath(); ctx.fill();

  if (after) {
    ctx.fillStyle = '#3a3a3a'; ctx.fillRect(200, 30, 80, 50);
    ctx.fillStyle = '#4a4a4a'; ctx.fillRect(215, 40, 50, 30);
    ctx.fillStyle = '#6b7280'; ctx.beginPath(); ctx.moveTo(0, 80); ctx.lineTo(320, 80); ctx.lineWidth = 6; ctx.strokeStyle = '#4b5563'; ctx.stroke();
    ctx.fillStyle = '#1e3a5f'; ctx.beginPath(); ctx.arc(100, 190, 25, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#8c7e6c'; ctx.fillRect(160, 140, 70, 60);
  }
}

function DemoCanvas({ mode, regions = [], selectedId, onSelect, showMask = true, appliedCategories = [] }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = DEMO_WIDTH;
    canvas.height = DEMO_HEIGHT;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (mode === 'before') {
      drawDemoScene(ctx, false);
    } else if (mode === 'after') {
      drawDemoScene(ctx, true);
    } else if (mode === 'change') {
      drawDemoScene(ctx, true);
      const activeRegions = regions.filter((r) => {
        if (!appliedCategories || appliedCategories.length === 0) return false;
        if (appliedCategories.includes('all')) return true;
        const type = (r.type || '').toUpperCase();
        if (appliedCategories.includes('roads') && (type.includes('ROAD') || type.includes('INFRASTRUCTURE'))) return true;
        if (appliedCategories.includes('construction') && (type.includes('CONSTRUCTION') || type.includes('BUILDING'))) return true;
        if (appliedCategories.includes('water') && (type.includes('WATER') || type.includes('RIVER'))) return true;
        if (appliedCategories.includes('vegetation') && (type.includes('VEGETATION') || type.includes('FOREST'))) return true;
        return false;
      });

      if (activeRegions.length > 0) {
        ctx.strokeStyle = '#ff3333';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);

        activeRegions.forEach((region) => {
          const isSelected = region.id === selectedId;
          ctx.strokeStyle = isSelected ? '#00ffcc' : (CATEGORY_COLORS[region.type] || '#ff3333');
          ctx.lineWidth = isSelected ? 3 : 2;
          ctx.fillStyle = region.id === selectedId ? 'rgba(255, 255, 255, 0.15)' : 'rgba(255, 255, 255, 0.03)';
          const [minX, minY, maxX, maxY] = region.pixel_bbox || [0,0,10,10];
          ctx.fillRect(minX, minY, maxX - minX, maxY - minY);
          ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
        });
      }
    }
  }, [mode, regions, selectedId, showMask, appliedCategories]);

  return <canvas ref={canvasRef} className="demo-scene-canvas" onClick={() => { if (mode === 'change' && onSelect && regions.length > 0) onSelect(regions[0]); }} />;
}

/* ──────────────────────────────────────────────
   MAIN APPLICATION CONTENT
─────────────────────────────────────────────── */
function AppContent() {
  const [viewMode, setViewMode] = useState('landing');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [theme, setTheme] = useState('space');
  const [showRawClassification, setShowRawClassification] = useState(false);
  
  // Analyst decisions state ({ [changeId]: 'confirmed' | 'rejected' })
  const [analystDecisions, setAnalystDecisions] = useState({});
  const [isProvenanceExpanded, setIsProvenanceExpanded] = useState(false);

  const [locations, setLocations] = useState([]);
  const [selectedLocId, setSelectedLocId] = useState(''); // empty = no AOI selected initially
  const [locationQuery, setLocationQuery] = useState('');
  const [locationDropdownOpen, setLocationDropdownOpen] = useState(false);
  const locationDropdownRef = useRef(null);
  
  // Multi-Select Change Detection Overlay state
  const [selectedCategories, setSelectedCategories] = useState([]); // checked in popover
  const [appliedCategories, setAppliedCategories] = useState([]);   // active on map (default [] clean After image)
  const [detectDropdownOpen, setDetectDropdownOpen] = useState(false);
  const detectDropdownRef = useRef(null);
  const [copiedCoord, setCopiedCoord] = useState(false);

  // User analysis execution state (default false on initial load)
  const [hasUserRunAnalysis, setHasUserRunAnalysis] = useState(false);

  const isCategoryChecked = (catId) => {
    if (catId === 'all') {
      return ALL_CAT_IDS.every((id) => selectedCategories.includes(id));
    }
    return selectedCategories.includes(catId);
  };

  const toggleCategory = (catId) => {
    if (catId === 'all') {
      if (isCategoryChecked('all')) {
        setSelectedCategories([]);
      } else {
        setSelectedCategories([...ALL_CAT_IDS, 'all']);
      }
      return;
    }

    setSelectedCategories((prev) => {
      let next;
      if (prev.includes(catId)) {
        next = prev.filter((c) => c !== catId && c !== 'all');
      } else {
        const added = [...prev, catId];
        if (ALL_CAT_IDS.every((id) => added.includes(id))) {
          next = [...added, 'all'];
        } else {
          next = added;
        }
      }
      return next;
    });
  };

  const handleApplyChanges = () => {
    setAppliedCategories([...selectedCategories]);
    setDetectDropdownOpen(false);
    if (selectedCategories.length > 0) {
      setHasUserRunAnalysis(true);
    }
  };

  const handleClearChanges = () => {
    setSelectedCategories([]);
    setAppliedCategories([]);
    setDetectDropdownOpen(false);
    setHasUserRunAnalysis(false);
    setSelectedChange(null);
  };
  
  const [scenes, setScenes] = useState([]);
  const [beforeSceneId, setBeforeSceneId] = useState('');
  const [afterSceneId, setAfterSceneId] = useState('');
  
  const [pipelineResult, setPipelineResult] = useState(null);
  const [changes, setChanges] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedChange, setSelectedChange] = useState(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scenesLoading, setScenesLoading] = useState(false);

  // Shared map view coordinates
  const [mapCenter, setMapCenter] = useState([26.1624, 91.7422]);
  const [mapZoom, setMapZoom] = useState(14);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (locationDropdownRef.current && !locationDropdownRef.current.contains(e.target)) {
        setLocationDropdownOpen(false);
      }
      if (detectDropdownRef.current && !detectDropdownRef.current.contains(e.target)) {
        setDetectDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Fetch locations on mount without auto-selecting any location
  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      const locRes = await fetch(`${API_BASE}/locations`);
      if (locRes.ok) {
        const locData = await locRes.json();
        if (Array.isArray(locData)) {
          setLocations(locData);
        }
      }
    } catch (err) {
      console.error('Fetch initial data error:', err);
    }
  };

  // Filter 200+ raw pixel detections into meaningful regions
  const filterMeaningfulChanges = (changeList) => {
    if (!Array.isArray(changeList) || changeList.length === 0) return [];
    return changeList.filter((c) => {
      const area = c.area_sqm || c.area_pixels || 0;
      const conf = c.confidence || 0.8;
      return area >= 35 && conf >= 0.45;
    });
  };

  const applyLocation = async (loc) => {
    if (!loc) return;
    setScenesLoading(true);
    setSelectedLocId(loc.location_id);
    setSelectedChange(null);
    setSelectedCategories([]);
    setAppliedCategories([]);
    setHasUserRunAnalysis(false);
    setPipelineResult(null);
    setSearchQuery('');
    setChanges([]);
    setSearchResults([]);
    
    if (loc.center && Array.isArray(loc.center) && loc.center.length === 2) {
      setMapCenter([loc.center[0], loc.center[1]]);
      setMapZoom(14);
    }
    
    try {
      const res = await fetch(`${API_BASE}/scenes?location=${loc.location_id}`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) {
          setScenes(data);
          if (data.length >= 2) {
            setBeforeSceneId(data[0].id || data[0].scene_id || '');
            setAfterSceneId(data[1].id || data[1].scene_id || '');
          } else if (data.length > 0) {
            setBeforeSceneId(data[0].id || data[0].scene_id || '');
            setAfterSceneId(data[0].id || data[0].scene_id || '');
          }
        }
      }

      const chRes = await fetch(`${API_BASE}/changes?location=${loc.location_id}`);
      if (chRes.ok) {
        const existingChanges = await chRes.json();
        const meaningful = filterMeaningfulChanges(existingChanges);
        if (meaningful.length > 0) {
          setChanges(meaningful);
          setSearchResults(meaningful);
        }
      }
    } catch (err) {
      console.error('Fetch location scenes error:', err);
    } finally {
      setScenesLoading(false);
    }
  };

  const handleLocationChange = (locId) => {
    setSelectedCategories([]);
    setAppliedCategories([]);
    setHasUserRunAnalysis(false);
    
    if (!locId) {
      setSelectedLocId('');
      setScenes([]);
      setBeforeSceneId('');
      setAfterSceneId('');
      return;
    }

    if (locId === CONCEPT_DEMO_ID) {
      setSelectedLocId(CONCEPT_DEMO_ID);
      setSelectedChange(null);
      setPipelineResult(null);
      setChanges([]);
      setSearchResults([]);
      setSearchQuery('');
      return;
    }

    setSelectedLocId(locId);
    const loc = Array.isArray(locations) ? locations.find(l => l && l.location_id === locId) : null;
    if (loc) {
      applyLocation(loc);
    }
  };

  const handleRunPipeline = async () => {
    if (!isConceptDemo && (!beforeSceneId || !afterSceneId)) return;
    setLoading(true);
    setError(null);
    setSelectedChange(null);
    setHasUserRunAnalysis(true);
    setSelectedCategories(['all']);
    setAppliedCategories(['all']);

    if (isConceptDemo) {
      window.setTimeout(() => {
        const demoResult = {
          aligned_target_url: '/api/placeholder/demo_aligned.png',
          change_mask_url: '/api/placeholder/demo_mask.png',
          changes_detected: [
            { id: 'c1', type: 'ROAD CHANGE', confidence: 0.94, pixel_bbox: [50, 75, 270, 85], is_demo: true, area_pixels: 220 },
            { id: 'c2', type: 'NEW CONSTRUCTION', confidence: 0.88, pixel_bbox: [200, 30, 280, 80], is_demo: true, area_pixels: 400 },
            { id: 'c3', type: 'BUILDING CHANGE', confidence: 0.91, pixel_bbox: [160, 140, 230, 200], is_demo: true, area_pixels: 420 },
            { id: 'c4', type: 'WATER EXTENT CHANGE', confidence: 0.82, pixel_bbox: [75, 165, 125, 215], is_demo: true, area_pixels: 250 },
          ]
        };
        setPipelineResult(demoResult);
        setChanges(demoResult.changes_detected);
        setSearchResults(demoResult.changes_detected);
        setLoading(false);
      }, 600);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/run-pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location_id: selectedLocId,
          reference_scene_id: beforeSceneId,
          target_scene_id: afterSceneId
        })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to execute AI pipeline');
      }

      const data = await res.json();
      setPipelineResult(data);
      
      const meaningful = filterMeaningfulChanges(data.changes_detected);
      setChanges(meaningful);
      setSearchResults(meaningful);
    } catch (err) {
      console.error('Pipeline error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSemanticSearch = async (e, overrideQuery) => {
    if (e && e.preventDefault) e.preventDefault();
    const query = (overrideQuery !== undefined ? overrideQuery : searchQuery).trim();
    if (!query) {
      setSearchResults(changes);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/search?query=${encodeURIComponent(query)}${selectedLocId ? `&location=${selectedLocId}` : ''}`, {
        method: 'POST'
      });
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          setSearchResults(data);
          return;
        }
      }
    } catch (err) {
      console.warn('Semantic search endpoint error, using fallback matching:', err);
    }

    // Natural language intent fallback matching
    const q = query.toLowerCase();
    const filtered = changes.filter((c) => {
      const type = (c.type || '').toLowerCase();
      const desc = (c.description || c.explanation || '').toLowerCase();
      
      const hasBuildIntent = q.includes('build') || q.includes('construct') || q.includes('structure') || q.includes('house');
      const hasRoadIntent = q.includes('road') || q.includes('highway') || q.includes('street') || q.includes('transport');
      const hasVegIntent = q.includes('veg') || q.includes('forest') || q.includes('tree') || q.includes('canopy') || q.includes('green');
      const hasWaterIntent = q.includes('water') || q.includes('river') || q.includes('lake') || q.includes('flood');

      if (hasBuildIntent && (type.includes('construction') || type.includes('building'))) return true;
      if (hasRoadIntent && type.includes('road')) return true;
      if (hasVegIntent && (type.includes('veg') || type.includes('forest'))) return true;
      if (hasWaterIntent && (type.includes('water') || type.includes('river'))) return true;

      return type.includes(q) || desc.includes(q);
    });

    setSearchResults(filtered);
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setSearchResults(changes);
  };

  const handleSelectChange = (change) => {
    setSelectedChange(change);
    if (change.center && Array.isArray(change.center) && change.center.length === 2) {
      setMapCenter([change.center[0], change.center[1]]);
      setMapZoom(16);
    }
  };

  const handleAnalystDecision = (changeId, decision) => {
    setAnalystDecisions((prev) => ({
      ...prev,
      [changeId]: prev[changeId] === decision ? null : decision,
    }));
  };

  const handleZoomToChange = () => {
    if (selectedChange && selectedChange.center) {
      setMapCenter([selectedChange.center[0], selectedChange.center[1]]);
      setMapZoom(17);
    }
  };

  const visibleOverlayChanges = searchResults.filter((r) => {
    if (!appliedCategories || appliedCategories.length === 0) return false;
    if (appliedCategories.includes('all')) return true;
    const type = (r.type || '').toUpperCase();
    if (appliedCategories.includes('roads') && (type.includes('ROAD') || type.includes('INFRASTRUCTURE'))) return true;
    if (appliedCategories.includes('construction') && (type.includes('CONSTRUCTION') || type.includes('BUILDING'))) return true;
    if (appliedCategories.includes('water') && (type.includes('WATER') || type.includes('RIVER'))) return true;
    if (appliedCategories.includes('vegetation') && (type.includes('VEGETATION') || type.includes('FOREST'))) return true;
    return false;
  });

  const handleExport = (format) => {
    window.open(`${API_BASE}/export?format=${format}&location=${selectedLocId}`, '_blank');
  };

  const getGeoJsonStyle = (changeItem) => {
    const isSelected = selectedChange && (
      (selectedChange.id && selectedChange.id === changeItem.id) ||
      (selectedChange.pixel_bbox && changeItem.pixel_bbox &&
       selectedChange.pixel_bbox.join() === changeItem.pixel_bbox.join())
    );
    
    const catColor = getCategoryColor(changeItem.type);
    const analystDecision = analystDecisions[changeItem.id];
    let strokeColor = catColor;
    
    if (analystDecision === 'confirmed') {
      strokeColor = '#10b981';
    } else if (analystDecision === 'rejected') {
      strokeColor = '#ef4444';
    }

    return {
      color: strokeColor,
      weight: isSelected ? 3.5 : 2.5,
      opacity: 1.0,
      dashArray: '6, 5',
      fillColor: strokeColor,
      fillOpacity: isSelected ? 0.2 : 0.05
    };
  };

  const hasLocationSelected = Boolean(selectedLocId);
  const isConceptDemo = selectedLocId === CONCEPT_DEMO_ID;
  const selectedLoc = isConceptDemo
    ? CONCEPT_DEMO_LOCATION
    : (Array.isArray(locations) ? locations.find(l => l && l.location_id === selectedLocId) : null) || null;
  const aoiLocations = [CONCEPT_DEMO_LOCATION, ...(Array.isArray(locations) ? locations : [])];
  
  const locationMatches = aoiLocations.filter((location) => {
    if (!location) return false;
    const name = location.name || location.location_id || '';
    const desc = location.description || '';
    const id = location.location_id || '';
    const query = (locationQuery || '').trim().toLowerCase();
    const searchable = `${name} ${desc} ${id}`.toLowerCase();
    return searchable.includes(query);
  });

  const beforeScene = Array.isArray(scenes) ? scenes.find(s => s && (s.id === beforeSceneId || s.scene_id === beforeSceneId)) : null;
  const afterScene = Array.isArray(scenes) ? scenes.find(s => s && (s.id === afterSceneId || s.scene_id === afterSceneId)) : null;
  const currentBounds = beforeScene?.bounds || selectedLoc?.leaflet_bounds || DEFAULT_BOUNDS;
  const beforeImgUrl = beforeScene?.image_url ? `${API_BASE}${beforeScene.image_url}` : null;
  const afterImgUrl = afterScene?.image_url ? `${API_BASE}${afterScene.image_url}` : null;
  const alignedImgUrl = pipelineResult ? `${API_BASE}${pipelineResult.aligned_target_url}` : null;

  if (viewMode === 'landing') {
    return (
      <>
        <div className="space-theme-starfield">
          <div className="starfield-stars" />
        </div>
        <LandingPage
          onLaunchWorkstation={() => setViewMode('workstation')}
          onSelectAOI={(locId) => {
            handleLocationChange(locId);
            setViewMode('workstation');
          }}
          locations={locations}
        />
      </>
    );
  }

  return (
    <>
      {/* Header */}
      <header>
        <div className="header-left">
          <button 
            className="back-to-home-btn"
            onClick={() => setViewMode('landing')}
            title="Return to Overview / Homepage"
          >
            <ArrowLeft size={14} />
            <span>Overview</span>
          </button>

          <button 
            className="menu-toggle-btn"
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            title="Toggle Control Panel"
          >
            {isSidebarOpen ? <X size={16} /> : <Menu size={16} />}
          </button>

          <div className="logo-section">
            <MoDLogo />
            <div>
              <span className="logo-title">DRISHTI</span>
              <span className="logo-subtitle" style={{ marginLeft: 8 }}>Satellite Intelligence Workstation</span>
            </div>
          </div>
        </div>

        <div className="header-right">
          <div className="health-pill">
            <span className="health-pulse-dot" />
            <span>SYSTEM ONLINE</span>
          </div>

          {(pipelineResult || changes.length > 0) && (
            <div className="export-btn-group">
              <button className="export-btn" onClick={() => handleExport('geojson')} title="Export GeoJSON">
                <Download size={12} />
                <span>GeoJSON</span>
              </button>
              <button className="export-btn" onClick={() => handleExport('csv')} title="Export CSV">
                <FileSpreadsheet size={12} />
                <span>CSV</span>
              </button>
              <button className="export-btn" onClick={() => handleExport('json')} title="Export Report">
                <FileJson size={12} />
                <span>Report</span>
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Workstation Layout */}
      <div className="workstation-flex-layout">
        
        {/* Sidebar */}
        <div className={`sidebar-flex-panel ${!isSidebarOpen ? 'collapsed' : ''}`}>
          <div className="sidebar-header">
            <span className="sidebar-title">
              <Search size={14} />
              <span>SEARCH & CONTROL</span>
            </span>
            <button className="menu-toggle-btn" style={{ width: 24, height: 24 }} onClick={() => setIsSidebarOpen(false)}>
              <X size={13} />
            </button>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-section-title">Filters</div>

            {/* Location Selector */}
            <div className="filter-group">
              <label className="filter-label">Target Location / AOI</label>
              <div className="location-select-container" ref={locationDropdownRef}>
                <div
                  className="location-search-container"
                  onClick={() => setLocationDropdownOpen(true)}
                >
                  <Search size={12} />
                  <input
                    value={locationQuery}
                    onChange={(event) => {
                      setLocationQuery(event.target.value);
                      setLocationDropdownOpen(true);
                    }}
                    onFocus={() => setLocationDropdownOpen(true)}
                    placeholder="Find location..."
                  />
                  <ChevronDown size={12} />
                </div>

                {locationDropdownOpen && (
                  <div className="location-dropdown">
                    {locationMatches.length === 0 ? (
                      <div style={{ padding: 8, fontSize: '0.72rem', color: '#666' }}>No location found.</div>
                    ) : (
                      locationMatches.map(loc => loc && (
                        <div
                          key={loc.location_id}
                          className={`location-dropdown-item ${selectedLocId === loc.location_id ? 'active' : ''}`}
                          onClick={() => {
                            handleLocationChange(loc.location_id);
                            setLocationQuery('');
                            setLocationDropdownOpen(false);
                          }}
                        >
                          <MapPin size={13} style={{ color: selectedLocId === loc.location_id ? 'var(--accent-cyan)' : 'var(--text-muted)', flexShrink: 0 }} />
                          <div>
                            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#f5f5f5' }}>{loc.name || loc.location_id}</div>
                            <div style={{ fontSize: '0.62rem', color: '#a0a0a0' }}>{loc.category || 'Observation Zone'}</div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Reference Date (T1) */}
            <div className="filter-group">
              <label className="filter-label">Reference Date (T₁ Before)</label>
              <select 
                className="select-box"
                value={beforeSceneId}
                onChange={(e) => setBeforeSceneId(e.target.value)}
                disabled={isConceptDemo || !hasLocationSelected}
              >
                {isConceptDemo ? (
                  <option>2024-02-10 (Demo)</option>
                ) : Array.isArray(scenes) && scenes.length > 0 ? (
                  scenes.map(s => <option key={s.id || s.scene_id} value={s.id || s.scene_id}>{s.date}</option>)
                ) : (
                  <option>Select AOI first</option>
                )}
              </select>
            </div>

            {/* Target Date (T2) */}
            <div className="filter-group">
              <label className="filter-label">Target Date (T₂ After)</label>
              <select 
                className="select-box"
                value={afterSceneId}
                onChange={(e) => setAfterSceneId(e.target.value)}
                disabled={isConceptDemo || !hasLocationSelected}
              >
                {isConceptDemo ? (
                  <option>2024-10-22 (Demo)</option>
                ) : Array.isArray(scenes) && scenes.length > 0 ? (
                  scenes.map(s => <option key={s.id || s.scene_id} value={s.id || s.scene_id}>{s.date}</option>)
                ) : (
                  <option>Select AOI first</option>
                )}
              </select>
            </div>

            {/* Search & Run Analytics Button */}
            <button 
              className="run-analytics-btn"
              onClick={handleRunPipeline}
              disabled={loading || (!hasLocationSelected && !isConceptDemo)}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 12, height: 12, marginBottom: 0, borderTopColor: '#0d0d0d' }} />
                  <span>Processing AI Pipeline...</span>
                </>
              ) : (
                <>
                  <Eye size={13} />
                  <span>SEARCH & RUN ANALYTICS</span>
                </>
              )}
            </button>
          </div>

          {/* Semantic Intelligence Search Section */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">
              SEMANTIC INTELLIGENCE SEARCH
            </div>
            
            <form onSubmit={handleSemanticSearch} className="search-input-box" style={{ marginBottom: 8 }}>
              <Search size={13} style={{ color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Search e.g. new construction, road..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  if (!e.target.value.trim()) {
                    setSearchResults(changes);
                  }
                }}
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={handleClearSearch}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 2 }}
                >
                  <X size={12} />
                </button>
              )}
            </form>

            <div className="quick-pills-grid">
              {[
                { label: 'New Construction', q: 'New Construction' },
                { label: 'Road Change', q: 'Road Change' },
                { label: 'Vegetation Loss', q: 'Vegetation Loss' },
                { label: 'Water Body', q: 'Water Extent' },
              ].map((pill) => (
                <button
                  key={pill.label}
                  type="button"
                  className={`quick-pill-btn ${searchQuery === pill.q ? 'active' : ''}`}
                  onClick={() => {
                    setSearchQuery(pill.q);
                    handleSemanticSearch(null, pill.q);
                  }}
                >
                  {pill.label}
                </button>
              ))}
            </div>
          </div>

          {/* Results Summary Section */}
          {!hasUserRunAnalysis ? (
            <div className="sidebar-section">
              <div className="sidebar-section-title">CHANGE RESULTS</div>
              <div style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-color)',
                padding: '12px',
                borderRadius: '6px',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  No analysis run yet.
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.4 }}>
                  Select a location to begin satellite analysis.
                </div>
              </div>
            </div>
          ) : searchResults.length > 0 ? (
            <div className="sidebar-section">
              <div className="sidebar-section-title">
                Meaningful Change Regions ({searchResults.length})
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {searchResults.map((change, idx) => (
                  <div 
                    key={idx}
                    onClick={() => handleSelectChange(change)}
                    style={{
                      background: selectedChange === change ? 'rgba(255,255,255,0.12)' : 'var(--bg-surface)',
                      border: '1px solid var(--border-color)',
                      padding: '8px 10px',
                      borderRadius: 6,
                      cursor: 'pointer'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', fontWeight: 600 }}>
                      <span>{change.type}</span>
                      {analystDecisions[change.id] && (
                        <span style={{ color: analystDecisions[change.id] === 'confirmed' ? '#10b981' : '#ef4444', fontSize: '0.66rem', fontWeight: 700 }}>
                          {analystDecisions[change.id].toUpperCase()}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: '0.66rem', color: 'var(--text-secondary)', marginTop: 2 }}>
                      {change.is_demo ? `${change.area_pixels} demo px` : change.area_sqm ? `${change.area_sqm.toLocaleString()} m²` : `${change.area_pixels} px`}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="sidebar-section">
              <div className="sidebar-section-title">CHANGE RESULTS</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'center', padding: '10px 0' }}>
                No changes detected for the current criteria.
              </div>
            </div>
          )}
        </div>

        {/* 2-Pane Main Analysis Area */}
        <div className="analysis-main-area">
          
          {/* PANE 1: BEFORE (T1) */}
          <div className="satellite-pane">
            <div className="pane-header-bar">
              <div className="scene-label-badge">
                <span className="tag">BEFORE</span>
                <span>{isConceptDemo ? '2024-02-10' : beforeScene ? beforeScene.date : (hasLocationSelected ? '—' : 'Select AOI')}</span>
              </div>
            </div>

            <div className="map-container-box">
              {!hasLocationSelected ? (
                <div className="map-placeholder no-aoi-placeholder">
                  <Crosshair size={28} style={{ color: '#404040', marginBottom: 12 }} />
                  <span className="placeholder-title">Select a location to load imagery</span>
                  <span className="placeholder-sub">Use the Search &amp; Control panel to select a Target Location / AOI</span>
                </div>
              ) : isConceptDemo ? (
                <DemoCanvas mode="before" />
              ) : scenesLoading ? (
                <div className="map-placeholder">
                  <span className="spinner" />
                  <span>LOADING SATELLITE IMAGERY...</span>
                </div>
              ) : beforeImgUrl ? (
                <MapContainer 
                  center={mapCenter} 
                  zoom={mapZoom} 
                  minZoom={12}
                  maxZoom={18}
                  zoomControl={false} 
                  className="demo-scene-canvas"
                >
                  <ImageOverlay url={beforeImgUrl} bounds={currentBounds} />
                  <MapBoundsFitter bounds={currentBounds} />
                  <MapResizer isSidebarOpen={isSidebarOpen} selectedChange={selectedChange} />
                  <MapSynchronizer center={mapCenter} zoom={mapZoom} onMapMoved={(c, z) => { setMapCenter(c); setMapZoom(z); }} />
                  <ZoomControl position="bottomright" />
                </MapContainer>
              ) : (
                <div className="map-placeholder">
                  <AlertCircle size={16} style={{ color: '#38bdf8', marginBottom: 6 }} />
                  <span>Imagery unavailable for this selection</span>
                </div>
              )}
            </div>
          </div>

          {/* PANE 2: AFTER (T2) */}
          <div className="satellite-pane">
            <div className="pane-header-bar">
              <div className="scene-label-badge">
                <span className="tag">AFTER</span>
                <span>{isConceptDemo ? '2024-10-22' : afterScene ? afterScene.date : (hasLocationSelected ? '—' : 'Select AOI')}</span>
              </div>

              {/* Multi-Select Control Popover */}
              <div className="detect-changes-popover-container" ref={detectDropdownRef}>
                <div className="detect-controls-row">
                  <button 
                    className={`detect-changes-trigger-btn ${appliedCategories.length > 0 ? 'active' : ''}`}
                    onClick={() => setDetectDropdownOpen(!detectDropdownOpen)}
                  >
                    <Layers size={13} />
                    <span>
                      {appliedCategories.length === 0
                        ? 'DETECT CHANGES ▾'
                        : appliedCategories.includes('all') || ALL_CAT_IDS.every(id => appliedCategories.includes(id))
                        ? 'ALL CHANGES ACTIVE ▾'
                        : `${appliedCategories.length} CATEGORIES ACTIVE ▾`}
                    </span>
                    <ChevronDown size={12} className={`dropdown-chevron ${detectDropdownOpen ? 'open' : ''}`} />
                  </button>

                  {appliedCategories.length > 0 && (
                    <button 
                      className="clear-overlay-btn"
                      onClick={handleClearChanges}
                      title="Clear Analysis Overlay"
                    >
                      <X size={12} />
                      <span>CLEAR</span>
                    </button>
                  )}
                </div>

                {detectDropdownOpen && (
                  <div className="detect-changes-dropdown">
                    <div className="detect-dropdown-header">DETECT CHANGES</div>
                    <div className="detect-dropdown-options">
                      {[
                        { id: 'roads', label: 'Roads', color: '#ffffff' },
                        { id: 'construction', label: 'New Construction', color: '#d97706' },
                        { id: 'water', label: 'Water Body', color: '#1d4ed8' },
                        { id: 'vegetation', label: 'Vegetation Loss', color: '#15803d' },
                        { id: 'all', label: 'All Changes', color: '#ffffff' },
                      ].map((opt) => {
                        const checked = isCategoryChecked(opt.id);
                        return (
                          <div
                            key={opt.id}
                            className={`detect-dropdown-option ${checked ? 'selected' : ''}`}
                            onClick={() => toggleCategory(opt.id)}
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {}}
                              className="detect-checkbox"
                            />
                            <span 
                              className="color-indicator-dot" 
                              style={{ 
                                backgroundColor: opt.color, 
                                boxShadow: opt.id === 'all' ? '0 0 6px rgba(255,255,255,0.8)' : `0 0 6px ${opt.color}` 
                              }} 
                            />
                            <span className="option-label-text">{opt.label}</span>
                          </div>
                        );
                      })}
                    </div>
                    <div className="detect-dropdown-actions">
                      <button className="apply-changes-btn" onClick={handleApplyChanges}>
                        APPLY CHANGES
                      </button>
                      <button className="clear-popover-btn" onClick={handleClearChanges}>
                        CLEAR
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="map-container-box">
              {!hasLocationSelected ? (
                <div className="map-placeholder no-aoi-placeholder">
                  <Crosshair size={28} style={{ color: '#404040', marginBottom: 12 }} />
                  <span className="placeholder-title">Select a location to load imagery</span>
                  <span className="placeholder-sub">Use the Search &amp; Control panel to select a Target Location / AOI</span>
                </div>
              ) : isConceptDemo ? (
                <DemoCanvas 
                  mode="change" 
                  regions={searchResults} 
                  selectedId={selectedChange?.id} 
                  onSelect={handleSelectChange} 
                  showMask={false} 
                  appliedCategories={appliedCategories}
                />
              ) : scenesLoading ? (
                <div className="map-placeholder">
                  <span className="spinner" />
                  <span>LOADING SATELLITE IMAGERY...</span>
                </div>
              ) : (alignedImgUrl || afterImgUrl) ? (
                <MapContainer 
                  center={mapCenter} 
                  zoom={mapZoom} 
                  minZoom={12}
                  maxZoom={18}
                  zoomControl={false} 
                  className="demo-scene-canvas"
                >
                  <ImageOverlay url={alignedImgUrl || afterImgUrl || beforeImgUrl} bounds={currentBounds} opacity={1.0} />
                  
                  {visibleOverlayChanges.map((change, idx) => (
                    <GeoJSON 
                      key={`${idx}-${appliedCategories.join('-')}-${selectedChange === change}-${analystDecisions[change.id]}`}
                      data={change.geometry}
                      style={() => getGeoJsonStyle(change)}
                      eventHandlers={{ click: () => handleSelectChange(change) }}
                    />
                  ))}
                  
                  <MapBoundsFitter bounds={currentBounds} />
                  <MapResizer isSidebarOpen={isSidebarOpen} selectedChange={selectedChange} />
                  <MapSynchronizer center={mapCenter} zoom={mapZoom} onMapMoved={(c, z) => { setMapCenter(c); setMapZoom(z); }} />
                  <ZoomControl position="bottomright" />
                </MapContainer>
              ) : (
                <div className="map-placeholder">
                  <AlertCircle size={16} style={{ color: '#38bdf8', marginBottom: 6 }} />
                  <span>Imagery unavailable for this selection</span>
                </div>
              )}
            </div>

            {/* Dedicated Legend Strip */}
            {appliedCategories.length > 0 && (
              <div className="pane-footer-legend-bar">
                <span className="legend-bar-title">LEGEND:</span>
                <div className="legend-bar-items">
                  {(appliedCategories.includes('all') || appliedCategories.includes('roads')) && (
                    <div className="legend-bar-item">
                      <span className="legend-dash-line" style={{ borderTopColor: '#ffffff' }} />
                      <span>Road</span>
                    </div>
                  )}
                  {(appliedCategories.includes('all') || appliedCategories.includes('water')) && (
                    <div className="legend-bar-item">
                      <span className="legend-dash-line" style={{ borderTopColor: '#1d4ed8' }} />
                      <span>Water</span>
                    </div>
                  )}
                  {(appliedCategories.includes('all') || appliedCategories.includes('vegetation')) && (
                    <div className="legend-bar-item">
                      <span className="legend-dash-line" style={{ borderTopColor: '#15803d' }} />
                      <span>Vegetation</span>
                    </div>
                  )}
                  {(appliedCategories.includes('all') || appliedCategories.includes('construction')) && (
                    <div className="legend-bar-item">
                      <span className="legend-dash-line" style={{ borderTopColor: '#d97706' }} />
                      <span>Construction</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Change Details Panel */}
        {selectedChange && (
          <div className="right-details-flex-panel">
            <div className="details-panel-header">
              <span className="details-panel-title">
                <Info size={13} />
                <span>FEATURE EVIDENCE DOSSIER</span>
              </span>
              <button 
                className="close-details-btn" 
                onClick={() => setSelectedChange(null)}
                title="Close dossier"
              >
                <X size={13} />
              </button>
            </div>

            <div className="details-panel-body">
              <div className="dossier-header-strip">
                <div className="dossier-type-title">{selectedChange.type}</div>
                <div className="dossier-id-badge">ID: {selectedChange.id || 'EV-DET-01'}</div>
              </div>

              <div className="dossier-grid">
                <div className="dossier-metric-card">
                  <span className="metric-label">SURFACE AREA</span>
                  <span className="metric-value">
                    {selectedChange.is_demo 
                      ? `${selectedChange.area_pixels} px` 
                      : selectedChange.area_sqm 
                      ? `${selectedChange.area_sqm.toLocaleString()} m²` 
                      : `${selectedChange.area_pixels || 450} px`}
                  </span>
                </div>
                <div className="dossier-metric-card">
                  <span className="metric-label">STATUS</span>
                  <span className="metric-value" style={{ 
                    color: analystDecisions[selectedChange.id] === 'confirmed' ? '#10b981' : analystDecisions[selectedChange.id] === 'rejected' ? '#ef4444' : '#38bdf8',
                    fontSize: '0.78rem' 
                  }}>
                    {analystDecisions[selectedChange.id] ? analystDecisions[selectedChange.id].toUpperCase() : 'PENDING REVIEW'}
                  </span>
                </div>
              </div>

              {/* LATITUDE & LONGITUDE Section */}
              {(() => {
                const coords = getChangeCoordinates(selectedChange, currentBounds);
                return (
                  <>
                    <div className="dossier-grid" style={{ marginTop: 2 }}>
                      <div className="dossier-metric-card">
                        <span className="metric-label">LATITUDE</span>
                        <span className="metric-value cyan" style={{ fontSize: '0.82rem', fontFamily: 'monospace' }}>
                          {coords.latStr}
                        </span>
                      </div>
                      <div className="dossier-metric-card">
                        <span className="metric-label">LONGITUDE</span>
                        <span className="metric-value cyan" style={{ fontSize: '0.82rem', fontFamily: 'monospace' }}>
                          {coords.lonStr}
                        </span>
                      </div>
                    </div>

                    <button 
                      className="copy-coords-btn"
                      style={{ marginTop: 2 }}
                      onClick={() => {
                        if (coords.latVal !== null && coords.lonVal !== null) {
                          navigator.clipboard.writeText(`${coords.latVal.toFixed(6)}, ${coords.lonVal.toFixed(6)}`);
                        }
                        setCopiedCoord(true);
                        setTimeout(() => setCopiedCoord(false), 2000);
                      }}
                    >
                      {copiedCoord ? <CheckCheck size={12} style={{ color: '#10b981' }} /> : <Copy size={12} />}
                      <span>{copiedCoord ? 'COORDINATES COPIED' : 'COPY LATITUDE & LONGITUDE'}</span>
                    </button>
                  </>
                );
              })()}

              {(selectedChange.explanation || selectedChange.description) && (
                <div style={{
                  padding: '8px 10px',
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 6,
                  fontSize: '0.72rem',
                  color: 'var(--text-secondary)',
                  fontStyle: 'italic',
                  lineHeight: 1.4
                }}>
                  "{selectedChange.explanation || selectedChange.description}"
                </div>
              )}

              <div className="analyst-action-strip">
                <span className="analyst-label font-mono">ANALYST DECISION:</span>
                <div className="analyst-btn-row">
                  <button 
                    className={`analyst-btn confirm ${analystDecisions[selectedChange.id] === 'confirmed' ? 'active' : ''}`}
                    onClick={() => handleAnalystDecision(selectedChange.id, 'confirmed')}
                  >
                    <CheckCircle2 size={12} />
                    <span>CONFIRM</span>
                  </button>
                  <button 
                    className={`analyst-btn reject ${analystDecisions[selectedChange.id] === 'rejected' ? 'active' : ''}`}
                    onClick={() => handleAnalystDecision(selectedChange.id, 'rejected')}
                  >
                    <Ban size={12} />
                    <span>REJECT</span>
                  </button>
                </div>
              </div>

              <div className="card-actions-row">
                {!selectedChange.is_demo && (
                  <button className="zoom-btn-action" onClick={handleZoomToChange}>
                    <ZoomIn size={12} />
                    <span>Zoom to Change</span>
                  </button>
                )}
                <button className="dismiss-btn-action" onClick={() => setSelectedChange(null)}>
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </>
  );
}

/* ──────────────────────────────────────────────
   EXPORTED WORKSTATION APP WITH ERROR BOUNDARY
─────────────────────────────────────────────── */
export default function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  );
}
