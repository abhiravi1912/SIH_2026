# 🌐 DRISHTI WebGIS Client — Frontend Workstation

This directory contains the user interface and interactive GIS workstation for **DRISHTI: AI-Powered Multi-Temporal Satellite Change Intelligence**.

Built with **React 19**, **Vite**, **Leaflet / React-Leaflet**, and **Lucide Icons**, the frontend operates as a tactical command dashboard supporting real-time multi-temporal satellite comparison, AI land-cover segmentation overlay, and spatial vector inspection.

---

## 🚀 Key Interface Features

### 1. Interactive 3D Earth & Hero Landing (`LandingPage.jsx`)
- **Starfield & Shooting Star Canvas**: Interactive, physics-based starry night sky canvas animation.
- **Custom WebGL 3D Earth Globe**: Fully rendered 3D rotating globe highlighting strategic Indian Earth Observation regions (Assam corridor, Amaravati, etc.).
- **Strategic AOI Quick-Launch Cards**: Direct entry points into pre-configured demonstration locations with observation dates and change statistics.
- **Capability Matrix**: Quick walkthrough of sub-pixel alignment, semantic segmentation, false-change suppression, and tactical reporting.

### 2. Dual-Pane Synchronized Leaflet Workstation (`App.jsx`)
- **Multi-Temporal Synchronization**: Side-by-side reference (baseline) vs. target (recent) view with synchronized pan and zoom.
- **Interactive Swipe Slider**: Smooth split-screen curtain allowing analysts to peel back the recent satellite pass over historical imagery.
- **Multi-Layer Controls**:
  - Raw True-Color RGB Satellite Ingestion Overlay
  - AI Land-Cover Classification Mask (Vegetation, Water, Built-up, Roads, Bare Soil)
  - Color-Coded Vector Change Overlays (New Construction, Forest Loss, River Migration, Road Change)
- **Vector Inspection Drawer**:
  - Click on any detected change polygon to inspect area in $\text{m}^2$, distance to transport routes, water proximity, and transition explanation.
  - Confidence scoring and analyst verification workflow (Approve / Reject classification).
- **Natural Language Spatial Search**:
  - Type plain English queries such as *"new construction near roads"* or *"deforestation"* to filter and spotlight matching polygons on the map in real time.
- **Tactical Dossier Export**:
  - One-click export to standard **GeoJSON**, **CSV**, or **JSON** for mission briefing and downstream GIS systems (QGIS, ArcGIS).

---

## 🛠️ Project Structure

```
frontend/
├── index.html            # Web application entry point
├── package.json          # Node dependencies and build scripts
├── vite.config.js        # Vite build and plugin configuration
├── public/               # Public assets, icons and textures
└── src/
    ├── main.jsx          # React DOM root entry
    ├── App.jsx           # Main WebGIS workstation, map panes, and controls
    ├── LandingPage.jsx   # 3D WebGL Earth & Starfield landing page
    ├── App.css           # Workstation layout, split panes & slider styling
    └── index.css         # Global design system, glassmorphism, buttons & themes
```

---

## 💻 Development & Scripts

### Prerequisites
- Node.js v18+ and npm

### Installation
```bash
cd frontend
npm install
```

### Run Local Development Server
```bash
npm run dev
```
The application will launch at `http://localhost:5173`.

### Production Build
```bash
npm run build
```
Creates an optimized production bundle in `frontend/dist`.

### Linting
```bash
npm run lint
```
Uses fast `oxlint` rules to enforce clean code and prevent errors.

---

## 🔗 Backend Connection

The frontend connects to the FastAPI backend running locally at `http://127.0.0.1:8000`. Ensure the backend server is running before launching detection tasks:

```bash
# In project root:
python backend/main.py
```
