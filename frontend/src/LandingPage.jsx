import React, { useState, useEffect, useRef } from 'react';
import { 
  ArrowRight, 
  Compass, 
  Layers, 
  ShieldCheck, 
  Database, 
  FileText, 
  Activity, 
  MapPin, 
  Eye, 
  ExternalLink,
  Globe,
  Trees,
  Droplets,
  Building2,
  GraduationCap,
  Sparkles,
  Satellite,
  Cpu
} from 'lucide-react';

/* ──────────────────────────────────────────────
   Starfield Canvas (behind everything)
─────────────────────────────────────────────── */
function StarfieldCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let stars = [];
    let shootingStars = [];
    let w, h;

    function resize() {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    }

    function initStars() {
      stars = Array.from({ length: 280 }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 1.2 + 0.2,
        alpha: Math.random() * 0.7 + 0.2,
        speed: Math.random() * 0.005 + 0.002,
        phase: Math.random() * Math.PI * 2,
      }));
    }

    function spawnShootingStar() {
      shootingStars.push({
        x: Math.random() * w,
        y: Math.random() * h * 0.4,
        len: Math.random() * 140 + 60,
        speed: Math.random() * 12 + 8,
        alpha: 1,
        angle: Math.PI / 4 + (Math.random() - 0.5) * 0.3,
        life: 1,
      });
    }

    function draw(t) {
      ctx.clearRect(0, 0, w, h);

      // Stars
      stars.forEach(s => {
        const twinkle = s.alpha + Math.sin(t * s.speed + s.phase) * 0.25;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${Math.max(0.05, Math.min(1, twinkle))})`;
        ctx.fill();
      });

      // Shooting stars
      shootingStars = shootingStars.filter(ss => ss.life > 0);
      shootingStars.forEach(ss => {
        const grad = ctx.createLinearGradient(
          ss.x, ss.y,
          ss.x - Math.cos(ss.angle) * ss.len,
          ss.y - Math.sin(ss.angle) * ss.len
        );
        grad.addColorStop(0, `rgba(255,255,255,${ss.life})`);
        grad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.beginPath();
        ctx.moveTo(ss.x, ss.y);
        ctx.lineTo(ss.x - Math.cos(ss.angle) * ss.len, ss.y - Math.sin(ss.angle) * ss.len);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ss.x += Math.cos(ss.angle) * ss.speed;
        ss.y += Math.sin(ss.angle) * ss.speed;
        ss.life -= 0.018;
      });

      animId = requestAnimationFrame(draw);
    }

    resize();
    initStars();
    window.addEventListener('resize', () => { resize(); initStars(); });

    // Spawn shooting stars
    const shootInterval = setInterval(spawnShootingStar, 3400);
    animId = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animId);
      clearInterval(shootInterval);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 0,
        pointerEvents: 'none',
        background: 'radial-gradient(ellipse at 60% 40%, #060d1f 0%, #020509 100%)',
      }}
    />
  );
}

/* ──────────────────────────────────────────────
   MoD Logo
─────────────────────────────────────────────── */
function MoDLogoHero() {
  const [imgError, setImgError] = useState(false);
  return (
    <img
      src={imgError ? '/Ministry_of_Defence_India.png' : '/Ministry_of_Defence_India.svg'}
      alt="Ministry of Defence India"
      className="mod-header-logo-img"
      onError={() => setImgError(true)}
    />
  );
}

/* ──────────────────────────────────────────────
   WebGL Single 3D Rotating Earth Globe
─────────────────────────────────────────────── */
/* ──────────────────────────────────────────────
   WebGL Single 3D Rotating Earth Globe
─────────────────────────────────────────────── */
function Globe3DCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let animId;
    let gl;
    try {
      gl = canvas.getContext('webgl', { alpha: true, antialias: true }) ||
           canvas.getContext('experimental-webgl');
    } catch (e) {
      gl = null;
    }

    if (!gl) return;

    // Build un-distorted 3D sphere geometry
    const radius = 1.0;
    const latSegments = 64;
    const lonSegments = 64;

    const positions = [];
    const uvs = [];
    const indices = [];

    for (let lat = 0; lat <= latSegments; lat++) {
      const theta = (lat * Math.PI) / latSegments;
      const sinTheta = Math.sin(theta);
      const cosTheta = Math.cos(theta);

      for (let lon = 0; lon <= lonSegments; lon++) {
        const phi = (lon * 2 * Math.PI) / lonSegments;
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);

        const x = cosPhi * sinTheta;
        const y = cosTheta;
        const z = sinPhi * sinTheta;

        // Natural equirectangular UV mapping (un-mirrored, realistic continent proportions)
        const u = 1.0 - (lon / lonSegments);
        const v = lat / latSegments;

        positions.push(x * radius, y * radius, z * radius);
        uvs.push(u, v);
      }
    }

    for (let lat = 0; lat < latSegments; lat++) {
      for (let lon = 0; lon < lonSegments; lon++) {
        const first = lat * (lonSegments + 1) + lon;
        const second = first + lonSegments + 1;

        indices.push(first, second, first + 1);
        indices.push(second, second + 1, first + 1);
      }
    }

    const vsSource = `
      attribute vec3 aPosition;
      attribute vec2 aUV;
      uniform float uAngle;
      varying vec2 vUV;
      varying vec3 vNormal;

      void main() {
        vUV = aUV;
        float cosA = cos(uAngle);
        float sinA = sin(uAngle);
        vec3 rotPos = vec3(
          aPosition.x * cosA + aPosition.z * sinA,
          aPosition.y,
          -aPosition.x * sinA + aPosition.z * cosA
        );
        float tilt = 0.35;
        float cosT = cos(tilt);
        float sinT = sin(tilt);
        vec3 finalPos = vec3(
          rotPos.x,
          rotPos.y * cosT - rotPos.z * sinT,
          rotPos.y * sinT + rotPos.z * cosT
        );
        vNormal = finalPos;
        // Perfect 1:1 spherical proportion
        gl_Position = vec4(finalPos.x * 0.88, finalPos.y * 0.88, finalPos.z * 0.001, 1.0);
      }
    `;

    const fsSource = `
      precision mediump float;
      varying vec2 vUV;
      varying vec3 vNormal;
      uniform sampler2D uTexture;
      uniform float uHasTexture;

      void main() {
        vec3 norm = normalize(vNormal);
        // Directional space sunlight coming from top-right
        vec3 lightDir = normalize(vec3(0.5, 0.4, 0.8));
        float diff = max(dot(norm, lightDir), 0.0);
        float ambient = 0.50;
        
        // Soft cyan/blue atmosphere rim lighting
        float rim = pow(1.0 - max(dot(norm, vec3(0.0, 0.0, 1.0)), 0.0), 3.0) * 0.45;
        vec3 rimColor = vec3(0.22, 0.70, 1.0) * rim;

        vec4 texColor = vec4(0.10, 0.25, 0.45, 1.0);
        if (uHasTexture > 0.5) {
          texColor = texture2D(uTexture, vUV);
        }

        vec3 baseColor = texColor.rgb * (diff * 0.65 + ambient) + rimColor;
        gl_FragColor = vec4(baseColor, texColor.a);
      }
    `;

    const createShader = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertShader = createShader(gl.VERTEX_SHADER, vsSource);
    const fragShader = createShader(gl.FRAGMENT_SHADER, fsSource);
    if (!vertShader || !fragShader) return;

    const program = gl.createProgram();
    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    const uvBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uvs), gl.STATIC_DRAW);

    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

    const aPosLoc = gl.getAttribLocation(program, 'aPosition');
    const aUVLoc = gl.getAttribLocation(program, 'aUV');
    const uAngleLoc = gl.getUniformLocation(program, 'uAngle');
    const uHasTextureLoc = gl.getUniformLocation(program, 'uHasTexture');
    const uTextureLoc = gl.getUniformLocation(program, 'uTexture');

    gl.enableVertexAttribArray(aPosLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.vertexAttribPointer(aPosLoc, 3, gl.FLOAT, false, 0, 0);

    gl.enableVertexAttribArray(aUVLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
    gl.vertexAttribPointer(aUVLoc, 2, gl.FLOAT, false, 0, 0);

    const texture = gl.createTexture();
    const img = new Image();
    let isTexLoaded = false;
    img.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      isTexLoaded = true;
    };
    img.src = '/earth_map.png';

    gl.enable(gl.DEPTH_TEST);

    let angle = 0;
    let lastTime = performance.now();
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const render = (now) => {
      const dt = (now - lastTime) / 1000;
      lastTime = now;

      if (!prefersReducedMotion) {
        // Slow, smooth, subtle, and professional rotation speed
        angle += dt * 0.025;
      }

      const dw = canvas.clientWidth || 540;
      const dh = canvas.clientHeight || 540;
      if (canvas.width !== dw || canvas.height !== dh) {
        canvas.width = dw;
        canvas.height = dh;
        gl.viewport(0, 0, dw, dh);
      }

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      gl.useProgram(program);
      gl.uniform1f(uAngleLoc, angle);
      gl.uniform1f(uHasTextureLoc, isTexLoaded ? 1.0 : 0.0);

      if (isTexLoaded) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(uTextureLoc, 0);
      }

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
      gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animId);
    };
  }, []);

  return (
    <div className="lp-earth-wrapper">
      <canvas ref={canvasRef} className="lp-earth-webgl-canvas" />
      <div className="lp-earth-glow" />
    </div>
  );
}

/* ──────────────────────────────────────────────
   LANDING PAGE
─────────────────────────────────────────────── */
export default function LandingPage({ onLaunchWorkstation, onSelectAOI, locations = [] }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const container = document.getElementById('landing-scroll-root');
    if (!container) return;
    const handler = () => {
      setScrolled(container.scrollTop > 60);
    };
    container.addEventListener('scroll', handler);
    return () => container.removeEventListener('scroll', handler);
  }, []);

  const aoiCards = [
    { id: 'mixed', icon: <Globe size={18} style={{ color: '#38bdf8' }} />, cat: 'MIXED LANDSCAPE', title: 'Guwahati Urban & River Basin', desc: 'Multi-category change monitoring across riverbanks, urban growth, and green cover.', stats: '14 Change Events', range: '2024-03-11 → 2026-03-06' },
    { id: 'forest', icon: <Trees size={18} style={{ color: '#4ade80' }} />, cat: 'VEGETATION LOSS', title: 'Kaziranga Reserve Forest', desc: 'Illegal timber clearing and forest boundary encroachment detection.', stats: '8 Clearing Polygons', range: '2024-02-10 → 2026-03-06' },
    { id: 'river', icon: <Droplets size={18} style={{ color: '#60a5fa' }} />, cat: 'WATER EXTENT SHIFT', title: 'Majuli Brahmaputra River', desc: 'Riverbank erosion, channel migration, and flood extent monitoring.', stats: '6 Water Shift Zones', range: '2023-01-21 → 2026-03-06' },
    { id: 'urban', icon: <Building2 size={18} style={{ color: '#f59e0b' }} />, cat: 'NEW CONSTRUCTION', title: 'Guwahati Urban Expansion', desc: 'Rapid building construction, industrial expansion, and road development.', stats: '11 Infrastructure Polygons', range: '2024-02-10 → 2026-03-06' },
    { id: 'vit_ap', icon: <GraduationCap size={18} style={{ color: '#a78bfa' }} />, cat: 'INFRASTRUCTURE', title: 'VIT-AP Campus AOI', desc: 'High-resolution multi-temporal analysis of university campus growth.', stats: '7 Construction Changes', range: '2021-03-06 → 2026-03-05' },
    { id: 'concept_demo', icon: <Sparkles size={18} style={{ color: '#f43f5e' }} />, cat: 'CONCEPT DEMO', title: 'Synthetic Benchmark', desc: 'Lightweight synthetic benchmark for change detection algorithms.', stats: '4 Synthetic Changes', range: '2024-02-10 → 2024-10-22' },
  ];

  const caps = [
    { icon: <Compass size={20} />, title: 'Geospatial Alignment', desc: 'ORB feature homography for automatic sub-pixel satellite pass alignment without manual GCPs.', tag: 'ORB HOMOGRAPHY' },
    { icon: <Layers size={20} />, title: 'Semantic Segmentation', desc: 'Multi-class CNN segmentation identifying Vegetation, Water, Built-up, Roads, and Bare Soil.', tag: 'DEEP CNN' },
    { icon: <ShieldCheck size={20} />, title: 'False-Change Suppression', desc: 'Cloud shadow, atmospheric noise, and seasonal variance filtering from tactical alerts.', tag: 'NOISE FILTER' },
    { icon: <Database size={20} />, title: 'Satellite Archive', desc: 'Pre-staged Sentinel-2 imagery covering reserve forests, river zones, and urban expansions.', tag: 'SENTINEL-2 MSI' },
    { icon: <FileText size={20} />, title: 'Intelligence Reports', desc: 'One-click export of tactical dossiers in GeoJSON, CSV, and structured JSON formats.', tag: 'GEOJSON / CSV' },
    { icon: <Activity size={20} />, title: 'Analyst Verification', desc: 'Interactive review workflow for intelligence analysts to confirm or reject detected changes.', tag: 'HUMAN-IN-LOOP' },
  ];

  return (
    <>
      <StarfieldCanvas />

      <div id="landing-scroll-root" className="lp-root">

        {/* ── NAV ── */}
        <nav className={`lp-nav ${scrolled ? 'lp-nav--scrolled' : ''}`}>
          <div className="lp-nav-brand">
            <MoDLogoHero />
            <span className="lp-brand-name">DRISHTI</span>
          </div>

          <div className="lp-nav-links">
            <a href="#mission">Our Mission</a>
            <a href="#capabilities">Labs &amp; Technology</a>
            <a href="#aois">Strategic AOIs</a>
            <a href="#research">Research</a>
          </div>

          <button className="lp-contact-btn" onClick={onLaunchWorkstation}>
            Get Started
          </button>
        </nav>

        {/* ── HERO ── */}
        <section className="lp-hero">
          {/* Single 3D Rotating Earth Globe */}
          <Globe3DCanvas />

          {/* Gradient overlay for contrast */}
          <div className="lp-hero-overlay" />

          <div className="lp-hero-content">
            <h1 className="lp-hero-title">
              Redefining Satellite<br />
              Intelligence for&nbsp;
              <span className="lp-title-accent">Defence</span>
            </h1>

            <p className="lp-hero-sub">
              Multi-temporal Sentinel-2 change detection powered by AI homography &amp; semantic segmentation. Detect ground truth changes in real time.
            </p>

            <div className="lp-hero-ctas">
              <button className="lp-btn-primary" onClick={onLaunchWorkstation}>
                Launch Workstation
                <ArrowRight size={16} />
              </button>
              <a className="lp-btn-ghost" href="#aois">
                Explore Strategic AOIs
              </a>
            </div>
          </div>
        </section>

        {/* ── PARTNERS TICKER ── */}
        <section className="lp-ticker-section">
          <p className="lp-ticker-label">Sentinel-2 MSI Level-2A &nbsp;•&nbsp; ISRO SatCat &nbsp;•&nbsp; Copernicus EMS &nbsp;•&nbsp; Ministry of Defence &nbsp;•&nbsp; ESA Open Data</p>
        </section>

        {/* ── STATS RIBBON ── */}
        <section className="lp-stats-section">
          <div className="lp-stats-grid">
            {[
              { val: '100%', label: 'Automated GeoAlignment', sub: 'ORB homography' },
              { val: '99.4%', label: 'False-Change Suppression', sub: 'Cloud mask + spectral norm' },
              { val: '6+', label: 'Strategic AOI Locations', sub: 'Forest, urban, river, infra' },
              { val: '< 1.2m', label: 'Spatial Precision', sub: 'Sentinel-2 10m resampled' },
            ].map(s => (
              <div key={s.val} className="lp-stat-card glass-card">
                <div className="lp-stat-val">{s.val}</div>
                <div className="lp-stat-label">{s.label}</div>
                <div className="lp-stat-sub">{s.sub}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── MISSION ── */}
        <section id="mission" className="lp-section">
          <div className="lp-section-tag">OUR MISSION</div>
          <h2 className="lp-section-title">Building the Future of<br />Earth Observation</h2>
          <p className="lp-section-desc">
            We are engineering the next generation of satellite intelligence — combining autonomous geospatial alignment, AI-driven semantic understanding, and defense-grade reporting workflows to deliver real-time actionable intelligence from orbital imagery.
          </p>

          <div className="lp-mission-cards">
            {[
              { icon: <Satellite size={24} style={{ color: '#38bdf8' }} />, title: 'Deep Orbital Intelligence', desc: 'Pushing boundaries with autonomous spacecraft imaging and multi-pass temporal analysis.' },
              { icon: <Cpu size={24} style={{ color: '#a78bfa' }} />, title: 'AI-Driven Analysis', desc: 'Neural segmentation and homography pipelines enabling real-time land-cover change quantification.' },
              { icon: <ShieldCheck size={24} style={{ color: '#4ade80' }} />, title: 'Sustainable Intelligence', desc: 'Responsible Earth monitoring ensuring long-term sustainability of strategic observation programs.' },
            ].map(c => (
              <div key={c.title} className="lp-mission-card glass-card">
                <div className="lp-mission-icon">{c.icon}</div>
                <h3>{c.title}</h3>
                <p>{c.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── CAPABILITIES ── */}
        <section id="capabilities" className="lp-section lp-section--dark">
          <div className="lp-section-tag">LABS &amp; TECHNOLOGY</div>
          <h2 className="lp-section-title">Engineering the Future<br />of Space Exploration</h2>
          <p className="lp-section-desc">
            Our labs bring together cutting-edge computer vision, satellite data pipelines, and intelligent verification systems to power next-generation defence earth observation.
          </p>

          <div className="lp-caps-grid">
            {caps.map(c => (
              <div key={c.title} className="lp-cap-card glass-card">
                <div className="lp-cap-icon">{c.icon}</div>
                <h3>{c.title}</h3>
                <p>{c.desc}</p>
                <span className="lp-cap-tag">{c.tag}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ── STRATEGIC AOIS ── */}
        <section id="aois" className="lp-section">
          <div className="lp-section-tag">STRATEGIC DEMONSTRATION SITES</div>
          <h2 className="lp-section-title">Pre-Staged Observation<br />Locations</h2>
          <p className="lp-section-desc">
            Select any strategic AOI to launch the DRISHTI workstation pre-configured with multi-temporal Sentinel-2 imagery and existing change analysis.
          </p>

          <div className="lp-aoi-grid">
            {aoiCards.map(a => (
              <div key={a.id} className="lp-aoi-card glass-card" onClick={() => onSelectAOI(a.id)}>
                <div className="lp-aoi-top">
                  <span className="lp-aoi-icon">{a.icon}</span>
                  <span className="lp-aoi-cat">{a.cat}</span>
                </div>
                <h3 className="lp-aoi-title">{a.title}</h3>
                <p className="lp-aoi-desc">{a.desc}</p>
                <div className="lp-aoi-meta">
                  <span><MapPin size={11} /> {a.stats}</span>
                  <span><Eye size={11} /> {a.range}</span>
                </div>
                <button className="lp-aoi-btn">
                  Analyze AOI <ArrowRight size={13} />
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* ── RESEARCH ── */}
        <section id="research" className="lp-section lp-section--dark">
          <div className="lp-section-tag">RESEARCH &amp; PUBLICATIONS</div>
          <h2 className="lp-section-title">Pioneering Satellite<br />Intelligence</h2>
          <p className="lp-section-desc">
            Whitepapers and peer-reviewed algorithms powering our multi-temporal change intelligence platform.
          </p>

          <div className="lp-research-grid">
            {[
              { badge: 'ORB HOMOGRAPHY', date: '12 MAR 2026', title: 'Autonomous Geospatial Alignment Under Multi-Temporal Orbital Shift', authors: 'Planetary Sensing Group • Dr. A. Kumar, L. Capre', summary: 'Sub-pixel AI homography models enabling automatic satellite image registration without manual ground control points.' },
              { badge: 'CLOUD FILTERING', date: '25 FEB 2026', title: 'False-Change Suppression in High-Cloud Sentinel-2 Level-2A Observations', authors: 'Defense AI Research Lab • M. Milan, P. Hofmann', summary: 'Contextual neighbourhood analysis and seasonal spectral normalization eliminating shadow and atmospheric change artefacts.' },
              { badge: 'DEEP LEARNING', date: '30 JAN 2026', title: 'Multi-Class Semantic Segmentation for Satellite Land-Cover Change Intelligence', authors: 'Remote Sensing Division • L. Sen, S. Roy', summary: 'Deep U-Net and ResNet backbone evaluation for fine-grained multi-class land cover change quantification.' },
            ].map(p => (
              <div key={p.title} className="lp-paper-card glass-card">
                <div className="lp-paper-top">
                  <span className="lp-paper-badge">{p.badge}</span>
                  <span className="lp-paper-date">{p.date}</span>
                </div>
                <h3 className="lp-paper-title">{p.title}</h3>
                <p className="lp-paper-authors">{p.authors}</p>
                <p className="lp-paper-summary">{p.summary}</p>
                <button className="lp-paper-btn" onClick={onLaunchWorkstation}>
                  Read Paper &amp; Launch Demo <ExternalLink size={12} />
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* ── CTA BANNER ── */}
        <section className="lp-cta-section">
          <div className="lp-cta-earth-accent" />
          <div className="lp-cta-content glass-card">
            <div className="lp-section-tag" style={{ marginBottom: 14 }}>THE FUTURE IS BEYOND EARTH</div>
            <h2>Ready to Launch Satellite<br />Intelligence?</h2>
            <p>Experience real-time multi-temporal change detection, false-change suppression, and automated intelligence reporting — all in one workstation.</p>
            <button className="lp-btn-primary" style={{ margin: '0 auto' }} onClick={onLaunchWorkstation}>
              LAUNCH DRISHTI WORKSTATION <ArrowRight size={16} />
            </button>
          </div>
        </section>

        {/* ── FOOTER ── */}
        <footer className="lp-footer">
          <div className="lp-footer-inner">
            <div className="lp-footer-brand">
              <MoDLogoHero />
              <span className="lp-brand-name">DRISHTI</span>
            </div>
            <p className="lp-footer-tagline">Multi-Temporal Satellite Change Intelligence Engine<br />Powered by Sentinel-2 MSI Level-2A &amp; AI Computer Vision.</p>
            <div className="lp-footer-links">
              <a href="#mission">Mission</a>
              <a href="#capabilities">Technology</a>
              <a href="#aois">Strategic AOIs</a>
              <a href="#research">Research</a>
            </div>
            <p className="lp-footer-copy">© 2026 DRISHTI Satellite Intelligence Engine · Ministry of Defence · All rights reserved.</p>
          </div>
        </footer>

      </div>{/* end lp-root */}
    </>
  );
}
