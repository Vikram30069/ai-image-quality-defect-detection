/**
 * VisionCheck Application Controller
 * -----------------------------------
 * Manages the 4-tab workflow (Home, Inspect, Previous Checks, Reports),
 * file uploads, realistic sample generation, and technical details toggle.
 */

import { ApiClient } from './api.js';
import { CanvasViewer } from './canvas_viewer.js';
import { ChartManager } from './charts.js';
import { UI } from './ui.js';

class App {
  constructor() {
    this.viewer = null;
    this.charts = new ChartManager();
    this.currentInspection = null;
    this.historyList = [];
    this.activeTab = 'homeTab';
  }

  init() {
    window.appInstance = this;

    // Initialize Canvas Viewer
    this.viewer = new CanvasViewer('inspectionCanvas');

    this.bindNavigation();
    this.bindUploadEvents();
    this.bindViewerControls();
    this.bindSampleChips();
    this.bindTechDetailsToggle();
    this.bindPreviousChecksControls();

    // Boot initial data
    this.boot();
  }

  bindNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const tabId = item.dataset.tab;
        if (!tabId) return;
        this.switchTab(tabId);
      });
    });
  }

  switchTab(tabId) {
    this.activeTab = tabId;

    // Update Nav
    document.querySelectorAll('.nav-item').forEach(n => {
      n.classList.toggle('active', n.dataset.tab === tabId);
    });

    // Update Panes
    document.querySelectorAll('.tab-pane').forEach(p => {
      p.classList.toggle('active', p.id === tabId);
    });

    // Update Header Text
    const titleEl = document.getElementById('pageHeaderTitle');
    const subEl = document.getElementById('pageHeaderSubtitle');

    if (tabId === 'homeTab') {
      if (titleEl) titleEl.textContent = 'AI Quality Check';
      if (subEl) subEl.textContent = 'Check any product image for defects in seconds';
    } else if (tabId === 'inspectTab') {
      if (titleEl) titleEl.textContent = 'Inspection Result';
      if (subEl) subEl.textContent = 'AI automated surface defect and quality assessment';
      if (this.viewer) setTimeout(() => this.viewer.resetView(), 60);
    } else if (tabId === 'previousChecksTab') {
      if (titleEl) titleEl.textContent = 'Previous Checks';
      if (subEl) subEl.textContent = 'History of all inspected parts and decisions';
      this.refreshPreviousChecks();
    } else if (tabId === 'reportsTab') {
      if (titleEl) titleEl.textContent = 'Quality Reports';
      if (subEl) subEl.textContent = 'Executive summary of defects and pass rates';
      this.refreshReports();
    }
  }

  bindUploadEvents() {
    const globalBtn = document.getElementById('globalUploadBtn');
    const homeBtn = document.getElementById('homeUploadBtn');
    const homeDropzone = document.getElementById('homeDropzone');
    const fileInput = document.getElementById('fileInput');
    const checkAnotherBtn = document.getElementById('checkAnotherBtn');

    const triggerUpload = () => fileInput?.click();

    globalBtn?.addEventListener('click', triggerUpload);
    homeBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      triggerUpload();
    });
    homeDropzone?.addEventListener('click', triggerUpload);

    checkAnotherBtn?.addEventListener('click', () => {
      this.switchTab('homeTab');
    });

    // Drag & Drop
    if (homeDropzone) {
      homeDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        homeDropzone.classList.add('dragover');
      });

      homeDropzone.addEventListener('dragleave', () => {
        homeDropzone.classList.remove('dragover');
      });

      homeDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        homeDropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          this.handleFileUpload(e.dataTransfer.files[0]);
        }
      });
    }

    fileInput?.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        this.handleFileUpload(e.target.files[0]);
      }
    });
  }

  bindViewerControls() {
    const btnDefects = document.getElementById('btnModeDefects');
    const btnOriginal = document.getElementById('btnModeOriginal');
    const btnHeatmap = document.getElementById('btnModeHeatmap');

    const setMode = (btn, mode) => {
      [btnDefects, btnOriginal, btnHeatmap].forEach(b => b?.classList.remove('active'));
      btn?.classList.add('active');
      this.viewer.setMode(mode);
    };

    btnDefects?.addEventListener('click', () => setMode(btnDefects, 'defects'));
    btnOriginal?.addEventListener('click', () => setMode(btnOriginal, 'original'));
    btnHeatmap?.addEventListener('click', () => setMode(btnHeatmap, 'heatmap'));

    document.getElementById('btnZoomIn')?.addEventListener('click', () => this.viewer.zoom(1.2));
    document.getElementById('btnZoomOut')?.addEventListener('click', () => this.viewer.zoom(0.8));
    document.getElementById('btnZoomFit')?.addEventListener('click', () => this.viewer.resetView());
  }

  bindSampleChips() {
    const sampleMap = {
      clean: 'sample_good.png',
      scratch: 'sample_scratched.png',
      crack: 'sample_crack_like.png',
      blemish: 'sample_blemish.png',
      contamination: 'sample_contamination.png'
    };

    document.querySelectorAll('.example-chip').forEach(chip => {
      chip.addEventListener('click', async () => {
        const sampleType = chip.dataset.sample || 'clean';
        UI.showToast(`Loading real sample for ${sampleType}...`, 'info');

        // Multi-tier candidate endpoints
        const candidates = [
          `/api/samples/${sampleType}`,
          `/sample_images/${sampleMap[sampleType] || 'sample_good.png'}`,
          `sample_images/${sampleMap[sampleType] || 'sample_good.png'}`
        ];

        let blob = null;
        for (const url of candidates) {
          try {
            const res = await fetch(url);
            if (res.ok) {
              blob = await res.blob();
              break;
            }
          } catch (e) {
            // try next candidate
          }
        }

        if (!blob) {
          // Client-side canvas fallback if network is completely unavailable
          blob = await this.generateClientFallbackBlob(sampleType);
        }

        if (blob) {
          const file = new File([blob], `dataset_${sampleType}.png`, { type: 'image/png' });
          this.handleFileUpload(file);
        } else {
          UI.showToast('Could not load sample image', 'error');
        }
      });
    });
  }

  async generateClientFallbackBlob(type) {
    const c = document.createElement('canvas');
    c.width = 300;
    c.height = 300;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#2a2f3a';
    ctx.fillRect(0, 0, 300, 300);
    ctx.strokeStyle = '#4a5568';
    ctx.lineWidth = 2;
    for (let i = 20; i < 300; i += 30) {
      ctx.strokeRect(i, i, 40, 40);
    }
    if (type === 'scratch') {
      ctx.strokeStyle = '#e53e3e';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(40, 60);
      ctx.lineTo(260, 240);
      ctx.stroke();
    } else if (type === 'crack') {
      ctx.strokeStyle = '#dd6b20';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(80, 50);
      ctx.lineTo(150, 150);
      ctx.lineTo(220, 260);
      ctx.stroke();
    } else if (type === 'blemish') {
      ctx.fillStyle = '#9b2c2c';
      ctx.beginPath();
      ctx.arc(150, 150, 25, 0, Math.PI * 2);
      ctx.fill();
    }
    return new Promise(resolve => c.toBlob(resolve, 'image/png'));
  }

  bindTechDetailsToggle() {
    const toggleBtn = document.getElementById('toggleTechDetailsBtn');
    const panel = document.getElementById('techDetailsPanel');
    const arrow = document.getElementById('techToggleArrow');

    toggleBtn?.addEventListener('click', () => {
      const isOpen = panel?.classList.toggle('open');
      if (arrow) arrow.textContent = isOpen ? '▴' : '▾';
    });
  }

  bindPreviousChecksControls() {
    document.getElementById('historyFilterSelect')?.addEventListener('change', () => this.refreshPreviousChecks());
    document.getElementById('refreshCardsBtn')?.addEventListener('click', () => this.refreshPreviousChecks());
  }

  async handleFileUpload(file) {
    try {
      UI.showToast(`AI analyzing ${file.name}...`, 'info');

      const result = await ApiClient.inspectImage(file);
      this.currentInspection = result;

      // Switch to Inspect Tab
      this.switchTab('inspectTab');

      // Load Image in Viewport
      this.viewer.loadImageData(
        result.image_url,
        result.heatmap_url,
        result.defect_summary ? result.defect_summary.defect_list : []
      );

      // Render Result Text
      UI.renderInspectionResult(result);

      const verdictTitle = (result.quality_label || 'ACCEPTABLE').toUpperCase();
      const verdictMsg = verdictTitle === 'ACCEPTABLE' ? 'Good Product (No defects)' : verdictTitle === 'DEGRADED' ? 'Check Required (Degraded)' : 'Defect Detected!';
      UI.showToast(`AI Scan Complete: ${verdictMsg}`, 'success');
    } catch (err) {
      console.error(err);
      UI.showToast(err.message, 'error');
    }
  }

  async boot() {
    try {
      const history = await ApiClient.fetchHistory(6, 0);
      this.historyList = history.items || [];
    } catch (err) {
      console.error('Boot err:', err);
    }
  }

  async refreshPreviousChecks() {
    try {
      const label = document.getElementById('historyFilterSelect')?.value || null;
      const data = await ApiClient.fetchHistory(30, 0, label);
      this.historyList = data.items || [];

      UI.renderPreviousChecksCards(this.historyList, async (id) => {
        UI.showToast(`Loading product #${id}...`, 'info');
        const detail = await ApiClient.fetchDetail(id);
        this.currentInspection = detail;
        this.switchTab('inspectTab');
        this.viewer.loadImageData(detail.image_url, detail.heatmap_url, detail.defect_summary ? detail.defect_summary.defect_list : []);
        UI.renderInspectionResult(detail);
      });
    } catch (err) {
      console.error(err);
    }
  }

  async refreshReports() {
    try {
      const analytics = await ApiClient.fetchAnalytics();
      UI.renderReportsMetrics(analytics);
      this.charts.renderReportsCharts(analytics.defect_distribution, this.historyList);
    } catch (err) {
      console.error(err);
    }
  }
}

// Bootstrap
document.addEventListener('DOMContentLoaded', () => {
  const app = new App();
  app.init();
});
