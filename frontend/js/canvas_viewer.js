/**
 * VisionCheck Clean Canvas Viewer
 * ---------------------------------
 * Renders product images with clean, uncluttered defect bounding boxes
 * and support for original image and heatmap overlay modes.
 */

export class CanvasViewer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');

    this.originalImg = null;
    this.heatmapImg = null;
    this.defects = [];

    // View Mode: 'defects' | 'original' | 'heatmap'
    this.mode = 'defects';

    // Pan & Zoom
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;

    this.initEvents();
  }

  initEvents() {
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      this.zoom(zoomFactor, e.offsetX, e.offsetY);
    });

    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.dragStartX = e.clientX - this.panX;
      this.dragStartY = e.clientY - this.panY;
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        this.panX = e.clientX - this.dragStartX;
        this.panY = e.clientY - this.dragStartY;
        this.render();
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });
  }

  loadImageData(originalUrl, heatmapUrl = null, defects = []) {
    this.defects = defects || [];

    this.originalImg = new Image();
    this.originalImg.crossOrigin = 'anonymous';

    this.originalImg.onload = () => {
      this.resetView();
      if (heatmapUrl) {
        this.heatmapImg = new Image();
        this.heatmapImg.crossOrigin = 'anonymous';
        this.heatmapImg.onload = () => this.render();
        this.heatmapImg.src = heatmapUrl;
      } else {
        this.heatmapImg = null;
        this.render();
      }
    };
    this.originalImg.src = originalUrl;
  }

  setMode(mode) {
    this.mode = mode;
    this.render();
  }

  resetView() {
    if (!this.originalImg) return;
    const parent = this.canvas.parentElement;
    this.canvas.width = parent.clientWidth || 700;
    this.canvas.height = parent.clientHeight || 420;

    const scaleX = this.canvas.width / this.originalImg.width;
    const scaleY = this.canvas.height / this.originalImg.height;
    this.scale = Math.min(scaleX, scaleY) * 0.92;

    this.panX = (this.canvas.width - this.originalImg.width * this.scale) / 2;
    this.panY = (this.canvas.height - this.originalImg.height * this.scale) / 2;

    this.render();
  }

  zoom(factor, centerX = this.canvas.width / 2, centerY = this.canvas.height / 2) {
    const newScale = Math.min(Math.max(this.scale * factor, 0.1), 8.0);
    this.panX = centerX - (centerX - this.panX) * (newScale / this.scale);
    this.panY = centerY - (centerY - this.panY) * (newScale / this.scale);
    this.scale = newScale;
    this.render();
  }

  render() {
    if (!this.originalImg) return;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this.ctx.save();
    this.ctx.translate(this.panX, this.panY);
    this.ctx.scale(this.scale, this.scale);

    // 1. Draw Base Product Image
    this.ctx.drawImage(this.originalImg, 0, 0);

    // 2. Draw Heatmap overlay if selected
    if (this.mode === 'heatmap' && this.heatmapImg) {
      this.ctx.globalAlpha = 0.65;
      this.ctx.drawImage(this.heatmapImg, 0, 0);
      this.ctx.globalAlpha = 1.0;
    }

    // 3. Draw Clean Bounding Boxes in 'defects' mode
    if (this.mode === 'defects' && this.defects && this.defects.length > 0) {
      // Limit to top 3 defects to avoid clutter
      const defectsToShow = this.defects.slice(0, 4);

      defectsToShow.forEach(d => {
        const b = d.bounding_box;
        const color = '#ef4444'; // Clean danger red

        // Box Outline
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2.5 / this.scale;
        this.ctx.strokeRect(b.x, b.y, b.width, b.height);

        // Highlight Fill
        this.ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
        this.ctx.fillRect(b.x, b.y, b.width, b.height);

        // Clean Pill Tag
        const label = `${d.type.replace('_LIKE', '')} • ${Math.round(d.confidence * 100)}%`;
        const fontSize = Math.max(9, Math.min(13, 12 / this.scale));
        this.ctx.font = `700 ${fontSize}px sans-serif`;

        const textWidth = this.ctx.measureText(label).width;
        const tagHeight = fontSize + 6;
        const tagY = Math.max(0, b.y - tagHeight - 2);

        this.ctx.fillStyle = color;
        this.ctx.fillRect(b.x, tagY, textWidth + 8, tagHeight);

        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillText(label, b.x + 4, tagY + fontSize);
      });
    }

    this.ctx.restore();
  }
}
