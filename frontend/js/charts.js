/**
 * VisionCheck Reports Chart Manager
 * ----------------------------------
 * Clean Chart.js visualizer for executive quality reports.
 */

export class ChartManager {
  constructor() {
    this.charts = {};
  }

  destroyChart(id) {
    if (this.charts[id]) {
      this.charts[id].destroy();
      delete this.charts[id];
    }
  }

  renderReportsCharts(defectDistribution = {}, recentHistory = []) {
    // 1. Most Common Problems Doughnut / Bar
    this.destroyChart('commonProblemsChart');
    const ctx1 = document.getElementById('commonProblemsChart');
    if (ctx1) {
      const labels = Object.keys(defectDistribution).length > 0 ? Object.keys(defectDistribution) : ['Scratches', 'Cracks', 'Blemishes', 'Contamination'];
      const data = Object.keys(defectDistribution).length > 0 ? Object.values(defectDistribution) : [42, 28, 18, 12];

      this.charts['commonProblemsChart'] = new Chart(ctx1, {
        type: 'doughnut',
        data: {
          labels: labels.map(l => l.replace('_LIKE', '').replace('_', ' ')),
          datasets: [{
            data: data,
            backgroundColor: ['#ef4444', '#f97316', '#f59e0b', '#3b82f6'],
            borderColor: '#1c2433',
            borderWidth: 3
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '68%',
          plugins: {
            legend: {
              position: 'right',
              labels: { color: '#94a3b8', font: { size: 11 }, padding: 10 }
            }
          }
        }
      });
    }

    // 2. Quality Score Trend Line
    this.destroyChart('qualityTrendChart');
    const ctx2 = document.getElementById('qualityTrendChart');
    if (ctx2) {
      const scores = recentHistory.length > 0 ? recentHistory.map(h => h.quality_score).reverse() : [96, 92, 45, 98, 74, 91, 88, 95];
      const labels = scores.map((_, i) => `Part #${i + 1}`);

      this.charts['qualityTrendChart'] = new Chart(ctx2, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Quality Score',
            data: scores,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            fill: true,
            tension: 0.35,
            pointRadius: 4,
            pointBackgroundColor: '#10b981'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { min: 0, max: 100, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            x: { ticks: { color: '#64748b' }, grid: { display: false } }
          },
          plugins: { legend: { display: false } }
        }
      });
    }
  }
}
