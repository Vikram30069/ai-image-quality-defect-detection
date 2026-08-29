/**
 * API Client Module
 * -----------------
 * Handles all asynchronous HTTP requests to the FastAPI backend.
 */

const API_BASE = '/api';

export const ApiClient = {
  /**
   * Uploads an image file for quality and defect inspection.
   * @param {File|Blob} file 
   * @returns {Promise<Object>} Inspection JSON response
   */
  async inspectImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/inspect`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned error ${response.status}`);
    }

    return await response.json();
  },

  /**
   * Fetches paginated inspection history.
   * @param {number} limit 
   * @param {number} offset 
   * @param {string|null} labelFilter 
   * @returns {Promise<Object>}
   */
  async fetchHistory(limit = 50, offset = 0, labelFilter = null) {
    let url = `${API_BASE}/history?limit=${limit}&offset=${offset}`;
    if (labelFilter) {
      url += `&label=${encodeURIComponent(labelFilter)}`;
    }

    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch history');
    return await response.json();
  },

  /**
   * Fetches specific inspection details by ID.
   * @param {number} id 
   * @returns {Promise<Object>}
   */
  async fetchDetail(id) {
    const response = await fetch(`${API_BASE}/history/${id}`);
    if (!response.ok) throw new Error(`Failed to fetch inspection #${id}`);
    return await response.json();
  },

  /**
   * Deletes an inspection record.
   * @param {number} id 
   */
  async deleteInspection(id) {
    const response = await fetch(`${API_BASE}/history/${id}`, {
      method: 'DELETE'
    });
    if (!response.ok) throw new Error(`Failed to delete inspection #${id}`);
    return await response.json();
  },

  /**
   * Fetches analytics summary metrics.
   * @returns {Promise<Object>}
   */
  async fetchAnalytics() {
    const response = await fetch(`${API_BASE}/analytics`);
    if (!response.ok) throw new Error('Failed to fetch analytics');
    return await response.json();
  },

  /**
   * Fetches system health and status.
   * @returns {Promise<Object>}
   */
  async checkHealth() {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) throw new Error('Health check failed');
    return await response.json();
  }
};
