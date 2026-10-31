const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001/api'

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl
  }

  private buildUrl(path: string, params?: Record<string, string | number | boolean | undefined>): string {
    const url = new URL(`${this.baseUrl}${path}`)
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          url.searchParams.append(key, String(value))
        }
      })
    }
    return url.toString()
  }

  async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const { params, ...fetchOptions } = options
    const url = this.buildUrl(path, params)

    const response = await fetch(url, {
      ...fetchOptions,
      headers: {
        'Content-Type': 'application/json',
        ...fetchOptions.headers,
      },
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new ApiError(response.status, errorData.detail || response.statusText)
    }

    if (response.status === 204) {
      return undefined as T
    }

    return response.json()
  }

  get<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
    return this.request<T>(path, { method: 'GET', params })
  }

  post<T>(path: string, data: unknown): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  delete<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: 'DELETE' })
  }

  // Datasets
  getDatasets(): Promise<{ datasets: import('../../types/api').DatasetInfo[] }> {
    return this.get('/datasets')
  }

  getDataset(id: string): Promise<import('../../types/api').DatasetDetailResponse> {
    return this.get(`/datasets/${id}`)
  }

  // Models
  getModels(taskType?: string): Promise<{ models: import('../../types/api').ModelInfo[] }> {
    return this.get('/models', { task_type: taskType })
  }

  getModel(id: string): Promise<{ model: import('../../types/api').ModelInfo }> {
    return this.get(`/models/${id}`)
  }

  // Experiments
  createExperiment(config: import('../../types/api').ExperimentConfig): Promise<import('../../types/api').ExperimentCreateResponse> {
    return this.post('/experiments', { config })
  }

  getExperimentStatus(id: string): Promise<import('../../types/api').ExperimentStatusResponse> {
    return this.get(`/experiments/${id}`)
  }

  getExperimentEvents(id: string, afterSequence = 0): Promise<{ events: import('../../types/api').WorkflowEvent[] }> {
    return this.get(`/experiments/${id}/events`, { after_sequence: afterSequence })
  }

  getExperimentResults(id: string): Promise<import('../../types/api').ExperimentResultsResponse> {
    return this.get(`/experiments/${id}/results`)
  }

  cancelExperiment(id: string): Promise<{ success: boolean; message: string }> {
    return this.post(`/experiments/${id}/cancel`, {})
  }

  // SSE stream URL
  getEventStreamUrl(experimentId: string): string {
    return `${this.baseUrl}/experiments/${experimentId}/events/stream`
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public message: string,
    public data?: unknown
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export const api = new ApiClient()