import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import type { DatasetInfo, DatasetDetailResponse, ModelInfo } from '../types/api'

export function useDatasets() {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: () => api.getDatasets(),
    staleTime: 1000 * 60 * 10, // 10 minutes
  })
}

export function useDataset(id: string | undefined) {
  return useQuery({
    queryKey: ['datasets', id],
    queryFn: () => api.getDataset(id!),
    enabled: !!id,
    staleTime: 1000 * 60 * 10,
  })
}

export function useModels(taskType?: string) {
  return useQuery({
    queryKey: ['models', taskType],
    queryFn: () => api.getModels(taskType),
    staleTime: 1000 * 60 * 10,
  })
}

export function useModel(id: string | undefined) {
  return useQuery({
    queryKey: ['models', id],
    queryFn: () => api.getModel(id!),
    enabled: !!id,
    staleTime: 1000 * 60 * 10,
  })
}

export function useCompatibleModels(datasetTaskType: string | undefined) {
  const { data: modelsResponse } = useModels(datasetTaskType)
  return modelsResponse?.models || []
}