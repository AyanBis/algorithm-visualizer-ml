import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../../lib/api'
import type { ExperimentConfig, ExperimentCreateResponse, ExperimentStatusResponse, ExperimentResultsResponse, WorkflowEvent } from '../../types/api'

export function useCreateExperiment() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (request: { config: ExperimentConfig }) => api.createExperiment(request.config),
    onSuccess: (data: ExperimentCreateResponse) => {
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
    },
  })
}

export function useExperimentStatus(experimentId: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ['experiments', experimentId, 'status'],
    queryFn: () => api.getExperimentStatus(experimentId!),
    enabled: enabled && !!experimentId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'running' || status === 'pending' ? 1000 : false
    },
  })
}

export function useExperimentResults(experimentId: string | undefined) {
  return useQuery({
    queryKey: ['experiments', experimentId, 'results'],
    queryFn: () => api.getExperimentResults(experimentId!),
    enabled: !!experimentId,
    staleTime: 1000 * 60 * 5,
  })
}

export function useExperimentEvents(experimentId: string | undefined, afterSequence = 0) {
  return useQuery({
    queryKey: ['experiments', experimentId, 'events', afterSequence],
    queryFn: () => api.getExperimentEvents(experimentId!, afterSequence),
    enabled: !!experimentId,
    refetchInterval: 1000,
  })
}

export function useCancelExperiment() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (experimentId: string) => api.cancelExperiment(experimentId),
    onSuccess: (_data, experimentId) => {
      queryClient.invalidateQueries({ queryKey: ['experiments', experimentId] })
    },
  })
}

export function getEventStreamUrl(experimentId: string): string {
  return api.getEventStreamUrl(experimentId)
}