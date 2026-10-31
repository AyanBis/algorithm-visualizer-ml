import { useEffect, useRef, useCallback, useState } from 'react'
import { getEventStreamUrl } from './hooks'
import type { WorkflowEvent, StageInfo } from '../../types/api'

export interface UseExperimentSSEOptions {
  experimentId: string | undefined
  onEvent?: (event: WorkflowEvent) => void
  onError?: (error: Event) => void
  onOpen?: () => void
  onClose?: () => void
  enabled?: boolean
  lastEventId?: number
}

export function useExperimentSSE({
  experimentId,
  onEvent,
  onError,
  onOpen,
  onClose,
  enabled = true,
  lastEventId = 0,
}: UseExperimentSSEOptions) {
  const eventSourceRef = useRef<EventSource | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [connectionError, setConnectionError] = useState<Event | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastEventIdRef = useRef(lastEventId)

  useEffect(() => {
    lastEventIdRef.current = lastEventId
  }, [lastEventId])

  const connect = useCallback(() => {
    if (!experimentId || !enabled) return

    // Clean up existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }

    try {
      const url = getEventStreamUrl(experimentId)
      const eventSource = new EventSource(url, { withCredentials: false })
      eventSourceRef.current = eventSource

      eventSource.onopen = () => {
        setIsConnected(true)
        setConnectionError(null)
        onOpen?.()
      }

      eventSource.onerror = (error) => {
        setIsConnected(false)
        setConnectionError(error)
        onError?.(error)
        
        // Attempt reconnection after 2 seconds
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
        }
        reconnectTimeoutRef.current = setTimeout(() => {
          if (enabled) {
            connect()
          }
        }, 2000)
      }

      // Handle different event types
      eventSource.addEventListener('stage_started', (event) => {
        const data = JSON.parse(event.data) as WorkflowEvent
        lastEventIdRef.current = Math.max(lastEventIdRef.current, data.sequence)
        onEvent?.(data)
      })

      eventSource.addEventListener('stage_completed', (event) => {
        const data = JSON.parse(event.data) as WorkflowEvent
        lastEventIdRef.current = Math.max(lastEventIdRef.current, data.sequence)
        onEvent?.(data)
      })

      eventSource.addEventListener('stage_failed', (event) => {
        const data = JSON.parse(event.data) as WorkflowEvent
        lastEventIdRef.current = Math.max(lastEventIdRef.current, data.sequence)
        onEvent?.(data)
      })

      eventSource.addEventListener('stage_skipped', (event) => {
        const data = JSON.parse(event.data) as WorkflowEvent
        lastEventIdRef.current = Math.max(lastEventIdRef.current, data.sequence)
        onEvent?.(data)
      })

      eventSource.addEventListener('experiment_completed', (event) => {
        const data = JSON.parse(event.data) as WorkflowEvent
        lastEventIdRef.current = Math.max(lastEventIdRef.current, data.sequence)
        onEvent?.(data)
      })

      eventSource.addEventListener('experiment_failed', (event) => {
        const data = JSON.parse(event.data) as WorkflowEvent
        lastEventIdRef.current = Math.max(lastEventIdRef.current, data.sequence)
        onEvent?.(data)
      })

      // Handle generic message events (fallback)
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WorkflowEvent
          lastEventIdRef.current = Math.max(lastEventIdRef.current, data.sequence)
          onEvent?.(data)
        } catch {
          // Ignore parse errors
        }
      }
    } catch (error) {
      console.error('Failed to create EventSource:', error)
      setConnectionError(error as Event)
    }
  }, [experimentId, enabled, onEvent, onError, onOpen])

  useEffect(() => {
    if (enabled && experimentId) {
      connect()
    }

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      onClose?.()
    }
  }, [enabled, experimentId, connect, onClose])

  // Expose reconnect function
  const reconnect = useCallback(() => {
    connect()
  }, [connect])

  return {
    isConnected,
    connectionError,
    reconnect,
    lastEventId: lastEventIdRef.current,
  }
}