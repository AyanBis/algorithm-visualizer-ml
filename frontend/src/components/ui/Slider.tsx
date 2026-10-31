import { forwardRef } from 'react'
import { cn } from '../../lib/utils'

export interface SliderProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  min?: number
  max?: number
  step?: number
  showValue?: boolean
}

export const Slider = forwardRef<HTMLInputElement, SliderProps>(
  ({ className, label, min = 0, max = 100, step = 1, showValue = true, id, ...props }, ref) => {
    const sliderId = id || label?.toLowerCase().replace(/\s+/g, '-')
    const value = props.value || props.defaultValue || min
    
    return (
      <div className="w-full">
        {label && (
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor={sliderId} className="text-sm font-medium text-foreground">
              {label}
            </label>
            {showValue && (
              <span className="text-sm text-muted font-mono">
                {typeof value === 'number' ? value.toFixed(step < 1 ? 2 : 0) : value}
              </span>
            )}
          </div>
        )}
        <input
          ref={ref}
          id={sliderId}
          type="range"
          min={min}
          max={max}
          step={step}
          className={cn(
            'w-full h-2 bg-border rounded-lg appearance-none cursor-pointer',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2',
            'disabled:cursor-not-allowed disabled:opacity-50',
            '[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-background [&::-webkit-slider-thumb]:shadow-sm',
            '[&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-accent [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-background',
            className
          )}
          {...props}
        />
      </div>
    )
  }
)
Slider.displayName = 'Slider'