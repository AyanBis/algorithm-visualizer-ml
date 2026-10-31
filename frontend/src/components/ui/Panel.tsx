"use client"
import * as React from "react"
import { cn } from "../../lib/utils"

interface PanelProps extends React.HTMLAttributes<HTMLDivElement> {
  position?: "top" | "bottom" | "left" | "right"
  className?: string
  children: React.ReactNode
}

export function Panel({ position = "bottom", className, children, ...props }: PanelProps) {
  const positions = {
    top: "inset-x-0 top-0",
    bottom: "inset-x-0 bottom-0",
    left: "inset-y-0 left-0",
    right: "inset-y-0 right-0",
  }

  return (
    <div
      className={cn(
        "fixed z-50 flex",
        positions[position],
        className
      )}
      {...props}
    >
      <div className={cn(
        "flex-1 bg-background border-border",
        position === "top" || position === "bottom" 
          ? "border-y" 
          : "border-x"
      )}>
        {children}
      </div>
    </div>
  )
}