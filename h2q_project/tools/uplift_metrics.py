#!/usr/bin/env python3
"""
Uplift Metrics Tracker - Phase 1 Track A Implementation

This module provides mechanisms to detect and respond to composite_uplift decay
using rolling windows and slope-based alarms. Used in:
- start_quantum_agi_highdim_evolution.py main loop
- Strategy persistence decisions
- Important-cycle trigger logic extension
"""

import json
import logging
from collections import deque
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


class RollingUpliftWindow:
    """
    Tracks composite scores over a sliding window to detect trends and trigger alarms.
    
    Key attributes:
    - window_size: Number of cycles to average (default 3)
    - alarm_threshold: Slope threshold to trigger alarm (default -0.005)
    - current_streak: Count of consecutive positive changes
    """
    
    def __init__(
        self,
        window_size: int = 3,
        alarm_threshold: float = -0.005,
        max_history: int = 100
    ):
        """
        Initialize window tracker.
        
        Args:
            window_size: Number of cycles per window (default 3)
            alarm_threshold: Slope < threshold triggers alarm (default -0.005)
            max_history: Max length of full history buffer (prevents unbounded growth)
        """
        self.window_size = window_size
        self.alarm_threshold = alarm_threshold
        self.max_history = max_history
        
        self.values = deque(maxlen=max_history)  # Full history
        self.window = deque(maxlen=window_size)   # Current sliding window
        self.current_streak = 0                    # Streak of positive changes
        self.slope_history = deque(maxlen=10)     # Last 10 slopes for debugging
        self.alarm_triggered_at_cycle = None      # Track when alarm last fired
        self.total_cycles_pushed = 0               # Running count
    
    def push_value(self, composite_score: float, cycle_idx: int) -> None:
        """
        Add new composite score to window and check for alarms.
        
        Args:
            composite_score: Current cycle's composite_score value
            cycle_idx: Current cycle index (for logging)
        """
        self.values.append(composite_score)
        self.window.append(composite_score)
        self.total_cycles_pushed += 1
        
        # Update streak based on change from previous value
        if len(self.values) >= 2:
            delta = self.values[-1] - self.values[-2]
            if delta > 0:
                self.current_streak += 1
            else:
                self.current_streak = 0  # Reset on any negative change
    
    def get_window_stats(self) -> Dict[str, float]:
        """
        Return current window average and other statistics.
        
        Returns:
            dict with keys: mean, min, max, values_list, valid
        """
        if len(self.window) < self.window_size:
            # Window not yet full
            return {
                "mean": (sum(self.window) / len(self.window)) if self.window else 0,
                "min": min(self.window) if self.window else 0,
                "max": max(self.window) if self.window else 0,
                "values": list(self.window),
                "valid": False,
                "reason": f"window not full ({len(self.window)}/{self.window_size})"
            }
        else:
            return {
                "mean": sum(self.window) / self.window_size,
                "min": min(self.window),
                "max": max(self.window),
                "values": list(self.window),
                "valid": True
            }
    
    def get_slope(self) -> Optional[float]:
        """
        Calculate slope between prior window and current window.
        
        Returns:
            float: (current_window_mean - prior_window_mean) or None if insuffient history
        """
        # Need at least 2*window_size values to compare windows
        if len(self.values) < 2 * self.window_size:
            return None
        
        # Prior window: values[-(2*window_size):-window_size]
        # Current window: values[-window_size:]
        prior_values = list(self.values)[-(2*self.window_size):-self.window_size]
        current_values = list(self.values)[-self.window_size:]
        
        prior_mean = sum(prior_values) / len(prior_values)
        current_mean = sum(current_values) / len(current_values)
        
        slope = current_mean - prior_mean
        self.slope_history.append(slope)
        
        return slope
    
    def is_alarm_triggered(self, cycle_idx: Optional[int] = None) -> bool:
        """
        Check if slope-based alarm should trigger.
        
        Args:
            cycle_idx: Optional cycle index for logging
        
        Returns:
            bool: True if slope < alarm_threshold
        """
        slope = self.get_slope()
        if slope is None:
            return False
        
        triggered = slope < self.alarm_threshold
        if triggered:
            self.alarm_triggered_at_cycle = cycle_idx
            logger.warning(
                f"[UPLIFT-ALARM] cycle={cycle_idx}, slope={slope:.6f} < {self.alarm_threshold} → TRIGGERED"
            )
        
        return triggered
    
    def should_be_in_critical_zone(self, uplift: float, uplift_threshold: float = -0.05) -> bool:
        """
        Check if system is in critical zone (uplift < threshold AND no positive streak).
        This triggers forced capability test + max boost strategy.
        
        Args:
            uplift: Current composite_uplift value
            uplift_threshold: Critical threshold (default -0.05)
        
        Returns:
            bool: True if uplift < threshold AND current_streak == 0
        """
        in_critical = (uplift < uplift_threshold) and (self.current_streak == 0)
        return in_critical
    
    def should_be_in_stability_zone(self, min_streak: int = 2) -> bool:
        """
        Check if system is in stability zone (positive streak >= min_streak).
        In this zone, curriculum-only mode is safe; no forced strategy application.
        
        Args:
            min_streak: Required consecutive positive changes (default 2)
        
        Returns:
            bool: True if current_streak >= min_streak
        """
        return self.current_streak >= min_streak
    
    def clear_alarm(self) -> None:
        """Reset alarm state."""
        self.alarm_triggered_at_cycle = None
    
    def to_dict(self) -> Dict:
        """Serialize state for checkpointing."""
        return {
            "window_size": self.window_size,
            "alarm_threshold": self.alarm_threshold,
            "current_streak": self.current_streak,
            "total_cycles_pushed": self.total_cycles_pushed,
            "values": list(self.values),
            "window": list(self.window),
            "slope_history": list(self.slope_history),
            "alarm_triggered_at_cycle": self.alarm_triggered_at_cycle
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RollingUpliftWindow":
        """Deserialize state from checkpoint."""
        obj = cls(
            window_size=data.get("window_size", 3),
            alarm_threshold=data.get("alarm_threshold", -0.005)
        )
        obj.current_streak = data.get("current_streak", 0)
        obj.total_cycles_pushed = data.get("total_cycles_pushed", 0)
        obj.alarm_triggered_at_cycle = data.get("alarm_triggered_at_cycle")
        
        # Restore deques
        for val in data.get("values", []):
            obj.values.append(val)
        for val in data.get("window", []):
            obj.window.append(val)
        for val in data.get("slope_history", []):
            obj.slope_history.append(val)
        
        return obj
    
    def get_report(self) -> Dict:
        """Generate diagnostic report for logging."""
        window_stats = self.get_window_stats()
        slope = self.get_slope()
        
        return {
            "total_cycles_processed": self.total_cycles_pushed,
            "window_size": self.window_size,
            "window_stats": window_stats,
            "slope": slope,
            "current_streak": self.current_streak,
            "alarm_threshold": self.alarm_threshold,
            "alarm_triggered_at_cycle": self.alarm_triggered_at_cycle,
            "slope_history": list(self.slope_history)
        }


class StrategyPersistenceManager:
    """
    Manages cross-window strategy persistence decisions.
    
    Tracks:
    - Base strategy lifetime (cycles_left)
    - Extension requests (from slope-alarm or critical-zone)
    - Stability zone exemption (curriculum-only safe)
    """
    
    def __init__(self):
        """Initialize strategy state."""
        self.strategy_active = False
        self.strategy_cycles_left = 0
        self.strategy_topic_boost = {}
        self.strategy_capability_interval = 3
        self.extension_count = 0  # Track how many times strategy was extended
        self.max_extensions = 3    # Cap total extensions to prevent infinite loops
    
    def set_strategy(
        self,
        topic_boost: Dict[str, float],
        capability_interval: int = 1,
        initial_duration: int = 2
    ) -> None:
        """Activate a new strategy."""
        self.strategy_active = True
        self.strategy_cycles_left = initial_duration
        self.strategy_topic_boost = topic_boost
        self.strategy_capability_interval = capability_interval
        self.extension_count = 0
        logger.info(
            f"[STRATEGY] Activated: boost={topic_boost}, "
            f"capability_interval={capability_interval}, duration={initial_duration}"
        )
    
    def extend_strategy(self, additional_cycles: int = 2) -> bool:
        """
        Request strategy extension.
        
        Args:
            additional_cycles: Cycles to extend (default 2)
        
        Returns:
            bool: True if extension granted, False if max extensions exceeded
        """
        if self.extension_count >= self.max_extensions:
            logger.warning(
                f"[STRATEGY] Extension DENIED: max_extensions={self.max_extensions} reached"
            )
            return False
        
        self.strategy_cycles_left += additional_cycles
        self.extension_count += 1
        logger.info(
            f"[STRATEGY] Extended: +{additional_cycles} cycles total_extension_count={self.extension_count}"
        )
        return True
    
    def decrement_and_check(self) -> bool:
        """
        Decrement strategy lifetime and check if still active.
        
        Returns:
            bool: True if strategy still active, False if expired
        """
        if self.strategy_active and self.strategy_cycles_left > 0:
            self.strategy_cycles_left -= 1
            if self.strategy_cycles_left == 0:
                logger.info("[STRATEGY] Expired")
                self.strategy_active = False
                return False
            return True
        return False
    
    def get_state(self) -> Dict:
        """Return current strategy state for checkpointing."""
        return {
            "strategy_active": self.strategy_active,
            "strategy_cycles_left": self.strategy_cycles_left,
            "strategy_topic_boost": self.strategy_topic_boost,
            "strategy_capability_interval": self.strategy_capability_interval,
            "extension_count": self.extension_count
        }
    
    def set_state(self, state: Dict) -> None:
        """Restore strategy state from checkpoint."""
        self.strategy_active = state.get("strategy_active", False)
        self.strategy_cycles_left = state.get("strategy_cycles_left", 0)
        self.strategy_topic_boost = state.get("strategy_topic_boost", {})
        self.strategy_capability_interval = state.get("strategy_capability_interval", 3)
        self.extension_count = state.get("extension_count", 0)
