"""
Military-Grade Human Behavioral Emulation System
Agent: Silus - Computational Psychology & Behavioral Biometrics

This module provides stateful, adaptive human behavioral emulation to defeat
continuous risk analysis engines through psychologically coherent simulation.
"""

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
try:
    from scipy import interpolate  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    interpolate = None  # type: ignore
    _HAVE_SCIPY = False


# ============================================================================
# PSYCHOLOGICAL STATE MODEL
# ============================================================================

class EmotionalState(Enum):
    """Discrete emotional states affecting behavioral outputs"""
    CALM = "calm"
    FOCUSED = "focused"
    STRESSED = "stressed"
    FATIGUED = "fatigued"
    ALERT = "alert"


@dataclass
class PsychologicalProfile:
    """
    Persistent psychological state that modulates all behavioral outputs.
    Models an agent's evolving mental state across interactions.
    """
    # Core psychological dimensions (0.0 to 1.0)
    confidence: float = 0.5
    stress: float = 0.2
    familiarity: float = 0.3
    fatigue: float = 0.1
    emotional_state: EmotionalState = EmotionalState.CALM
    base_reaction_time: float = 0.35
    base_typing_speed: float = 0.22
    base_mouse_jitter: float = 1.2
    action_history: List[bool] = field(default_factory=list)

    def get_reaction_multiplier(self) -> float:
        """Return multiplier applied to reaction-driven delays"""
        state_mods = {
            EmotionalState.CALM: 1.0,
            EmotionalState.FOCUSED: 0.85,
            EmotionalState.STRESSED: 1.3,
            EmotionalState.FATIGUED: 1.4,
            EmotionalState.ALERT: 0.9
        }
        base_mod = state_mods.get(self.emotional_state, 1.0)
        stress_penalty = 1.0 + (self.stress * 0.5)
        fatigue_penalty = 1.0 + (self.fatigue * 0.6)
        confidence_bonus = 1.0 - (self.confidence * 0.15)
        return base_mod * stress_penalty * fatigue_penalty * confidence_bonus

    def get_typing_multiplier(self) -> float:
        """Return multiplier for typing speed variation"""
        return 1.0 + (self.fatigue * 0.4) - (self.confidence * 0.2)

    def get_mouse_jitter_multiplier(self) -> float:
        """Return multiplier for mouse movement precision"""
        return 1.0 + (self.stress * 0.8) + (self.fatigue * 0.3)

    def update_from_action(self, success: bool):
        """Update psychological state based on action outcome"""
        self.action_history.append(success)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        recent_success_rate = sum(self.action_history) / max(len(self.action_history), 1)
        
        # Confidence adjustment
        if success:
            self.confidence = min(1.0, self.confidence + 0.05)
            self.stress = max(0.0, self.stress - 0.03)
        else:
            self.confidence = max(0.0, self.confidence - 0.08)
            self.stress = min(1.0, self.stress + 0.05)
        
        # Gradual fatigue accumulation
        self.fatigue = min(1.0, self.fatigue + 0.002)
        
        # State transitions based on metrics
        if self.stress > 0.7:
            self.emotional_state = EmotionalState.STRESSED
        elif self.fatigue > 0.6:
            self.emotional_state = EmotionalState.FATIGUED
        elif self.confidence > 0.7 and recent_success_rate > 0.7:
            self.emotional_state = EmotionalState.FOCUSED
        elif self.stress < 0.3 and self.fatigue < 0.3:
            self.emotional_state = EmotionalState.CALM
        else:
            self.emotional_state = EmotionalState.ALERT


# ============================================================================
# BEHAVIORAL TIMING MODELS
# ============================================================================

class HumanTimingModel:
    """Models realistic human timing patterns with psychological coherence"""
    
    def __init__(self, profile: PsychologicalProfile):
        self.profile = profile
        self.last_action_time = time.time()
    
    def get_reaction_delay(self, base_delay: Optional[float] = None) -> float:
        """
        Generate psychologically coherent reaction time.
        Uses shifted gamma distribution for realistic human response modeling.
        """
        if base_delay is None:
            base_delay = self.profile.base_reaction_time
        
        multiplier = self.profile.get_reaction_multiplier()
        
        # Shifted gamma distribution (shape=2, scale varies with state)
        shape = 2.0
        scale = base_delay * multiplier / shape
        
        delay = random.gammavariate(shape, scale)
        
        # Add micro-variations (neurological noise)
        noise = random.gauss(0, base_delay * 0.05)
        delay += noise
        
        # Clamp to reasonable bounds
        return max(0.1, min(delay, 3.0))
    
    def get_typing_delay(self, char_type: str = "normal") -> float:
        """
        Generate realistic inter-keystroke delay.
        Models human typing rhythm with contextual variation.
        """
        base_speed = self.profile.base_typing_speed
        multiplier = self.profile.get_typing_multiplier()
        
        # Character-specific modifiers
        char_modifiers = {
            "normal": 1.0,
            "shift": 1.2,      # Shift keys slightly slower
            "punctuation": 1.15,
            "number": 1.1,
            "space": 0.9        # Spaces slightly faster
        }
        
        char_mod = char_modifiers.get(char_type, 1.0)
        mean_delay = base_speed * multiplier * char_mod
        
        # Log-normal distribution for typing (realistic human pattern)
        sigma = 0.3
        delay = random.lognormvariate(math.log(mean_delay), sigma)
        
        return max(0.05, min(delay, 0.8))
    
    def get_pause_delay(self, context: str = "thinking") -> float:
        """
        Generate realistic cognitive pause durations.
        Models different types of human thinking/reading pauses.
        """
        pause_bases = {
            "thinking": 1.5,
            "reading": 2.0,
            "hesitation": 0.8,
            "correction": 1.2,
            "decision": 2.5
        }
        
        base = pause_bases.get(context, 1.5)
        multiplier = self.profile.get_reaction_multiplier()
        
        # Exponential distribution with contextual mean
        mean_pause = base * multiplier
        delay = random.expovariate(1.0 / mean_pause)
        
        return max(0.3, min(delay, 8.0))


# ============================================================================
# MOUSE MOVEMENT BIOMETRICS
# ============================================================================

class MouseMovementModel:
    """
    Generates human-like mouse trajectories with micro-corrections.
    Implements Fitts's Law and motor control noise.
    """
    
    def __init__(self, profile: PsychologicalProfile):
        self.profile = profile
    
    def generate_trajectory(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 20
    ) -> List[Tuple[float, float]]:
        """
        Generate psychologically realistic mouse movement path.
        Includes sub-movements, corrections, and motor noise.
        """
        x0, y0 = start
        x1, y1 = end
        
        # Calculate distance and direction
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
        # Fitts's Law: movement time depends on distance and target size
        # MT = a + b * log2(D/W + 1), simplified here
        movement_time = 0.2 + 0.15 * math.log2(max(distance / 10, 1) + 1)
        movement_time *= self.profile.get_reaction_multiplier()
        
        # Generate base trajectory with overshoot/correction
        points = []
        jitter_mult = self.profile.get_mouse_jitter_multiplier()
        
        # Primary movement phase (80% of distance)
        overshoot = 1.0 + random.uniform(0, 0.1) * jitter_mult
        
        for i in range(int(num_points * 0.8)):
            t = (i / num_points) ** 0.7  # Non-linear progression (acceleration)
            
            # Interpolate position
            x = x0 + (x1 - x0) * t * overshoot
            y = y0 + (y1 - y0) * t * overshoot
            
            # Add motor noise (increases with speed and stress)
            noise_x = random.gauss(0, jitter_mult * 2.0)
            noise_y = random.gauss(0, jitter_mult * 2.0)
            
            points.append((x + noise_x, y + noise_y))
        
        # Correction phase (final 20%)
        for i in range(int(num_points * 0.2)):
            t = 0.8 + (i / num_points) * 0.2
            
            # Converge to target with decreasing noise
            convergence = 1.0 - (i / (num_points * 0.2))
            x = x0 + (x1 - x0) * (0.8 * overshoot + 0.2 * t)
            y = y0 + (y1 - y0) * (0.8 * overshoot + 0.2 * t)
            
            noise_x = random.gauss(0, jitter_mult * convergence * 0.5)
            noise_y = random.gauss(0, jitter_mult * convergence * 0.5)
            
            points.append((x + noise_x, y + noise_y))
        
        # Ensure final point is exactly the target
        points.append((x1, y1))
        
        return points
    
    def get_click_delay(self) -> float:
        """Time between mouse-down and mouse-up events"""
        base_click = 0.08  # ~80ms average human click duration
        variance = random.gauss(0, 0.02)
        return max(0.03, base_click + variance)


# ============================================================================
# BEHAVIORAL PATTERN INTEGRATION
# ============================================================================

class StealthBrowserConfig:
    """
    Configuration object for integrating stealth behaviors into browser-use.
    Provides delays and behavioral patterns for natural interaction.
    """
    
    def __init__(self, profile: Optional[PsychologicalProfile] = None):
        self.profile = profile or PsychologicalProfile()
        self.timing = HumanTimingModel(self.profile)
        self.mouse = MouseMovementModel(self.profile)
    
    async def before_action(self, action_type: str):
        """Insert realistic delay before browser action"""
        if action_type in ["click", "type", "scroll"]:
            delay = self.timing.get_reaction_delay()
            await asyncio.sleep(delay)
    
    async def after_action(self, action_type: str, success: bool = True):
        """Update psychological state after action"""
        self.profile.update_from_action(success)
        
        # Small pause after action (processing/observation)
        if random.random() < 0.3:  # 30% chance of micro-pause
            await asyncio.sleep(self.timing.get_pause_delay("thinking") * 0.2)
    
    def get_typing_delays(self, text: str) -> List[float]:
        """Generate realistic delays for typing a string"""
        delays = []
        for i, char in enumerate(text):
            if char.isupper() or char in "!@#$%^&*()":
                char_type = "shift"
            elif char in ".,;:!?":
                char_type = "punctuation"
            elif char.isdigit():
                char_type = "number"
            elif char == " ":
                char_type = "space"
            else:
                char_type = "normal"
            
            delays.append(self.timing.get_typing_delay(char_type))
            
            # Occasional longer pauses (thinking/reading)
            if random.random() < 0.05:  # 5% chance
                delays[-1] += self.timing.get_pause_delay("thinking")
        
        return delays


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def create_stealth_config(
    confidence: float = 0.5,
    stress: float = 0.2,
    emotional_state: str = "calm"
) -> StealthBrowserConfig:
    """
    Factory function to create stealth configuration with custom parameters.
    
    Args:
        confidence: Agent confidence level (0.0 to 1.0)
        stress: Stress level affecting behavior (0.0 to 1.0)
        emotional_state: One of: calm, focused, stressed, fatigued, alert
    
    Returns:
        Configured StealthBrowserConfig instance
    """
    profile = PsychologicalProfile(
        confidence=confidence,
        stress=stress,
        emotional_state=EmotionalState(emotional_state.lower())
    )
    
    return StealthBrowserConfig(profile)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_stealth_usage():
    """Demonstrate stealth integration with browser-use"""
    stealth = create_stealth_config(
        confidence=0.7,
        stress=0.3,
        emotional_state="focused"
    )
    
    # Before clicking
    await stealth.before_action("click")
    # ... perform actual click ...
    await stealth.after_action("click", success=True)
    
    # For typing with realistic delays
    text = "Hello World"
    delays = stealth.get_typing_delays(text)
    for char, delay in zip(text, delays):
        # Type character
        await asyncio.sleep(delay)
        # ... type char ...


if __name__ == "__main__":
    # Quick test
    asyncio.run(example_stealth_usage())
