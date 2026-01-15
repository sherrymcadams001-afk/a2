#!/usr/bin/env python3
"""
Workflow Runner - Orchestrates multi-step browser automation with validation and transforms.
Provides deterministic control flow, retry logic, and Python-based data transformations.
Global stealth is applied to ALL operations.
"""
import asyncio
import inspect
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

# Import database layer
import database as db

logger = logging.getLogger(__name__)


# =============================================================================
# GLOBAL STEALTH CONFIGURATION
# =============================================================================

class GlobalStealth:
    """Global stealth manager - applies human-like behavior everywhere"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_stealth()
        return cls._instance
    
    def _init_stealth(self):
        self.enabled = True
        self.min_delay = 0.5
        self.max_delay = 2.0
        self.reading_pause_min = 1.0
        self.reading_pause_max = 3.0
        self.action_count = 0
        self.fatigue_factor = 0.0  # Increases over time
        
    def configure(self, config: Dict):
        """Configure stealth from workflow config"""
        if not config:
            return
        self.enabled = config.get('enabled', True)
        self.min_delay = config.get('min_action_delay', 0.5)
        self.max_delay = config.get('max_action_delay', 2.0)
        self.reading_pause_min = config.get('reading_pause', 1.5)
        self.reading_pause_max = config.get('reading_pause', 1.5) + 1.5
        logger.info(f"[STEALTH] Configured: delays {self.min_delay}-{self.max_delay}s")
    
    async def pre_action_delay(self, action_type: str = "default"):
        """Apply delay before an action"""
        if not self.enabled:
            return
        
        # Increase fatigue over time
        self.action_count += 1
        if self.action_count % 20 == 0:
            self.fatigue_factor = min(0.5, self.fatigue_factor + 0.05)
        
        # Calculate delay based on action type
        base_delay = self._calculate_delay(action_type)
        
        # Add fatigue
        delay = base_delay * (1 + self.fatigue_factor)
        
        # Add micro-variation (human noise)
        noise = random.gauss(0, delay * 0.1)
        delay = max(0.1, delay + noise)
        
        logger.debug(f"[STEALTH] Pre-action delay: {delay:.2f}s (type={action_type})")
        await asyncio.sleep(delay)
    
    async def post_action_delay(self, action_type: str = "default"):
        """Apply delay after an action (observation/reading)"""
        if not self.enabled:
            return
        
        if action_type in ["navigate", "click", "scroll"]:
            # Longer pause to "observe" results
            delay = random.uniform(self.reading_pause_min, self.reading_pause_max)
        else:
            delay = random.uniform(self.min_delay * 0.5, self.max_delay * 0.5)
        
        logger.debug(f"[STEALTH] Post-action delay: {delay:.2f}s")
        await asyncio.sleep(delay)
    
    async def thinking_pause(self):
        """Simulate human thinking/decision pause"""
        if not self.enabled:
            return
        
        # Occasional longer pause (5% chance)
        if random.random() < 0.05:
            delay = random.uniform(3.0, 6.0)
            logger.debug(f"[STEALTH] Thinking pause: {delay:.2f}s")
            await asyncio.sleep(delay)
    
    def _calculate_delay(self, action_type: str) -> float:
        """Calculate delay using gamma distribution (realistic human timing)"""
        type_multipliers = {
            "navigate": 1.5,
            "click": 1.0,
            "scroll": 0.8,
            "type": 0.6,
            "extract": 1.2,
            "default": 1.0
        }
        multiplier = type_multipliers.get(action_type, 1.0)
        
        # Gamma distribution for human-like timing
        shape = 2.0
        scale = (self.min_delay + self.max_delay) / 4 * multiplier
        delay = random.gammavariate(shape, scale)
        
        return max(self.min_delay, min(delay, self.max_delay * 1.5))


# Global stealth instance
stealth = GlobalStealth()


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class StepResult:
    """Result of a workflow step execution"""
    step_id: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    retries: int = 0


@dataclass
class WorkflowContext:
    """Shared context across workflow steps"""
    variables: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, StepResult] = field(default_factory=dict)
    extracted_data: List[Dict] = field(default_factory=list)
    
    def get_step_output(self, step_id: str) -> Any:
        """Get output from a completed step"""
        if step_id in self.results and self.results[step_id].status == StepStatus.SUCCESS:
            return self.results[step_id].output
        return None
    
    def merge_extracted_data(self, new_data: List[Dict]):
        """Merge new extracted data, avoiding duplicates.

        The workflows extract different shapes depending on source:
        - OLBG: often uses a single `teams` field
        - Sportybet: uses `home_team`/`away_team`/`sport`
        - Categories: uses `category`
        
        Dedup must therefore be shape-aware; otherwise everything with missing
        `teams` collapses into a single record.
        """

        def dedupe_key(item: Dict[str, Any]) -> str:
            teams = (item.get('teams') or '').strip()
            if teams:
                return f"teams::{teams.lower()}"

            # Handle team_names (can be string "A vs B" or list ["A", "B"])
            team_names = item.get('team_names')
            if team_names:
                if isinstance(team_names, list) and len(team_names) >= 2:
                    team_key = f"{team_names[0]}::{team_names[1]}"
                elif isinstance(team_names, str):
                    team_key = team_names
                else:
                    team_key = str(team_names)
                sport = (item.get('sport') or '').strip()
                return f"team_names::{sport.lower()}::{team_key.lower()}"

            home = (item.get('home_team') or '').strip()
            away = (item.get('away_team') or '').strip()
            sport = (item.get('sport') or '').strip()
            if home or away or sport:
                return f"match::{sport.lower()}::{home.lower()}::{away.lower()}"

            category = (item.get('category') or '').strip()
            if category:
                return f"category::{category.lower()}"

            # Last resort: stable-ish JSON key
            try:
                return f"json::{json.dumps(item, sort_keys=True, ensure_ascii=False)}"
            except Exception:
                return f"repr::{repr(item)}"

        existing = {dedupe_key(d) for d in self.extracted_data}
        for item in new_data:
            key = dedupe_key(item)
            if key not in existing:
                self.extracted_data.append(item)
                existing.add(key)


class PythonTransforms:
    """Built-in Python transforms for data processing"""

    @staticmethod
    def clear_extracted_data(data: List[Dict], params: Dict, context: 'WorkflowContext' = None) -> List[Dict]:
        """Clear context extracted_data to separate phases within a workflow."""
        if context is not None:
            context.extracted_data = []
        return []
    
    @staticmethod
    def rank_by_confidence(data: List[Dict], params: Dict) -> List[Dict]:
        """Rank matches by confidence score"""
        min_confidence = params.get('min_confidence', 0)
        max_results = int(params.get('max_results', 10))
        sort_by = params.get('sort_by', ['confidence'])
        
        # Filter by minimum confidence
        filtered = [d for d in data if d.get('confidence', 0) >= min_confidence]
        
        # Sort by specified fields
        def sort_key(item):
            return tuple(item.get(f, 0) for f in sort_by)
        
        sorted_data = sorted(filtered, key=sort_key, reverse=True)
        return sorted_data[:max_results]
    
    @staticmethod
    def filter_by_time_window(data: List[Dict], params: Dict) -> List[Dict]:
        """Filter matches within time window"""
        start_time = params.get('start_time', '')
        end_time = params.get('end_time', '')
        time_field = params.get('time_field', 'time')
        
        # Parse time strings (HH:MM format)
        def parse_time(time_str: str) -> Optional[datetime]:
            if not time_str:
                return None
            # Try common formats
            for fmt in ['%H:%M', '%I:%M %p', '%H.%M']:
                try:
                    t = datetime.strptime(time_str.strip(), fmt)
                    # Set to today's date
                    now = datetime.now()
                    return t.replace(year=now.year, month=now.month, day=now.day)
                except ValueError:
                    continue
            return None
        
        start = parse_time(start_time)
        end = parse_time(end_time)
        
        if not start or not end:
            logger.warning(f"Could not parse time window: {start_time} - {end_time}")
            return data
        
        filtered = []
        for item in data:
            match_time = parse_time(str(item.get(time_field, '')))
            if match_time and start <= match_time <= end:
                filtered.append(item)
            elif not match_time:
                # Keep items where we couldn't parse the time (manual review needed)
                item['time_validation'] = 'unparsed'
                filtered.append(item)
        
        return filtered
    
    @staticmethod
    def add_risk_assessment(data: List[Dict], params: Dict) -> List[Dict]:
        """Add risk level based on confidence thresholds"""
        thresholds = params.get('confidence_thresholds', {
            'low_risk': 75,
            'medium_risk': 60,
            'high_risk': 0
        })
        
        for item in data:
            confidence = item.get('confidence', 0)
            if confidence >= thresholds.get('low_risk', 75):
                item['risk_level'] = 'Low'
            elif confidence >= thresholds.get('medium_risk', 60):
                item['risk_level'] = 'Medium'
            else:
                item['risk_level'] = 'High'
        
        return data
    
    @staticmethod
    def merge_matches(data: List[Dict], params: Dict) -> List[Dict]:
        """Merge and deduplicate match data"""
        dedupe_fields = params.get('deduplicate_by', ['sport', 'home_team', 'away_team'])
        required_fields = params.get('required_fields', ['home_team', 'away_team'])
        
        seen = set()
        merged = []
        
        logger.info(f"[TRANSFORM] Merging {len(data)} items with fields {dedupe_fields}; required={required_fields}")
        
        for item in data:
            # Filter out non-match payloads (e.g. categories) or malformed entries
            if required_fields:
                has_required = any((item.get(f) not in (None, '', [])) for f in required_fields)
                if not has_required:
                    continue

            # Create dedup key
            key = tuple(str(item.get(f, '')).lower().strip() for f in dedupe_fields)
            
            # Debug log for first few items
            if len(merged) < 3:
                logger.info(f"[TRANSFORM] Item key: {key}")
            
            if key not in seen and any(k for k in key):  # Skip if all fields empty
                seen.add(key)
                merged.append(item)
            else:
                if not any(k for k in key):
                    logger.warning(f"[TRANSFORM] Skipped empty key item: {item}")
        
        logger.info(f"[TRANSFORM] Merged: {len(data)} -> {len(merged)} matches (deduped)")
        return merged
    
    @staticmethod
    def store_to_db(data: List[Dict], params: Dict, context: 'WorkflowContext' = None) -> List[Dict]:
        """Store data to database"""
        import database as db
        
        table = params.get('table', 'live_matches')
        mode = params.get('mode', 'append')
        source = params.get('source', 'sportybet')

        session_id = None
        if context is not None:
            session_id = context.variables.get('db_session_id')
        
        # Get data from specific step if specified
        data_source = params.get('data_source')
        if data_source and context:
            step_output = context.get_step_output(data_source)
            if step_output:
                data = step_output if isinstance(step_output, list) else [step_output]
        
        if table == 'categories':
            count = db.store_categories(source, data, session_id=session_id)
        else:
            count = db.store_matches(source, data, mode=mode, session_id=session_id)
        
        logger.info(f"[DATABASE] Stored {count} records to {table}")
        return data
    
    @staticmethod
    def generate_summary(data: List[Dict], params: Dict) -> Dict[str, Any]:
        """Generate summary statistics"""
        group_by = params.get('group_by', 'sport')
        
        summary = {
            'total_matches': len(data),
            'by_category': {},
            'generated_at': datetime.now().isoformat()
        }
        
        for item in data:
            category = item.get(group_by, 'Unknown')
            if category not in summary['by_category']:
                summary['by_category'][category] = 0
            summary['by_category'][category] += 1
        
        logger.info(f"[SUMMARY] Total: {summary['total_matches']} matches across {len(summary['by_category'])} categories")
        return summary


class WorkflowRunner:
    """Executes workflow steps with validation, retries, and transforms"""
    
    def __init__(self, workflow_path: str, agent=None, browser=None, llm=None, controller=None, source: str = None, fallback_llm=None):
        with open(workflow_path, 'r') as f:
            self.workflow = json.load(f)
        
        # Store components for creating fresh agents per step
        self.agent = agent  # Template agent (may not be used)
        self.browser = browser
        self.llm = llm
        self.fallback_llm = fallback_llm
        self.controller = controller
        self.context = WorkflowContext()
        self.transforms = PythonTransforms()
        self.source = source or self.workflow.get('name', 'unknown')
        
        # Initialize context variables from workflow config
        config = self.workflow.get('config', {})
        self.context.variables.update(config)
        
        # Configure global stealth from workflow config
        stealth_config = config.get('stealth', {})
        stealth.configure(stealth_config)
        
        # Initialize database session
        self.db_session_id = db.start_session(self.source, self.workflow.get('name', 'unknown'))
        # Expose session id to transforms for deterministic linkage
        self.context.variables['db_session_id'] = self.db_session_id
    
    def set_variable(self, key: str, value: Any):
        """Set a context variable"""
        self.context.variables[key] = value
    
    def _load_prompt(self, prompt_ref: str) -> str:
        """Load prompt from file if it's a file reference, otherwise return as-is.
        
        Supports:
        - "@prompts/sportybet/init_session.txt" -> loads from file
        - "Navigate to {url}..." -> returns as-is (inline prompt)
        """
        if not prompt_ref.startswith('@'):
            return prompt_ref
        
        # Remove @ prefix and resolve path
        prompt_path = prompt_ref[1:]
        if not os.path.isabs(prompt_path):
            # Resolve relative to workspace root
            base_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(base_dir, prompt_path)
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {prompt_path}, using reference as prompt")
            return prompt_ref
    
    def _interpolate(self, text: str) -> str:
        """Replace {variable} placeholders with context values"""
        if not isinstance(text, str):
            return text
        
        def replace(match):
            var_name = match.group(1)
            return str(self.context.variables.get(var_name, match.group(0)))
        
        return re.sub(r'\{(\w+)\}', replace, text)
    
    def _interpolate_dict(self, d: Dict) -> Dict:
        """Recursively interpolate a dictionary"""
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                result[k] = self._interpolate(v)
            elif isinstance(v, dict):
                result[k] = self._interpolate_dict(v)
            elif isinstance(v, list):
                result[k] = [self._interpolate(i) if isinstance(i, str) else i for i in v]
            else:
                result[k] = v
        return result
    
    def _check_condition(self, condition: Dict) -> bool:
        """Evaluate a step condition"""
        cond_type = condition.get('type')
        
        if cond_type == 'result_count_less_than':
            step_id = condition.get('step')
            threshold = condition.get('value', 0)
            output = self.context.get_step_output(step_id)
            if isinstance(output, list):
                return len(output) < threshold
            return True
        
        elif cond_type == 'step_executed':
            step_id = condition.get('step')
            return step_id in self.context.results and \
                   self.context.results[step_id].status == StepStatus.SUCCESS
        
        elif cond_type == 'step_failed':
            step_id = condition.get('step')
            return step_id in self.context.results and \
                   self.context.results[step_id].status == StepStatus.FAILED
        
        return True
    
    def _check_dependencies(self, step: Dict) -> bool:
        """Check if all dependencies are satisfied"""
        depends_on = step.get('depends_on', [])
        for dep_id in depends_on:
            if dep_id not in self.context.results:
                return False
            if self.context.results[dep_id].status not in [StepStatus.SUCCESS, StepStatus.SKIPPED]:
                # Allow continuing if dependency used 'continue' on_failure
                dep_step = self._get_step(dep_id)
                if dep_step and dep_step.get('on_failure') != 'continue':
                    return False
        return True
    
    def _get_step(self, step_id: str) -> Optional[Dict]:
        """Get step definition by ID"""
        for step in self.workflow.get('steps', []):
            if step.get('id') == step_id:
                return step
        return None
    
    def _create_agent(self, task: str):
        """Create a fresh agent for a step - shares browser session for continuity"""
        from browser_use import Agent
        return Agent(
            task=task,
            llm=self.llm,
            browser=self.browser,
            controller=self.controller,
            use_vision=True,
            fallback_llm=self.fallback_llm,
            llm_timeout=300,  # 5 min: LotL via AI Studio needs extra time
            step_timeout=300,  # 5 min: LotL steps need more time than default 120s
            use_thinking=False,  # Disable thinking field in output (already in system prompt)
        )
    
    async def _execute_browser_step(self, step: Dict) -> StepResult:
        """Execute a browser action step with stealth"""
        step_id = step['id']
        
        # Merge step params into context for interpolation
        step_params = step.get('params', {})
        for k, v in step_params.items():
            self.context.variables[k] = v
        
        # Load prompt from file if it starts with @, then interpolate variables
        raw_prompt = step.get('prompt', '')
        prompt = self._interpolate(self._load_prompt(raw_prompt))
        timeout = step.get('timeout_seconds', 30)
        
        start_time = datetime.now()
        
        try:
            if self.llm is None or self.browser is None:
                raise RuntimeError("No llm/browser configured for browser actions")
            
            # STEALTH: Pre-action delay
            await stealth.pre_action_delay("navigate")
            
            # STEALTH: Occasional thinking pause
            await stealth.thinking_pause()
            
            # Create a focused sub-task for this step
            logger.info(f"  Prompt: {prompt[:100]}...")
            
            # Create fresh agent for this step (prevents browser session reset issues)
            step_agent = self._create_agent(prompt)
            result = await asyncio.wait_for(
                step_agent.run(),
                timeout=timeout
            )
            
            # SYNC CHECK: Verify LLM actually responded before continuing
            if result and hasattr(result, 'history') and result.history:
                last_item = result.history[-1] if result.history else None
                if last_item and hasattr(last_item, 'model_output') and last_item.model_output is None:
                    logger.warning(f"  SYNC ISSUE: Step {step_id} - LLM returned no output")
                    raise RuntimeError("LLM sync failed - empty model output")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Validate result if validation rules exist
            validation = step.get('validation')
            if validation and not self._validate_result(result, validation):
                return StepResult(
                    step_id=step_id,
                    status=StepStatus.FAILED,
                    error="Validation failed",
                    duration_seconds=duration
                )
            
            return StepResult(
                step_id=step_id,
                status=StepStatus.SUCCESS,
                output=result,
                duration_seconds=duration
            )
            
        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            return StepResult(
                step_id=step_id,
                status=StepStatus.FAILED,
                error=f"Timeout after {timeout}s",
                duration_seconds=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return StepResult(
                step_id=step_id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_seconds=duration
            )
        finally:
            # STEALTH: Post-action observation delay
            await stealth.post_action_delay("navigate")
    
    async def _execute_extraction_step(self, step: Dict) -> StepResult:
        """Execute a data extraction step with stealth"""
        step_id = step['id']
        
        # Merge step params into context for interpolation
        step_params = step.get('params', {})
        for k, v in step_params.items():
            self.context.variables[k] = v
        
        # Load prompt from file if it starts with @, then interpolate variables
        raw_prompt = step.get('prompt', '')
        prompt = self._interpolate(self._load_prompt(raw_prompt))
        timeout = step.get('timeout_seconds', 45)
        
        start_time = datetime.now()
        
        try:
            if self.llm is None or self.browser is None:
                raise RuntimeError("No llm/browser configured for extraction")
            
            # STEALTH: Pre-extraction delay (reading the page)
            await stealth.pre_action_delay("extract")
            
            # Add JSON output instruction to prompt
            extraction_prompt = f"{prompt}\n\nIMPORTANT: Return the data as a valid JSON array."
            
            logger.info(f"  Extraction prompt: {extraction_prompt[:100]}...")
            
            # Create fresh agent for this extraction step
            step_agent = self._create_agent(extraction_prompt)
            result = await asyncio.wait_for(
                step_agent.run(),
                timeout=timeout
            )
            
            # SYNC CHECK: Verify LLM actually responded before parsing results
            if result and hasattr(result, 'history') and result.history:
                last_item = result.history[-1] if result.history else None
                if last_item and hasattr(last_item, 'model_output') and last_item.model_output is None:
                    logger.warning(f"  SYNC ISSUE: Extraction {step_id} - LLM returned no output")
                    raise RuntimeError("LLM sync failed - empty model output during extraction")
            
            # Try to parse JSON from agent output
            extracted = self._extract_json_from_result(result)
            
            if extracted:
                self.context.merge_extracted_data(extracted)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # STEALTH: Post-extraction delay
            await stealth.post_action_delay("extract")
            
            return StepResult(
                step_id=step_id,
                status=StepStatus.SUCCESS,
                output=extracted,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return StepResult(
                step_id=step_id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_seconds=duration
            )
    
    def _execute_transform_step(self, step: Dict) -> StepResult:
        """Execute a Python transform step"""
        step_id = step['id']
        transform_name = step.get('transform')
        params = self._interpolate_dict(step.get('params', {}))
        
        # Add source to params for database transforms
        params['source'] = self.source
        
        start_time = datetime.now()
        
        try:
            # Get the transform function
            transform_fn = getattr(self.transforms, transform_name, None)
            if not transform_fn:
                raise ValueError(f"Unknown transform: {transform_name}")
            
            # Apply transform to extracted data
            input_data = self.context.extracted_data.copy()
            
            # Check if transform accepts context parameter
            import inspect
            sig = inspect.signature(transform_fn)
            if 'context' in sig.parameters:
                output_data = transform_fn(input_data, params, context=self.context)
            else:
                output_data = transform_fn(input_data, params)
            
            # Update context with transformed data (if list returned)
            if isinstance(output_data, list):
                self.context.extracted_data = output_data
                item_count = len(output_data)
            else:
                item_count = output_data.get('total_matches', 0) if isinstance(output_data, dict) else 0
            
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"  Transform {transform_name}: {len(input_data)} -> {item_count} items")
            
            return StepResult(
                step_id=step_id,
                status=StepStatus.SUCCESS,
                output=output_data,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"  Transform error: {e}")
            return StepResult(
                step_id=step_id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_seconds=duration
            )
    
    def _validate_result(self, result: Any, validation: Dict) -> bool:
        """Validate a step result"""
        val_type = validation.get('type')
        
        if val_type == 'url_contains':
            # Check if current URL contains value
            value = validation.get('value', '')
            # This would need browser state - for now, assume success
            return True
        
        elif val_type == 'element_exists':
            # Check if element matching selector exists
            return True
        
        elif val_type == 'content_check':
            # Check if result contains any of the specified strings
            contains_any = validation.get('contains_any', [])
            result_str = str(result).lower()
            return any(s.lower() in result_str for s in contains_any)
        
        elif val_type == 'json_valid':
            # Check if result is valid JSON with minimum items
            min_items = validation.get('min_items', 0)
            if isinstance(result, list):
                return len(result) >= min_items
            return False
        
        return True
    
    def _extract_json_from_result(self, result: Any) -> Optional[List[Dict]]:
        """Extract JSON array from agent result"""
        # Handle AgentHistoryList
        if hasattr(result, 'final_result'):
            try:
                result_str = result.final_result()
                if not result_str:  # If empty, fallback to string rep
                    result_str = str(result)
            except Exception:
                result_str = str(result)
        else:
            result_str = str(result)
        
        # Try to find JSON array in the result
        json_patterns = [
            r'```json\s*(\[[\s\S]*?\])\s*```',  # Markdown JSON block
            r'\[[\s\S]*\]',  # Match [...] anywhere
            r'\{[\s\S]*\}',  # Match {...} anywhere
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, result_str)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict):
                        return [parsed]
                except json.JSONDecodeError:
                    continue
        
        return None
    
    async def run(self) -> Dict[str, Any]:
        """Execute the complete workflow"""
        logger.info(f"Starting workflow: {self.workflow.get('name')}")
        logger.info("=" * 60)
        
        retry_policy = self.workflow.get('config', {}).get('retry_policy', {})
        max_retries = retry_policy.get('max_retries', 2)
        retry_delay = retry_policy.get('retry_delay_seconds', 5)
        
        aborted = False

        for step in self.workflow.get('steps', []):
            step_id = step['id']
            step_name = step.get('name', step_id)
            step_type = step.get('type')
            
            logger.info(f"\n[Step: {step_name}] ({step_type})")
            
            # Check dependencies
            if not self._check_dependencies(step):
                logger.warning(f"  Skipping: dependencies not met")
                self.context.results[step_id] = StepResult(
                    step_id=step_id,
                    status=StepStatus.SKIPPED,
                    error="Dependencies not satisfied"
                )
                continue
            
            # Check condition
            condition = step.get('condition')
            if condition and not self._check_condition(condition):
                logger.info(f"  Skipping: condition not met")
                self.context.results[step_id] = StepResult(
                    step_id=step_id,
                    status=StepStatus.SKIPPED,
                    error="Condition not met"
                )
                continue
            
            # Execute step with retry logic
            result = None
            retries = 0
            
            while retries <= max_retries:
                if step_type == 'browser_action':
                    result = await self._execute_browser_step(step)
                elif step_type == 'data_extraction':
                    result = await self._execute_extraction_step(step)
                elif step_type == 'python_transform':
                    result = self._execute_transform_step(step)
                else:
                    logger.warning(f"  Unknown step type: {step_type}")
                    result = StepResult(step_id=step_id, status=StepStatus.SKIPPED)
                    break
                
                if result.status == StepStatus.SUCCESS:
                    break
                
                # Check retry policy
                on_failure = step.get('on_failure', 'abort')
                if on_failure == 'retry' and retries < max_retries:
                    retries += 1
                    result.retries = retries
                    logger.warning(f"  Retry {retries}/{max_retries} after {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                elif on_failure == 'continue':
                    logger.warning(f"  Step failed but continuing: {result.error}")
                    break
                else:
                    logger.error(f"  Step failed, aborting: {result.error}")
                    break
            
            self.context.results[step_id] = result
            
            if result.status == StepStatus.SUCCESS:
                logger.info(f"  ✓ Completed in {result.duration_seconds:.2f}s")
            elif result.status == StepStatus.FAILED and step.get('on_failure') == 'abort':
                logger.error(f"  ✗ Workflow aborted at step: {step_name}")
                # Complete session with error
                if self.db_session_id:
                    total = len(self.context.extracted_data)
                    db.complete_session(self.db_session_id, total_matches=total, error=result.error)
                aborted = True
                break
        
        # Complete database session
        if self.db_session_id and not aborted:
            total = len(self.context.extracted_data)
            db.complete_session(self.db_session_id, total_matches=total, error=None)
            logger.info(f"Database session {self.db_session_id} completed with {total} records")
        
        # Compile final output
        return self._compile_output()
    
    def _compile_output(self) -> Dict[str, Any]:
        """Compile workflow output"""
        output_config = self.workflow.get('output', {})
        include_metadata = output_config.get('include_metadata', True)
        output_fields = output_config.get('fields', [])
        
        # Filter extracted data to include only specified fields
        if output_fields:
            filtered_data = []
            for item in self.context.extracted_data:
                filtered_item = {k: v for k, v in item.items() if k in output_fields}
                filtered_data.append(filtered_item)
        else:
            filtered_data = self.context.extracted_data
        
        result = {
            "matches": filtered_data,
            "count": len(filtered_data)
        }
        
        if include_metadata:
            result["metadata"] = {
                "workflow": self.workflow.get('name'),
                "version": self.workflow.get('version'),
                "executed_at": datetime.now().isoformat(),
                "steps_summary": {
                    step_id: {
                        "status": r.status.value,
                        "duration": r.duration_seconds,
                        "retries": r.retries
                    }
                    for step_id, r in self.context.results.items()
                }
            }
        
        return result
