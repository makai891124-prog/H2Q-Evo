# src/agent.py

import json
from typing import Callable, Dict, Any

from core.observation import Observation
from core.action import Action
from core.environment import Environment
from core.llm_client import LLMClient

class Agent:
    def __init__(self, llm_client: LLMClient, task_prompt: str):
        self.llm_client = llm_client
        self.task_prompt = task_prompt
        self.history = [] # Store (observation, action) tuples

    def run_step(self, observation: Observation, environment: Environment) -> Action:
        # 1. Formulate Prompt
        prompt = self._create_prompt(observation)

        # 2. Get LLM Response
        action_string = self.llm_client.generate(prompt)

        # 3. Parse Action (Assumes Action is JSON formatted)
        try:
            action_data = json.loads(action_string)
            action = Action(**action_data)
        except json.JSONDecodeError:
            print(f"Error decoding action: {action_string}")
            action = Action(command="noop", argument="{}") # Default no-op action

        # 4. Store History
        self.history.append((observation, action))

        # 5. (NEW) Self-Reflection (Lightweight)
        self._self_reflect(observation, action, action_string, environment)

        return action

    def _create_prompt(self, observation: Observation) -> str:
        # This is a placeholder, replace with your actual prompt creation logic
        return f"You are an agent. Your task: {self.task_prompt}.\nCurrent Observation: {observation.content}\nPrevious History: {self.history}\nWhat action should you take? (Respond in JSON format)"

    def _self_reflect(self, observation: Observation, action: Action, action_string: str, environment: Environment) -> None:
        """Lightweight Self-Reflection Mechanism."""
        if len(self.history) > 1: # Reflect after the first action.
            # Get the previous observation and action.
            prev_observation, prev_action = self.history[-2]

            # Formulate a simple reflection prompt.
            reflection_prompt = f"Previous Observation: {prev_observation.content}\n" \
                                f"Previous Action: {prev_action}\n" \
                                f"Current Observation: {observation.content}\n" \
                                f"Current Action (String): {action_string}\n" \
                                f"Current Action (Parsed): {action}\n" \
                                f"Environment Info: {environment.get_info()}\n" \
                                f"Was the previous action helpful in achieving the task: {self.task_prompt}? Why or why not?  Suggest a small improvement for the next action. (Keep it concise.)"

            # Generate the reflection using the LLM.
            reflection = self.llm_client.generate(reflection_prompt)

            print(f"Self-Reflection: {reflection}\n") # Print for now, can be extended to modify behavior.

    def reset(self) -> None:
        self.history = []
