import json
import re
import asyncio
from typing import List, Callable, Any, Awaitable

from flowllm.schema.message import Message, Trajectory
from flowllm.utils.llm_utils import merge_messages_content as merge_messages_content_flowllm
from loguru import logger


def merge_messages_content(messages: List[Message | dict]) -> str:
    return merge_messages_content_flowllm(messages)


def parse_json_experience_response(response: str) -> List[dict]:
    """Parse JSON formatted experience response"""
    try:
        # Extract JSON blocks
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_blocks = re.findall(json_pattern, response)

        if json_blocks:
            parsed = json.loads(json_blocks[0])

            # Handle array format
            if isinstance(parsed, list):
                experiences = []
                for exp_data in parsed:
                    if isinstance(exp_data, dict) and (
                            ("when_to_use" in exp_data and "experience" in exp_data) or
                            ("condition" in exp_data and "experience" in exp_data)
                    ):
                        experiences.append(exp_data)

                return experiences


            # Handle single object
            elif isinstance(parsed, dict) and (
                    ("when_to_use" in parsed and "experience" in parsed) or
                    ("condition" in parsed and "experience" in parsed)
            ):
                return [parsed]

        # Fallback: try to parse entire response
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON experience response: {e}")

    return []


def get_trajectory_context(trajectory: Trajectory, step_sequence: List[Message]) -> str:
    """Get context of step sequence within trajectory"""
    try:
        # Find position of step sequence in trajectory
        start_idx = 0
        for i, step in enumerate(trajectory.messages):
            if step == step_sequence[0]:
                start_idx = i
                break

        # Extract before and after context
        context_before = trajectory.messages[max(0, start_idx - 2):start_idx]
        context_after = trajectory.messages[start_idx + len(step_sequence):start_idx + len(step_sequence) + 2]

        context = f"Query: {trajectory.metadata.get('query', 'N/A')}\n"

        if context_before:
            context += "Previous steps:\n" + "\n".join(
                [f"- {step.content[:100]}..." for step in context_before]) + "\n"

        if context_after:
            context += "Following steps:\n" + "\n".join([f"- {step.content[:100]}..." for step in context_after])

        return context

    except Exception as e:
        logger.error(f"Error getting trajectory context: {e}")
        return f"Query: {trajectory.metadata.get('query', 'N/A')}"

def parse_update_insight_response(response_text: str, language: str = "en") -> str:
    """Parse update insight response to extract updated insight content"""
    import re

    # Pattern to match both Chinese and English insight formats
    # Chinese: {user_name}的资料: <信息>
    # English: {user_name}'s profile: <Information>
    if language in ["zh", "cn"]:
        pattern = r"的资料[：:]\s*<([^<>]+)>"
    else:
        pattern = r"profile[：:]\s*<([^<>]+)>"

    matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)

    if matches:
        insight_content = matches[0].strip()
        logger.info(f"Parsed insight content: {insight_content}")
        return insight_content

    # Fallback: try to find content between angle brackets
    fallback_pattern = r"<([^<>]+)>"
    fallback_matches = re.findall(fallback_pattern, response_text)
    if fallback_matches:
        # Get the last match as it's likely the final answer
        insight_content = fallback_matches[-1].strip()
        logger.info(f"Parsed insight content (fallback): {insight_content}")
        return insight_content

    logger.warning("No insight content found in response")
    return ""


async def handle_429_error(
    llm_call: Callable[[], Awaitable[Any]],
    max_retries: int = 3,
    sleep_seconds: int = 30
) -> Any:
    """
    Handle 429 (rate limit) errors by sleeping and retrying.
    
    Args:
        llm_call: An async callable that makes an LLM API call
        max_retries: Maximum number of retries (default: 3)
        sleep_seconds: Seconds to sleep when 429 error occurs (default: 30)
    
    Returns:
        The result of the LLM call
    
    Raises:
        Exception: If the error is not a 429 error, or if max retries are exceeded
    """
    for attempt in range(max_retries):
        try:
            return await llm_call()
        except Exception as e:
            error_str = str(e)
            # Check if it's a 429 error (rate limit)
            if "429" in error_str or "Error code: 429" in error_str or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limit error (429) encountered. Sleeping {sleep_seconds} seconds before retry. "
                        f"Attempt {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(sleep_seconds)
                    continue
                else:
                    logger.error(f"Rate limit error (429) persisted after {max_retries} attempts")
                    raise
            else:
                # Not a 429 error, re-raise immediately
                raise
