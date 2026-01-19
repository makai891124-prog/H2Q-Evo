from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, model_validator
import json

class EvolutionTask(BaseModel):
    """
    H2Q 进化任务的标准契约。
    所有进入系统的任务（无论是 AI 生成还是人工注入）必须符合此定义。
    """
    id: int = Field(default=-1, description="Unique Task ID (Auto-assigned if -1)")
    task: str = Field(..., description="The executable instruction string")
    priority: Literal['critical', 'high', 'medium', 'low'] = Field(default='medium')
    status: Literal['pending', 'completed', 'failed'] = Field(default='pending')
    source: Literal['user', 'ai'] = Field(default='ai')
    retry_count: int = Field(default=0)
    
    # 动态对齐逻辑：处理 AI 的各种幻觉字段
    @model_validator(mode='before')
    @classmethod
    def align_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
            
        # 1. 映射任务描述字段
        # AI 可能会用 description, goal, objective, summary 来代替 task
        if 'task' not in data:
            for alias in ['description', 'goal', 'objective', 'summary', 'title']:
                if alias in data and isinstance(data[alias], str):
                    data['task'] = data[alias]
                    break
        
        # 2. 映射优先级
        if 'priority' in data:
            p = data['priority'].lower()
            if p not in ['critical', 'high', 'medium', 'low']:
                data['priority'] = 'medium' # 默认降级
                
        # 3. 确保 ID 是整数
        if 'id' in data and isinstance(data['id'], str):
            if data['id'].isdigit():
                data['id'] = int(data['id'])
            else:
                data['id'] = -1 # 重置无效 ID
                
        return data

    @classmethod
    def get_prompt_schema(cls) -> str:
        """生成动态 Schema 供 Prompt 使用"""
        schema = cls.model_json_schema()
        # 简化 Schema 以节省 Token，只保留关键部分
        simplified = {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The specific coding task"},
                "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]}
            },
            "required": ["task"]
        }
        return json.dumps(simplified, indent=2)