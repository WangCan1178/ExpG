"""
Dump all tool schemas under ./apis into a single JSON file.

Output format:
{
  "ApiName": {
    "name": "ApiName",
    "description": "...",
    "input_parameters": {...},
    "output_parameters": {...}
  },
  ...
}
"""

import json
from tool_manager import ToolManager


def collect_all_schemas():
    tm = ToolManager()
    schemas = {}
    for name in tm.list_all_apis():
        desc_str = tm.get_api_description(name)
        try:
            desc_obj = json.loads(desc_str)
        except json.JSONDecodeError:
            # Fallback: store raw string if it isn't valid JSON
            desc_obj = desc_str
        schemas[name] = desc_obj
    return schemas


def main(output_path: str = "all_api_schemas.json"):
    schemas = collect_all_schemas()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schemas, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(schemas)} tool schemas to {output_path}")


if __name__ == "__main__":
    main()

