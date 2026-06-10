import json

log_path = r"C:\Users\HP\.gemini\antigravity\brain\2a0fe99c-f42e-442e-84d0-53dd7ea5ba3f\.system_generated\logs\transcript.jsonl"
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        idx = obj.get("step_index")
        if idx in range(73, 86):
            print(f"=== STEP {idx} ===")
            if obj.get("content"):
                print("CONTENT:")
                print(obj["content"][:800])
            if obj.get("tool_calls"):
                for tc in obj["tool_calls"]:
                    print(f"TOOL: {tc['name']}")
                    args = tc.get("args", {})
                    if "CommandLine" in args:
                        print("COMMAND:")
                        print(args["CommandLine"])
                    if "CodeContent" in args:
                        print("CODE CONTENT:")
                        print(args["CodeContent"][:300])
            print("-" * 50)
