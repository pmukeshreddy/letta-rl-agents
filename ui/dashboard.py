"""
Gradio Dashboard

Visual interface for the RL skill selector.
"""

import gradio as gr
import httpx
import os
import pandas as pd
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60.0


# API helpers
async def api_get(endpoint: str):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_URL}{endpoint}", timeout=TIMEOUT)
        if resp.status_code != 200:
            return {"error": resp.text}
        return resp.json()


async def api_post(endpoint: str, data: dict = None):
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_URL}{endpoint}", json=data or {}, timeout=TIMEOUT)
        if resp.status_code != 200:
            return {"error": resp.text}
        return resp.json()


# Task execution
async def run_task(task: str, use_rl: bool, max_skills: int, use_mock: bool):
    if not task.strip():
        return "Please enter a task", [], [], 0, 0, ""
    
    result = await api_post("/task", {
        "task": task,
        "use_rl": use_rl,
        "max_skills": int(max_skills),
        "mock": use_mock,
    })
    
    if "error" in result:
        return f"Error: {result['error']}", [], [], 0, 0, ""
    
    return (
        result["response"],
        result["skills_used"],
        result["confidence_scores"],
        result["tokens_used"],
        result["latency_ms"],
        result["task_id"],
    )


async def submit_feedback(task_id: str, success: bool):
    if not task_id:
        return "No task ID - run a task first"
    
    result = await api_post("/feedback", {
        "task_id": task_id,
        "success": success,
    })
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    return f"✓ Feedback recorded for task {task_id}"


# Skills management
async def get_skills():
    result = await api_get("/skills")
    if "error" in result:
        return pd.DataFrame()
    
    if not result:
        return pd.DataFrame(columns=["Name", "Success Rate", "Uses", "Tokens"])
    
    df = pd.DataFrame([
        {
            "ID": s["id"],
            "Name": s["name"],
            "Success Rate": f"{s['success_rate']:.1%}",
            "Uses": s["usage_count"],
            "Tokens": s["token_count"],
        }
        for s in result
    ])
    return df


async def get_skill_detail(skill_id: str):
    if not skill_id:
        return "Select a skill to view details"
    
    result = await api_get(f"/skills/{skill_id}")
    if "error" in result:
        return f"Error: {result['error']}"
    
    return f"""# {result['name']}

**ID:** {result['id']}
**Success Rate:** {result['success_rate']:.1%}
**Uses:** {result['usage_count']}
**Tokens:** {result['token_count']}

---

{result['content']}
"""


# Metrics
async def get_metrics():
    result = await api_get("/metrics")
    if "error" in result:
        return {}, [], 0, 0
    
    return (
        result,
        result.get("top_skills", []),
        result.get("total_outcomes", 0),
        result.get("avg_success_rate", 0),
    )


async def get_training_history():
    result = await api_get("/policy/history")
    if "error" in result:
        return pd.DataFrame()
    
    history = result.get("history", [])
    if not history:
        return pd.DataFrame(columns=["Step", "Policy Loss", "Value Loss", "Entropy"])
    
    df = pd.DataFrame(history)
    return df


# Training
async def trigger_training(min_samples: int):
    result = await api_post(f"/train?min_samples={int(min_samples)}")
    if "error" in result:
        return f"Error: {result['error']}"
    
    if result["status"] == "insufficient_data":
        samples = result["metrics"].get("samples", 0)
        required = result["metrics"].get("required", min_samples)
        return f"⚠️ Need more data: {samples}/{required} samples"
    
    metrics = result["metrics"]
    return f"""✓ Training complete!

**Step:** {metrics.get('training_step', 0)}
**Policy Loss:** {metrics.get('policy_loss', 0):.4f}
**Value Loss:** {metrics.get('value_loss', 0):.4f}
**Entropy:** {metrics.get('entropy', 0):.4f}
**Samples Used:** {metrics.get('samples_used', 0)}
"""


# Build the UI
with gr.Blocks(title="Letta RL Skill Selector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Letta RL Skill Selector")
    gr.Markdown("RL-optimized skill selection for Letta agents — [GitHub](https://github.com/pmukeshreddy/letta-rl-agents)")
    
    # Store task_id across interactions
    current_task_id = gr.State("")
    
    with gr.Tabs():
        # Tab 1: Run Tasks
        with gr.TabItem("🚀 Run Task"):
            with gr.Row():
                with gr.Column(scale=2):
                    task_input = gr.Textbox(
                        label="Task Description",
                        placeholder="Describe what you want the agent to do...",
                        lines=3,
                    )
                    with gr.Row():
                        use_rl = gr.Checkbox(label="Use RL Selection", value=True)
                        use_mock = gr.Checkbox(label="Mock Mode (no API)", value=True)
                        max_skills = gr.Slider(
                            label="Max Skills", minimum=1, maximum=5, value=3, step=1
                        )
                    
                    with gr.Row():
                        run_btn = gr.Button("▶️ Run Task", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear")
                
                with gr.Column(scale=2):
                    response_output = gr.Textbox(label="Response", lines=8)
                    
                    with gr.Row():
                        skills_output = gr.JSON(label="Skills Used")
                        confidence_output = gr.JSON(label="Confidence")
                    
                    with gr.Row():
                        tokens_output = gr.Number(label="Tokens")
                        latency_output = gr.Number(label="Latency (ms)")
                    
                    task_id_display = gr.Textbox(label="Task ID", interactive=False)
            
            with gr.Row():
                gr.Markdown("### Feedback")
                feedback_success = gr.Button("👍 Success", variant="primary")
                feedback_fail = gr.Button("👎 Failed", variant="secondary")
                feedback_result = gr.Textbox(label="Feedback Status", interactive=False)
            
            # Event handlers
            run_btn.click(
                run_task,
                inputs=[task_input, use_rl, max_skills, use_mock],
                outputs=[response_output, skills_output, confidence_output, 
                        tokens_output, latency_output, task_id_display],
            )
            
            clear_btn.click(
                lambda: ("", [], [], 0, 0, "", ""),
                outputs=[task_input, response_output, skills_output, confidence_output,
                        tokens_output, latency_output, task_id_display, feedback_result],
            )
            
            feedback_success.click(
                lambda tid: submit_feedback(tid, True),
                inputs=[task_id_display],
                outputs=[feedback_result],
            )
            
            feedback_fail.click(
                lambda tid: submit_feedback(tid, False),
                inputs=[task_id_display],
                outputs=[feedback_result],
            )
        
        # Tab 2: Skills Library
        with gr.TabItem("📚 Skills"):
            with gr.Row():
                with gr.Column(scale=1):
                    skills_table = gr.Dataframe(
                        label="Registered Skills",
                        interactive=False,
                    )
                    refresh_skills_btn = gr.Button("🔄 Refresh")
                
                with gr.Column(scale=1):
                    skill_id_input = gr.Textbox(label="Skill ID", placeholder="Enter skill ID")
                    view_skill_btn = gr.Button("View Skill")
                    skill_detail = gr.Markdown(label="Skill Details")
            
            refresh_skills_btn.click(get_skills, outputs=[skills_table])
            view_skill_btn.click(get_skill_detail, inputs=[skill_id_input], outputs=[skill_detail])
            demo.load(get_skills, outputs=[skills_table])
        
        # Tab 3: Training
        with gr.TabItem("🎯 Training"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Train Policy")
                    gr.Markdown("Train the RL policy on collected feedback.")
                    
                    min_samples_input = gr.Slider(
                        label="Minimum Samples",
                        minimum=8, maximum=128, value=32, step=8,
                    )
                    train_btn = gr.Button("🔄 Train Policy", variant="primary")
                    train_result = gr.Markdown(label="Training Result")
                
                with gr.Column():
                    gr.Markdown("### Training History")
                    training_history = gr.Dataframe(label="Recent Training Runs")
                    refresh_history_btn = gr.Button("Refresh History")
            
            train_btn.click(trigger_training, inputs=[min_samples_input], outputs=[train_result])
            refresh_history_btn.click(get_training_history, outputs=[training_history])
        
        # Tab 4: Metrics
        with gr.TabItem("📊 Metrics"):
            with gr.Row():
                total_outcomes = gr.Number(label="Total Tasks")
                avg_success = gr.Number(label="Avg Success Rate")
            
            metrics_json = gr.JSON(label="Full Metrics")
            top_skills_json = gr.JSON(label="Top Skills")
            
            refresh_metrics_btn = gr.Button("🔄 Refresh Metrics")
            
            async def refresh_all_metrics():
                metrics, top_skills, outcomes, success_rate = await get_metrics()
                return metrics, top_skills, outcomes, success_rate
            
            refresh_metrics_btn.click(
                refresh_all_metrics,
                outputs=[metrics_json, top_skills_json, total_outcomes, avg_success],
            )
            demo.load(
                refresh_all_metrics,
                outputs=[metrics_json, top_skills_json, total_outcomes, avg_success],
            )
    
    gr.Markdown("---")
    gr.Markdown("Built with 🧠 by [Mukesh Reddy](https://github.com/pmukeshreddy)")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
