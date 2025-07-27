import gradio as gr
from gradio.themes.utils import fonts

import src.pages.execution_progress as execution_progress
import src.pages.pipeline_performance as pipeline_performance
import src.pages.pipeline_optimization as pipeline_optimization
import src.pages.pipeline_comparison as pipeline_comparison
from src.database import MongoDBClient
from src.utils.common_utils import load_yaml_config

import matplotlib
matplotlib.use('Agg')


# Load configuration once at startup
yaml_config = load_yaml_config()
print("✅ YAML configuration loaded successfully")

# Extract pipeline optimization parameters from config
exp_config_name = yaml_config.get('common_args', {}).get('exp_config_name', 'gradio_demo')
run_num = yaml_config.get('common_args', {}).get('run_nums', [1])[0]  # Take first run number
max_trials = yaml_config.get('optimisation_args', {}).get('max_total_pipelines_num', 10)
ref_point = yaml_config.get('optimisation_args', {}).get('ref_point', [0.20, 0.20])

# Initialize database client
try:
    db_client = MongoDBClient()
    print("✅ MongoDB connection established successfully")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {e}")
    db_client = None


with gr.Blocks(theme=gr.themes.Soft(font=(fonts.LocalFont("IBM Plex Mono"), "ui-sans-serif", "system-ui", "sans-serif")),
               title="VirnyFlow Demo",
               css="""
               .main-container {
                   max-width: 1200px;
                   margin: 0 auto;
                   padding: 0 20px;
               }
               """) as demo:
    with gr.Row(elem_classes=["main-container"]):
        with gr.Tabs():
            with gr.Tab("Execution Progress"):
                execution_progress.create_execution_progress_page(db_client, yaml_config)
            with gr.Tab("Pipeline Performance"):
                pipeline_performance.create_pipeline_performance_page(
                    exp_config_name=exp_config_name,
                    run_num=run_num
                )
            with gr.Tab("Pipeline Optimization"):
                pipeline_optimization.create_pipeline_optimization_page(
                    exp_config_name=exp_config_name,
                    run_num=run_num,
                    max_trials=max_trials,
                    ref_point=ref_point
                )
            with gr.Tab("Pipeline Comparison"):
                pipeline_comparison.create_pipeline_comparison_page(
                    exp_config_name=exp_config_name,
                    run_num=run_num
                )


if __name__ == "__main__":
    try:
        demo.launch()
    finally:
        # Clean up database connection
        if db_client:
            db_client.close()
            print("🔌 MongoDB connection closed")
