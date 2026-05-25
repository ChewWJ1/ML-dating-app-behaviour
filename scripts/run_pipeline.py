import os
import sys
import subprocess
import time
import asyncio

# Set Windows Selector Event Loop Policy to suppress Tornado/ZMQ event loop warnings
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

def install_if_missing():
    print("Checking required runner dependencies...")
    try:
        import nbformat
        import nbconvert
        print("Required packages are available. ✅")
    except ImportError:
        print("Installing nbformat and nbconvert for headless execution...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nbformat", "nbconvert"])
        print("Dependencies installed successfully! ✅")

def run_notebook():
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_file = os.path.join(script_dir, 'ML_dating_app_behaviour.ipynb')
    print(f"\n🔄 Loading {notebook_file}...")
    
    with open(notebook_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Allow up to 3 hours for the entire execution to finish
    ep = ExecutePreprocessor(timeout=10800, kernel_name='python3')

    print("\n🚀 Starting execution... You can safely go to sleep! 😴")
    print("Please keep your computer turned on and plugged into a power source.")
    print("This will take about 7 to 10 minutes (or slightly longer on older hardware).\n")

    start_time = time.time()
    try:
        # Run execution relative to the script's directory
        ep.preprocess(nb, {'metadata': {'path': script_dir}})
        duration = time.time() - start_time
        print(f"🎉 Notebook executed successfully in {duration/60:.1f} minutes!")
        
        # Save the executed notebook back to disk
        print("💾 Saving all outputs, plots, and tables back to the notebook...")
        with open(notebook_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print(f"\n✅ All outputs have been saved to '{os.path.basename(notebook_file)}'.")
        print("When you wake up, you can open the notebook and see all completed charts and tables instantly!")
        print("The baseline, learning curves, and tuning model checkpoints are also saved to disk.")
        
    except Exception as e:
        print(f"\n❌ Error occurred during execution: {e}")
        print("Any cells executed before the error occurred have been preserved.")

if __name__ == '__main__':
    install_if_missing()
    run_notebook()
