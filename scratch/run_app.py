import sys
from streamlit.web import cli as stcli
import os

if __name__ == '__main__':
    script_path = os.path.join("streamlit_app_v2", "1_📊_Overview.py")
    sys.argv = ["streamlit", "run", script_path, "--server.headless=true", "--server.port=8501"]
    sys.exit(stcli.main())
