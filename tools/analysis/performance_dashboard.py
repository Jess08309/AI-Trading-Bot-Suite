import os
import runpy


if __name__ == "__main__":
    root_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "performance_dashboard.py",
    )
    runpy.run_path(root_script, run_name="__main__")
