"""PyInstaller entry point - imports app.simple_gui as a package (needed for
its `from app.core...` absolute imports to resolve) rather than running
app/simple_gui.py directly as a script, which would put app/ itself on
sys.path instead of the project root."""

from app.simple_gui import main

if __name__ == "__main__":
    main()
