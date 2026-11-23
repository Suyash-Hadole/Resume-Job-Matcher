from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path("notebooks")
for path in sorted(root.glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    print(f"{path.name}: executed, code_cells={sum(c.cell_type == 'code' for c in notebook.cells)}, errors={len(errors)}")
    for error in errors:
        print("  ", error.get("ename"), error.get("evalue"))
