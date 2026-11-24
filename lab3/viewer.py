"""Окно с вкладками для графиков и пояснениями к ним."""

def show_gallery(title, pages, add_controls=None):
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title(title)
    width = min(1120, root.winfo_screenwidth() - 55)
    height = min(800, root.winfo_screenheight() - 100)
    root.geometry(f"{width}x{height}+20+20")
    root.minsize(780, 530)
    style = ttk.Style(root)
    style.configure("TNotebook.Tab", padding=(10, 6))
    ttk.Label(root, text=title, font=("Segoe UI", 19, "bold")).pack(
        anchor="w", padx=16, pady=(12, 6)
    )
    if add_controls:
        controls = ttk.Frame(root, padding=(16, 5))
        controls.pack(fill="x")
        add_controls(controls)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=12, pady=(4, 12))
    canvases = []
    for _, tab_title, description, figure in pages:
        tab = ttk.Frame(notebook, padding=5)
        notebook.add(tab, text=tab_title)
        caption = ttk.Label(tab, text=description, padding=(10, 8), wraplength=800)
        caption.pack(fill="x")
        caption.bind("<Configure>", lambda event: event.widget.configure(
            wraplength=max(200, event.width - 25)
        ))
        canvas = FigureCanvasTkAgg(figure, master=tab)
        toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        canvases.append(canvas)

    # Первая вкладка может обновляться, например после нового прогона СМО.
    if add_controls:
        controls.redraw = canvases[0].draw_idle
    root.mainloop()
