import os
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from tkinter import Tk, filedialog, messagebox, Toplevel, Label, ttk
import threading

class ProgressWindow:
    def __init__(self, total_cells):
        self.window = Toplevel()
        self.window.title("Ejecutando Notebook")
        self.window.geometry("500x150")
        self.window.resizable(False, False)
        self.window.attributes('-topmost', True)
        
        # Centrar ventana
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.window.winfo_screenheight() // 2) - (150 // 2)
        self.window.geometry(f"500x150+{x}+{y}")
        
        # Etiqueta de estado
        self.label = Label(self.window, text="Preparando ejecución...", font=("Arial", 10))
        self.label.pack(pady=20)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(
            self.window, 
            length=450, 
            mode='determinate',
            maximum=total_cells
        )
        self.progress.pack(pady=10)
        
        # Etiqueta de progreso
        self.progress_label = Label(self.window, text="0 / 0 celdas ejecutadas", font=("Arial", 9))
        self.progress_label.pack(pady=5)
        
        self.total_cells = total_cells
        self.current_cell = 0
    
    def update(self, cell_index, cell_type="code"):
        self.current_cell = cell_index + 1
        self.progress['value'] = self.current_cell
        self.label.config(text=f"Ejecutando celda {self.current_cell} de {self.total_cells} ({cell_type})...")
        self.progress_label.config(text=f"{self.current_cell} / {self.total_cells} celdas procesadas")
        self.window.update()
    
    def set_text(self, text):
        self.label.config(text=text)
        self.window.update()
    
    def close(self):
        self.window.destroy()

class CustomExecutePreprocessor(ExecutePreprocessor):
    def __init__(self, progress_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_window = progress_window
    
    def preprocess_cell(self, cell, resources, cell_index):
        # Actualizar ventana de progreso
        if self.progress_window:
            self.progress_window.update(cell_index, cell.cell_type)
        return super().preprocess_cell(cell, resources, cell_index)

def export_notebook():
    # Ocultar la ventana principal de Tkinter
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Seleccionar el notebook de entrada
    print("Selecciona el archivo Jupyter Notebook (.ipynb)...")
    ipynb_path = filedialog.askopenfilename(
        title="Seleccionar Notebook",
        filetypes=[("Jupyter Notebooks", "*.ipynb"), ("Todos los archivos", "*.*")]
    )
    
    if not ipynb_path:
        print("❌ No se seleccionó ningún archivo.")
        root.destroy()
        return
    
    print(f"✔ Archivo seleccionado: {ipynb_path}")
    
    # ==================== NUEVA PREGUNTA: EJECUTAR O NO ====================
    ejecutar_notebook = messagebox.askyesno(
        "Modo de exportación",
        "¿Deseas EJECUTAR el notebook antes de exportar?\n\n"
        "• SÍ: Ejecutará todas las celdas (requiere dependencias instaladas)\n"
        "• NO: Solo convertirá el notebook tal como está (más rápido)"
    )
    # ========================================================================
    
    # Preguntar si mostrar el código
    mostrar_codigo = messagebox.askyesno(
        "Mostrar código",
        "¿Deseas incluir el código en el informe HTML?"
    )
    
    # Seleccionar ubicación de salida
    print("Selecciona dónde guardar el archivo HTML...")
    output_path = filedialog.asksaveasfilename(
        title="Guardar informe HTML como",
        defaultextension=".html",
        filetypes=[("HTML", "*.html"), ("Todos los archivos", "*.*")],
        initialfile=f"{os.path.splitext(os.path.basename(ipynb_path))[0]}_informe.html"
    )
    
    if not output_path:
        print("❌ No se seleccionó ubicación de salida.")
        root.destroy()
        return
    
    print(f"✔ Guardando en: {output_path}")
    
    progress_window = None
    
    try:
        # Leer el notebook
        print("\n📖 Leyendo notebook...")
        nb = nbformat.read(ipynb_path, as_version=4)
        
        # Contar celdas de código
        total_cells = len(nb.cells)
        print(f"📊 Total de celdas: {total_cells}")
        
        # ==================== EJECUCIÓN CONDICIONAL ====================
        if ejecutar_notebook:
            # Crear ventana de progreso
            progress_window = ProgressWindow(total_cells)
            progress_window.set_text("Iniciando ejecución del notebook...")
            
            # Ejecutar el notebook con seguimiento de progreso
            print("⚙️ Ejecutando notebook...")
            ep = CustomExecutePreprocessor(
                progress_window,
                timeout=600, 
                kernel_name='python3'
            )
            
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(ipynb_path) or '.'}})
            
            progress_window.set_text("✔ Notebook ejecutado correctamente")
            print("✔ Notebook ejecutado correctamente")
        else:
            print("⏩ Saltando ejecución (modo conversión directa)")
        # ================================================================
        
        # Exportar a HTML
        if progress_window:
            progress_window.set_text("📄 Generando HTML...")
        print("📄 Generando HTML...")
        exporter = HTMLExporter()
        exporter.exclude_input = not mostrar_codigo
        
        body, _ = exporter.from_notebook_node(nb)
        
        # Guardar el archivo
        if progress_window:
            progress_window.set_text("💾 Guardando archivo...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(body)
        
        if progress_window:
            progress_window.close()
            progress_window = None
        
        modo = "ejecutado y exportado" if ejecutar_notebook else "exportado sin ejecutar"
        print(f"\n✅ ¡Informe {modo} exitosamente!")
        print(f"📁 Ubicación: {output_path}")
        
        # Preguntar si abrir el archivo
        if messagebox.askyesno("Éxito", f"Informe {modo} correctamente.\n¿Deseas abrirlo ahora?"):
            os.startfile(output_path)  # Windows
            # Para Linux usar: os.system(f'xdg-open "{output_path}"')
            # Para Mac usar: os.system(f'open "{output_path}"')
        
    except Exception as e:
        if progress_window:
            progress_window.close()
        error_msg = f"❌ Error al procesar el notebook:\n{str(e)}"
        print(error_msg)
        messagebox.showerror("Error", error_msg)
    
    finally:
        if progress_window:
            progress_window.close()
        root.destroy()

if __name__ == "__main__":
    print("=== EXPORTADOR DE NOTEBOOKS A HTML ===\n")
    export_notebook()