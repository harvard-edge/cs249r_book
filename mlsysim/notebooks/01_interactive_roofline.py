import marimo

__generated_with = "0.19.6"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import mlsysim
    import matplotlib.pyplot as plt
    return mlsysim, mo, plt

@app.cell
def __(mo):
    mo.md(
        """
        # Interactive Roofline Analysis
        
        This notebook demonstrates the **SingleNodeModel** (Tier 1: Mechanistic Evaluation). 
        By changing the batch size or the hardware target, you can observe the "Memory Wall" in real-time.
        
        The Roofline model tells us if a workload is **Compute-Bound** (limited by FLOPs) or **Memory-Bound** (limited by bandwidth). 
        The turning point is called the **Ridge Point**.
        """
    )
    return

@app.cell
def __(mo):
    hw_dropdown = mo.ui.dropdown(
        options=["A100", "H100", "MI300X", "Mac_M2_Ultra", "ESP32_S3"],
        value="H100",
        label="Hardware Target"
    )
    
    model_dropdown = mo.ui.dropdown(
        options=["ResNet50", "Llama3_8B", "GPT3", "WakeVision_CNN"],
        value="Llama3_8B",
        label="Workload Architecture"
    )
    
    batch_slider = mo.ui.slider(
        start=1, stop=2048, step=1, value=1, 
        label="Batch Size"
    )
    
    mo.md(
        f"""
        ### 1. Configure the System
        *Select the hardware and workload below, then drag the batch size slider.*
        
        {hw_dropdown}
        {model_dropdown}
        {batch_slider}
        """
    )
    return batch_slider, hw_dropdown, model_dropdown

@app.cell
def __(batch_slider, hw_dropdown, mlsysim, model_dropdown, mo, plt):
    # 1. Resolve Hardware
    hw_obj = getattr(mlsysim.Hardware.Cloud, hw_dropdown.value, None)
    if hw_obj is None:
        hw_obj = getattr(mlsysim.Hardware.Edge, hw_dropdown.value, None)
        if hw_obj is None:
            hw_obj = getattr(mlsysim.Hardware.Tiny, hw_dropdown.value, None)
            
    # 2. Resolve Model
    if model_dropdown.value == "ResNet50":
        mod_obj = mlsysim.Models.ResNet50
    elif model_dropdown.value == "Llama3_8B":
        mod_obj = mlsysim.Models.Language.Llama3_8B
    elif model_dropdown.value == "GPT3":
        mod_obj = mlsysim.Models.Language.GPT3
    elif model_dropdown.value == "WakeVision_CNN":
        mod_obj = mlsysim.Models.Vision.WakeVision_CNN
        
    # 3. Evaluate using the SingleNodeModel (Tier 1)
    solver = mlsysim.SingleNodeModel()
    
    try:
        profile = solver.solve(
            model=mod_obj, 
            hardware=hw_obj, 
            batch_size=batch_slider.value,
            raise_errors=True # Will throw OOM if it doesn't fit
        )
        
        # Format the output scorecard
        scorecard = mo.md(
            f"""
            ### 2. Performance Scorecard
            
            ```text
            {profile.summary()}
            ```
            """
        )
        
        # Render the Plot
        fig = profile.plot(mode="roofline")
        
        result_ui = mo.vstack([scorecard, mo.as_html(fig)])
        
    except mlsysim.core.exceptions.OOMError as e:
        result_ui = mo.md(f"""### 🚨 Out of Memory Error

**{str(e)}**""")
        
    result_ui
    return hw_obj, mod_obj, profile, result_ui, solver

if __name__ == "__main__":
    app.run()
