import marimo

__generated_with = "0.19.6"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import mlsysim
    import pandas as pd
    import matplotlib.pyplot as plt
    return mlsysim, mo, pd, plt

@app.cell
def __(mo):
    mo.md(
        """
        # The Global Placement Search (Sustainable AI)
        
        This notebook demonstrates the **PlacementOptimizer** (Tier 3: Engineering Engine). 
        
        When training a large model, the physical location of the datacenter changes the carbon footprint by up to 40x. But greener grids (like Quebec) might have higher base electricity costs than carbon-heavy grids (like Iowa).
        
        By using an Optimizer, we can search the entire `InfraZoo` to find the location that minimizes the combined objective function: `TCO + (Carbon * Carbon Tax)`.
        
        **Adjust the Carbon Tax slider below to see how the optimal datacenter location shifts!**
        """
    )
    return

@app.cell
def __(mo):
    tax_slider = mo.ui.slider(
        start=0, stop=500, step=10, value=0, 
        label="Carbon Tax ($ per ton CO2)"
    )
    
    mo.md(
        f"""
        ### 1. Set the Policy Variable
        {tax_slider}
        """
    )
    return tax_slider,

@app.cell
def __(mlsysim, mo, pd, plt, tax_slider):
    # 1. Setup the Demand and Fleet
    model = mlsysim.Models.Language.GPT3
    
    fleet = mlsysim.Systems.Clusters.H100_1K
    duration = 30.0 # days
    
    # 2. Run the Optimizer
    optimizer = mlsysim.PlacementOptimizer()
    
    regions = ["US_Avg", "Quebec", "Iowa", "Norway", "Poland"]
    
    result = optimizer.solve(
        fleet=fleet,
        duration_days=duration,
        regions=regions,
        carbon_tax_per_ton=tax_slider.value
    )
    
    # 3. Format the Results
    best_region = result.best_region
    
    df = pd.DataFrame(result.top_candidates)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    x = range(len(df))
    ax1.bar(x, df["objective"], color=["#10b981" if r == best_region else "#94a3b8" for r in df["region"]])
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["region"])
    ax1.set_ylabel("Total Objective Cost ($)")
    ax1.set_title(f"Optimal Placement at ${tax_slider.value}/ton Carbon Tax")
    
    # Add a twin axis for just carbon
    ax2 = ax1.twinx()
    ax2.plot(x, df["carbon"], color="#ef4444", marker="o", linestyle="dashed", linewidth=2)
    ax2.set_ylabel("Carbon Footprint (Tons CO2)", color="#ef4444")
    ax2.tick_params(axis="y", labelcolor="#ef4444")
    
    plt.tight_layout()
    
    scorecard = mo.md(
        f"""
        ### 2. Optimizer Scorecard
        
        **Winner:** `{best_region}`
        
        **Objective Cost:** `${result.objective_value:,.0f}`
        * TCO (Hardware + Energy): `${result.lowest_tco:,.0f}`
        * Carbon Footprint: `{result.carbon_footprint:,.0f} tons`
        
        At a carbon tax of `${tax_slider.value}/ton`, the penalty is `${(result.carbon_footprint * tax_slider.value):,.0f}`.
        """
    )
    
    mo.vstack([scorecard, mo.as_html(fig)])
    return ax1, ax2, best_region, df, duration, fleet, fig, model, optimizer, regions, result, scorecard, x

if __name__ == "__main__":
    app.run()
