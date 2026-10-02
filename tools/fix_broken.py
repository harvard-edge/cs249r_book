import yaml

with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

fixes = {
    "\\EndtoendRequestLatency": "L_{\\text{lat, total}}",
    "\\TakeoverResponseTime": "T_{\\Sigma, 1}",
    "\\TakeoverResponseTimeThree": "T_{\\Sigma, 2}",
    "\\SystemHazardRate": "\\lambda_{\\text{system}, h}",
    "\\HazardousProposalRate": "\\lambda_{\\text{proposal}, h}",
    "\\NormalContactForce": "f_{k, \\perp}",
    "\\TangentialContactShear": "\\mathbf{f}_{k, \\parallel}",
    "\\AllowableCollisionEnergy": "E_{k, \\text{allow}}"
}

deletes = ["\\EndtoendRequestLatencyTwo", "\\TakeoverResponseTimeTwo", "\\TakeoverResponseTimeFour", 
           "\\SystemHazardRateTwo", "\\HazardousProposalRateTwo", "\\NormalContactForceTwo", 
           "\\TangentialContactShearTwo", "\\AllowableCollisionEnergyTwo"]

new_vars = []
for var in data["variables"]:
    if var["macro"] in deletes:
        continue
    if var["macro"] in fixes:
        var["latex"] = fixes[var["macro"]]
    new_vars.append(var)

data["variables"] = new_vars

with open("books/shared/_includes/notation_registry.yml", "w") as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)
