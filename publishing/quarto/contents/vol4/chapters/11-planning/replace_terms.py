import re

file_path = '/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/11-planning/11-planning.qmd'
with open(file_path, 'r') as f:
    content = f.read()

replacements = [
    (r'\*\*\*Physical AI Systems\*\*\*\\index\{Physical AI systems!definition\}', r'***Physical AI systems***\\index{Physical AI systems!definition}'),
    (r'\*\*action chunk\*\*(?!\\index)', r'**action chunk**\\index{Action chunk!definition}'),
    (r'\*\*seam behavior\*\*(?!\\index)', r'**seam behavior**\\index{Seam behavior!definition}'),
    (r'\*\*configuration space\*\*(?!\\index)', r'**configuration space**\\index{Configuration space!definition}'),
    (r'\*\*executable trajectory\*\*(?!\\index)', r'**executable trajectory**\\index{Executable trajectory!definition}'),
    (r'\*\*trajectory packet\*\*(?!\\index)', r'**trajectory packet**\\index{Trajectory packet!definition}'),
    (r'\*\*precomputed stopping suffix\*\*(?!\\index)', r'**precomputed stopping suffix**\\index{Precomputed stopping suffix!definition}'),
    (r'\*\*embedded execution buffer\*\*(?!\\index)', r'**embedded execution buffer**\\index{Embedded execution buffer!definition}'),
    (r'\*\*Probabilistic Roadmap \(PRM\)\*\*(?!\\index)', r'**Probabilistic Roadmap (PRM)**\\index{Probabilistic Roadmap (PRM)!definition}'),
    (r'\*\*Rapidly-Exploring Random Tree \(RRT\)\*\*(?!\\index)', r'**Rapidly-Exploring Random Tree (RRT)**\\index{Rapidly-Exploring Random Tree (RRT)!definition}'),
    (r'\*\*receding-horizon action chunking\*\*(?!\\index)', r'**receding-horizon action chunking**\\index{Receding-horizon action chunking!definition}'),
    (r'\*\*end-to-end replacement latency\*\*(?!\\index)', r'**end-to-end replacement latency**\\index{End-to-end replacement latency!definition}'),
    (r'\*\*future-state conditioning\*\*(?!\\index)', r'**future-state conditioning**\\index{Future-state conditioning!definition}'),
    (r'\*\*feasibility\*\*(?!\\index)', r'**feasibility**\\index{Feasibility!definition}'),
    (r'\*\*Covariant Hamiltonian Optimization for Motion Planning \(CHOMP\)\*\*(?!\\index)', r'**Covariant Hamiltonian Optimization for Motion Planning (CHOMP)**\\index{Covariant Hamiltonian Optimization for Motion Planning (CHOMP)!definition}'),
    (r'\*\*TrajOpt\*\*(?!\\index)', r'**TrajOpt**\\index{TrajOpt!definition}'),
    (r'\*\*Model Predictive Path Integral \(MPPI\)\*\*(?!\\index)', r'**Model Predictive Path Integral (MPPI)**\\index{Model Predictive Path Integral (MPPI)!definition}'),
    (r'\*\*\$C\^2\$ trajectory spline blending\*\*(?!\\index)', r'**$C^2$ trajectory spline blending**\\index{C2 trajectory spline blending@$C^2$ trajectory spline blending!definition}'),
    (r'\*\*jerk profile\*\*(?!\\index)', r'**jerk profile**\\index{Jerk profile!definition}'),
    (r'\*\*seam lateness\*\*(?!\\index)', r'**seam lateness**\\index{Seam lateness!definition}'),
    (r'\*\*fallback suffix\*\*(?!\\index)', r'**fallback suffix**\\index{Fallback suffix!definition}'),
    (r'\*\*trajectory record\*\*(?!\\index)', r'**trajectory record**\\index{Trajectory record!definition}')
]

for pat, repl in replacements:
    content, count = re.subn(pat, repl, content)
    print(f"Replaced {count} occurrences of {pat}")

with open(file_path, 'w') as f:
    f.write(content)

