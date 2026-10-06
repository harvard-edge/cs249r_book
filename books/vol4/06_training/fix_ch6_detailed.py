import re

with open("/private/tmp/MLSysBook-dev-notation/books/vol4/06_training/06_training.qmd", "r") as f:
    content = f.read()

# Helper for first occurrence replacement
def replace_first(pattern, replacement, text):
    return re.sub(pattern, replacement, text, count=1)

# Fix acronyms
content = replace_first(r'\b(ODD)\b', r'Operational Design Domain (ODD)', content)
content = replace_first(r'\b(MCU)\b', r'Microcontroller Unit (MCU)', content)
content = replace_first(r'\b(ODE)\b', r'Ordinary Differential Equation (ODE)', content)
content = replace_first(r'\b(back-EMF)\b', r'back electromotive force (back-EMF)', content)
content = replace_first(r'\b(IMU)\b', r'Inertial Measurement Unit (IMU)', content)
content = replace_first(r'\b(RMS)\b', r'root-mean-square (RMS)', content)
content = replace_first(r'\b(DoF)\b', r'degrees of freedom (DoF)', content)
content = replace_first(r'\b(BC)\b', r'behavioral cloning (BC)', content)
content = replace_first(r'\b(RL)\b', r'reinforcement learning (RL)', content)

# Fix Concepts
content = replace_first(r'policy manifest \(@sec-training-policy-manifest\)', r'policy manifest (a formal declaration of its supported operating domain)', content)
content = replace_first(r'\(@sec-training-privileged-distillation\)', r'(a process where a student model learns to replicate a teacher using only deployable sensors)', content)
content = replace_first(r'contact deadline alongside', r'contact yield deadline (the strict time limit before mechanical damage occurs) alongside', content)


# Fix Forward Connectors to Later Chapters
content = re.sub(r' \(@sec-(evaluation|perception|memory|intent|planning|enforcement|placement|intervention|verification|release|frontier).*?\)', '', content)
content = re.sub(r' @sec-(evaluation|perception|memory|intent|planning|enforcement|placement|intervention|verification|release|frontier)[^\s,.]*', ' a subsequent evaluation', content)
content = re.sub(r'Chapters? 7.*?(?:1[1-7])', 'subsequent chapters', content)
content = re.sub(r'Chapters? [789]|Chapter 1[0-7]', 'a subsequent chapter', content)
content = re.sub(r'↩.*?@sec-(evaluation|perception|memory|intent|planning|enforcement|placement|intervention|verification|release|frontier).*?\n', '', content)
content = re.sub(r'\\ref\{nbk-(evaluation|perception|memory|intent|planning|enforcement|placement|intervention|verification|release|frontier).*?\}', 'a future notebook', content)

# Fix Intra-Chapter Forward References
content = re.sub(r' \(@sec-training-contact-exploit\)', '', content)
content = re.sub(r' \(@sec-training-learning-by-trial\)', '', content)
content = re.sub(r' \(@sec-training-offline-rl\)', '', content)
content = re.sub(r' \(@sec-training-decoder-costs\)', '', content)
content = re.sub(r' \(@sec-training-sim-to-real\)', '', content)
content = re.sub(r' \(@sec-training-parallel-simulation\)', '', content)
content = re.sub(r' \(@sec-training-real-world-rl\)', '', content)
content = re.sub(r' \(@sec-training-contact-model\)', '', content)
content = re.sub(r' @sec-training-contact-exploit', ' a later section', content)


with open("/private/tmp/MLSysBook-dev-notation/books/vol4/06_training/06_training.qmd", "w") as f:
    f.write(content)

print("Done")
