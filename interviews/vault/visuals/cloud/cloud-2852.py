import os
import matplotlib.pyplot as plt

def draw_diagram():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Timeline
    ax.plot([5, 95], [5, 5], color='black', linewidth=2)
    
    # Checkpoints
    ax.plot([25, 25], [4.5, 5.5], color='#4a90c4', linewidth=3)
    ax.text(25, 6, 'Checkpoint N-1', ha='center', color='#4a90c4', fontweight='bold')
    
    ax.plot([60, 60], [4.5, 5.5], color='#4a90c4', linewidth=3)
    ax.text(60, 6, 'Checkpoint N', ha='center', color='#4a90c4', fontweight='bold')
    
    # Failure
    ax.plot([80, 80], [4.5, 5.5], color='#c87b2a', linewidth=3, linestyle='--')
    ax.text(80, 6, 'Failure Crash', ha='center', color='#c87b2a', fontweight='bold')
    
    # Recovery
    ax.plot([90, 90], [4.5, 5.5], color='#3d9e5a', linewidth=3)
    ax.text(90, 6, 'Training\nResumes', ha='center', color='#3d9e5a', fontweight='bold')
    
    # Annotations
    # Checkpoint Interval
    ax.annotate('', xy=(25, 3.5), xytext=(60, 3.5), arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(42.5, 2.5, 'Interval (T)', ha='center', color='gray')
    
    # Lost work
    ax.annotate('', xy=(60, 3.5), xytext=(80, 3.5), arrowprops=dict(arrowstyle='<->', color='#c87b2a'))
    ax.text(70, 2.5, 'Lost Work (avg T/2)', ha='center', color='#c87b2a')
    
    # Recovery time
    ax.annotate('', xy=(80, 3.5), xytext=(90, 3.5), arrowprops=dict(arrowstyle='<->', color='#3d9e5a'))
    ax.text(85, 2.5, 'Recovery', ha='center', color='#3d9e5a')
    
    plt.tight_layout()
    out_path = os.environ.get('VISUAL_OUT_PATH', 'visual.svg')
    plt.savefig(out_path, format='svg', bbox_inches='tight')

if __name__ == '__main__':
    draw_diagram()