import math

content = r"""\documentclass[border=10pt,tikz]{standalone}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{tikz}
\usepackage{tikz-3dplot}
\usetikzlibrary{calc,positioning,shapes.geometric,arrows.meta,decorations.pathreplacing}

\definecolor{emeraldpine}{HTML}{1A4D3E}
\definecolor{harvardcrimson}{HTML}{A51C30}
\definecolor{ethdarkblue}{HTML}{1F407A}
\definecolor{ethpetrol}{HTML}{007A87}
\definecolor{ethbronze}{HTML}{B87333}
\definecolor{ink}{HTML}{1A202C}
\definecolor{cardbg}{HTML}{F8FAFC}
\definecolor{shadecolor}{HTML}{F4F6F9}

\begin{document}
\tdplotsetmaincoords{65}{110}
\begin{tikzpicture}[tdplot_main_coords, font=\sffamily, line cap=round, line join=round]

\def\layerdz{4.5}
\def\w{8}
\def\d{6}

% Layer 1: Mesh
\begin{scope}[shift={(0,0,0)}]
    \draw[ethpetrol, thick, fill=ethpetrol!10, opacity=0.8] (0,0,0) -- (\w,0,0) -- (\w,\d,0) -- (0,\d,0) -- cycle;
    \foreach \x in {0,2,...,\w} \draw[ethpetrol!50, thin] (\x,0,0) -- (\x,\d,0);
    \foreach \y in {0,2,...,\d} \draw[ethpetrol!50, thin] (0,\y,0) -- (\w,\y,0);
    % Diagonal triangulation
    \foreach \x in {0,2,...,6} \draw[ethpetrol!50, thin] (\x,0,0) -- (\x+2,\d,0);
    \foreach \x in {2,4,...,8} \draw[ethpetrol!50, thin] (\x,0,0) -- (\x-2,\d,0);
    \node[anchor=north east, text=ethpetrol, font=\Large\bfseries] at (\w,\d,0) {Tier 1: Mesh};
\end{scope}

% Layer 2: Objects
\begin{scope}[shift={(0,0,\layerdz)}]
    \draw[black!10, dashed] (0,0,0) -- (\w,0,0) -- (\w,\d,0) -- (0,\d,0) -- cycle;
    \node[anchor=north east, text=ethbronze, font=\Large\bfseries] at (\w,\d,0) {Tier 2: Objects};
    
    \def\obj#1#2#3#4#5#6#7{
        \draw[ethbronze, thick, fill=ethbronze!20, opacity=0.8] (#1,#2,0) -- ++(#3,0,0) -- ++(0,#4,0) -- ++(-#3,0,0) -- cycle;
        \draw[ethbronze, thick, fill=ethbronze!20, opacity=0.8] (#1,#2,#5) -- ++(#3,0,0) -- ++(0,#4,0) -- ++(-#3,0,0) -- cycle;
        \draw[ethbronze, thick] (#1,#2,0) -- ++(0,0,#5);
        \draw[ethbronze, thick] (#1+#3,#2,0) -- ++(0,0,#5);
        \draw[ethbronze, thick] (#1,#2+#4,0) -- ++(0,0,#5);
        \draw[ethbronze, thick] (#1+#3,#2+#4,0) -- ++(0,0,#5);
        \draw[->, ethbronze, very thick] (#1+#3/2,#2+#4/2,#5) -- ++(0.5,0,0);
        \draw[->, ethbronze, very thick] (#1+#3/2,#2+#4/2,#5) -- ++(0,0.5,0);
        \draw[->, ethbronze, very thick] (#1+#3/2,#2+#4/2,#5) -- ++(0,0,0.5);
        \node[text=ethbronze, font=\small, anchor=south] at (#1+#3/2,#2+#4/2,#5+0.5) {#6};
        \coordinate (O#7B) at (#1+#3/2,#2+#4/2,0);
        \coordinate (O#7T) at (#1+#3/2,#2+#4/2,#5);
    }
    
    \obj{1}{1}{0.8}{0.8}{0.8}{Mug}{1}
    \obj{1.5}{3}{1.2}{1.2}{1.5}{Chair}{2}
    \obj{3.5}{2.5}{2.5}{2.5}{1.2}{Table}{3}
    \obj{6}{1}{1.5}{2}{1}{Sofa}{4}
    
    \draw[ethbronze, thick] (O1B) ++(0,0,0.4) -- ($(O2B)+(0,0,0.75)$);
    \draw[ethbronze, thick] (O2B) ++(0,0,0.75) -- ($(O3B)+(0,0,0.6)$);
    \draw[ethbronze, thick] (O3B) ++(0,0,0.6) -- ($(O4B)+(0,0,0.5)$);
\end{scope}

% Layer 3: Places
\begin{scope}[shift={(0,0,2*\layerdz)}]
    \draw[black!10, dashed] (0,0,0) -- (\w,0,0) -- (\w,\d,0) -- (0,\d,0) -- cycle;
    \node[anchor=north east, text=ethdarkblue, font=\Large\bfseries] at (\w,\d,0) {Tier 3: Places};
    
    \def\place#1#2#3#4#5#6#7{
        \draw[ethdarkblue, thick, fill=ethdarkblue!10, opacity=0.7] (#1,#2,0) -- ++(#3,0,0) -- ++(0,#4,0) -- ++(-#3,0,0) -- cycle;
        \draw[ethdarkblue, thick, fill=ethdarkblue!10, opacity=0.7] (#1,#2,#5) -- ++(#3,0,0) -- ++(0,#4,0) -- ++(-#3,0,0) -- cycle;
        \draw[ethdarkblue, thick] (#1,#2,0) -- ++(0,0,#5);
        \draw[ethdarkblue, thick] (#1+#3,#2,0) -- ++(0,0,#5);
        \draw[ethdarkblue, thick] (#1,#2+#4,0) -- ++(0,0,#5);
        \draw[ethdarkblue, thick] (#1+#3,#2+#4,0) -- ++(0,0,#5);
        \node[text=ethdarkblue, font=\bfseries] at (#1+#3/2,#2+#4/2,#5+0.3) {#6};
        \coordinate (P#7B) at (#1+#3/2,#2+#4/2,0);
        \coordinate (P#7T) at (#1+#3/2,#2+#4/2,#5);
    }
    
    \place{0.5}{0.5}{5.8}{5.0}{2}{Kitchen Zone}{1}
    \place{5.0}{0.5}{2.5}{3.5}{2}{Office Zone}{2}
    
    \draw[ethdarkblue, thick, dashed] (P1B) ++(0,0,1) -- ($(P2B)+(0,0,1)$);
\end{scope}

% Layer 4: Rooms
\begin{scope}[shift={(0,0,3*\layerdz)}]
    \draw[black!10, dashed] (0,0,0) -- (\w,0,0) -- (\w,\d,0) -- (0,\d,0) -- cycle;
    \node[anchor=north east, text=emeraldpine, font=\Large\bfseries] at (\w,\d,0) {Tier 4: Rooms};
    
    \def\room#1#2#3#4#5#6#7{
        \draw[emeraldpine, very thick, fill=emeraldpine!5, opacity=0.6] (#1,#2,0) -- ++(#3,0,0) -- ++(0,#4,0) -- ++(-#3,0,0) -- cycle;
        \draw[emeraldpine, very thick, fill=emeraldpine!5, opacity=0.6] (#1,#2,#5) -- ++(#3,0,0) -- ++(0,#4,0) -- ++(-#3,0,0) -- cycle;
        \draw[emeraldpine, very thick] (#1,#2,0) -- ++(0,0,#5);
        \draw[emeraldpine, very thick] (#1+#3,#2,0) -- ++(0,0,#5);
        \draw[emeraldpine, very thick] (#1,#2+#4,0) -- ++(0,0,#5);
        \draw[emeraldpine, very thick] (#1+#3,#2+#4,0) -- ++(0,0,#5);
        \node[text=emeraldpine, font=\Large\bfseries] at (#1+#3/2,#2+#4/2,#5+0.5) {#6};
        \coordinate (R#7B) at (#1+#3/2,#2+#4/2,0);
        \coordinate (R#7T) at (#1+#3/2,#2+#4/2,#5);
    }
    
    \room{0}{0}{8}{6}{2.5}{Room A}{1}
\end{scope}

% Layer 5: Building
\begin{scope}[shift={(0,0,4*\layerdz)}]
    \draw[black!10, dashed] (0,0,0) -- (\w,0,0) -- (\w,\d,0) -- (0,\d,0) -- cycle;
    \node[anchor=north east, text=harvardcrimson, font=\Large\bfseries] at (\w,\d,0) {Tier 5: Building};
    
    \def\bld#1#2#3#4#5#6#7{
        \draw[harvardcrimson, very thick, dashed, fill=harvardcrimson!2, opacity=0.5] (#1,#2,0) -- ++(#3,0,0) -- ++(0,#4,0) -- ++(-#3,0,0) -- cycle;
        \draw[harvardcrimson, very thick, dashed, fill=harvardcrimson!2, opacity=0.5] (#1,#2,#5) -- ++(#3,0,0) -- ++(0,#4,0) -- ++(-#3,0,0) -- cycle;
        \draw[harvardcrimson, very thick, dashed] (#1,#2,0) -- ++(0,0,#5);
        \draw[harvardcrimson, very thick, dashed] (#1+#3,#2,0) -- ++(0,0,#5);
        \draw[harvardcrimson, very thick, dashed] (#1,#2+#4,0) -- ++(0,0,#5);
        \draw[harvardcrimson, very thick, dashed] (#1+#3,#2+#4,0) -- ++(0,0,#5);
        \node[text=harvardcrimson, font=\huge\bfseries] at (#1+#3/2,#2+#4/2,#5+0.5) {#6};
        \coordinate (B#7B) at (#1+#3/2,#2+#4/2,0);
    }
    \bld{-0.5}{-0.5}{9}{7}{3}{Building}{1}
\end{scope}

% Connections
% Building to Room
\draw[harvardcrimson, thick, dashed] (B1B) -- (R1T);

% Room to Places
\draw[emeraldpine, thick, dashed] (R1B) -- (P1T);
\draw[emeraldpine, thick, dashed] (R1B) -- (P2T);

% Places to Objects
\draw[ethdarkblue, thick, dashed] (P1B) -- (O1T);
\draw[ethdarkblue, thick, dashed] (P1B) -- (O2T);
\draw[ethdarkblue, thick, dashed] (P1B) -- (O3T);
\draw[ethdarkblue, thick, dashed] (P2B) -- (O4T);

% Objects to Mesh
\draw[ethbronze, thick, dashed] (O1B) -- ($(O1B)-(0,0,\layerdz)$);
\draw[ethbronze, thick, dashed] (O2B) -- ($(O2B)-(0,0,\layerdz)$);
\draw[ethbronze, thick, dashed] (O3B) -- ($(O3B)-(0,0,\layerdz)$);
\draw[ethbronze, thick, dashed] (O4B) -- ($(O4B)-(0,0,\layerdz)$);

\end{tikzpicture}
\end{document}
"""

with open('fig09_dynamic_scene_graph_3d.tex', 'w') as f:
    f.write(content)

