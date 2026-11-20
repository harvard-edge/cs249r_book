% studysheet.cls: one double-sided chapter study sheet (front: learn, back: practice).
% Built with lualatex, like the book, with the book's notation macros and fonts.
% A sheet is plain LaTeX content; its numbers come from \val{key}, set by the chapter's
% numbers file (\setnum{key}{value}), which a small Python script computes from mlsysim.
% Guidance: AIConfigs/projects/MLSysBook/.claude/rules/studysheet.md
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{studysheet}[2026/11/19 MLSysBook chapter study sheet]
\LoadClass[10pt]{article}

\RequirePackage[letterpaper, top=0.42in, bottom=0.55in, left=0.55in, right=0.55in, footskip=0.28in]{geometry}
\RequirePackage{fontspec}
\RequirePackage{mathptmx}          % the book's math
\RequirePackage{newpxtext}         % the book's body text
\IfFontExistsTF{Helvetica}{\setsansfont[Scale=0.92]{Helvetica}}{\IfFontExistsTF{TeX Gyre Heros}{\setsansfont[Scale=0.92]{TeX Gyre Heros}}{}}
\RequirePackage{amsmath, amssymb}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta}
\RequirePackage[most]{tcolorbox}
\RequirePackage{enumitem}
\RequirePackage{multicol}
\RequirePackage{graphicx}
\RequirePackage{fancyhdr}
\RequirePackage{calc}
\RequirePackage{colortbl}
\RequirePackage{booktabs, tabularx, array}
\RequirePackage{qrcode}                 % chapter link in the header (build.py passes \SheetURL)
\RequirePackage[hidelinks]{hyperref}

% ---- the book's notation macros (books/shared/_includes/_macros.tex, generated from the registry)
\InputIfFileExists{_macros.tex}{}{\ClassError{studysheet}{_macros.tex not found; run tools/build_notation.py}{}}

% ---- numbers: \setnum{key}{value} in the numbers file, \val{key} in the sheet
\newcommand{\setnum}[2]{\expandafter\gdef\csname cs@num@#1\endcsname{#2}}
\newcommand{\val}[1]{\ifcsname cs@num@#1\endcsname\csname cs@num@#1\endcsname\else\errmessage{studysheet: number `#1' is not set}\fi}

% ---- colors. The accent is the volume's own `accentcolor` (books/volN/tex/theme-colors.tex);
% build.py reads it and passes it as \SheetAccent, so the sheet always matches its book.
\definecolor{ink}{HTML}{1C2430}
\definecolor{muted}{HTML}{5B6676}
\definecolor{rulegray}{HTML}{D9DEE6}
\newcommand{\cs@setaccent}{%
  \ifdefined\SheetAccent\else\ClassError{studysheet}{No volume accent. Build with books/shared/studysheet/build.py}{}\def\SheetAccent{000000}\fi
  \definecolor{accent}{HTML}{\SheetAccent}\colorlet{accentsoft}{accent!8!white}}

% ---- which sheet: \sheetvolume{1} \sheetchapter{1}{Introduction} \sheetsources{...}
\newcommand{\sheetvolume}[1]{\cs@setaccent\ifcase#1\or
  \def\cs@book{Introduction to Machine Learning Systems}\def\cs@roman{I}\or
  \def\cs@book{Scaling Machine Learning Systems}\def\cs@roman{II}\or
  \def\cs@book{Agentic Machine Learning Systems}\def\cs@roman{III}\or
  \def\cs@book{Physical AI}\def\cs@roman{IV}\fi}
\newcommand{\sheetchapter}[2]{\def\cs@ch{#1}\def\cs@title{#2}}
\newcommand{\sheetsources}[1]{\def\cs@sources{#1}}
\def\cs@sources{}

% ---- type
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\linespread{1.08}
\newcommand{\sheetsans}{\sffamily}
\newcommand{\res}[1]{{\color{accent}\bfseries #1}}
\AtBeginDocument{\color{ink}}

% ---- footer
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.4pt}
\renewcommand{\footrule}{{\color{rulegray}\hrule width\headwidth height\footrulewidth}}
\fancyfoot[L]{\parbox[b]{0.68\textwidth}{\raggedright\sheetsans\fontsize{6.8}{8}\selectfont\color{muted}\cs@sources}}
\fancyfoot[R]{\sheetsans\fontsize{6.8}{8}\selectfont\color{muted}Free online at
  \href{https://mlsysbook.ai}{\textcolor{accent}{mlsysbook.ai}} \textperiodcentered{}
  \href{https://github.com/harvard-edge/cs249r_book}{star it on GitHub}}

% ---- headers: \frontside, \backside
\newcommand{\cs@header}{%
  \noindent\begin{minipage}[b]{0.62\linewidth}\raggedright
    {\sheetsans\fontsize{7.6}{9}\selectfont\color{accent}\bfseries CHAPTER \cs@ch{} \textperiodcentered{} STUDY SHEET}\\[2pt]
    {\sheetsans\fontsize{20}{22}\selectfont\bfseries \cs@title}
  \end{minipage}\hfill
  \begin{minipage}[b]{0.36\linewidth}\raggedleft
    \def\cs@which{{\sheetsans\fontsize{7.2}{9}\selectfont\color{muted}\cs@book\\Volume \cs@roman}}%
    \ifdefined\SheetURL
      % Chapter link: the QR code and its label both open the chapter online (build.py passes \SheetURL).
      % Two lines, as without the link, and a smashed code, so the header keeps its height.
      \parbox[b]{\dimexpr\linewidth-0.42in-5pt\relax}{\raggedleft\sheetsans\fontsize{7.2}{9}\selectfont
        {\color{muted}\cs@book}\\{\color{muted}Volume \cs@roman{} \textperiodcentered{}}
        \href{\SheetURL}{\textcolor{accent}{Scan to read the chapter}}}%
      \hspace{5pt}\smash{\raisebox{\depth}{\href{\SheetURL}{\qrcode[height=0.42in, level=L, nolink]{\SheetURL}}}}%
    \else\cs@which\fi
  \end{minipage}\par\vspace{4pt}
  {\color{accent}\hrule height 1.6pt}\vspace{8pt}}
\newcommand{\frontside}{\cs@header}
\newcommand{\backside}{\newpage\cs@header}

% ---- big idea
\newtcolorbox{bigidea}{enhanced, colback=accentsoft, colframe=accentsoft, boxrule=0pt, arc=0pt,
  borderline west={2.8pt}{0pt}{accent}, left=9pt, right=9pt, top=5pt, bottom=5pt,
  fontupper=\fontsize{10.6}{14}\selectfont, after skip=10pt}
\newcommand{\hl}[1]{{\color{accent}\bfseries #1}}

% ---- section label
\newcommand{\sheetsec}[1]{{\sheetsans\fontsize{9}{11}\selectfont\bfseries\color{accent}\MakeUppercase{#1}}\par\vspace{4pt}}

% ---- key ideas
\newlist{ideas}{enumerate}{1}
\setlist[ideas]{label=\textcolor{accent}{\sheetsans\bfseries\arabic*.}, leftmargin=13pt, labelsep=4pt, itemsep=3.5pt, topsep=0pt, parsep=0pt}
% Each key idea is typeset once in a measuring box; the build fails an idea longer than 3 lines.
\newsavebox{\cs@ideabox}
\newcommand{\idea}[2]{\item\sbox{\cs@ideabox}{\parbox[t]{\linewidth}{\strut{\sheetsans\bfseries #1} #2\strut}}%
  \typeout{STUDYSHEET-IDEA lines=\the\numexpr(\ht\cs@ideabox+\dp\cs@ideabox)/\baselineskip\relax}%
  {\sheetsans\bfseries #1} #2}

% ---- formula card: \fcard{name}{display math}{what it says}{when it holds}{figure}
\newtcolorbox{eqbox}{colback=accentsoft, colframe=accentsoft, boxrule=0pt, arc=2pt, left=4pt, right=4pt, top=2pt, bottom=2pt, halign=center}
\newif\iffirstcard \firstcardtrue
\newcommand{\fcard}[5]{%
  \iffirstcard\global\firstcardfalse\else\par\vspace{6pt}{\color{rulegray}\hrule height 0.5pt}\vspace{6pt}\fi
  {\sheetsans\fontsize{7.4}{9}\selectfont\bfseries\color{muted}\MakeUppercase{#1}}\par\vspace{3pt}
  \begin{eqbox}$\displaystyle #2$\end{eqbox}\par\vspace{2pt}
  #3\par
  \if\relax\detokenize{#4}\relax\else\vspace{1pt}{\fontsize{8.6}{11}\selectfont\color{muted}#4}\par\fi
  \if\relax\detokenize{#5}\relax\else\vspace{4pt}{\centering #5\par}\fi}
\newcommand{\under}[2]{\underset{\textstyle\text{\sheetsans\fontsize{7.6}{9}\selectfont\itshape\color{accent}#2}}{#1}}

% ---- number tiles: \tile{value}{label} inside \begin{tiles}...\end{tiles}
\newcommand{\tile}[2]{\begin{tcolorbox}[colback=white, colframe=rulegray, boxrule=0.5pt, arc=2pt, left=5pt, right=5pt, top=4pt, bottom=4pt,
    width=\dimexpr(\linewidth-6pt)/2\relax, nobeforeafter, box align=top, equal height group=tiles]
  {\fontsize{14}{16}\selectfont\bfseries #1}\par\vspace{1pt}{\raggedright\fontsize{8.2}{10}\selectfont\color{muted}#2\par}\end{tcolorbox}}
\newenvironment{tiles}{\noindent}{\par}

% ---- tables: \begin{sheettable}{L C C} \thead{..} & .. \\ \midrule .. \end{sheettable}
% L = wrapping left column, C = wrapping centered column, l/c/r as usual; width is the column.
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\newenvironment{sheettable}[1]{\par\noindent\begingroup\fontsize{8.4}{10.2}\selectfont
  \renewcommand{\arraystretch}{1.3}\setlength{\tabcolsep}{3pt}\arrayrulecolor{rulegray}
  \tabularx{\dimexpr\linewidth-1.5pt\relax}{#1}\toprule}{\bottomrule\endtabularx\endgroup\par}
\newcommand{\thead}[1]{\leavevmode{\sheetsans\fontsize{7.4}{9}\selectfont\bfseries\color{muted}\MakeUppercase{#1}}}

% ---- practice boxes
\newtcolorbox{practicebox}{colback=white, colframe=rulegray, boxrule=0.6pt, arc=2pt, left=7pt, right=7pt, top=5pt, bottom=6pt}
\newcommand{\boxtitle}[1]{{\bfseries #1}\par\vspace{2pt}}
\newcommand{\writelines}[1]{\par\vspace{2pt}\foreach \i in {1,...,#1}{\par\vspace{12pt}\noindent{\color{muted}\leaders\hbox to 2.5pt{\hss.\hss}\hfill\null}\par}}
\newlist{steps}{enumerate}{1}
\setlist[steps]{label=\arabic*., leftmargin=13pt, itemsep=1pt, topsep=2pt}
\newcommand{\takeaway}[1]{\par\vspace{3pt}{\color{accent}\bfseries Takeaway.} #1}

% ---- mistakes and questions
\newcommand{\mistake}[3]{{\itshape #1}\par #2\par\if\relax\detokenize{#3}\relax\else\vspace{3pt}{\centering #3\par}\fi\vspace{7pt}}
\newlist{qs}{enumerate}{1}
\setlist[qs]{label=\arabic*., leftmargin=13pt, itemsep=5pt, topsep=0pt}
\newcommand{\assumed}{{\itshape Assumed:}}

% ---- drawing styles for TikZ devices (draw in pt; a column is about 250pt wide)
\tikzset{
  sheet/.style={x=1pt, y=1pt, baseline=(current bounding box.north)},
  base/.style={execute at begin node=\hyphenpenalty 10000\relax, font=\sheetsans\fontsize{7.2}{8.5}\selectfont, text=ink, inner sep=2.5pt, rounded corners=1pt, line width=0.5pt},
  accbox/.style={base, fill=accentsoft, draw=accent},
  hotbox/.style={base, fill=accent, draw=accent, text=white},
  nbox/.style={base, fill=white, draw=muted, line width=0.45pt},
  note/.style={font=\sheetsans\itshape\fontsize{7}{8.5}\selectfont, text=muted, inner sep=0},
  small/.style={font=\sheetsans\fontsize{7}{8.4}\selectfont, text=muted, inner sep=0},
  lab/.style={font=\sheetsans\fontsize{7.2}{8.5}\selectfont, text=ink, inner sep=0},
  dim/.style={muted, line width=0.4pt},
  gridline/.style={rulegray, line width=0.35pt},
  arr/.style={-{Stealth[length=3.2pt]}, ink, line width=0.5pt},
}

% ---- the two columns, measured: leftcol then rightcol; reports free space to the log
\newlength{\colw}\setlength{\colw}{\dimexpr(\textwidth-22pt)/2\relax}
\newlength{\sheetreserve}\setlength{\sheetreserve}{0pt}
\newsavebox{\colA}\newsavebox{\colB}
\newenvironment{leftcol}{\par\global\firstcardtrue\begin{lrbox}{\colA}\begin{minipage}[t]{\colw}}{\end{minipage}\end{lrbox}\global\setbox\colA=\box\colA}
\newenvironment{rightcol}{\global\firstcardtrue\begin{lrbox}{\colB}\begin{minipage}[t]{\colw}}{\end{minipage}\end{lrbox}\global\setbox\colB=\box\colB\aftergroup\cs@placecols}
\newcommand{\cs@placecols}{%
  \dimen0=\dimexpr\pagegoal-\pagetotal-\sheetreserve-\baselineskip\relax
  \typeout{STUDYSHEET-FREE page=\thepage\space left=\the\dimexpr\dimen0-\ht\colA-\dp\colA\relax\space right=\the\dimexpr\dimen0-\ht\colB-\dp\colB\relax}%
  \noindent\usebox{\colA}\hfill\usebox{\colB}\par}

% ---- answers, printed upside down at the foot of the back. Put the answers environment
% right after \backside so the columns are measured against the space it takes.
\newsavebox{\ansbox}
\newenvironment{answers}{\begin{lrbox}{\ansbox}\begin{minipage}{\textwidth}\fontsize{8.2}{10.4}\selectfont\color{muted}
    {\color{rulegray}\hrule height 0.6pt}\vspace{4pt}
    \begin{multicols}{2}\raggedright{\color{ink}\bfseries Answers} (turn the sheet)\par}
  {\end{multicols}\end{minipage}\end{lrbox}\global\setbox\ansbox=\box\ansbox
   \global\sheetreserve=\dimexpr\ht\ansbox+\dp\ansbox+10pt\relax}
\newcommand{\ans}[2]{{\color{ink}\bfseries #1.} #2\par\vspace{2pt}}
\AtEndDocument{\ifdim\wd\ansbox>0pt\par\vfill\noindent\rotatebox{180}{\usebox{\ansbox}}\par\fi}
