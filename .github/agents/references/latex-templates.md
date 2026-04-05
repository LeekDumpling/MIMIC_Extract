# LaTeX Templates

Use this file when the user asks for equation formatting, table formatting, figure insertion, references, theorem-like environments, algorithm blocks, or text that can be pasted directly into a LaTeX thesis.

## Escaping reminders

Escape these characters when they are part of normal text rather than LaTeX commands:

- `_`
- `%`
- `&`
- `#`

Do not escape characters blindly inside valid LaTeX commands, labels, URLs, or environments.

## Equation template

Reuse this equation format exactly when the user asks for the same style:

```latex
\begin{equation}
    L_{\text{evo}}^{(r)} = \frac{1}{|\mathcal{K}^{(r)}|} \sum_{\mathbf{x} \in \mathcal{K}^{(r)}} \min_{\mathbf{y} \in \hat{\mathcal{K}}} \| \mathbf{x} - \mathbf{y} \|_2^2 + \frac{1}{|\hat{\mathcal{K}}|} \sum_{\mathbf{y} \in \hat{\mathcal{K}}} \min_{\mathbf{x} \in \mathcal{K}^{(r)}} \| \mathbf{x} - \mathbf{y} \|_2^2,
    \label{eq:chamfer}
\end{equation}
```

Generic displayed equation pattern from the thesis template:

```latex
\begin{equation}
St=\frac{fd}{v}=\frac{f\overline{d_{32}}}{\overline{Q^{''}}}
\label{eq:St}
\end{equation}
```

## Figure templates

Single figure:

```latex
\begin{figure}[!ht]
\centering
\includegraphics{figures/figure4}
\bicaption{图片}{Picture}  \label{fig4:diagram}
\end{figure}
```

Two side-by-side subfigures:

```latex
\begin{figure}[!ht]
\centering
\subfigure[热流耦合数值模拟]{\includegraphics[width=0.45\textwidth]{figures/figure8}}
\subfigure[热固耦合数值模拟]{\includegraphics[width=0.45\textwidth]{figures/figure9}}
\bicaption{数值模拟图像}{Numerical simulation image}
\end{figure}
```

Use `\centering` for centered figures. Use `\includegraphics` to control image inclusion and size. Add `\label{...}` immediately after the bilingual caption when cross-reference is needed.

## Table templates

Default table pattern in the template uses `\small`, centered layout, bilingual captions through `\bicaption`, and three-line tables with `\toprule`, `\midrule`, and `\bottomrule`.

Representative table template:

```latex
\begin{table}[!ht]
    \small
    \centering
    \bicaption{国内外各返回式航天器热控设计情况}{Design of thermal control systems of spacecraft for different countries} \label{tab3:category}
    \begin{tabular}{m{5em}<{\centering}m{5em}<{\centering}m{5em}<{\centering}m{5em}<{\centering}m{5em}<{\centering}m{5em}<{\centering}}
        \toprule[2pt]
        项目、指标 &月地高速再入返回器&传统返回式卫星回收舱&神舟飞船&国外载人飞船&航天飞机 \\
        \midrule[1pt]
        回收舱气密性&半密封舱&非密封舱&密封舱&密封舱&密封舱\\
        回收舱长期热耗(W)&整器150&5--25&约1000(含宇航员)&约1000(含宇航员)&1500以上\\
        热控方案&基于柔性自适应“热开关”的新型热控方案&被动热控设计为主、电加热主动热控设计为辅&泵驱单相流体回路+对流通风&泵驱单相流体回路+对流通风&泵驱单相流体回路+对流通风+主动式相变系统\\
        \bottomrule[2pt]
    \end{tabular}
\end{table}
```

Useful table rules extracted from the template:

- Use `&` for column alignment.
- Use `\multicolumn` for horizontal cell merging.
- Use `\multirow` for vertical cell merging.
- Keep the default style as a three-line table unless the user explicitly asks otherwise.

## References

The thesis template assumes a `.bib` bibliography database and a `.bst` style file. The provided style is `GBT7714-2005NLang.bst`.

Citation patterns:

```latex
\cite{Jiang2005Size}
\upcite{Takahashi1996Structure,Xia2002Analysis,Jiang1989}
```

Important rule extracted from the template:

- The `.bib` file stores bibliography content.
- The `.bst` file controls bibliography appearance.
- In `GBT7714-2005NLang.bst`, entries with a non-empty `Language` field are treated as Chinese references.

Do not invent bibliography metadata. If the user provides only partial reference information, ask for or preserve the missing fields rather than fabricating them.

## Theorem and definition environments

The template includes theorem-like environments such as `thm` and `defn`.

Examples:

```latex
\begin{thm}
设函数$y=f(x)$在区间(a,b)上可导，它对应曲线是向上凹(或向下凹)的充分必要条件是：导数 $y=f^{'}(x)$在区间(a,b)上是单调增(或单调减)。
\end{thm}
```

```latex
\begin{defn}[函数极值]
设函数$f(x)$在区间(a,b)内有定义，$x_0$是(a,b)内一点。\par
若存在着$x_0$点的一个邻域，对于这个邻域内任何点$x$($x_0$点除外)，$f(x)<f(x_{0})$均成立，则说$f(x_{0})$ 是函数 $f(x)$的一个极大值；若存在着$x_0$点的一个邻域，对于这个邻域内任何点$x$($x_0$点除外)，$f(x)>f(x_{0})$均成立，则说$f(x_{0})$ 是函数$f(x)$ 的一个极小值。函数的极大值与极小值统称为函数的极值。
\end{defn}
```

## Algorithm environment

The template supports algorithm pseudocode with `algorithm` and `algorithmic`.

Pattern:

```latex
\begin{algorithm}[h]
   \caption{Pseudocode of Simulated Annealing Algorithm}
   \begin{algorithmic}[1]
     \Require
       $x_0$: initial individual or state;
       $T_0$: a high enough initial temperature;
       $T_{min}$: the lowest limit of temperature;
     \Ensure
       optimal state or approximate optimal state;
       \State set $x_0 = x_{best}$, compute initial energy function $E(x_0)$;
   \end{algorithmic}
\end{algorithm}
```

## Output rule

When the user asks for thesis-ready prose, return text that can be pasted directly into LaTeX with minimal further editing.
