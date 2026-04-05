# Thesis Template Guide

Use this file when the user's school or lab thesis template affects chapter commands, folder organization, bilingual headings, or formatting conventions beyond ordinary LaTeX.

## Template structure

Extracted directory and file roles:

- The `.cls` file controls the thesis body format.
- The `.bst` file controls bibliography style.
- The main file, such as `MA Thesis.tex`, assembles the distributed content into one document and usually should not be edited for normal writing tasks.
- The `chapters` directory contains the main thesis body, including Chinese and English abstracts, notation lists, chapter text, appendices, acknowledgements, achievements during the degree period, and author profile.
- The `figures` directory stores inserted image files such as `PNG`, `JPG`, `PDF`, and `EPS`. If there are many figures, organize them by chapter in subdirectories.
- The `reference` directory stores bibliography databases such as `.bib`.

Use these conventions when the user asks where content should live in the thesis project.

## Bilingual structural commands

The template uses custom bilingual sectioning commands. Keep them exactly when the user's project already uses them.

Examples:

```latex
\BiChapter{Latex模板简介}{The Introduction of Latex Template}
\BiSection{认识模板组成}{Understanding Template Composition}
\BiSubsection{单张图片的插入}{Insertion of a Single Image}
```

Do not replace these with plain `\chapter`, `\section`, or `\subsection` unless the user explicitly asks for template-independent LaTeX.

## Bilingual captions

The template uses `\bicaption{Chinese title}{English title}` for figures and tables.

Rules:

- Keep both Chinese and English captions when the template expects them.
- Place `\label{...}` after the bilingual caption when cross-reference is needed.
- Maintain consistent naming and numbering style across the same thesis.

## Terminology and expression rules extracted from the template

- Use officially approved scientific and engineering terminology where possible.
- If the author creates a local term, explain it at first occurrence.
- Keep the same concept named consistently across the whole thesis.
- Foreign personal names may remain in the original language.
- The full name of an institution, organization, school, research institute, or company should be written out at first occurrence.

## Symbol, letter, and unit conventions

- Follow national standards for numbers and units when the user needs formal thesis wording.
- Use legal unit symbols rather than casual Chinese unit names in technical contexts.
- Distinguish clearly between italic and upright Latin letters.
- Italic letters are typically used for variables, varying subscripts, function variables, and symbols representing quantities.
- Upright letters are typically used for standard function names such as `sin`, `exp`, and `ln`, for mathematical constants and defined operators, for legal measurement units, and for non-quantity abbreviations.
- Scalars should be upright if the user's template or discipline requires that convention.
- Vectors should be bold.

These are not universal across every discipline. Follow the user's template and field conventions first, then keep notation consistent within the document.
