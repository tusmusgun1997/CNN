# Padding

When a filter slides over an image without help, two annoying things happen: the
output **shrinks**, and the **edge pixels get under-used**. **Padding** fixes
both by adding a border around the input. This sub-folder explains it fully.

| File | What it teaches |
|------|-----------------|
| [01 — What is padding?](01-what-is-padding.md) | The two problems and the core idea |
| [02 — Valid vs. Same padding](02-valid-vs-same-padding.md) | The two modes you'll use constantly |
| [03 — Formulas & examples](03-padding-formulas-and-examples.md) | Computing padding amounts and output sizes |

**The one-line summary:**

> Padding wraps the input in a border (usually zeros) so the output can stay the
> same size and the corner/edge pixels get a fair amount of attention.
