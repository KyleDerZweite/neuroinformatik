"""Static help content for the TUI."""

HELP_MARKDOWN = """\
# Neuroinformatik TUI

This interface is intentionally narrow and debugging-oriented.

## Main flows

- Configure a preset and network structure in the left sidebar.
- Use **Run** / **Pause** for continuous training.
- Use manual stepping for forward, backward, and layer-wise inspection.
- Read current values in the **Overview** and **Inspect** tabs.

## Shortcuts

- `r`: run continuous training
- `p`: pause continuous training
- `e`: step one epoch
- `f`: step one forward pass
- `b`: step one backward pass
- `l`: step one layer forward
- `k`: step one layer backward
- `c`: configure from form values
- `x`: reset current network
- `q`: quit

## Presets

- **XOR**: nonlinear binary classification sanity check
- **Sine**: normalized regression task from `x = 0..7`
"""
