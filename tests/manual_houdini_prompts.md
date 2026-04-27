# Manual Houdini Prompt Tests

Eight live-provider prompts that exercise the bridge end-to-end. These are
**manual** tests — they hit the real OpenAI / Anthropic / LM Studio backend
configured by your `.env`, so they're not part of the automated `pytest`
suite. Use them to sanity-check a new model or a new system prompt.

The list below is kept in sync with `scripts/run_manual_test.py`. If you
edit one, edit the other.

## How to run

Generate plans only (no scene side effects, default):

    python scripts/run_manual_test.py

Run the full pipeline (safety gate + execution via the configured Houdini
transport — needs `HYTHON_PATH` for hython mode, or a running receiver for
socket mode):

    python scripts/run_manual_test.py --execute --mode direct

Single prompt by id:

    python scripts/run_manual_test.py --only sphere

Pick a provider explicitly (defaults to `DEFAULT_PROVIDER` from `.env`):

    python scripts/run_manual_test.py --provider lmstudio

Save a JSON transcript:

    python scripts/run_manual_test.py --output runs/manual.json

## Prompts

### 1. Sphere
- **Id:** `sphere`
- **Prompt:** `Create a sphere`
- **Expected intent (any of):** `create_sphere`, `create_geometry`
- **Expected risk:** `low`
- **Verify in Houdini:**
  - A new geo container under `/obj` with a sphere SOP inside.
  - The sphere SOP has its display + render flags set.
  - No prior nodes destroyed.

### 2. Procedural rock
- **Id:** `procedural_rock`
- **Prompt:** `Make a procedural rock`
- **Expected intent (any of):** `create_rock`, `create_procedural_rock`
- **Expected risk:** `low` or `medium`
- **Verify in Houdini:**
  - Geometry container with a sphere/platonic base displaced by a Mountain
    SOP (or noise) and ideally cleaned with a Remesh SOP.
  - Output node has display flag set; result looks irregular, not a sphere.

### 3. Desert scatter
- **Id:** `desert_scatter`
- **Prompt:** `Scatter a few rocks across a desert ground plane`
- **Expected intent (any of):** `create_scatter`, `create_desert_scatter`,
  `scatter_rocks`
- **Expected risk:** `medium`
- **Verify in Houdini:**
  - A ground plane (Grid SOP) with a Scatter SOP feeding a Copy-to-Points
    (or instancer) using a small rock SOP as the source.
  - Reasonable point count (not millions); seed is set.

### 4. Explosion setup (no sim run)
- **Id:** `explosion_setup`
- **Prompt:** `Set up an explosion without running the sim`
- **Expected intent (any of):** `create_explosion_placeholder`,
  `create_pyro_setup`
- **Expected risk:** `medium`
- **Verify in Houdini:**
  - Emitter geometry (sphere/box) wired into a Pyro Source SOP.
  - A sibling DOP network with a Smoke/Pyro Object, Source Volume, and a
    Pyro Solver — but the script must **not** call `cook()` or otherwise
    advance the simulation.
  - No warnings about long-running cooks.

### 5. Camera
- **Id:** `camera`
- **Prompt:** `Add a camera looking at the origin`
- **Expected intent (any of):** `create_camera`, `create_camera_and_light`
- **Expected risk:** `low`
- **Verify in Houdini:**
  - A new `cam` node under `/obj` with sensible focal length (35–75 mm)
    and translation pulled back from the origin.
  - Render resolution defaulted to a common value (e.g. 1920×1080).

### 6. Scale selected
- **Id:** `scale_selected`
- **Prompt:** `Scale the selected nodes by 2`
- **Expected intent (any of):** `modify_selected`, `scale_selected`
- **Expected risk:** `medium`
- **Verify in Houdini:**
  - Code reads `hou.selectedNodes()`, filters to nodes with a scale parm
    tuple, and multiplies their scale by 2.
  - Wrapped in `hou.undos.group(...)` so the change is one Ctrl+Z away.
  - If nothing is selected, script raises `hou.OperationFailed` rather
    than silently doing nothing.

### 7. Assign material
- **Id:** `assign_material`
- **Prompt:** `Assign a default principled shader to the selected geometry`
- **Expected intent (any of):** `assign_material`, `create_material`,
  `apply_principled`
- **Expected risk:** `medium`
- **Verify in Houdini:**
  - A Material Library or `/mat` Principled Shader is created if missing.
  - The selected geo's material parm is set to that shader's path.
  - No errors on selections without a material parm — they're skipped.

### 8. Network organization
- **Id:** `network_organization`
- **Prompt:** `Lay out and color-code the /obj network so rendering nodes
  are blue and geometry is yellow`
- **Expected intent (any of):** `organize_network`, `layout_network`,
  `color_nodes`
- **Expected risk:** `low` or `medium`
- **Verify in Houdini:**
  - `/obj` children are laid out (`obj.layoutChildren()`).
  - Cameras/lights coloured one hue, geometry containers another via
    `node.setColor(hou.Color((r, g, b)))`.
  - Existing node positions are not arbitrarily reset elsewhere.

## What to look for across prompts

- Plans should never contain `os.system`, `subprocess`, `urllib`, sockets,
  `eval`, or `exec`. The safety gate will block these — if you see
  `verdict=blocked` it's the LLM's fault, not the bridge's.
- Risk levels should match the table above. A `low` plan that touches
  `hou.hipFile.clear()` is a bug in the prompt or the model.
- Clarification responses are valid: when a request is genuinely
  ambiguous, the model may return
  `{"intent": "clarification", "requires_houdini": false, "question": "…"}`.
  Treat that as a pass; ask the question and re-prompt.
