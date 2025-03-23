# Custom Instructions for HAL Projects
- Refer to me as Jarrod when addressing me directly.
- Project: HAL - A sharp AI assistant.

## Python - HAL Core (src/hal/*)
- Use Python 3.12 conventions and type hints where possible.
- Follow snake_case for variable and function names.
- Prefer concise, functional code over verbose implementations.
- Structure code to match FastAPI patterns for API endpoints.
- Keep docstrings brief and focused on intent, not implementation details.

## JavaScript/HTML/CSS - HAL UI (hal-ui/*)
- Use camelCase for JavaScript variables and functions.
- Follow modern ES6+ syntax, avoiding legacy patterns.
- Stick to Tauri conventions for `src-tauri/*` integration.
- Keep CSS minimal and scoped to components where possible.
- Prefer functional components over class-based in `src/*`.
- Optimize UI for a retro *WarGames*-style aesthetic.

## General Guidelines
- Prioritize performance and readability across both projects.
- Avoid unnecessary comments—code should be self-explanatory unless complex.
