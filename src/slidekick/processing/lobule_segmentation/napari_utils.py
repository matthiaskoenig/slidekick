"""Napari UI utilities for interactive pipeline previews.

Extracted from lob_utils.py so that image-processing helpers and napari/Qt
wiring live in separate modules.  Import napari and magicgui lazily inside
each function so that headless (non-GUI) environments never have to install
them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from slidekick.processing.lobule_segmentation.lob_utils import downsample_to_max_side


def add_napari_controls_dock(
    viewer: Any,
    controls_widget: Any,
    *,
    on_update: Optional[Callable[[], None]] = None,
    on_confirm: Optional[Callable[[], None]] = None,
    on_apply_values: Optional[Callable[[], None]] = None,
    on_back: Optional[Callable[[], None]] = None,
    on_reset: Optional[Callable[[], None]] = None,
    on_abort: Optional[Callable[[], None]] = None,
    update_text: str = "Preview/Recalculate",
    confirm_text: str = "Confirm Current View",
    apply_values_text: str = "continue and apply current values without preview",
    back_text: str = "Back",
    reset_text: str = "Reset Parameters",
    abort_text: str = "Abort",
    include_update: bool = True,
    include_confirm: bool = True,
    include_apply_values: bool = True,
    include_back: bool = True,
    include_reset: bool = True,
    include_abort: bool = True,
    dock_area: str = "right",
) -> Dict[str, Any]:
    """
    Unify napari preview UI wiring across the codebase.

    Default button set (in this exact order)
    ----------------------------------------
      1) Preview/Recalculate
      2) Confirm Current View: uses last previewed result
      3) Apply Slider Values & Continue: applies current slider values (even if not previewed)
      4) Back
      5) Reset Parameters
      6) Abort

    Notes on environment compatibility
    ---------------------------------
    Different napari/magicgui versions expose PushButton signals differently:
      - some versions use `changed`
      - others use `clicked`
      - some require wiring through the underlying Qt widget (`.native`)

    Critical robustness fix
    ----------------------
    Some Qt bindings (notably some PySide setups) can drop callbacks connected via
    anonymous lambdas if no Python reference is kept. We therefore create a
    wrapper function and store it on the button to prevent GC.

    Returns
    -------
    Dict[str, Any]
        Mapping of button names to PushButton widgets for optional external wiring.
    """
    import warnings
    from magicgui.widgets import Container, PushButton

    def _btn_text(btn: Any) -> str:
        txt = getattr(btn, "text", None)
        try:
            return str(txt()) if callable(txt) else str(txt)
        except Exception:
            return "<unknown>"

    def _wire_button(btn: Any, cb: Optional[Callable[[], None]]) -> None:
        """Best-effort connect for magicgui/Qt push buttons across versions (GC-safe)."""
        if cb is None:
            return

        def _invoke_cb(*_args: Any, **_kwargs: Any) -> None:
            try:
                cb()
            except Exception as e:
                warnings.warn(
                    f"add_napari_controls_dock: callback error for button '{_btn_text(btn)}': {e!r}",
                    RuntimeWarning,
                )

        # Keep a strong reference to avoid GC issues in some environments.
        try:
            setattr(btn, "_napari_controls_cb_ref", _invoke_cb)
        except Exception:
            pass

        # Prefer underlying Qt signal if available (most reliable across versions)
        native = getattr(btn, "native", None)
        if native is not None:
            for sig_name in ("clicked", "pressed", "released"):
                sig = getattr(native, sig_name, None)
                if sig is None:
                    continue
                try:
                    sig.connect(_invoke_cb)
                    return
                except Exception:
                    pass

        # Fall back to magicgui-level signals
        for sig_name in ("clicked", "changed"):
            sig = getattr(btn, sig_name, None)
            if sig is None:
                continue
            try:
                sig.connect(_invoke_cb)
                return
            except Exception:
                pass

        warnings.warn(
            f"add_napari_controls_dock: could not connect callback for button '{_btn_text(btn)}'.",
            RuntimeWarning,
        )

    widgets = [controls_widget]
    out: Dict[str, Any] = {}

    # IMPORTANT: keep button order stable across ALL previews.
    if include_update:
        update_btn = PushButton(text=str(update_text))
        _wire_button(update_btn, on_update)
        widgets.append(update_btn)
        out["update"] = update_btn

    if include_confirm:
        confirm_btn = PushButton(text=str(confirm_text))
        _wire_button(confirm_btn, on_confirm)
        widgets.append(confirm_btn)
        out["confirm"] = confirm_btn

    if include_apply_values:
        apply_values_btn = PushButton(text=str(apply_values_text))
        _wire_button(apply_values_btn, on_apply_values)
        widgets.append(apply_values_btn)
        out["apply_values"] = apply_values_btn

    if include_back:
        back_btn = PushButton(text=str(back_text))
        _wire_button(back_btn, on_back)
        widgets.append(back_btn)
        out["back"] = back_btn

    if include_reset:
        reset_btn = PushButton(text=str(reset_text))
        _wire_button(reset_btn, on_reset)
        widgets.append(reset_btn)
        out["reset"] = reset_btn

    if include_abort:
        abort_btn = PushButton(text=str(abort_text))
        _wire_button(abort_btn, on_abort)
        widgets.append(abort_btn)
        out["abort"] = abort_btn

    dock = Container(widgets=widgets, layout="vertical")
    viewer.window.add_dock_widget(dock, area=str(dock_area))
    return out


def run_napari_preview_action(
    viewer: Any,
    controls_widget: Any,
    *,
    require_confirm: bool = False,
    on_update: Optional[Callable[[], None]] = None,
    on_reset: Optional[Callable[[], None]] = None,
    on_confirm: Optional[Callable[[], None]] = None,
    on_apply_values: Optional[Callable[[], None]] = None,
    on_back: Optional[Callable[[], None]] = None,
    on_abort: Optional[Callable[[], None]] = None,
    include_update: bool = True,
    include_confirm: bool = True,
    include_apply_values: bool = True,
    include_back: bool = True,
    include_reset: bool = True,
    include_abort: bool = True,
    update_text: str = "Preview/Recalculate",
    confirm_text: str = "Confirm Current View",
    apply_values_text: str = "continue and apply current values without preview",
    back_text: str = "Back",
    reset_text: str = "Reset Parameters",
    abort_text: str = "Abort",
    dock_area: str = "right",
) -> str:
    """
    Centralized napari preview runner for "button-only apply" previews.

    - Wires the unified 6-button dock via add_napari_controls_dock
    - Runs napari event loop
    - Returns action in {"confirm","apply_values","back","abort","closed"}

    "confirm": user accepted the last previewed result (current view).
    "apply_values": user wants to apply current slider values even if not
                    previewed via Recalculate.

    Closing the viewer without pressing a button maps to:
      - "closed" if require_confirm=True
      - "confirm" if require_confirm=False
    """
    import napari

    nav: Dict[str, Optional[str]] = {"action": None}

    def _safe_call(cb: Optional[Callable[[], None]]) -> None:
        if cb is None:
            return
        cb()

    def _on_update() -> None:
        _safe_call(on_update)

    def _on_reset() -> None:
        _safe_call(on_reset)

    def _on_confirm() -> None:
        _safe_call(on_confirm)
        nav["action"] = "confirm"
        viewer.close()

    def _on_apply_values() -> None:
        _safe_call(on_apply_values)
        nav["action"] = "apply_values"
        viewer.close()

    def _on_back() -> None:
        _safe_call(on_back)
        nav["action"] = "back"
        viewer.close()

    def _on_abort() -> None:
        _safe_call(on_abort)
        nav["action"] = "abort"
        viewer.close()

    add_napari_controls_dock(
        viewer,
        controls_widget,
        on_update=_on_update if include_update else None,
        on_confirm=_on_confirm if include_confirm else None,
        on_apply_values=_on_apply_values if include_apply_values else None,
        on_back=_on_back if include_back else None,
        on_reset=_on_reset if include_reset else None,
        on_abort=_on_abort if include_abort else None,
        include_update=bool(include_update),
        include_confirm=bool(include_confirm),
        include_apply_values=bool(include_apply_values),
        include_back=bool(include_back),
        include_reset=bool(include_reset),
        include_abort=bool(include_abort),
        update_text=str(update_text),
        confirm_text=str(confirm_text),
        apply_values_text=str(apply_values_text),
        back_text=str(back_text),
        reset_text=str(reset_text),
        abort_text=str(abort_text),
        dock_area=str(dock_area),
    )

    # Disable napari's background status-checker thread for the duration of the
    # event loop. In napari 0.6.x that thread reads a layer's size/_indices_view
    # off the main thread while a point delete is mutating them, raising an
    # IndexError in Points._view_size; napari catches it but surfaces the
    # traceback as a notification on every hover after a seed delete. The
    # appearance.update_status_based_on_layer setting is the supported switch:
    # assigning False terminates the thread and stops it restarting. The only
    # cost is the hover status-bar readout. The prior value is restored so the
    # user's global setting is left unchanged, and the import is guarded so a
    # future napari that renames the setting falls back to a plain run.
    _status_setting = _prev_status = None
    try:
        from napari.settings import get_settings
        _status_setting = get_settings().appearance
        _prev_status = _status_setting.update_status_based_on_layer
        _status_setting.update_status_based_on_layer = False
    except Exception:
        _status_setting = None

    try:
        napari.run()
    finally:
        if _status_setting is not None:
            _status_setting.update_status_based_on_layer = _prev_status

    action = nav["action"]
    if action is None:
        action = "confirm" if not require_confirm else "closed"
    return str(action)


def preview_images_napari(
    images: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    *,
    require_confirm: bool = False,
    return_action: bool = False,
    include_confirm: bool = True,
    include_back: bool = False,
    include_abort: bool = True,
    confirm_text: str = "Confirm and Continue",
    back_text: str = "Back",
    abort_text: str = "Abort",
    dock_area: str = "right",
) -> Optional[str]:
    """
    Small helper to preview one or more images in napari.

    Images are downsampled (max side 2048 px) for responsiveness.

    Behavior
    --------
      - Always uses the unified 5-button dock:
          Preview/Recalculate, Confirm and Continue, Back, Reset Parameters, Abort
      - Returns an action string in {"confirm", "back", "abort", "closed"} if return_action=True.

    Parameters
    ----------
    require_confirm:
        If True, closing the viewer without pressing Confirm returns "closed".
        If False, closing the viewer maps to "confirm".
    return_action:
        If True, return one of {"confirm","back","abort","closed"}.
        If False, preserve legacy behavior and return None.
    include_confirm / include_back / include_abort:
        Still supported for call-site compatibility, but the dock always shows all five buttons.
        If a callback is not enabled, the button is wired as a no-op.
    confirm_text / back_text / abort_text:
        Button labels.
    dock_area:
        Napari dock area; usually "right".
    """
    if not images:
        return ("confirm" if not require_confirm else "closed") if return_action else None

    import napari
    from magicgui.widgets import Label

    thumbs = [downsample_to_max_side(np.asarray(im), 2048) for im in images]
    viewer = napari.Viewer()

    for idx, im in enumerate(thumbs):
        name = str(titles[idx]) if titles and idx < len(titles) else f"Layer_{idx}"
        if im.ndim == 2:
            viewer.add_image(im, name=name, colormap="gray")
        elif im.ndim == 3 and im.shape[-1] in (3, 4):
            viewer.add_image(im, name=name, rgb=True)
        else:
            viewer.add_image(im, name=name)

    def on_update() -> None:
        # No parameters to recompute here; treat as a "refresh" button for consistency.
        try:
            viewer.reset_view()
        except Exception:
            pass

    def on_reset() -> None:
        # No parameters to reset; treat as a view reset.
        try:
            viewer.reset_view()
        except Exception:
            pass

    # Minimal "controls" widget so we can reuse the shared dock builder.
    controls_stub = Label(value="")

    action = run_napari_preview_action(
        viewer,
        controls_stub,
        require_confirm=bool(require_confirm),
        on_update=on_update,
        on_reset=on_reset,
        on_confirm=None,  # no state to apply
        on_apply_values=None,
        on_back=None,     # no state to apply
        on_abort=None,    # no state to apply
        include_update=True,
        include_confirm=True,
        include_apply_values=False,  # no sliders in image preview
        include_back=True,
        include_reset=True,
        include_abort=True,
        update_text="Preview/Recalculate",
        confirm_text=str(confirm_text),
        back_text=str(back_text),
        reset_text="Reset Parameters",
        abort_text=str(abort_text),
        dock_area=str(dock_area),
    )

    if not return_action:
        return None

    # Preserve call-site compatibility: disable actions by mapping to no-op wiring only.
    # Note: the UI still shows all buttons (by design), but callers can still request
    # whether a back/abort action is meaningfully used.
    if (action == "back") and (not include_back):
        return "confirm"
    if (action == "abort") and (not include_abort):
        return "confirm"
    if (action == "confirm") and (not include_confirm):
        return "confirm"

    return action
