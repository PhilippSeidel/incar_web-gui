"""Small UI building blocks shared by the GUI pages (styling lives in static/theme.css)."""
from contextlib import contextmanager

from nicegui import ui

PILL_STATES = {'positive': 'ok', 'warning': 'warn', 'negative': 'bad', 'grey': 'idle'}  # Quasar colour -> dot


@contextmanager
def card(title: str, icon: str):
    with ui.card().classes('w-full glass gap-3').props('flat') as element:
        heading(title, icon)
        yield element


def pill(icon: str) -> tuple[ui.element, ui.label]:
    """Status pill: glowing dot (state via data-state, see theme.css), icon and text."""
    with ui.element('div').classes('pill') as element:
        ui.element('span').classes('dot')
        ui.icon(icon)
        text = ui.label()
    return element, text


def set_pill(element: ui.element, text: ui.label, label: str, state: str) -> None:
    text.set_text(label)
    set_props(element, f'data-state={state}')


def set_props(element: ui.element, props: str) -> None:
    """element.props(...) that skips unchanged values: props() and style() always push to the browser."""
    if getattr(element, '_last_props', None) != props:
        element._last_props = props
        element.props(props)


def set_style(element: ui.element, style: str) -> None:
    if getattr(element, '_last_style', None) != style:
        element._last_style = style
        element.style(style)


def icon_button(icon: str, label: str, on_click) -> ui.button:
    return ui.button(icon=icon, on_click=on_click).props(f'flat round dense aria-label="{label}"').tooltip(label)


def heading(title: str, icon: str) -> None:
    with ui.row().classes('items-center gap-3 no-wrap'):
        with ui.element('div').classes('card-icon'):
            ui.icon(icon)
        ui.label(title).classes('card-title')
