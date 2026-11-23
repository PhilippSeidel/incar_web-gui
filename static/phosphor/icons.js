// Phosphor duotone icons for the whole GUI: the code names them like 'ph-robot' (the font needs 'ph-duotone ph-robot'),
// and Quasar's own Material icons (dropdown arrows, clear buttons, table pagination, ...) get Phosphor look-alikes.
(() => {
  const quasar = {
    arrow_drop_down: 'ph-caret-down', keyboard_arrow_down: 'ph-caret-down', expand_more: 'ph-caret-down',
    keyboard_arrow_up: 'ph-caret-up', expand_less: 'ph-caret-up', arrow_drop_up: 'ph-caret-up',
    chevron_left: 'ph-caret-left', chevron_right: 'ph-caret-right', keyboard_arrow_left: 'ph-caret-left',
    keyboard_arrow_right: 'ph-caret-right', first_page: 'ph-caret-double-left', last_page: 'ph-caret-double-right',
    arrow_upward: 'ph-arrow-up', arrow_downward: 'ph-arrow-down', cancel: 'ph-x-circle', close: 'ph-x', clear: 'ph-x',
    check: 'ph-check', warning: 'ph-warning', error: 'ph-warning-circle', info: 'ph-info', search: 'ph-magnifying-glass',
    visibility: 'ph-eye', visibility_off: 'ph-eye-slash', menu: 'ph-list', more_vert: 'ph-dots-three-vertical',
    remove: 'ph-minus', add: 'ph-plus-circle', event: 'ph-calendar-blank', access_time: 'ph-clock', refresh: 'ph-arrows-clockwise',
  };
  const map = (name) => {
    const icon = name.startsWith('ph-') ? name : quasar[name];
    return icon ? {cls: 'ph-duotone ' + icon} : undefined;
  };
  (function install() {  // NiceGUI mounts its Vue app asynchronously; Quasar re-renders icons once this is set
    if (typeof mounted_app === 'undefined' || !mounted_app?.$q) return setTimeout(install, 5);
    mounted_app.$q.iconMapFn = map;
  })();
})();
