package tray

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"os"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/godbus/dbus/v5"
)

// Linux backend: a StatusNotifierItem (the tray icon) and a com.canonical.dbusmenu menu,
// exported on a private connection to the session bus. Closing the connection drops the bus
// name, and the watcher removes the icon.

const (
	sniPath       = dbus.ObjectPath("/StatusNotifierItem")
	menuPath      = dbus.ObjectPath("/MenuBar")
	sniIface      = "org.kde.StatusNotifierItem"
	menuIface     = "com.canonical.dbusmenu"
	propsIface    = "org.freedesktop.DBus.Properties"
	introIface    = "org.freedesktop.DBus.Introspectable"
	watcherName   = "org.kde.StatusNotifierWatcher"
	watcherPath   = dbus.ObjectPath("/StatusNotifierWatcher")
	notifyName    = "org.freedesktop.Notifications"
	notifyPath    = dbus.ObjectPath("/org/freedesktop/Notifications")
	ownerChanged  = "org.freedesktop.DBus.NameOwnerChanged"
	sniCategory   = "ApplicationStatus"
	sniStatus     = "Active"
	menuVersion   = uint32(3)
	menuRootID    = int32(0)
	notifyTimeout = int32(-1) // the server's default
)

var itemSeq atomic.Uint32

type sniBackend struct {
	conn  *dbus.Conn
	name  string // our bus name, org.kde.StatusNotifierItem-<pid>-<n>
	title string
	icon  []pixmap
	click func(string)
	sigs  chan *dbus.Signal
	done  chan struct{}

	mu       sync.Mutex // guards the fields below
	tooltip  string
	items    []Item
	revision uint32
}

// pixmap is one size of the icon: width, height, ARGB32 in network byte order.
type pixmap struct {
	W, H int32
	Data []byte
}

// toolTip is the SNI ToolTip property: icon name, icon pixmaps, title, description.
type toolTip struct {
	IconName string
	Icon     []pixmap
	Title    string
	Text     string
}

// layout is one dbusmenu node: id, properties, children (each a variant holding a layout).
type layout struct {
	ID       int32
	Props    map[string]dbus.Variant
	Children []dbus.Variant
}

type idProps struct {
	ID    int32
	Props map[string]dbus.Variant
}

type menuEvent struct {
	ID        int32
	EventID   string
	Data      dbus.Variant
	Timestamp uint32
}

func newBackend(spec Spec, click func(string)) (backend, error) {
	conn, err := dbus.SessionBusPrivate()
	if err != nil {
		return nil, fmt.Errorf("tray: no session bus (%v): %w", err, ErrNoTray)
	}
	fail := func(err error) (backend, error) {
		conn.Close()
		return nil, err
	}
	if err := conn.Auth(nil); err != nil {
		return fail(fmt.Errorf("tray: session bus auth (%v): %w", err, ErrNoTray))
	}
	if err := conn.Hello(); err != nil {
		return fail(fmt.Errorf("tray: session bus hello (%v): %w", err, ErrNoTray))
	}
	var has bool
	if err := conn.BusObject().Call("org.freedesktop.DBus.NameHasOwner", 0, watcherName).Store(&has); err != nil || !has {
		return fail(fmt.Errorf("tray: no %s on the session bus: %w", watcherName, ErrNoTray))
	}
	b := &sniBackend{
		conn:    conn,
		name:    fmt.Sprintf("org.kde.StatusNotifierItem-%d-%d", os.Getpid(), itemSeq.Add(1)),
		title:   spec.Title,
		icon:    pixmaps(spec.Icon.PNG),
		click:   click,
		sigs:    make(chan *dbus.Signal, 8),
		done:    make(chan struct{}),
		tooltip: spec.Tooltip,
		items:   cloneItems(spec.Items),
	}
	if err := b.export(); err != nil {
		return fail(fmt.Errorf("tray: export: %w", err))
	}
	reply, err := conn.RequestName(b.name, dbus.NameFlagDoNotQueue)
	if err != nil || reply != dbus.RequestNameReplyPrimaryOwner {
		return fail(fmt.Errorf("tray: request bus name %s: %v", b.name, err))
	}
	// Watch the watcher: when it restarts (the panel crashed, the desktop reloaded), register again.
	if err := conn.AddMatchSignal(dbus.WithMatchInterface("org.freedesktop.DBus"),
		dbus.WithMatchMember("NameOwnerChanged"), dbus.WithMatchArg(0, watcherName)); err != nil {
		return fail(fmt.Errorf("tray: watch %s: %w", watcherName, err))
	}
	conn.Signal(b.sigs)
	if err := b.register(); err != nil {
		return fail(fmt.Errorf("tray: register with %s (%v): %w", watcherName, err, ErrNoTray))
	}
	go b.watch()
	return b, nil
}

func (b *sniBackend) register() error {
	return b.conn.Object(watcherName, watcherPath).Call(watcherName+".RegisterStatusNotifierItem", 0, b.name).Err
}

// watch re-registers the item whenever a new watcher takes the name. It ends with the connection.
func (b *sniBackend) watch() {
	for {
		select {
		case <-b.done:
			return
		case s, ok := <-b.sigs:
			if !ok {
				return
			}
			if s.Name != ownerChanged || len(s.Body) != 3 {
				continue
			}
			if name, _ := s.Body[0].(string); name != watcherName {
				continue
			}
			if owner, _ := s.Body[2].(string); owner != "" {
				b.register() // best effort: the new watcher may refuse; the old icon is gone either way
			}
		}
	}
}

func (b *sniBackend) export() error {
	item := &sniObject{b}
	menu := &menuObject{b}
	for _, e := range []struct {
		v     any
		path  dbus.ObjectPath
		iface string
	}{
		{item, sniPath, sniIface},
		{sniProps{b}, sniPath, propsIface},
		{introspectable(sniIntrospection), sniPath, introIface},
		{menu, menuPath, menuIface},
		{menuProps{b}, menuPath, propsIface},
		{introspectable(menuIntrospection), menuPath, introIface},
	} {
		if err := b.conn.Export(e.v, e.path, e.iface); err != nil {
			return err
		}
	}
	return nil
}

func (b *sniBackend) setMenu(items []Item) {
	b.mu.Lock()
	b.items = items
	b.revision++
	rev := b.revision
	b.mu.Unlock()
	b.conn.Emit(menuPath, menuIface+".LayoutUpdated", rev, menuRootID)
}

func (b *sniBackend) setTooltip(s string) {
	b.mu.Lock()
	b.tooltip = s
	b.mu.Unlock()
	b.conn.Emit(sniPath, sniIface+".NewToolTip")
}

func (b *sniBackend) notify(title, body string) error {
	var id uint32
	err := b.conn.Object(notifyName, notifyPath).Call(notifyName+".Notify", 0,
		b.title, uint32(0), "", title, body, []string{}, map[string]dbus.Variant{}, notifyTimeout).Store(&id)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrNoNotify, err)
	}
	return nil
}

func (b *sniBackend) close() {
	close(b.done)
	b.conn.Close()
}

// sniObject is org.kde.StatusNotifierItem. ItemIsMenu is true, so hosts open the menu on a
// click and these methods have nothing to do.
type sniObject struct{ b *sniBackend }

func (sniObject) Activate(x, y int32) *dbus.Error          { return nil }
func (sniObject) SecondaryActivate(x, y int32) *dbus.Error { return nil }
func (sniObject) ContextMenu(x, y int32) *dbus.Error       { return nil }
func (sniObject) Scroll(delta int32, dir string) *dbus.Error {
	return nil
}

func (b *sniBackend) sniProperties() map[string]dbus.Variant {
	b.mu.Lock()
	tip := b.tooltip
	b.mu.Unlock()
	return map[string]dbus.Variant{
		"Category":            dbus.MakeVariant(sniCategory),
		"Id":                  dbus.MakeVariant(itemID(b.title)),
		"Title":               dbus.MakeVariant(b.title),
		"Status":              dbus.MakeVariant(sniStatus),
		"WindowId":            dbus.MakeVariant(int32(0)),
		"IconName":            dbus.MakeVariant(""),
		"IconPixmap":          dbus.MakeVariant(b.icon),
		"OverlayIconName":     dbus.MakeVariant(""),
		"OverlayIconPixmap":   dbus.MakeVariant([]pixmap{}),
		"AttentionIconName":   dbus.MakeVariant(""),
		"AttentionIconPixmap": dbus.MakeVariant([]pixmap{}),
		"AttentionMovieName":  dbus.MakeVariant(""),
		"ToolTip":             dbus.MakeVariant(toolTip{Icon: []pixmap{}, Title: tip}),
		"ItemIsMenu":          dbus.MakeVariant(true),
		"Menu":                dbus.MakeVariant(menuPath),
	}
}

func menuProperties() map[string]dbus.Variant {
	return map[string]dbus.Variant{
		"Version":       dbus.MakeVariant(menuVersion),
		"TextDirection": dbus.MakeVariant("ltr"),
		"Status":        dbus.MakeVariant("normal"),
		"IconThemePath": dbus.MakeVariant([]string{}),
	}
}

type sniProps struct{ b *sniBackend }

func (p sniProps) Get(iface, name string) (dbus.Variant, *dbus.Error) {
	return getProp(iface, sniIface, name, p.b.sniProperties())
}
func (p sniProps) GetAll(iface string) (map[string]dbus.Variant, *dbus.Error) {
	return getAll(iface, sniIface, p.b.sniProperties())
}
func (sniProps) Set(iface, name string, v dbus.Variant) *dbus.Error { return errReadOnly }

type menuProps struct{ b *sniBackend }

func (menuProps) Get(iface, name string) (dbus.Variant, *dbus.Error) {
	return getProp(iface, menuIface, name, menuProperties())
}
func (menuProps) GetAll(iface string) (map[string]dbus.Variant, *dbus.Error) {
	return getAll(iface, menuIface, menuProperties())
}
func (menuProps) Set(iface, name string, v dbus.Variant) *dbus.Error { return errReadOnly }

var (
	errReadOnly = dbus.NewError("org.freedesktop.DBus.Error.PropertyReadOnly", []any{"read only"})
	errNoProp   = dbus.NewError("org.freedesktop.DBus.Error.UnknownProperty", []any{"no such property"})
	errNoIface  = dbus.NewError("org.freedesktop.DBus.Error.UnknownInterface", []any{"no such interface"})
	errNoMenuID = dbus.NewError("org.freedesktop.DBus.Error.InvalidArgs", []any{"no such menu item"})
)

func getProp(iface, want, name string, all map[string]dbus.Variant) (dbus.Variant, *dbus.Error) {
	if iface != want {
		return dbus.Variant{}, errNoIface
	}
	v, ok := all[name]
	if !ok {
		return dbus.Variant{}, errNoProp
	}
	return v, nil
}

func getAll(iface, want string, all map[string]dbus.Variant) (map[string]dbus.Variant, *dbus.Error) {
	if iface != "" && iface != want {
		return nil, errNoIface
	}
	return all, nil
}

// menuObject is com.canonical.dbusmenu: a flat menu whose item i has id i+1 under the root 0.
type menuObject struct{ b *sniBackend }

func (m menuObject) GetLayout(parent, depth int32, names []string) (uint32, layout, *dbus.Error) {
	m.b.mu.Lock()
	items, rev := cloneItems(m.b.items), m.b.revision
	m.b.mu.Unlock()
	if parent != menuRootID {
		if parent < 1 || int(parent) > len(items) {
			return 0, layout{}, errNoMenuID
		}
		return rev, layout{ID: parent, Props: filterProps(itemProps(items[parent-1]), names), Children: []dbus.Variant{}}, nil
	}
	root := layout{ID: menuRootID, Props: filterProps(rootProps(), names), Children: []dbus.Variant{}}
	if depth != 0 {
		for i, it := range items {
			root.Children = append(root.Children, dbus.MakeVariant(layout{
				ID: int32(i + 1), Props: filterProps(itemProps(it), names), Children: []dbus.Variant{},
			}))
		}
	}
	return rev, root, nil
}

func (m menuObject) GetGroupProperties(ids []int32, names []string) ([]idProps, *dbus.Error) {
	m.b.mu.Lock()
	items := cloneItems(m.b.items)
	m.b.mu.Unlock()
	out := []idProps{}
	for _, id := range ids {
		switch {
		case id == menuRootID:
			out = append(out, idProps{id, filterProps(rootProps(), names)})
		case id >= 1 && int(id) <= len(items):
			out = append(out, idProps{id, filterProps(itemProps(items[id-1]), names)})
		}
	}
	return out, nil
}

func (m menuObject) GetProperty(id int32, name string) (dbus.Variant, *dbus.Error) {
	m.b.mu.Lock()
	items := cloneItems(m.b.items)
	m.b.mu.Unlock()
	var props map[string]dbus.Variant
	switch {
	case id == menuRootID:
		props = rootProps()
	case id >= 1 && int(id) <= len(items):
		props = itemProps(items[id-1])
	default:
		return dbus.Variant{}, errNoMenuID
	}
	v, ok := props[name]
	if !ok {
		return dbus.Variant{}, errNoProp
	}
	return v, nil
}

func (m menuObject) Event(id int32, eventID string, data dbus.Variant, ts uint32) *dbus.Error {
	if eventID != "clicked" {
		return nil
	}
	m.b.mu.Lock()
	var it Item
	ok := id >= 1 && int(id) <= len(m.b.items)
	if ok {
		it = m.b.items[id-1]
	}
	m.b.mu.Unlock()
	if !ok {
		return errNoMenuID
	}
	if !it.Separator && !it.Disabled && it.ID != "" {
		m.b.click(it.ID)
	}
	return nil
}

func (m menuObject) EventGroup(events []menuEvent) ([]int32, *dbus.Error) {
	bad := []int32{}
	for _, e := range events {
		if err := m.Event(e.ID, e.EventID, e.Data, e.Timestamp); err != nil {
			bad = append(bad, e.ID)
		}
	}
	return bad, nil
}

func (menuObject) AboutToShow(id int32) (bool, *dbus.Error) { return false, nil }

func (menuObject) AboutToShowGroup(ids []int32) ([]int32, []int32, *dbus.Error) {
	return []int32{}, []int32{}, nil
}

func rootProps() map[string]dbus.Variant {
	return map[string]dbus.Variant{"children-display": dbus.MakeVariant("submenu")}
}

func itemProps(it Item) map[string]dbus.Variant {
	if it.Separator {
		return map[string]dbus.Variant{"type": dbus.MakeVariant("separator")}
	}
	p := map[string]dbus.Variant{
		"label":   dbus.MakeVariant(strings.ReplaceAll(it.Title, "_", "__")), // "_" marks a mnemonic
		"enabled": dbus.MakeVariant(!it.Disabled && it.ID != ""),
		"visible": dbus.MakeVariant(true),
	}
	if it.Checkable {
		state := int32(0)
		if it.Checked {
			state = 1
		}
		p["toggle-type"] = dbus.MakeVariant("checkmark")
		p["toggle-state"] = dbus.MakeVariant(state)
	}
	return p
}

// filterProps keeps the named properties; no names means all of them.
func filterProps(p map[string]dbus.Variant, names []string) map[string]dbus.Variant {
	if len(names) == 0 {
		return p
	}
	out := make(map[string]dbus.Variant, len(names))
	for _, n := range names {
		if v, ok := p[n]; ok {
			out[n] = v
		}
	}
	return out
}

// itemID makes the SNI Id from the title: lower case letters, digits and dashes.
func itemID(title string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(title) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	if b.Len() == 0 {
		return "tray"
	}
	return b.String()
}

// pixmaps decodes the PNG into the SNI's ARGB32 (network byte order) form. The host scales it.
func pixmaps(pngData []byte) []pixmap {
	if len(pngData) == 0 {
		return []pixmap{}
	}
	img, err := png.Decode(bytes.NewReader(pngData))
	if err != nil {
		return []pixmap{}
	}
	return []pixmap{argb(img)}
}

func argb(img image.Image) pixmap {
	r := img.Bounds()
	nrgba := image.NewNRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(nrgba, nrgba.Bounds(), img, r.Min, draw.Src)
	out := make([]byte, 0, 4*r.Dx()*r.Dy())
	for i := 0; i < len(nrgba.Pix); i += 4 {
		p := nrgba.Pix[i : i+4]
		out = append(out, p[3], p[0], p[1], p[2])
	}
	return pixmap{W: int32(r.Dx()), H: int32(r.Dy()), Data: out}
}

type introspectable string

func (s introspectable) Introspect() (string, *dbus.Error) { return string(s), nil }

const introHead = `<!DOCTYPE node PUBLIC "-//freedesktop//DTD D-BUS Object Introspection 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd">
<node>
 <interface name="org.freedesktop.DBus.Introspectable"><method name="Introspect"><arg name="xml" type="s" direction="out"/></method></interface>
 <interface name="org.freedesktop.DBus.Properties">
  <method name="Get"><arg type="s" direction="in"/><arg type="s" direction="in"/><arg type="v" direction="out"/></method>
  <method name="GetAll"><arg type="s" direction="in"/><arg type="a{sv}" direction="out"/></method>
  <method name="Set"><arg type="s" direction="in"/><arg type="s" direction="in"/><arg type="v" direction="in"/></method>
  <signal name="PropertiesChanged"><arg type="s"/><arg type="a{sv}"/><arg type="as"/></signal>
 </interface>
`

const sniIntrospection = introHead + ` <interface name="org.kde.StatusNotifierItem">
  <property name="Category" type="s" access="read"/>
  <property name="Id" type="s" access="read"/>
  <property name="Title" type="s" access="read"/>
  <property name="Status" type="s" access="read"/>
  <property name="WindowId" type="i" access="read"/>
  <property name="IconName" type="s" access="read"/>
  <property name="IconPixmap" type="a(iiay)" access="read"/>
  <property name="OverlayIconName" type="s" access="read"/>
  <property name="OverlayIconPixmap" type="a(iiay)" access="read"/>
  <property name="AttentionIconName" type="s" access="read"/>
  <property name="AttentionIconPixmap" type="a(iiay)" access="read"/>
  <property name="AttentionMovieName" type="s" access="read"/>
  <property name="ToolTip" type="(sa(iiay)ss)" access="read"/>
  <property name="ItemIsMenu" type="b" access="read"/>
  <property name="Menu" type="o" access="read"/>
  <method name="ContextMenu"><arg name="x" type="i" direction="in"/><arg name="y" type="i" direction="in"/></method>
  <method name="Activate"><arg name="x" type="i" direction="in"/><arg name="y" type="i" direction="in"/></method>
  <method name="SecondaryActivate"><arg name="x" type="i" direction="in"/><arg name="y" type="i" direction="in"/></method>
  <method name="Scroll"><arg name="delta" type="i" direction="in"/><arg name="orientation" type="s" direction="in"/></method>
  <signal name="NewTitle"/>
  <signal name="NewIcon"/>
  <signal name="NewAttentionIcon"/>
  <signal name="NewOverlayIcon"/>
  <signal name="NewToolTip"/>
  <signal name="NewStatus"><arg name="status" type="s"/></signal>
 </interface>
</node>`

const menuIntrospection = introHead + ` <interface name="com.canonical.dbusmenu">
  <property name="Version" type="u" access="read"/>
  <property name="TextDirection" type="s" access="read"/>
  <property name="Status" type="s" access="read"/>
  <property name="IconThemePath" type="as" access="read"/>
  <method name="GetLayout"><arg type="i" name="parentId" direction="in"/><arg type="i" name="recursionDepth" direction="in"/><arg type="as" name="propertyNames" direction="in"/><arg type="u" name="revision" direction="out"/><arg type="(ia{sv}av)" name="layout" direction="out"/></method>
  <method name="GetGroupProperties"><arg type="ai" name="ids" direction="in"/><arg type="as" name="propertyNames" direction="in"/><arg type="a(ia{sv})" name="properties" direction="out"/></method>
  <method name="GetProperty"><arg type="i" name="id" direction="in"/><arg type="s" name="name" direction="in"/><arg type="v" name="value" direction="out"/></method>
  <method name="Event"><arg type="i" name="id" direction="in"/><arg type="s" name="eventId" direction="in"/><arg type="v" name="data" direction="in"/><arg type="u" name="timestamp" direction="in"/></method>
  <method name="EventGroup"><arg type="a(isvu)" name="events" direction="in"/><arg type="ai" name="idErrors" direction="out"/></method>
  <method name="AboutToShow"><arg type="i" name="id" direction="in"/><arg type="b" name="needUpdate" direction="out"/></method>
  <method name="AboutToShowGroup"><arg type="ai" name="ids" direction="in"/><arg type="ai" name="updatesNeeded" direction="out"/><arg type="ai" name="idErrors" direction="out"/></method>
  <signal name="ItemsPropertiesUpdated"><arg type="a(ia{sv})" name="updatedProps"/><arg type="a(ias)" name="removedProps"/></signal>
  <signal name="LayoutUpdated"><arg type="u" name="revision"/><arg type="i" name="parent"/></signal>
  <signal name="ItemActivationRequested"><arg type="i" name="id"/><arg type="u" name="timestamp"/></signal>
 </interface>
</node>`
