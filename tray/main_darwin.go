//go:build darwin

package tray

import (
	"runtime"

	"github.com/ebitengine/purego/objc"
)

// The main goroutine must stay on the process's main thread: AppKit runs its loop there only.
func init() { runtime.LockOSThread() }

// Main keeps the main thread, runs run in a goroutine and drives the NSApplication loop until
// run returns. Call it from main().
func Main(run func()) {
	if err := loadRuntime(); err != nil {
		// No UI loop possible; the tray reports ErrNoTray and the program still runs.
		run()
		return
	}
	app := class("NSApplication").Send(sel("sharedApplication"))
	// NSApplicationActivationPolicyAccessory: no Dock icon.
	app.Send(sel("setActivationPolicy:"), 1)

	mainMu.Lock()
	loopRunning = true
	mainMu.Unlock()

	go func() {
		defer func() {
			// Ask the loop to end: stop: only takes effect after an event, so post a dummy one.
			enqueueMain(func() { stopLoop(app) })
		}()
		run()
	}()

	app.Send(sel("run"))

	mainMu.Lock()
	loopRunning = false
	mainMu.Unlock()
	drainMain()
}

func stopLoop(app objc.ID) {
	app.Send(sel("stop:"), objc.ID(0))
	// NSEventTypeApplicationDefined = 15
	ev := class("NSEvent").Send(
		sel("otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:"),
		15, nsPoint{0, 0}, 0, float64(0), 0, objc.ID(0), 0, 0, 0)
	app.Send(sel("postEvent:atStart:"), ev, true)
}
