//go:build darwin

package tray

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// Helpers for talking to the Objective-C runtime and to the main thread, without cgo.

type nsSize struct{ W, H float64 }
type nsPoint struct{ X, Y float64 }

var (
	dispatchAsyncF func(queue, ctx, fn uintptr)
	pthreadMainNP  func() int32
	mainQueue      uintptr
	mainCallback   uintptr

	// mainMu guards loopRunning and mainQ.
	mainMu      sync.Mutex
	loopRunning bool
	mainQ       []func()

	loadOnce sync.Once
	loadErr  error
)

// loadRuntime opens the frameworks and the libSystem entry points once.
func loadRuntime() error {
	loadOnce.Do(func() {
		loadErr = func() error {
			for _, p := range []string{
				"/System/Library/Frameworks/Foundation.framework/Foundation",
				"/System/Library/Frameworks/AppKit.framework/AppKit",
			} {
				if _, err := purego.Dlopen(p, purego.RTLD_GLOBAL|purego.RTLD_NOW); err != nil {
					return fmt.Errorf("tray: load %s: %w", p, err)
				}
			}
			sys, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_GLOBAL|purego.RTLD_NOW)
			if err != nil {
				return fmt.Errorf("tray: load libSystem: %w", err)
			}
			q, err := purego.Dlsym(sys, "_dispatch_main_q")
			if err != nil {
				return fmt.Errorf("tray: find the main queue: %w", err)
			}
			mainQueue = q
			purego.RegisterLibFunc(&dispatchAsyncF, sys, "dispatch_async_f")
			purego.RegisterLibFunc(&pthreadMainNP, sys, "pthread_main_np")
			mainCallback = purego.NewCallback(func(ctx uintptr) {
				mainMu.Lock()
				var f func()
				if len(mainQ) > 0 {
					f = mainQ[0]
					mainQ = mainQ[1:]
				}
				mainMu.Unlock()
				if f != nil {
					f()
				}
			})
			return nil
		}()
	})
	return loadErr
}

func isMainThread() bool { return pthreadMainNP() != 0 }

// enqueueMain schedules f on the main thread's queue; false when the loop is not running.
func enqueueMain(f func()) bool {
	mainMu.Lock()
	if !loopRunning {
		mainMu.Unlock()
		return false
	}
	mainQ = append(mainQ, f)
	mainMu.Unlock()
	dispatchAsyncF(mainQueue, 0, mainCallback)
	return true
}

// onMain runs f on the main thread and waits for it. It runs f directly when already on the
// main thread (a dispatch_sync there would deadlock). It reports false when the loop is gone.
func onMain(f func()) bool {
	if isMainThread() {
		f()
		return true
	}
	done := make(chan struct{})
	if !enqueueMain(func() {
		defer close(done)
		f()
	}) {
		return false
	}
	<-done
	return true
}

// drainMain runs whatever is still queued once the loop has ended (main thread only), so no
// waiter hangs.
func drainMain() {
	for {
		mainMu.Lock()
		if len(mainQ) == 0 {
			mainMu.Unlock()
			return
		}
		f := mainQ[0]
		mainQ = mainQ[1:]
		mainMu.Unlock()
		f()
	}
}

func sel(name string) objc.SEL { return objc.RegisterName(name) }

func class(name string) objc.ID { return objc.ID(objc.GetClass(name)) }

// nsString makes an autoreleased NSString from s (cut at a NUL byte).
func nsString(s string) objc.ID {
	b := append([]byte(s), 0)
	id := class("NSString").Send(sel("stringWithUTF8String:"), &b[0])
	runtime.KeepAlive(b)
	return id
}

// nsImage makes an owned (retained) NSImage from PNG bytes, or 0.
func nsImage(png []byte, template bool) objc.ID {
	if len(png) == 0 {
		return 0
	}
	data := class("NSData").Send(sel("dataWithBytes:length:"), &png[0], len(png))
	runtime.KeepAlive(png)
	if data == 0 {
		return 0
	}
	img := class("NSImage").Send(sel("alloc")).Send(sel("initWithData:"), data)
	if img == 0 {
		return 0
	}
	img.Send(sel("setSize:"), nsSize{18, 18})
	if template {
		img.Send(sel("setTemplate:"), true)
	}
	return img
}
