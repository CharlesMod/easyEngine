//go:build !darwin

package tray

// Main runs run and returns when it returns. On macOS it also runs the UI loop on the main
// thread (see the package documentation); here there is nothing else to do.
func Main(run func()) { run() }
