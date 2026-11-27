package shadow

import "time"

var _ = time.Second

// The import is shadowed by a local inside this function.
func Shadowed() {
	time := napper{}
	time.Sleep()
}
