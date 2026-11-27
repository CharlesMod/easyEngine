package openfile

import (
	"fmt"
	"net/url"
)

// OpenURL opens an http or https address in the default browser (xdg-open, open, ShellExecute).
// Like Open it waits for the opener, so call it from a goroutine when the caller must not wait.
func OpenURL(addr string) error {
	if err := checkAddr(addr); err != nil {
		return err
	}
	return open(addr)
}

// checkAddr: addr must be an http or https address with a host.
func checkAddr(addr string) error {
	u, err := url.Parse(addr)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("openfile: %q is not an http(s) address", addr)
	}
	return nil
}
