package run

import (
	"errors"
	"os/exec"
	"syscall"
)

// ErrAppControl: the operating system's application control refused to run downloaded code. On Windows this is
// Smart App Control (or a WDAC policy): it blocks unsigned programs and DLLs that have no reputation yet, which a
// llama.cpp build published in the last day or so usually lacks. Every backend from the same release is blocked the
// same way, so the install stops at the first one instead of downloading the others.
var ErrAppControl = errors.New("Windows Smart App Control blocked the model server: this llama.cpp build is not code-signed and is too new to have a reputation yet")

const (
	statusIntegrityPolicyViolation = 0xC0E90002 // NTSTATUS: a DLL the program needs was refused at load
	errorIntegrityPolicyViolation  = 4551       // Win32: the program itself was refused at start
)

// appControlBlocked reports whether err (from starting or running a program) is an application-control refusal.
func appControlBlocked(err error) bool {
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return blockedExitCode(ee.ExitCode())
	}
	var en syscall.Errno
	if errors.As(err, &en) {
		return uint32(en) == errorIntegrityPolicyViolation
	}
	return false
}

func blockedExitCode(code int) bool { return uint32(code) == statusIntegrityPolicyViolation }
